#include <algorithm>
#include <array>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <parquet/arrow/reader.h>
#include <sstream>
#include <thread>
#include <vector>

struct StateProbs {
	std::array<double, 21> P_bid;
	std::array<double, 21> P_ask;
	std::array<double, 21> P_rest;
};

StateProbs load_probs_csv(const std::string &path) {
	StateProbs probs{};
	std::ifstream file(path);
	std::string line;
	std::getline(file, line); // skip header

	while (std::getline(file, line)) {
		std::istringstream ss(line);
		std::string token;
		std::getline(ss, token, ','); // imb (skip)
		std::getline(ss, token, ',');
		int idx = std::stoi(token); // imb_bin
		std::getline(ss, token, ',');
		probs.P_ask[idx] = std::stod(token);
		std::getline(ss, token, ',');
		probs.P_bid[idx] = std::stod(token);
		std::getline(ss, token, ',');
		probs.P_rest[idx] = std::stod(token);
	}
	return probs;
}

struct LikelihoodData {
	std::vector<double> ts;
	std::vector<bool> day_starts;
	std::vector<int32_t> imb_bin;
	std::vector<double> trade_sign;
	std::vector<double> trade_size;
	std::vector<bool> is_bid_trade;
	std::vector<bool> is_ask_trade;
};

LikelihoodData load_parquet(const std::string &path) {
	auto infile = arrow::io::ReadableFile::Open(path).ValueOrDie();
	auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool()).ValueOrDie();

	std::shared_ptr<arrow::Table> table;
	(void)reader->ReadTable(&table);

	int64_t N = table->num_rows();
	LikelihoodData data;
	data.ts.resize(N);
	data.day_starts.resize(N);
	data.imb_bin.resize(N);
	data.trade_sign.resize(N);
	data.trade_size.resize(N);
	data.is_bid_trade.resize(N);
	data.is_ask_trade.resize(N);

	auto get_double = [&](const std::string &name) -> std::shared_ptr<arrow::DoubleArray> {
		return std::static_pointer_cast<arrow::DoubleArray>(table->GetColumnByName(name)->chunk(0));
	};
	auto get_int64 = [&](const std::string &name) -> std::shared_ptr<arrow::Int64Array> {
		return std::static_pointer_cast<arrow::Int64Array>(table->GetColumnByName(name)->chunk(0));
	};
	auto get_int32 = [&](const std::string &name) -> std::shared_ptr<arrow::Int32Array> {
		return std::static_pointer_cast<arrow::Int32Array>(table->GetColumnByName(name)->chunk(0));
	};

	auto ts_col = get_double("ts");
	auto ds_col = get_double("day_starts");
	auto imb_col = get_int32("imb_bin");
	auto sign_col = get_double("trade_sign");
	auto size_col = get_double("trade_size");
	auto bid_col = get_int64("is_bid_trade");
	auto ask_col = get_int64("is_ask_trade");

	for (int64_t i = 0; i < N; i++) {
		data.ts[i] = ts_col->Value(i);
		data.day_starts[i] = ds_col->Value(i) != 0.0;
		data.imb_bin[i] = imb_col->Value(i);
		data.trade_sign[i] = sign_col->Value(i);
		data.trade_size[i] = size_col->Value(i);
		data.is_bid_trade[i] = bid_col->Value(i) != 0;
		data.is_ask_trade[i] = ask_col->Value(i) != 0;
	}
	return data;
}

struct Kernel {
	std::vector<double> kappa;
	std::vector<double> weights;
};

Kernel init_kernel(double beta, double tau, size_t K = 12) {
	Kernel kernel;
	kernel.kappa.resize(K);
	kernel.weights.resize(K);

	for (size_t k = 0; k < K; k++) {
		double frac = static_cast<double>(k) / (K - 1);
		double half_life_sec = std::pow(10.0, -2.0 + 5.0 * frac);
		kernel.kappa[k] = std::log(2.0) / half_life_sec;
	}

	std::array<double, 100> t;
	std::array<double, 100> Y;
	std::vector<double> X(100 * K);
	for (size_t i = 0; i < 100; i++) {
		double frac = static_cast<double>(i) / 99;
		t[i] = std::pow(10.0, -2.0 + 5.0 * frac);
		Y[i] = std::pow(1 + t[i] / tau, -beta);
		for (size_t j = 0; j < K; j++) {
			X[j + i * K] = std::exp(-kernel.kappa[j] * t[i]);
		}
	}

	std::vector<double> XtX(K * K, 0.0);
	std::vector<double> XtY(K, 0.0);
	for (size_t i = 0; i < K; i++) {
		for (size_t j = 0; j < K; j++) {
			for (size_t l = 0; l < 100; l++) {
				XtX[i * K + j] += X[l * K + i] * X[l * K + j];
			}
		}
	}
	for (size_t i = 0; i < K; i++) {
		for (size_t j = 0; j < 100; j++) {
			XtY[i] += X[j * K + i] * Y[j];
		}
	}

	// Cholesky: L L^T = XtX
	std::vector<double> L(K * K, 0.0);
	for (size_t i = 0; i < K; i++) {
		for (size_t j = 0; j <= i; j++) {
			double s = XtX[i * K + j];
			for (size_t p = 0; p < j; p++)
				s -= L[i * K + p] * L[j * K + p];
			if (i == j)
				L[i * K + j] = std::sqrt(std::max(s, 1e-15));
			else
				L[i * K + j] = s / L[j * K + j];
		}
	}

	// Forward sub: L y = XtY
	std::vector<double> y(K);
	for (size_t i = 0; i < K; i++) {
		y[i] = XtY[i];
		for (size_t p = 0; p < i; p++)
			y[i] -= L[i * K + p] * y[p];
		y[i] /= L[i * K + i];
	}

	// Back sub: L^T w = y
	for (int i = K - 1; i >= 0; i--) {
		kernel.weights[i] = y[i];
		for (size_t p = i + 1; p < K; p++)
			kernel.weights[i] -= L[p * K + i] * kernel.weights[p];
		kernel.weights[i] /= L[i * K + i];
	}

	// Clamp negatives
	for (size_t k = 0; k < K; k++)
		kernel.weights[k] = std::max(kernel.weights[k], 0.0);

	return kernel;
}

std::vector<double> compute_phi(const LikelihoodData &data, const Kernel &kernel) {
	size_t N = data.ts.size();
	size_t K = kernel.weights.size();
	std::vector<double> phi(N, 0.0);
	std::vector<double> phi_components(K, 0.0);

	for (size_t i = 1; i < N; i++) {
		if (data.day_starts[i]) {
			for (size_t k = 0; k < K; k++) {
				phi_components[k] = 0.0;
			}
		} else {
			double dt = data.ts[i] - data.ts[i - 1];
			for (size_t k = 0; k < K; k++) {
				phi_components[k] *= std::exp(-dt * kernel.kappa[k]);
				phi[i] += kernel.weights[k] * phi_components[k];
			}
		}

		if (data.trade_sign[i] != 0.0) {
			for (size_t k = 0; k < K; k++) {
				phi_components[k] += data.trade_sign[i] * std::sqrt(data.trade_size[i]);
			}
		}
	}
	return phi;
}

double nll(const StateProbs &probs, const LikelihoodData &data, const std::vector<double> &phi,
		   double m) {
	size_t N = phi.size();
	double nll = 0.0;
	for (size_t i = 0; i < N; i++) {
		double b = phi[i] * m;
		int32_t ib = data.imb_bin[i];
		double Z;
		if (b > 0) {
			Z = probs.P_bid[ib] * std::exp(b) + probs.P_ask[ib] + probs.P_rest[ib];
		} else {
			Z = probs.P_bid[ib] + probs.P_ask[ib] * std::exp(-b) + probs.P_rest[ib];
		}
		nll += std::log(Z);
		if (data.is_bid_trade[i] && b > 0) {
			nll -= b;
		} else if (data.is_ask_trade[i] && b < 0) {
			nll += b;
		}
	}
	return nll;
}

int main(int argc, char *argv[]) {
	std::string ticker;
	for (int i = 1; i < argc; i++) {
		if (std::string(argv[i]) == "--ticker" && i + 1 < argc) {
			ticker = argv[i + 1];
			i++;
		}
	}
	if (ticker.empty()) {
		std::cerr << "Usage: " << argv[0] << " --ticker <TICKER>\n";
		return 1;
	}

	std::string base_path = "/home/labcmap/saad.souilmi/dev_cpp/qr/data/" + ticker + "/impact_calibration/";
	std::string prob_path = base_path + "probabilities.csv";
	std::string data_path = base_path + "data.parquet";

	StateProbs probs = load_probs_csv(prob_path);
	LikelihoodData data = load_parquet(data_path);

	std::printf("Loaded %zu events\n", data.ts.size());

	// Grid: tau logspaced 1e-2 to 1e2, beta linspaced 0.1 to 2.5
	const int N_tau = 50;
	const int N_beta = 50;
	std::vector<double> taus(N_tau);
	std::vector<double> betas(N_beta);
	for (int i = 0; i < N_tau; i++) {
		double frac = static_cast<double>(i) / (N_tau - 1);
		taus[i] = std::pow(10.0, -2.0 + frac * std::log10(300.0 / 0.01));
	}
	for (int i = 0; i < N_beta; i++) {
		double frac = static_cast<double>(i) / (N_beta - 1);
		betas[i] = 0.1 + 4.9 * frac;
	}

	int total_jobs = N_tau * N_beta;
	std::vector<double> best_m(total_jobs);
	std::vector<double> best_nll(total_jobs);

	std::atomic<int> job_idx{0};
	std::atomic<int> done_count{0};
	std::mutex print_mutex;

	auto t_start = std::chrono::steady_clock::now();

	auto worker = [&]() {
		while (true) {
			int idx = job_idx.fetch_add(1);
			if (idx >= total_jobs)
				break;

			int i_tau = idx / N_beta;
			int i_beta = idx % N_beta;
			double tau = taus[i_tau];
			double beta = betas[i_beta];

			Kernel kernel = init_kernel(beta, tau);
			std::vector<double> phi = compute_phi(data, kernel);

			// Golden section search for m in [0, 1]
			double a = 0.0, b_hi = 1.0;
			const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
			double c = b_hi - gr * (b_hi - a);
			double d = a + gr * (b_hi - a);
			double fc = nll(probs, data, phi, c);
			double fd = nll(probs, data, phi, d);

			for (int iter = 0; iter < 50; iter++) {
				if (fc < fd) {
					b_hi = d;
					d = c;
					fd = fc;
					c = b_hi - gr * (b_hi - a);
					fc = nll(probs, data, phi, c);
				} else {
					a = c;
					c = d;
					fc = fd;
					d = a + gr * (b_hi - a);
					fd = nll(probs, data, phi, d);
				}
			}

			double m_opt = (a + b_hi) / 2.0;
			double nll_opt = nll(probs, data, phi, m_opt);

			best_m[idx] = m_opt;
			best_nll[idx] = nll_opt;

			int completed = done_count.fetch_add(1) + 1;
			if (completed % 5 == 0 || completed == total_jobs) {
				auto now = std::chrono::steady_clock::now();
				double elapsed =
					std::chrono::duration<double>(now - t_start).count();
				double per_job = elapsed / completed;
				double remaining = per_job * (total_jobs - completed);
				std::lock_guard<std::mutex> lock(print_mutex);
				std::printf("[%3d/%d] tau=%.4f beta=%.3f => m=%.6f nll=%.2f "
							"(%.1fs elapsed, ~%.1fs left)\n",
							completed, total_jobs, tau, beta, m_opt, nll_opt,
							elapsed, remaining);
			}
		}
	};

	int n_threads = std::max(1u, std::thread::hardware_concurrency());
	std::printf("Starting grid search: %d jobs on %d threads\n", total_jobs, n_threads);

	std::vector<std::thread> threads;
	for (int t = 0; t < n_threads; t++) {
		threads.emplace_back(worker);
	}
	for (auto &t : threads) {
		t.join();
	}

	auto t_end = std::chrono::steady_clock::now();
	double total_time = std::chrono::duration<double>(t_end - t_start).count();
	std::printf("Grid search done in %.1fs\n", total_time);

	// Find global best
	int best_idx = 0;
	for (int i = 1; i < total_jobs; i++) {
		if (best_nll[i] < best_nll[best_idx])
			best_idx = i;
	}
	int best_i_tau = best_idx / N_beta;
	int best_i_beta = best_idx % N_beta;
	std::printf("Best: tau=%.6f beta=%.4f m=%.6f nll=%.2f\n", taus[best_i_tau],
				betas[best_i_beta], best_m[best_idx], best_nll[best_idx]);

	// Save results CSV
	std::string out_dir = "/home/labcmap/saad.souilmi/dev_cpp/qr/data/results/" + ticker + "/impact_calibration";
	std::filesystem::create_directories(out_dir);
	std::string out_path = out_dir + "/calibration_results.csv";
	std::ofstream out(out_path);
	out << "tau,beta,m,nll\n";
	for (int i = 0; i < total_jobs; i++) {
		int i_tau = i / N_beta;
		int i_beta = i % N_beta;
		out << taus[i_tau] << "," << betas[i_beta] << "," << best_m[i] << ","
			<< best_nll[i] << "\n";
	}
	out.close();
	std::printf("Results saved to %s\n", out_path.c_str());

	return 0;
}