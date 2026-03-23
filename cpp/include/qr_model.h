#pragma once
#include "orderbook.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

namespace qr {

// Key: (imb_bin, event_type, signed_queue) -> cumulative probability vector
struct SizeDistributions {
	static constexpr int NUM_IMB_BINS = 21;
	static constexpr int32_t MAX_SIZE = 50;

	// Spread=1: [imb_bin][event_type: 0=Add, 1=Cancel, 2=Trade][queue: 4 signed slots]
	std::array<std::array<std::array<std::vector<double>, 4>, 3>, NUM_IMB_BINS> cum_probs{};

	// Spread>=2: [imb_bin][0=Create_Bid, 1=Create_Ask]
	std::array<std::array<std::vector<double>, 2>, NUM_IMB_BINS> cum_probs_create{};

	SizeDistributions() = default;
	SizeDistributions(const std::string &csv_path);

	// queue_nbr is signed: -2, -1, 1, 2
	int32_t sample_size(int imb_bin, OrderType type, int queue_nbr, std::mt19937_64 &rng) const {
		const std::vector<double> *cp = nullptr;

		if (type == OrderType::CreateBid) {
			cp = &cum_probs_create[imb_bin][0];
		} else if (type == OrderType::CreateAsk) {
			cp = &cum_probs_create[imb_bin][1];
		} else {
			int type_idx = (type == OrderType::Add) ? 0 : (type == OrderType::Cancel) ? 1 : 2;
			cp = &cum_probs[imb_bin][type_idx][q4_idx(queue_nbr)];
		}

		if (!cp || cp->empty())
			return 1;

		double u = std::uniform_real_distribution<>(0.0, 1.0)(rng);
		auto it = std::lower_bound(cp->begin(), cp->end(), u);
		return static_cast<int32_t>(it - cp->begin()) + 1; // 1-based sizes
	}

  private:
	static int q4_idx(int q) {
		switch (q) {
		case -2:
			return 0;
		case -1:
			return 1;
		case 1:
			return 2;
		case 2:
			return 3;
		}
		return 1; // fallback
	}
};

struct Event {
	OrderType type;
	Side side;
	int queue_nbr; // -2, -1, 1, 2, 0
};

struct StateParams {
	std::vector<Event> events;
	std::vector<double> base_probs; // immutable do not modify

	std::vector<double> probs;
	std::vector<double> cum_probs;
	double total;
	double lambda;

	void bias(double b) {
		// Exponential bias: always positive, no clamping needed
		// bid trades *= exp(b), ask trades *= exp(-b)
		total = 0.0;
		for (size_t i = 0; i < events.size(); i++) {
			if (events[i].type == OrderType::Trade) {
				double factor = 1;
				if ((events[i].side == Side::Bid) && (b > 0)) {
					factor = std::exp(b);
				} else if ((events[i].side == Side::Ask) && (b < 0)) {
					factor = std::exp(-b);
				}
				probs[i] = base_probs[i] * factor;
			} else
				probs[i] = base_probs[i];
			total += probs[i];
		}
		cum_probs[0] = probs[0];
		for (size_t i = 1; i < probs.size(); i++)
			cum_probs[i] = cum_probs[i - 1] + probs[i];
	}

	const Event &sample_event(std::mt19937_64 &rng) const {
		double u = std::uniform_real_distribution<>(0, total)(rng);
		auto it = std::lower_bound(cum_probs.begin(), cum_probs.end(), u);
		return events[it - cum_probs.begin()];
	}

	long long sample_dt(std::mt19937_64 &rng) const {
		double dt = std::exponential_distribution<>(lambda)(rng);
		return static_cast<long long>(std::ceil(dt));
	}
};

struct QRParams {
	// 21 imbalance bins: -1.0, -0.9, ..., 0.0, ..., 0.9, 1.0
	// 2 spread states: spread=1 (index 0), spread>=2 (index 1)
	static constexpr int NUM_IMB_BINS = 21;
	static constexpr int NUM_TOTAL_LVL_BINS = 5;

	// 2D state params: [imb_bin][spread] - used for time sampling and default event sampling
	std::array<std::array<StateParams, 2>, NUM_IMB_BINS> state_params;

	// 3D state params: [imb_bin][spread][total_lvl] - used for event sampling when
	// use_total_lvl=true
	std::array<std::array<std::array<StateParams, NUM_TOTAL_LVL_BINS>, 2>, NUM_IMB_BINS>
		event_state_params_3d;

	// Quantile edges for total_lvl binning (6 values for 5 bins)
	std::array<double, NUM_TOTAL_LVL_BINS + 1> total_lvl_edges{};

	// Flag to enable 3D event sampling
	bool use_total_lvl = false;

	QRParams(const std::string &data_path);

	// Load 3D event probabilities from CSV
	void load_event_probabilities_3d(const std::string &csv_path);

	// Load total_lvl quantile edges from CSV
	void load_total_lvl_quantiles(const std::string &csv_path);

	StateParams &get(uint8_t imbalance_bin, int32_t spread) {
		return state_params[imbalance_bin][spread];
	}

	StateParams &get_3d(uint8_t imbalance_bin, int32_t spread, uint8_t total_lvl_bin) {
		return event_state_params_3d[imbalance_bin][spread][total_lvl_bin];
	}
};

inline const char *order_type_to_string(OrderType type) {
	switch (type) {
	case OrderType::Add:
		return "Add";
	case OrderType::Cancel:
		return "Cancel";
	case OrderType::Trade:
		return "Trade";
	case OrderType::CreateBid:
		return "CreateBid";
	case OrderType::CreateAsk:
		return "CreateAsk";
	default:
		return "Unknown";
	}
}

class MarketImpact {
  public:
	virtual ~MarketImpact() = default;
	// Advance time (decay for time-based models, no-op for others)
	virtual void step(int64_t time) = 0;
	// Add trade impact (size in real shares, normalized internally by mes)
	void add_trade(Side side, int32_t size = 1) {
		add_trade_impl(side, std::max(1, size / mes_best_));
	}
	// Get current bias factor
	virtual double bias_factor() const = 0;

	void set_mes(int32_t m) { mes_best_ = m; }

	// Convenience: step + add_trade in one call
	void update(Side side, int64_t time, int32_t size = 1) {
		step(time);
		add_trade(side, size);
	}

  protected:
	virtual void add_trade_impl(Side side, int32_t size) = 0;
	int32_t mes_best_ = 1;
};

class NoImpact : public MarketImpact {
  public:
	void step(int64_t /*time*/) override {}
	void add_trade_impl(Side /*side*/, int32_t /*size*/) override {}
	double bias_factor() const override { return 0.0; }
};

// Power-law impact approximated as a sum of exponentials:
//   K(t) ≈ Σ_k w_k * exp(-λ_k * t)
// Each component is an EMA with its own decay rate and weight.
// half_lives_sec and weights are parsed from config.
// m: overall bias multiplier applied to the sum.
// O(K) per step, no trade history stored.
class PowerLawImpact : public MarketImpact {
  public:
	// half_lives_sec: per-component half-lives in seconds
	// weights: per-component weights (from NNLS fit)
	// m: overall bias multiplier
	PowerLawImpact(const std::vector<double> &half_lives_sec, const std::vector<double> &weights,
				   double m)
		: m_(m), last_time_(0) {
		size_t K = half_lives_sec.size();
		kappas_ns_.resize(K);
		weights_.resize(K);
		phis_.resize(K, 0.0);
		for (size_t k = 0; k < K; k++) {
			kappas_ns_[k] = std::log(2.0) / (half_lives_sec[k] * 1e9);
			weights_[k] = weights[k];
		}
	}

	PowerLawImpact(double beta, double tau, double m, size_t K = 20, bool verbose = false)
		: m_(m), last_time_(0) {
		kappas_ns_.resize(K);
		phis_.resize(K, 0.0);
		for (size_t k = 0; k < K; k++) {
			double frac = static_cast<double>(k) / (K - 1);
			double half_life_sec = std::pow(10.0, -2.0 + 5.0 * frac);
			kappas_ns_[k] = std::log(2.0) / (half_life_sec * 1e9);
		}

		std::array<double, 100> t;
		std::array<double, 100> Y;
		std::vector<double> X(100 * K);
		for (size_t i = 0; i < 100; i++) {
			double frac = static_cast<double>(i) / 99;
			t[i] = std::pow(10.0, -2.0 + 5.0 * frac);
			Y[i] = std::pow(1 + t[i] / tau, -beta);
			for (size_t j = 0; j < K; j++) {
				X[j + i * K] = std::exp(-kappas_ns_[j] * t[i] * 1e9);
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
		weights_.resize(K);
		for (int i = K - 1; i >= 0; i--) {
			weights_[i] = y[i];
			for (size_t p = i + 1; p < K; p++)
				weights_[i] -= L[p * K + i] * weights_[p];
			weights_[i] /= L[i * K + i];
		}

		// Clamp negatives
		for (size_t k = 0; k < K; k++)
			weights_[k] = std::max(weights_[k], 0.0);

		if (verbose) {
			std::cout << "PowerLawImpact(tau=" << tau << ", beta=" << beta << ", m=" << m
					  << ", K=" << K << ")\n";
			std::cout << "  half_lives: [";
			for (size_t k = 0; k < K; k++) {
				double hl = std::log(2.0) / (kappas_ns_[k] * 1e9);
				std::cout << (k ? ", " : "") << hl;
			}
			std::cout << "]\n  weights:    [";
			for (size_t k = 0; k < K; k++) {
				std::cout << (k ? ", " : "") << weights_[k];
			}
			std::cout << "]\n";
		}
	}

	void step(int64_t time) override {
		if (last_time_ > 0 && time > last_time_) {
			double dt = static_cast<double>(time - last_time_);
			for (size_t k = 0; k < phis_.size(); k++) {
				phis_[k] *= std::exp(-kappas_ns_[k] * dt);
			}
		}
		last_time_ = time;
	}

	void add_trade_impl(Side side, int32_t size) override {
		double sign = (side == Side::Ask) ? 1.0 : -1.0;
		double impact = sign * std::sqrt(static_cast<double>(size));
		for (size_t k = 0; k < phis_.size(); k++) {
			phis_[k] += weights_[k] * impact;
		}
	}

	double bias_factor() const override {
		double sum = 0.0;
		for (size_t k = 0; k < phis_.size(); k++) {
			sum += phis_[k];
		}
		return m_ * sum;
	}



  private:
	double m_;
	int64_t last_time_;
	std::vector<double> kappas_ns_; // decay rates per ns
	std::vector<double> weights_;	// per-component weights
	std::vector<double> phis_;		// per-component accumulators
};

class DeltaT {
  public:
	virtual ~DeltaT() = default;
	virtual int64_t sample(int imb_bin, int spread, const Event &event,
						   std::mt19937_64 &rng) const = 0;
};

// Exponential dt using lambda from QRParams (the original approach)
class ExponentialDeltaT : public DeltaT {
  public:
	ExponentialDeltaT(const QRParams &params) : params_(params) {}

	int64_t sample(int imb_bin, int spread, const Event & /*event*/,
				   std::mt19937_64 &rng) const override {
		double lambda =
			params_.state_params[std::clamp(imb_bin, 0, 20)][std::clamp(spread, 0, 1)].lambda;
		double dt = std::exponential_distribution<>(lambda)(rng);
		return static_cast<int64_t>(std::ceil(dt));
	}

  private:
	const QRParams &params_;
};

// 5-component Gaussian mixture for dt (log10 space), per (imbalance, spread, event, queue)
class MixtureDeltaT : public DeltaT {
  public:
	static constexpr int K = 5;
	// Add: 4 queues, Cancel: 4 queues, Trade: 2 queues, Create_Bid, Create_Ask
	static constexpr int NUM_SLOTS = 12;

	MixtureDeltaT(const std::string &csv_path) {
		std::ifstream file(csv_path);
		if (!file.is_open()) {
			throw std::runtime_error("Cannot open mixture CSV: " + csv_path);
		}

		std::string line;
		std::getline(file, line); // skip header

		while (std::getline(file, line)) {
			std::istringstream ss(line);
			std::string token;

			// imbalance (0.0 to 1.0)
			std::getline(ss, token, ',');
			double imb_val = std::stod(token);
			int imb_idx = static_cast<int>(std::round(imb_val * 10.0)) + 10;
			imb_idx = std::clamp(imb_idx, 10, 20);

			// spread (1 or 2)
			std::getline(ss, token, ',');
			int spread = std::stoi(token) - 1;

			// event
			std::getline(ss, token, ',');
			std::string event_str = token;

			// queue
			std::getline(ss, token, ',');
			int queue = std::stoi(token);

			// side (skip)
			std::getline(ss, token, ',');

			// Read K components: w_i, mu_i, sig_i
			MixtureParams mp{};
			for (int i = 0; i < K; ++i) {
				std::getline(ss, token, ',');
				mp.w[i] = std::stod(token);
				std::getline(ss, token, ',');
				mp.mu[i] = std::stod(token);
				std::getline(ss, token, ',');
				mp.sig[i] = std::stod(token);
			}
			mp.loaded = true;

			int slot = str_to_slot(event_str, queue);
			if (slot < 0)
				continue;

			params_[imb_idx][spread][slot] = mp;

			// Mirror to negative imbalance
			if (imb_idx != 10) {
				int neg_idx = 20 - imb_idx;
				int neg_slot = mirror_slot(event_str, queue);
				params_[neg_idx][spread][neg_slot] = mp;
			}
		}
	}

	int64_t sample(int imb_bin, int spread, const Event &event,
				   std::mt19937_64 &rng) const override {
		int slot = event_to_slot(event.type, event.queue_nbr);
		const auto &p = params_[std::clamp(imb_bin, 0, 20)][std::clamp(spread, 0, 1)][slot];

		if (!p.loaded) {
			// Fallback: exponential with 100ms mean
			return static_cast<int64_t>(std::exponential_distribution<>(1e-8)(rng));
		}

		// Choose component
		double u = std::uniform_real_distribution<>(0.0, 1.0)(rng);
		double cum = 0.0;
		int k = K - 1;
		for (int i = 0; i < K; ++i) {
			cum += p.w[i];
			if (u < cum) {
				k = i;
				break;
			}
		}

		double log_dt = std::max(1.0, std::normal_distribution<>(p.mu[k], p.sig[k])(rng));
		return static_cast<int64_t>(std::pow(10.0, log_dt));
	}

  private:
	struct MixtureParams {
		double w[5]{}, mu[5]{}, sig[5]{};
		bool loaded = false;
	};

	// [imb 0-20][spread 0-1][slot 0-11]
	std::array<std::array<std::array<MixtureParams, NUM_SLOTS>, 2>, 21> params_{};

	// Slot layout:
	//  0-3: Add    q={-2,-1,1,2}
	//  4-7: Cancel q={-2,-1,1,2}
	//  8-9: Trade  q={-1,1}
	// 10:   Create_Bid
	// 11:   Create_Ask

	static int q4_idx(int q) {
		switch (q) {
		case -2:
			return 0;
		case -1:
			return 1;
		case 1:
			return 2;
		case 2:
			return 3;
		}
		return 1;
	}

	static int q2_idx(int q) { return (q == -1) ? 0 : 1; }

	static int event_to_slot(OrderType type, int q) {
		if (type == OrderType::CreateBid)
			return 10;
		if (type == OrderType::CreateAsk)
			return 11;
		if (type == OrderType::Trade)
			return 8 + q2_idx(q);
		int t = (type == OrderType::Add) ? 0 : 4;
		return t + q4_idx(q);
	}

	static int str_to_slot(const std::string &s, int q) {
		if (s == "Create_Bid")
			return 10;
		if (s == "Create_Ask")
			return 11;
		if (s == "Trade")
			return 8 + q2_idx(q);
		int t = (s == "Add") ? 0 : 4;
		return t + q4_idx(q);
	}

	// Mirror: flip queue sign, swap Create_Bid/Ask
	static int mirror_slot(const std::string &s, int q) {
		if (s == "Create_Bid")
			return 11;
		if (s == "Create_Ask")
			return 10;
		if (s == "Trade")
			return 8 + q2_idx(-q);
		int t = (s == "Add") ? 0 : 4;
		return t + q4_idx(-q);
	}
};

class Alpha {
  public:
	virtual ~Alpha() = default;
	virtual void step(int64_t dt_ns) = 0;
	virtual double value() const = 0;
	virtual void reset() = 0;
	virtual void consume(double fraction) = 0;
	virtual double scale() const { return 1.0; }
};

class NoAlpha : public Alpha {
  public:
	void step(int64_t) override {}
	double value() const override { return 0.0; }
	void reset() override {}
	void consume(double) override {}
	double scale() const override { return 0.0; }
};

class OUAlpha : public Alpha {
  public:
	// kappa_per_min: mean reversion rate in min^-1
	// s: stationary standard deviation
	// scale: multiplier applied to the OU output
	OUAlpha(double kappa_per_min, double s, uint64_t seed, double scale = 1.0)
		: kappa_(kappa_per_min / (60.0 * 1e9)), sigma_(s * std::sqrt(2.0 * kappa_)), scale_(scale),
		  alpha_(0.0), rng_(seed) {}

	void step(int64_t dt_ns) override {
		double dt = static_cast<double>(dt_ns);
		double decay = std::exp(-kappa_ * dt);
		double var = (sigma_ * sigma_) * (1.0 - decay * decay) / (2.0 * kappa_);
		alpha_ = alpha_ * decay + std::sqrt(var) * normal_(rng_);
	}

	double value() const override { return alpha_; }
	double scale() const override { return scale_; }
	void reset() override { alpha_ = 0.0; }
	void consume(double fraction) override {
		// Reduce alpha toward 0 by fraction (info acted upon)
		alpha_ *= (1.0 - fraction);
	}

  private:
	double kappa_; // in ns^-1
	double sigma_; // σ = s·√(2κ)
	double scale_;
	double alpha_;
	std::mt19937_64 rng_;
	std::normal_distribution<> normal_{0.0, 1.0};
};

class QRModel {
  public:
	QRModel(OrderBook *lob, const QRParams &params, uint64_t seed = 42)
		: lob_(lob), params_(params), size_dists_(nullptr), delta_t_(nullptr), rng_(seed) {}

	QRModel(OrderBook *lob, const QRParams &params, const SizeDistributions &size_dists,
			uint64_t seed = 42)
		: lob_(lob), params_(params), size_dists_(&size_dists), delta_t_(nullptr), rng_(seed) {}

	QRModel(OrderBook *lob, const QRParams &params, const SizeDistributions &size_dists,
			const DeltaT &delta_t, uint64_t seed = 42)
		: lob_(lob), params_(params), size_dists_(&size_dists), delta_t_(&delta_t), rng_(seed) {}

	void set_mes(const std::array<int32_t, 4> &m) { mes_ = m; }

	Order sample_order(int64_t current_time) {
		uint8_t imb_bin = get_imbalance_bin(lob_->imbalance());
		int32_t spread = std::min(lob_->spread() - 1, 1);

		// Get state params - either 2D or 3D depending on use_total_lvl flag
		StateParams *state_params_ptr;
		if (params_.use_total_lvl) {
			uint8_t total_lvl_bin = get_total_lvl_bin();
			state_params_ptr = &params_.get_3d(imb_bin, spread, total_lvl_bin);
		} else {
			state_params_ptr = &params_.get(imb_bin, spread);
		}
		StateParams &state_params = *state_params_ptr;
		const Event &event = state_params.sample_event(rng_);
		last_event_ = event;

		int32_t size = 1;
		if (size_dists_) {
			if (lob_->spread() == 1) {
				size = size_dists_->sample_size(imb_bin, event.type, event.queue_nbr, rng_);
				int level_idx = std::min(std::abs(event.queue_nbr) - 1, 3);
				size *= mes_[level_idx];
			} else if (event.type == OrderType::CreateBid || event.type == OrderType::CreateAsk) {
				size = size_dists_->sample_size(imb_bin, event.type, 0, rng_);
				size *= mes_[0];
			}
		}

		Order order(event.type, event.side, get_price(event), size, current_time);
		return order;
	}

	const Event &last_event() const { return last_event_; }

	int64_t sample_dt(const Event &event) {
		uint8_t imb_bin = get_imbalance_bin(lob_->imbalance());
		int spread = std::min(lob_->spread() - 1, 1);
		if (delta_t_) {
			return delta_t_->sample(imb_bin, spread, event, rng_);
		}
		return state_params_at(imb_bin, spread).sample_dt(rng_);
	}

	void bias(double b) {
		uint8_t imb_bin = get_imbalance_bin(lob_->imbalance());
		int32_t spread = std::min(lob_->spread() - 1, 1);

		if (params_.use_total_lvl) {
			// Apply bias to 3D state params (used for event sampling)
			uint8_t total_lvl_bin = get_total_lvl_bin();
			params_.get_3d(imb_bin, spread, total_lvl_bin).bias(b);
		} else {
			// Apply bias to 2D state params
			params_.get(imb_bin, spread).bias(b);
		}
	}

  private:
	OrderBook *lob_;
	QRParams params_;
	const SizeDistributions *size_dists_;
	const DeltaT *delta_t_;
	std::mt19937_64 rng_;
	Event last_event_{};
	std::array<int32_t, 4> mes_{1, 1, 1, 1};

	// 21 bins: -1.0 (idx 0), -0.9 (idx 1), ..., 0.0 (idx 10), ..., 0.9 (idx 19), 1.0 (idx 20)
	// Bin boundaries:
	//   -1.0: [-1, -0.9)    -> idx 0
	//   -0.9: [-0.9, -0.8)  -> idx 1
	//   ...
	//   -0.1: [-0.1, 0)     -> idx 9
	//   0.0:  {0}           -> idx 10
	//   0.1:  (0, 0.1]      -> idx 11
	//   ...
	//   1.0:  (0.9, 1]      -> idx 20
	uint8_t get_imbalance_bin(double imbalance) {
		imbalance = std::clamp(imbalance, -1.0, 1.0);

		if (imbalance == 0.0) {
			return 10;
		}
		if (imbalance < 0.0) {
			// [-1, -0.9) -> 0, [-0.9, -0.8) -> 1, ..., [-0.1, 0) -> 9
			int bin = static_cast<int>(std::floor(imbalance * 10.0)) + 10;
			return static_cast<uint8_t>(std::clamp(bin, 0, 9));
		} else {
			// (0, 0.1] -> 11, (0.1, 0.2] -> 12, ..., (0.9, 1] -> 20
			int bin = static_cast<int>(std::ceil(imbalance * 10.0)) + 10;
			return static_cast<uint8_t>(std::clamp(bin, 11, 20));
		}
	}

	// Get total_lvl bin based on current order book state
	// total_lvl = Q_bid + Q_ask (at best levels)
	uint8_t get_total_lvl_bin() {
		double total_lvl = static_cast<double>(lob_->best_bid_vol() + lob_->best_ask_vol());
		const auto &edges = params_.total_lvl_edges;

		// Find bin: edges[i] <= total_lvl < edges[i+1]
		for (uint8_t i = 0; i < QRParams::NUM_TOTAL_LVL_BINS - 1; ++i) {
			if (total_lvl <= edges[i + 1]) {
				return i;
			}
		}
		return QRParams::NUM_TOTAL_LVL_BINS - 1; // Last bin
	}
	StateParams &state_params_at(int imb_bin, int spread) { return params_.get(imb_bin, spread); }
	int32_t get_price(const Event &event) const {
		if (event.type == OrderType::CreateBid)
			return lob_->best_bid() + 1;
		if (event.type == OrderType::CreateAsk)
			return lob_->best_ask() - 1;
		if (event.side == Side::Bid)
			return lob_->best_bid() + 1 + event.queue_nbr; // queue_nbr is -1 or -2
		return lob_->best_ask() - 1 + event.queue_nbr;	   // queue_nbr is 1 or 2
	}
};
};