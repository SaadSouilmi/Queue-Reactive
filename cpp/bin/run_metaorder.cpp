#include <iostream>
#include <vector>
#include <array>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <memory>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"
#include "json_helpers.h"

using namespace qr;

struct MetaOrder {
    std::vector<int64_t> timestamps;
    std::vector<int32_t> sizes;
    Side side;
};

// --- LOB configs (volumes only, prices hardcoded) ---

struct LOBConfig {
    std::array<int32_t, 4> bid_vols;
    std::array<int32_t, 4> ask_vols;
};

constexpr std::array<int32_t, 4> BID_PRICES = {99, 98, 97, 96};
constexpr std::array<int32_t, 4> ASK_PRICES = {100, 101, 102, 103};

std::vector<LOBConfig> load_lob_configs(const std::string& path) {
    std::vector<LOBConfig> configs;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }

    std::string line;
    std::getline(file, line);  // skip header: q_4,q_3,q_2,q_1,q_-1,q_-2,q_-3,q_-4

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        LOBConfig cfg;

        // Ask vols: q_4, q_3, q_2, q_1 (far to near)
        for (int i = 3; i >= 0; i--) {
            std::getline(ss, token, ',');
            cfg.ask_vols[i] = std::stoi(token);
        }
        // Bid vols: q_-1, q_-2, q_-3, q_-4 (near to far)
        for (int i = 0; i < 4; i++) {
            std::getline(ss, token, ',');
            cfg.bid_vols[i] = std::stoi(token);
        }

        configs.push_back(cfg);
    }
    return configs;
}

// --- Accumulator ---

struct Accumulator {
    std::vector<int64_t> grid;
    std::vector<double> mid_sum;
    std::vector<double> mid_sum_sq;
    std::vector<double> bias_sum;
    std::vector<double> bias_sum_sq;
    std::vector<double> vol_sum;
    int count = 0;

    Accumulator(int64_t duration, int64_t step) {
        for (int64_t t = 0; t <= duration; t += step) {
            grid.push_back(t);
            mid_sum.push_back(0.0);
            mid_sum_sq.push_back(0.0);
            bias_sum.push_back(0.0);
            bias_sum_sq.push_back(0.0);
            vol_sum.push_back(0.0);
        }
    }

    void add(const std::vector<double>& grid_mid,
             const std::vector<double>& grid_bias,
             const std::vector<int32_t>& grid_vol) {
        double mid0 = grid_mid[0];
        for (size_t i = 0; i < grid.size(); i++) {
            double dm = grid_mid[i] - mid0;
            mid_sum[i] += dm;
            mid_sum_sq[i] += dm * dm;
            bias_sum[i] += grid_bias[i];
            bias_sum_sq[i] += grid_bias[i] * grid_bias[i];
            vol_sum[i] += grid_vol[i];
        }
        count++;
    }

    void merge(const Accumulator& other) {
        for (size_t i = 0; i < grid.size(); i++) {
            mid_sum[i] += other.mid_sum[i];
            mid_sum_sq[i] += other.mid_sum_sq[i];
            bias_sum[i] += other.bias_sum[i];
            bias_sum_sq[i] += other.bias_sum_sq[i];
            vol_sum[i] += other.vol_sum[i];
        }
        count += other.count;
    }

    void save_csv(const std::string& path) {
        std::ofstream file(path);
        file << "timestamp,avg_mid_price_change,mid_price_change_se,avg_bias,bias_se,avg_meta_vol\n";
        for (size_t i = 0; i < grid.size(); i++) {
            double mid_mean = mid_sum[i] / count;
            double mid_var = mid_sum_sq[i] / count - mid_mean * mid_mean;
            double mid_se = std::sqrt(mid_var / count);

            double bias_mean = bias_sum[i] / count;
            double bias_var = bias_sum_sq[i] / count - bias_mean * bias_mean;
            double bias_se = std::sqrt(bias_var / count);

            double vol_mean = vol_sum[i] / count;

            file << grid[i] << "," << mid_mean << "," << mid_se << "," << bias_mean << "," << bias_se << "," << vol_mean << "\n";
        }
    }
};

// --- Metaorder builder ---

MetaOrder build_metaorder(int32_t total_vol, Side side, int32_t max_order_size, int64_t exec_duration_ns) {
    MetaOrder metaorder;
    metaorder.side = side;

    int num_orders = (total_vol + max_order_size - 1) / max_order_size;
    if (num_orders == 0) num_orders = 1;

    int64_t interval = (num_orders > 1) ? exec_duration_ns / (num_orders - 1) : 0;

    int32_t remaining = total_vol;
    for (int i = 0; i < num_orders; i++) {
        int64_t t = static_cast<int64_t>(i) * interval;
        int32_t size = std::min(remaining, max_order_size);
        metaorder.timestamps.push_back(t);
        metaorder.sizes.push_back(size);
        remaining -= size;
    }

    return metaorder;
}

// --- Per-simulation worker ---

struct SimConfig {
    const QueueDistributions* dists;
    const DeltaT* delta_t;
    const std::vector<LOBConfig>* lob_configs;
    const QRParams* params;
    const SizeDistributions* size_dists;

    // MES
    std::array<int32_t, 4> mes{1, 1, 1, 1};

    // Impact
    std::string impact_type;
    double impact_m;
    std::vector<double> pl_half_lives;
    std::vector<double> pl_weights;
    double pl_tau;
    double pl_beta;
    size_t pl_K;

    // Metaorder
    const MetaOrder* metaorder;

    // Simulation
    int64_t duration;
    int64_t grid_step;
};

void run_and_accumulate(const SimConfig& cfg, uint64_t seed, Accumulator& acc) {
    std::mt19937_64 rng(seed);
    size_t config_idx = rng() % cfg.lob_configs->size();
    const LOBConfig& lob_cfg = (*cfg.lob_configs)[config_idx];

    OrderBook lob(*cfg.dists, 4, seed);
    lob.init(std::vector<int32_t>(BID_PRICES.begin(), BID_PRICES.end()),
             std::vector<int32_t>(lob_cfg.bid_vols.begin(), lob_cfg.bid_vols.end()),
             std::vector<int32_t>(ASK_PRICES.begin(), ASK_PRICES.end()),
             std::vector<int32_t>(lob_cfg.ask_vols.begin(), lob_cfg.ask_vols.end()));

    std::unique_ptr<QRModel> model_ptr;
    if (cfg.delta_t) {
        model_ptr = std::make_unique<QRModel>(&lob, *cfg.params, *cfg.size_dists, *cfg.delta_t, seed);
    } else {
        model_ptr = std::make_unique<QRModel>(&lob, *cfg.params, *cfg.size_dists, seed);
    }
    model_ptr->set_mes(cfg.mes);
    QRModel& model = *model_ptr;

    // Impact model
    std::unique_ptr<MarketImpact> impact_ptr;
    if (cfg.impact_type == "power_law") {
        if (cfg.pl_half_lives.empty()) {
            impact_ptr = std::make_unique<PowerLawImpact>(cfg.pl_beta, cfg.pl_tau, cfg.impact_m, cfg.pl_K);
        } else {
            impact_ptr = std::make_unique<PowerLawImpact>(cfg.pl_half_lives, cfg.pl_weights, cfg.impact_m);
        }
    } else {
        impact_ptr = std::make_unique<NoImpact>();
    }
    impact_ptr->set_mes(cfg.mes[0]);
    MarketImpact& impact = *impact_ptr;

    const MetaOrder& metaorder = *cfg.metaorder;

    // Grid arrays for on-the-fly projection
    size_t grid_size = static_cast<size_t>(cfg.duration / cfg.grid_step) + 1;
    std::vector<double> grid_mid(grid_size, 0.0);
    std::vector<double> grid_bias(grid_size, 0.0);
    std::vector<int32_t> grid_vol(grid_size, 0);
    size_t grid_idx = 0;

    std::vector<Fill> fills;
    int64_t time = 0;
    size_t meta_i = 0;
    size_t meta_n = metaorder.timestamps.size();
    double current_bias = 0.0;
    int32_t cum_meta_vol = 0;
    double current_mid = (lob.best_bid() + lob.best_ask()) / 2.0;
    Order order;
    bool is_meta = false;

    while (time < cfg.duration) {
        impact.step(time);
        current_bias = impact.bias_factor();
        model.bias(current_bias);

        order = model.sample_order(time);
        int64_t dt = model.sample_dt(model.last_event());
        if (meta_i < meta_n && time + dt >= metaorder.timestamps[meta_i]) {
            time = metaorder.timestamps[meta_i];
            int32_t price = (metaorder.side == Side::Bid)
                ? std::max(1, lob.best_bid() - 4)
                : lob.best_ask() + 4;
            order = Order(OrderType::Trade, metaorder.side, price, metaorder.sizes[meta_i] * cfg.mes[0], time);
            meta_i++;
            is_meta = true;
        }
        else {
            time += dt;
            order.ts = time;
            is_meta = false;
        }

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);
            int32_t filled = 0;
            for (const auto& f : fills) filled += f.size;
            if (is_meta) cum_meta_vol += filled / cfg.mes[0];
            if (filled > 0) impact.update(order.side, order.ts, filled);
        } else {
            lob.process(order);
        }

        current_mid = (lob.best_bid() + lob.best_ask()) / 2.0;

        // Project onto grid: fill all grid points up to current time
        while (grid_idx < grid_size && static_cast<int64_t>(grid_idx) * cfg.grid_step <= time) {
            grid_mid[grid_idx] = current_mid;
            grid_bias[grid_idx] = current_bias;
            grid_vol[grid_idx] = cum_meta_vol;
            grid_idx++;
        }
    }

    // Fill any remaining grid points with final state
    while (grid_idx < grid_size) {
        grid_mid[grid_idx] = current_mid;
        grid_bias[grid_idx] = current_bias;
        grid_vol[grid_idx] = cum_meta_vol;
        grid_idx++;
    }

    acc.add(grid_mid, grid_bias, grid_vol);
}

// --- Main ---

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.json>\n";
        std::cout << R"(
Example config.json:
{
  "ticker": "PFE",
  "use_mixture": false,
  "use_total_lvl": false,
  "duration_min": 30,
  "num_sims": 100000,
  "grid_step_ms": 500,

  "impact": {"type": "no_impact"},

  "metaorder_pcts": [10.0, 5.0, 2.5],
  "hourly_vol": 1000,
  "max_order_size": 2
}
)";
        return 1;
    }

    // Read and hash config
    std::string config_content = read_file(argv[1]);
    std::string config_hash = hash_config(config_content);

    rj::Document doc;
    doc.Parse(config_content.c_str());
    if (doc.HasParseError()) {
        std::cerr << "JSON parse error at offset " << doc.GetErrorOffset() << "\n";
        return 1;
    }

    std::string base_path = "/home/labcmap/saad.souilmi/dev_cpp/qr/data";
    std::string ticker = get_string(doc, "ticker", "PFE");
    std::string data_path = base_path + "/" + ticker;
    std::string params_path = data_path + "/qr_params";
    std::string results_path = base_path + "/results/" + ticker + "/metaorder/";

    bool use_mixture = get_bool(doc, "use_mixture", false);
    bool use_total_lvl = get_bool(doc, "use_total_lvl", false);
    int duration_min = get_int(doc, "duration_min", 30);
    int num_sims = get_int(doc, "num_sims", 100'000);
    int64_t grid_step = static_cast<int64_t>(get_int(doc, "grid_step_ms", 500)) * 1'000'000;
    int32_t hourly_vol = get_int(doc, "hourly_vol", 1000);
    int32_t max_order_size = get_int(doc, "max_order_size", 2);
    int exec_min = get_int(doc, "exec_duration_min", 5);
    int64_t exec_duration_ns = static_cast<int64_t>(exec_min) * 60 * 1'000'000'000LL;

    // Impact config
    std::string impact_type = "no_impact";
    double impact_m = 4.5;
    std::vector<double> pl_half_lives;
    std::vector<double> pl_weights;
    double pl_tau = 50.0;
    double pl_beta = 1.5;
    size_t pl_K = 20;
    if (doc.HasMember("impact") && doc["impact"].IsObject()) {
        const auto& imp = doc["impact"];
        impact_type = get_string(imp, "type", "no_impact");
        impact_m = get_double(imp, "m", 4.5);
        if (imp.HasMember("half_lives") && imp["half_lives"].IsArray()) {
            for (const auto& v : imp["half_lives"].GetArray())
                pl_half_lives.push_back(v.GetDouble());
        }
        if (imp.HasMember("weights") && imp["weights"].IsArray()) {
            for (const auto& v : imp["weights"].GetArray())
                pl_weights.push_back(v.GetDouble());
        }
        pl_tau = get_double(imp, "tau", 50.0);
        pl_beta = get_double(imp, "beta", 1.5);
        pl_K = static_cast<size_t>(get_double(imp, "K", 20));
    }

    // Metaorder percentages
    std::vector<double> metaorder_pcts = {10.0, 5.0, 2.5};
    if (doc.HasMember("metaorder_pcts") && doc["metaorder_pcts"].IsArray()) {
        metaorder_pcts.clear();
        for (const auto& v : doc["metaorder_pcts"].GetArray()) {
            metaorder_pcts.push_back(v.GetDouble());
        }
    }

    int64_t duration = static_cast<int64_t>(duration_min) * 60 * 1'000'000'000LL;
    int num_threads = std::thread::hardware_concurrency();

    // Print config
    std::cout << "Config: " << argv[1] << "\n";
    std::cout << "Hash: " << config_hash << "\n";
    std::cout << "Ticker: " << ticker << "\n";
    std::cout << "Delta_t: " << (use_mixture ? "mixture" : "exponential") << "\n";
    std::cout << "Total lvl: " << (use_total_lvl ? "on" : "off") << "\n";
    std::cout << "Impact: " << impact_type;
    if (impact_type == "power_law") {
        if (pl_half_lives.empty()) {
            std::cout << " (tau=" << pl_tau << ", beta=" << pl_beta << ", m=" << impact_m << ")";
        } else {
            std::cout << " (K=" << pl_half_lives.size() << " components, m=" << impact_m << ")";
        }
    }
    std::cout << "\n";
    std::cout << "Duration: " << duration_min << " min, Metaorder execution: " << exec_min << " min\n";
    std::cout << "Hourly vol: " << hourly_vol << ", Max order size: " << max_order_size << "\n";
    std::cout << "Sims: " << num_sims << ", Threads: " << num_threads << "\n";
    std::cout << "Grid: " << (duration / grid_step + 1) << " points (" << (grid_step / 1'000'000) << "ms spacing)\n";

    // Load MES from params.json
    std::array<int32_t, 4> mes{1, 1, 1, 1};
    {
        std::ifstream mes_file(data_path + "/daily_estimates/params.json");
        if (mes_file.is_open()) {
            rj::IStreamWrapper mes_isw(mes_file);
            rj::Document mes_doc;
            mes_doc.ParseStream(mes_isw);
            if (!mes_doc.HasParseError() && mes_doc.HasMember("median_event_sizes")) {
                const auto& m = mes_doc["median_event_sizes"];
                for (int i = 0; i < 4; i++) {
                    std::string key = std::to_string(i + 1);
                    if (m.HasMember(key.c_str())) mes[i] = m[key.c_str()].GetInt();
                }
            }
        }
    }

    // Load shared data (loaded once, shared read-only across all threads)
    QueueDistributions dists(params_path + "/invariant_distributions_qmax100.csv");
    dists.set_mes(mes);

    std::unique_ptr<MixtureDeltaT> delta_t_ptr;
    if (use_mixture) {
        delta_t_ptr = std::make_unique<MixtureDeltaT>(params_path + "/delta_t_gmm.csv");
    }
    const DeltaT* delta_t = delta_t_ptr.get();

    std::vector<LOBConfig> lob_configs = load_lob_configs(params_path + "/random_lob.csv");
    std::cout << "Loaded " << lob_configs.size() << " LOB configs\n";

    QRParams params(params_path);
    if (use_total_lvl) {
        params.load_total_lvl_quantiles(params_path + "/total_lvl_quantiles.csv");
        params.load_event_probabilities_3d(params_path + "/event_probabilities_3d.csv");
    }
    SizeDistributions size_dists(params_path + "/size_distrib.csv");
    std::cout << "Loaded QRParams and SizeDistributions\n";

    std::filesystem::create_directories(results_path);

    // Build SimConfig (shared across all workers)
    SimConfig sim_cfg;
    sim_cfg.mes = mes;
    sim_cfg.dists = &dists;
    sim_cfg.delta_t = delta_t;
    sim_cfg.lob_configs = &lob_configs;
    sim_cfg.params = &params;
    sim_cfg.size_dists = &size_dists;
    sim_cfg.impact_type = impact_type;
    sim_cfg.impact_m = impact_m;
    sim_cfg.pl_half_lives = pl_half_lives;
    sim_cfg.pl_weights = pl_weights;
    sim_cfg.pl_tau = pl_tau;
    sim_cfg.pl_beta = pl_beta;
    sim_cfg.pl_K = pl_K;
    sim_cfg.duration = duration;
    sim_cfg.grid_step = grid_step;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (double pct : metaorder_pcts) {
        int32_t metaorder_vol = static_cast<int32_t>(hourly_vol * pct / 100.0);

        std::cout << "\n=== Metaorder " << pct << "% of hourly vol (" << metaorder_vol << " shares) ===\n";

        MetaOrder metaorder = build_metaorder(metaorder_vol, Side::Ask, max_order_size, exec_duration_ns);
        sim_cfg.metaorder = &metaorder;

        std::cout << "  Orders: " << metaorder.timestamps.size() << ", sizes: ";
        for (size_t i = 0; i < std::min(metaorder.sizes.size(), size_t(5)); i++) {
            std::cout << metaorder.sizes[i] << " ";
        }
        if (metaorder.sizes.size() > 5) std::cout << "...";
        std::cout << "\n";

        Accumulator acc(duration, grid_step);

        auto start = std::chrono::high_resolution_clock::now();

        std::atomic<int> next_sim(0);
        std::atomic<int> done_sims(0);
        std::vector<std::thread> workers;
        std::vector<Accumulator> local_accs;
        workers.reserve(num_threads);
        local_accs.reserve(num_threads);
        for (int t = 0; t < num_threads; t++) {
            local_accs.emplace_back(duration, grid_step);
        }

        for (int t = 0; t < num_threads; t++) {
            workers.emplace_back([&, t]() {
                Accumulator& local = local_accs[t];
                while (true) {
                    int i = next_sim.fetch_add(1);
                    if (i >= num_sims) break;
                    run_and_accumulate(sim_cfg, static_cast<uint64_t>(i), local);
                    int d = done_sims.fetch_add(1) + 1;
                    if (d % 10000 == 0) {
                        std::cout << "  Progress: " << d << "/" << num_sims << "\n";
                    }
                }
            });
        }
        for (auto& w : workers) w.join();

        for (const auto& local : local_accs) {
            acc.merge(local);
        }

        std::string out_path = results_path + config_hash + "_pct_" + std::to_string(static_cast<int>(pct * 10)) + ".csv";
        acc.save_csv(out_path);
        std::cout << "  Output: " << out_path << "\n";

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  Done in " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s\n";
    }

    // Update registry
    std::string registry_path = results_path + "registry.json";
    update_registry(registry_path, config_hash, config_content);

    auto total_end = std::chrono::high_resolution_clock::now();
    std::cout << "\nAll done in " << std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count() << "s\n";
    std::cout << "Registry: " << registry_path << "\n";
}
