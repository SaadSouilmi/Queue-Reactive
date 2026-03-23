#include <iostream>
#include <memory>
#include <random>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"
#include "json_helpers.h"

using namespace qr;

std::string fmt_double(double v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

// ============================================================================
// Hardcoded risk parameter grid (edit and recompile to change)
// ============================================================================
struct GridPoint {
    int32_t q_max;
    int32_t max_inventory;
    double threshold;
};

std::vector<GridPoint> build_grid() {
    std::vector<int32_t> q_maxes = {1, 3, 5, 7, 10};
    std::vector<int32_t> max_inventories;
    for (int32_t i=2; i<=30; i+=2){
        max_inventories.push_back(i);
    }
    std::vector<double> thresholds = {0.4237, 0.5237, 0.6501, 0.8394};

    std::vector<GridPoint> grid;
    for (auto q : q_maxes)
        for (auto inv : max_inventories)
            for (auto thr : thresholds)
                grid.push_back({q, inv, thr});
    return grid;
}

// Result collected from each thread
struct GridResult {
    size_t index;
    GridPoint gp;
    int64_t n_trades;
    int64_t total_volume;
    int64_t n_round_trips;
    double realized_pnl;
    int64_t volume_at_last_zero;
    double edge_per_share;
    int32_t final_inventory;
    double final_pnl;
    long long elapsed_s;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.json>\n";
        std::cout << "Config has market params only (no strategy block).\n";
        std::cout << "Risk params (q_max, max_inventory, threshold) are a hardcoded grid.\n";
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

    std::string base_path = "../data";
    std::string ticker = get_string(doc, "ticker", "AAPL");
    std::string data_path = base_path + "/" + ticker;
    std::string results_dir = base_path + "/results/" + ticker + "/strategy/";
    std::string results_path = results_dir + config_hash + ".parquet";

    uint64_t master_seed = get_uint64(doc, "seed", std::random_device{}());
    bool use_mixture = get_bool(doc, "use_mixture", true);
    bool use_total_lvl = get_bool(doc, "use_total_lvl", false);
    double duration_hours = get_double(doc, "duration_hours", 1000.0);

    // Impact config
    std::string impact_type = "no_impact";
    rj::Value impact_cfg(rj::kObjectType);
    if (doc.HasMember("impact") && doc["impact"].IsObject()) {
        impact_cfg.CopyFrom(doc["impact"], doc.GetAllocator());
        impact_type = get_string(impact_cfg, "type", "no_impact");
    }
    double impact_m = get_double(impact_cfg, "m", 4.0);
    std::vector<double> pl_half_lives, pl_weights;
    double pl_tau = get_double(impact_cfg, "tau", 50.0);
    double pl_beta = get_double(impact_cfg, "beta", 1.5);
    size_t pl_K = static_cast<size_t>(get_double(impact_cfg, "K", 20));
    if (impact_cfg.HasMember("half_lives") && impact_cfg["half_lives"].IsArray()) {
        for (const auto& v : impact_cfg["half_lives"].GetArray())
            pl_half_lives.push_back(v.GetDouble());
    }
    if (impact_cfg.HasMember("weights") && impact_cfg["weights"].IsArray()) {
        for (const auto& v : impact_cfg["weights"].GetArray())
            pl_weights.push_back(v.GetDouble());
    }

    bool use_alpha = get_bool(doc, "use_alpha", false);

    // Alpha config
    double kappa = 0.5;
    double sigma = 0.5;
    double alpha_scale = 1.0;
    if (doc.HasMember("alpha") && doc["alpha"].IsObject()) {
        const auto& ou = doc["alpha"];
        kappa = get_double(ou, "kappa", 0.5);
        sigma = get_double(ou, "sigma", 0.5);
        alpha_scale = get_double(ou, "scale", 1.0);
    }

    int64_t duration = static_cast<int64_t>(duration_hours * 3600.0 * 1e9);

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

    // ========================================================================
    // Load shared read-only data (loaded once, shared across all threads)
    // ========================================================================
    std::string params_path = data_path + "/qr_params";
    QueueDistributions dists(params_path + "/invariant_distributions_qmax100.csv");
    dists.set_mes(mes);

    std::unique_ptr<MixtureDeltaT> delta_t_ptr;
    if (use_mixture) {
        delta_t_ptr = std::make_unique<MixtureDeltaT>(params_path + "/delta_t_gmm.csv");
    }

    QRParams qr_params(params_path);
    if (use_total_lvl) {
        qr_params.load_total_lvl_quantiles(params_path + "/total_lvl_quantiles.csv");
        qr_params.load_event_probabilities_3d(params_path + "/event_probabilities_3D.csv");
    }
    SizeDistributions size_dists(params_path + "/size_distrib.csv");

    // ========================================================================
    // Build grid and create output directory
    // ========================================================================
    auto grid = build_grid();
    std::filesystem::create_directories(results_dir);

    // Print config
    std::cout << "Config: " << argv[1] << "\n";
    std::cout << "Hash: " << config_hash << "\n";
    std::cout << "Ticker: " << ticker << "\n";
    std::cout << "Duration: " << duration_hours << " hours\n";
    std::cout << "Delta_t: " << (use_mixture ? "mixture" : "exponential") << "\n";
    std::cout << "Impact: " << impact_type << "\n";
    if (use_alpha) {
        std::cout << "OU: kappa=" << kappa << ", sigma=" << sigma
                  << ", scale=" << alpha_scale << "\n";
    }
    unsigned int num_workers = std::thread::hardware_concurrency();
    if (num_workers == 0) num_workers = 4;

    std::cout << "Seed: " << master_seed << "\n";
    std::cout << "Grid points: " << grid.size() << "\n";
    std::cout << "Workers: " << num_workers << "\n\n";

    // ========================================================================
    // Run grid with worker pool (hardware_concurrency threads)
    // ========================================================================
    std::vector<GridResult> results(grid.size());
    std::mutex print_mutex;
    std::atomic<size_t> next_job{0};
    size_t total = grid.size();

    auto worker = [&]() {
        while (true) {
            size_t i = next_job.fetch_add(1);
            if (i >= total) break;

            const auto& gp = grid[i];

            // Deterministic seeds per grid point
            std::mt19937_64 seed_rng(master_seed + i);
            uint64_t lob_seed = seed_rng();
            uint64_t model_seed = seed_rng();
            uint64_t alpha_seed = seed_rng();

            // Per-task OrderBook
            OrderBook lob(dists, 4, lob_seed);
            lob.init({1516, 1517, 1518, 1519},
                      {4, 1, 10, 5},
                      {1520, 1521, 1522, 1523},
                      {6, 17, 22, 23});

            // Per-task QRModel (copies qr_params internally)
            std::unique_ptr<QRModel> model_ptr;
            if (delta_t_ptr) {
                model_ptr = std::make_unique<QRModel>(&lob, qr_params, size_dists, *delta_t_ptr, model_seed);
            } else {
                model_ptr = std::make_unique<QRModel>(&lob, qr_params, size_dists, model_seed);
            }
            model_ptr->set_mes(mes);

            // Per-task Alpha
            std::unique_ptr<OUAlpha> alpha_ptr;
            if (use_alpha) {
                alpha_ptr = std::make_unique<OUAlpha>(kappa, sigma, alpha_seed, alpha_scale);
            }

            // Per-task Impact
            std::unique_ptr<MarketImpact> impact_ptr;
            if (impact_type == "power_law") {
                if (pl_half_lives.empty()) {
                    impact_ptr = std::make_unique<PowerLawImpact>(pl_beta, pl_tau, impact_m, pl_K);
                } else {
                    impact_ptr = std::make_unique<PowerLawImpact>(pl_half_lives, pl_weights, impact_m);
                }
            }
            if (impact_ptr) impact_ptr->set_mes(mes[0]);

            // Strategy trader for this grid point
            StrategyParams strat_params{gp.q_max * mes[0], gp.max_inventory * mes[0], gp.threshold};
            StrategyTrader strategy(strat_params);

            // Run simulation — stats accumulate in strategy trader
            auto t0 = std::chrono::high_resolution_clock::now();
            run_simulation_with_strategy(
                lob, *model_ptr, duration,
                alpha_ptr.get(), impact_ptr.get(),
                strategy);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto secs = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();

            // Collect result from accumulators
            results[i] = {i, gp,
                          strategy.n_trades(), strategy.total_volume(),
                          strategy.n_round_trips(), strategy.realized_pnl(),
                          strategy.volume_at_last_zero(), strategy.edge_per_share(),
                          strategy.inventory(), strategy.pnl(), secs};

            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[" << (i + 1) << "/" << total << "] "
                          << "q=" << gp.q_max
                          << " inv=" << gp.max_inventory
                          << " thr=" << gp.threshold
                          << " | trades=" << strategy.n_trades()
                          << " vol=" << strategy.total_volume()
                          << " rtrips=" << strategy.n_round_trips()
                          << " edge=" << std::fixed << std::setprecision(4) << strategy.edge_per_share()
                          << " | " << secs << " s\n";
            }
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int w = 0; w < num_workers; w++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();

    // ========================================================================
    // Write single summary parquet
    // ========================================================================
    {
        auto q_max_b = std::make_shared<arrow::Int32Builder>();
        auto max_inv_b = std::make_shared<arrow::Int32Builder>();
        auto threshold_b = std::make_shared<arrow::DoubleBuilder>();
        auto n_trades_b = std::make_shared<arrow::Int64Builder>();
        auto total_volume_b = std::make_shared<arrow::Int64Builder>();
        auto n_round_trips_b = std::make_shared<arrow::Int64Builder>();
        auto realized_pnl_b = std::make_shared<arrow::DoubleBuilder>();
        auto vol_at_zero_b = std::make_shared<arrow::Int64Builder>();
        auto edge_b = std::make_shared<arrow::DoubleBuilder>();
        auto final_inv_b = std::make_shared<arrow::Int32Builder>();
        auto final_pnl_b = std::make_shared<arrow::DoubleBuilder>();
        auto elapsed_b = std::make_shared<arrow::Int64Builder>();

        for (const auto& r : results) {
            (void)q_max_b->Append(r.gp.q_max);
            (void)max_inv_b->Append(r.gp.max_inventory);
            (void)threshold_b->Append(r.gp.threshold);
            (void)n_trades_b->Append(r.n_trades);
            (void)total_volume_b->Append(r.total_volume);
            (void)n_round_trips_b->Append(r.n_round_trips);
            (void)realized_pnl_b->Append(r.realized_pnl);
            (void)vol_at_zero_b->Append(r.volume_at_last_zero);
            (void)edge_b->Append(r.edge_per_share);
            (void)final_inv_b->Append(r.final_inventory);
            (void)final_pnl_b->Append(r.final_pnl);
            (void)elapsed_b->Append(r.elapsed_s);
        }

        std::shared_ptr<arrow::Array> a_q, a_inv, a_thr, a_nt, a_tv, a_rt, a_rp, a_vz, a_e, a_fi, a_fp, a_el;
        (void)q_max_b->Finish(&a_q);
        (void)max_inv_b->Finish(&a_inv);
        (void)threshold_b->Finish(&a_thr);
        (void)n_trades_b->Finish(&a_nt);
        (void)total_volume_b->Finish(&a_tv);
        (void)n_round_trips_b->Finish(&a_rt);
        (void)realized_pnl_b->Finish(&a_rp);
        (void)vol_at_zero_b->Finish(&a_vz);
        (void)edge_b->Finish(&a_e);
        (void)final_inv_b->Finish(&a_fi);
        (void)final_pnl_b->Finish(&a_fp);
        (void)elapsed_b->Finish(&a_el);

        auto schema = arrow::schema({
            arrow::field("q_max", arrow::int32()),
            arrow::field("max_inventory", arrow::int32()),
            arrow::field("threshold", arrow::float64()),
            arrow::field("n_trades", arrow::int64()),
            arrow::field("total_volume", arrow::int64()),
            arrow::field("n_round_trips", arrow::int64()),
            arrow::field("realized_pnl", arrow::float64()),
            arrow::field("volume_at_last_zero", arrow::int64()),
            arrow::field("edge_per_share", arrow::float64()),
            arrow::field("final_inventory", arrow::int32()),
            arrow::field("final_pnl", arrow::float64()),
            arrow::field("elapsed_s", arrow::int64()),
        });

        auto table = arrow::Table::Make(schema,
            {a_q, a_inv, a_thr, a_nt, a_tv, a_rt, a_rp, a_vz, a_e, a_fi, a_fp, a_el});

        auto outfile = arrow::io::FileOutputStream::Open(results_path).ValueOrDie();
        auto props = parquet::WriterProperties::Builder().compression(parquet::Compression::SNAPPY)->build();
        (void)parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, table->num_rows(), props);

        std::cout << "\nSummary parquet: " << results_path << "\n";
    }

    // ========================================================================
    // Print summary table
    // ========================================================================
    std::cout << "\n=== Summary ===\n";
    std::cout << std::left
              << std::setw(6) << "q_max"
              << std::setw(8) << "max_inv"
              << std::setw(8) << "thr"
              << std::setw(10) << "trades"
              << std::setw(10) << "volume"
              << std::setw(8) << "rtrips"
              << std::setw(14) << "realized_pnl"
              << std::setw(12) << "edge/share"
              << std::setw(10) << "final_inv"
              << std::setw(10) << "ms"
              << "\n";
    std::cout << std::string(96, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(6) << r.gp.q_max
                  << std::setw(8) << r.gp.max_inventory
                  << std::setw(8) << std::defaultfloat << r.gp.threshold
                  << std::setw(10) << r.n_trades
                  << std::setw(10) << r.total_volume
                  << std::setw(8) << r.n_round_trips
                  << std::setw(14) << std::fixed << std::setprecision(0) << r.realized_pnl
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.edge_per_share
                  << std::setw(10) << r.final_inventory
                  << std::setw(10) << r.elapsed_s
                  << "\n";
    }

    std::cout << "\nOutput: " << results_path << "\n";

    // Update registry
    std::string registry_path = base_path + "/results/" + ticker + "/strategy/registry.json";
    update_registry(registry_path, config_hash, config_content, master_seed);
}
