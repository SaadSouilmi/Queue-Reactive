#include <iostream>
#include <memory>
#include <random>
#include <filesystem>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"
#include "json_helpers.h"

using namespace qr;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.json>\n";
        std::cout << R"(
Example config.json:
{
  "ticker": "AAPL",
  "duration_hours": 1000,
  "seed": 12345,
  "use_mixture": true,
  "use_total_lvl": false,
  "use_alpha": true,

  "impact": {"type": "no_impact"},

  "alpha": {
    "kappa": 0.5,
    "sigma": 0.5,
    "scale": 1.0
  }
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

    std::string base_path = "../data";
    std::string ticker = get_string(doc, "ticker", "AAPL");
    std::string data_path = base_path + "/" + ticker;
    std::string results_path = base_path + "/results/" + ticker + "/samples/";

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

    bool use_alpha = get_bool(doc, "use_alpha", false);
    bool do_alpha_pnl = get_bool(doc, "compute_alpha_pnl", false);

    // OU config (only used if race is enabled)
    double kappa = 0.5;
    double sigma = 0.5;
    double alpha_scale = 1.0;
    if (doc.HasMember("alpha") && doc["alpha"].IsObject()) {
        const auto& ou = doc["alpha"];
        kappa = get_double(ou, "kappa", 0.5);
        sigma = get_double(ou, "sigma", 0.5);
        alpha_scale = get_double(ou, "scale", 1.0);
    }

    // Output
    std::string output_path = results_path + config_hash + ".parquet";
    std::string alpha_pnl_path = results_path + config_hash + "_alpha_pnl.csv";
    std::string registry_path = results_path + "registry.json";

    // Print config
    std::cout << "Config: " << argv[1] << "\n";
    std::cout << "Hash: " << config_hash << "\n";
    std::cout << "Ticker: " << ticker << "\n";
    std::cout << "Duration: " << duration_hours << " hours\n";
    std::cout << "Delta_t: " << (use_mixture ? "mixture" : "exponential") << "\n";
    std::cout << "Total lvl: " << (use_total_lvl ? "on" : "off") << "\n";
    std::cout << "Impact: " << impact_type;
    if (impact_type == "power_law") {
        if (impact_cfg.HasMember("tau")) {
            std::cout << " (tau=" << get_double(impact_cfg, "tau", 50.0)
                      << ", beta=" << get_double(impact_cfg, "beta", 1.5)
                      << ", m=" << get_double(impact_cfg, "m", 4.0) << ")";
        } else {
            int K = impact_cfg.HasMember("half_lives") ? impact_cfg["half_lives"].GetArray().Size() : 0;
            std::cout << " (K=" << K << " components, m=" << get_double(impact_cfg, "m", 4.0) << ")";
        }
    }
    std::cout << "\n";
    if (use_alpha) {
        std::cout << "OU: kappa=" << kappa << ", sigma=" << sigma << ", scale=" << alpha_scale << "\n";
    } else {
        std::cout << "Alpha: off\n";
    }
    std::cout << "Seed: " << master_seed << "\n";

    // Generate seeds
    std::mt19937_64 seed_rng(master_seed);
    uint64_t lob_seed = seed_rng();
    uint64_t model_seed = seed_rng();
    uint64_t alpha_seed = seed_rng();

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

    // Load data
    std::string params_path = data_path + "/qr_params";
    QueueDistributions dists(params_path + "/invariant_distributions_qmax100.csv");
    dists.set_mes(mes);

    std::unique_ptr<MixtureDeltaT> delta_t_ptr;
    if (use_mixture) {
        delta_t_ptr = std::make_unique<MixtureDeltaT>(params_path + "/delta_t_gmm.csv");
    }

    // Initialize order book
    OrderBook lob(dists, 4, lob_seed);
    lob.init({1516, 1517, 1518, 1519},
              {4, 1, 10, 5},
              {1520, 1521, 1522, 1523},
              {6, 17, 22, 23});

    QRParams params(params_path);
    if (use_total_lvl) {
        params.load_total_lvl_quantiles(params_path + "/total_lvl_quantiles.csv");
        params.load_event_probabilities_3d(params_path + "/event_probabilities_3D.csv");
    }
    SizeDistributions size_dists(params_path + "/size_distrib.csv");

    // Create model
    std::unique_ptr<QRModel> model_ptr;
    if (delta_t_ptr) {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, *delta_t_ptr, model_seed);
    } else {
        model_ptr = std::make_unique<QRModel>(&lob, params, size_dists, model_seed);
    }
    QRModel& model = *model_ptr;
    model.set_mes(mes);

    int64_t duration = static_cast<int64_t>(duration_hours * 3600.0 * 1e9);

    // Create optional components
    std::unique_ptr<OUAlpha> alpha_ptr;
    std::unique_ptr<MarketImpact> impact_ptr;

    if (use_alpha) {
        alpha_ptr = std::make_unique<OUAlpha>(kappa, sigma, alpha_seed, alpha_scale);
    }

    if (impact_type == "power_law") {
        double m = get_double(impact_cfg, "m", 4.0);
        if (impact_cfg.HasMember("tau")) {
            double tau = get_double(impact_cfg, "tau", 50.0);
            double beta = get_double(impact_cfg, "beta", 1.5);
            size_t K = static_cast<size_t>(get_double(impact_cfg, "K", 20));
            impact_ptr = std::make_unique<PowerLawImpact>(beta, tau, m, K);
        } else {
            std::vector<double> half_lives, weights;
            if (impact_cfg.HasMember("half_lives") && impact_cfg["half_lives"].IsArray()) {
                for (const auto& v : impact_cfg["half_lives"].GetArray())
                    half_lives.push_back(v.GetDouble());
            }
            if (impact_cfg.HasMember("weights") && impact_cfg["weights"].IsArray()) {
                for (const auto& v : impact_cfg["weights"].GetArray())
                    weights.push_back(v.GetDouble());
            }
            impact_ptr = std::make_unique<PowerLawImpact>(half_lives, weights, m);
        }
    } else if (impact_type != "no_impact") {
        std::cerr << "Unknown impact type: " << impact_type << "\n";
        return 1;
    }
    if (impact_ptr) impact_ptr->set_mes(mes[0]);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    Buffer result = run_simulation(lob, model, duration,
                                    alpha_ptr.get(), impact_ptr.get());
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Simulation: " << elapsed.count() << " s\n";
    std::cout << "Events: " << result.num_events() << "\n";

    // Save
    std::filesystem::create_directories(results_path);
    auto start_save = std::chrono::high_resolution_clock::now();
    result.save_parquet(output_path);
    auto end_save = std::chrono::high_resolution_clock::now();

    auto save_time = std::chrono::duration_cast<std::chrono::seconds>(end_save - start_save);
    std::cout << "Save: " << save_time.count() << " s\n";
    std::cout << "Output: " << output_path << "\n";

    if (do_alpha_pnl){
        std::vector<int64_t> lags_ns;
        for (int s = 30; s <= 30 * 60; s += 30) {
            lags_ns.push_back(static_cast<int64_t>(s) * 1'000'000'000);
        }
        std::vector<double> quantiles;
        for (int i = 1; i <= 9; i++){
            quantiles.push_back((double)i / 10.0);
        }
        auto start = std::chrono::high_resolution_clock::now();
        AlphaPnL result_alpha = compute_alpha_pnl(result, lags_ns, quantiles);
        result_alpha.save_csv(alpha_pnl_path);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Alpha P&L Computation: " << elapsed.count() << "s\n";
        
    }

    // Update registry
    update_registry(registry_path, config_hash, config_content, master_seed);
    std::cout << "Registry: " << registry_path << "\n";
}
