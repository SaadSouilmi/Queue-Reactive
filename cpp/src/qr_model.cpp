#include "qr_model.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace qr {

namespace {
    OrderType parse_event_type(const std::string& s) {
        if (s == "Add") return OrderType::Add;
        if (s == "Cancel") return OrderType::Cancel;
        if (s == "Trade") return OrderType::Trade;
        if (s == "Create_Bid") return OrderType::CreateBid;
        if (s == "Create_Ask") return OrderType::CreateAsk;
        throw std::runtime_error("Unknown event type: " + s);
    }

    Side parse_side(const std::string& s) {
        if (s == "1") return Side::Ask;
        if (s == "-1") return Side::Bid;
        throw std::runtime_error("Unknown side: " + s);
    }

    OrderType flip_event_type(OrderType t) {
        if (t == OrderType::CreateBid) return OrderType::CreateAsk;
        if (t == OrderType::CreateAsk) return OrderType::CreateBid;
        return t;
    }

    Side flip_side(Side s) {
        return (s == Side::Bid) ? Side::Ask : Side::Bid;
    }

    // CSV has symmetric imbalance 0.0-1.0 -> index 10-20
    int imb_bin_to_index(double imb_val) {
        int idx = static_cast<int>(std::round(imb_val * 10.0)) + 10;
        return std::clamp(idx, 10, 20);
    }

    // Mirror index: 13 (imb=0.3) -> 7 (imb=-0.3)
    int mirror_imb_index(int idx) {
        return 20 - idx;
    }

}

QRParams::QRParams(const std::string& path) {
    // Load event probabilities
    // Format: imbalance,spread,event,queue,side,probability
    std::ifstream file(path + "/event_probabilities.csv");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open event_probabilities.csv");
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // imbalance (0.0 to 1.0)
        std::getline(ss, token, ',');
        double imb_val = std::stod(token);
        int imb_idx = imb_bin_to_index(imb_val);

        // spread (1 or 2)
        std::getline(ss, token, ',');
        int spread = std::stoi(token) - 1; // 1->0, 2->1

        // event type
        std::getline(ss, token, ',');
        OrderType type = parse_event_type(token);

        // queue
        std::getline(ss, token, ',');
        int queue_nbr = std::stoi(token);

        // side (1 or -1)
        std::getline(ss, token, ',');
        Side side = parse_side(token);

        // probability
        std::getline(ss, token, ',');
        double prob = std::stod(token);

        // Store at positive index
        StateParams& sp = state_params[imb_idx][spread];
        sp.events.push_back({type, side, queue_nbr});
        sp.base_probs.push_back(prob);

        // Mirror to negative index (skip imbalance=0.0 which is idx 10)
        if (imb_idx != 10) {
            int neg_idx = mirror_imb_index(imb_idx);
            StateParams& sp_neg = state_params[neg_idx][spread];
            sp_neg.events.push_back({flip_event_type(type), flip_side(side), -queue_nbr});
            sp_neg.base_probs.push_back(prob);
        }
    }
    file.close();

    // Load intensities
    // Format: imbalance,spread,average_dt
    std::ifstream ifile(path + "/delta_t_exponential.csv");
    if (!ifile.is_open()) {
        throw std::runtime_error("Cannot open delta_t_exponential.csv");
    }

    std::getline(ifile, line); // skip header

    while (std::getline(ifile, line)) {
        std::istringstream ss(line);
        std::string token;

        // imbalance (0.0 to 1.0)
        std::getline(ss, token, ',');
        double imb_val = std::stod(token);
        int imb_idx = imb_bin_to_index(imb_val);

        // spread (1 or 2)
        std::getline(ss, token, ',');
        int spread = std::stoi(token) - 1; // 1->0, 2->1

        // average_dt (mean inter-arrival time)
        std::getline(ss, token, ',');
        double dt_mean = std::stod(token);

        double lambda = 1.0 / dt_mean;
        state_params[imb_idx][spread].lambda = lambda;

        // Mirror to negative index (lambda is symmetric)
        if (imb_idx != 10) {
            state_params[mirror_imb_index(imb_idx)][spread].lambda = lambda;
        }
    }
    ifile.close();

    // Initialize working vectors for each state
    for (auto& row : state_params) {
        for (auto& sp : row) {
            sp.probs = sp.base_probs;
            sp.cum_probs.resize(sp.probs.size());
            sp.total = 0.0;
            for (size_t i = 0; i < sp.probs.size(); i++) {
                sp.total += sp.probs[i];
            }
            if (!sp.cum_probs.empty()) {
                sp.cum_probs[0] = sp.probs[0];
                for (size_t i = 1; i < sp.probs.size(); i++) {
                    sp.cum_probs[i] = sp.cum_probs[i-1] + sp.probs[i];
                }
            }
        }
    }
}

void QRParams::load_total_lvl_quantiles(const std::string& csv_path) {
    // Load total_lvl quantile edges
    // Format: bin,lower,upper,percentile_lower,percentile_upper
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open total_lvl_quantiles.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line); // skip header

    // First edge is the lower of bin 0
    bool first_row = true;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // bin
        std::getline(ss, token, ',');
        int bin = std::stoi(token);

        // lower
        std::getline(ss, token, ',');
        double lower = std::stod(token);

        // upper
        std::getline(ss, token, ',');
        double upper = std::stod(token);

        if (bin >= 0 && bin < NUM_TOTAL_LVL_BINS) {
            if (first_row) {
                total_lvl_edges[0] = lower;
                first_row = false;
            }
            total_lvl_edges[bin + 1] = upper;
        }
    }
    file.close();
}

void QRParams::load_event_probabilities_3d(const std::string& csv_path) {
    // Load 3D event probabilities
    // Format: imb_bin,spread,total_lvl_bin,event,event_q,len,event_side,proba
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open event_probabilities_3d.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // imb_bin (float like -1.0, -0.9, 0.0, etc.)
        std::getline(ss, token, ',');
        double imb_bin_val = std::stod(token);
        int imb_bin = imb_bin_to_index(imb_bin_val);

        // spread (1 or 2)
        std::getline(ss, token, ',');
        int spread = std::stoi(token) - 1; // 1->0, 2->1

        // total_lvl_bin (0-4)
        std::getline(ss, token, ',');
        int total_lvl_bin = std::stoi(token);

        // event type
        std::getline(ss, token, ',');
        OrderType type = parse_event_type(token);

        // event_q (queue number)
        std::getline(ss, token, ',');
        int queue_nbr = std::stoi(token);

        // len (skip - just count)
        std::getline(ss, token, ',');

        // event_side
        std::getline(ss, token, ',');
        Side side = parse_side(token);

        // proba
        std::getline(ss, token, ',');
        double prob = std::stod(token);

        // Validate indices
        if (imb_bin < 0 || imb_bin >= NUM_IMB_BINS) continue;
        if (spread < 0 || spread > 1) continue;
        if (total_lvl_bin < 0 || total_lvl_bin >= NUM_TOTAL_LVL_BINS) continue;

        // Add to 3D state params
        StateParams& sp = event_state_params_3d[imb_bin][spread][total_lvl_bin];
        sp.events.push_back({type, side, queue_nbr});
        sp.base_probs.push_back(prob);
    }
    file.close();

    // Initialize working vectors for each 3D state
    for (auto& imb_row : event_state_params_3d) {
        for (auto& spread_row : imb_row) {
            for (auto& sp : spread_row) {
                sp.probs = sp.base_probs;
                sp.cum_probs.resize(sp.probs.size());
                sp.total = 0.0;
                for (size_t i = 0; i < sp.probs.size(); i++) {
                    sp.total += sp.probs[i];
                }
                if (!sp.cum_probs.empty()) {
                    sp.cum_probs[0] = sp.probs[0];
                    for (size_t i = 1; i < sp.probs.size(); i++) {
                        sp.cum_probs[i] = sp.cum_probs[i-1] + sp.probs[i];
                    }
                }
            }
        }
    }

    use_total_lvl = true;
}

SizeDistributions::SizeDistributions(const std::string& csv_path) {
    // Load empirical size distributions
    // Format: imbalance,spread,event,queue,side,1,2,...,50
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open size_distrib.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // imbalance (0.0 to 1.0)
        std::getline(ss, token, ',');
        double imb_val = std::stod(token);
        int imb_idx = imb_bin_to_index(imb_val);

        // spread (1 or 2)
        std::getline(ss, token, ',');
        int spread = std::stoi(token);

        // event type
        std::getline(ss, token, ',');
        std::string event_str = token;

        // queue (signed: -2, -1, 1, 2, or 0 for creates)
        std::getline(ss, token, ',');
        int queue = std::stoi(token);

        // side (skip)
        std::getline(ss, token, ',');

        // Read MAX_SIZE probabilities, convert to CDF
        std::vector<double> cdf(MAX_SIZE);
        double cumsum = 0.0;
        for (int i = 0; i < MAX_SIZE; i++) {
            std::getline(ss, token, ',');
            cumsum += std::stod(token);
            cdf[i] = cumsum;
        }

        if (imb_idx < 0 || imb_idx >= NUM_IMB_BINS) continue;
        int neg_idx = mirror_imb_index(imb_idx);

        if (spread == 1) {
            int type_idx;
            if (event_str == "Add") type_idx = 0;
            else if (event_str == "Cancel") type_idx = 1;
            else if (event_str == "Trade") type_idx = 2;
            else continue;

            int qi = q4_idx(queue);
            int qi_neg = q4_idx(-queue);
            cum_probs[imb_idx][type_idx][qi] = cdf;
            if (imb_idx != 10) {
                cum_probs[neg_idx][type_idx][qi_neg] = cdf;
            }
        } else if (spread == 2) {
            int create_idx;
            if (event_str == "Create_Bid") create_idx = 0;
            else if (event_str == "Create_Ask") create_idx = 1;
            else continue;

            cum_probs_create[imb_idx][create_idx] = cdf;
            if (imb_idx != 10) {
                cum_probs_create[neg_idx][1 - create_idx] = cdf;
            }
        }
    }
    file.close();
}

}
