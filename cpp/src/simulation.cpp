#include "simulation.h"
#include "strategy.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <fstream>
#include <cmath>

namespace qr {

void Buffer::save_parquet(const std::string& path) const {
    // Build columns
    arrow::Int64Builder sequence_builder;
    arrow::Int32Builder best_bid_price_builder;
    arrow::Int32Builder best_bid_vol_builder;
    arrow::Int32Builder best_ask_price_builder;
    arrow::Int32Builder best_ask_vol_builder;
    arrow::Int32Builder second_bid_price_builder;
    arrow::Int32Builder second_bid_vol_builder;
    arrow::Int32Builder second_ask_price_builder;
    arrow::Int32Builder second_ask_vol_builder;
    arrow::DoubleBuilder imbalance_builder;
    arrow::DoubleBuilder mid_builder;
    arrow::Int64Builder timestamp_builder;
    arrow::StringBuilder type_builder;
    arrow::Int8Builder side_builder;
    arrow::Int32Builder price_builder;
    arrow::Int32Builder volume_builder;
    arrow::Int32Builder filled_size_builder;
    arrow::BooleanBuilder rejected_builder;
    arrow::BooleanBuilder partial_builder;
    arrow::DoubleBuilder bias_builder;
    arrow::DoubleBuilder alpha_builder;
    arrow::Int8Builder source_builder;

    for (const auto& r : records) {
        (void)sequence_builder.Append(r.sequence);
        (void)best_bid_price_builder.Append(r.best_bid_price);
        (void)best_bid_vol_builder.Append(r.best_bid_vol);
        (void)best_ask_price_builder.Append(r.best_ask_price);
        (void)best_ask_vol_builder.Append(r.best_ask_vol);
        (void)second_bid_price_builder.Append(r.second_bid_price);
        (void)second_bid_vol_builder.Append(r.second_bid_vol);
        (void)second_ask_price_builder.Append(r.second_ask_price);
        (void)second_ask_vol_builder.Append(r.second_ask_vol);
        (void)imbalance_builder.Append(r.imbalance);
        (void)mid_builder.Append(r.mid);
        (void)timestamp_builder.Append(r.timestamp);
        (void)type_builder.Append(r.type);
        (void)side_builder.Append(static_cast<int8_t>(r.side));
        (void)price_builder.Append(r.price);
        (void)volume_builder.Append(r.volume);
        (void)filled_size_builder.Append(r.filled_size);
        (void)rejected_builder.Append(r.rejected);
        (void)partial_builder.Append(r.partial);
        (void)bias_builder.Append(r.bias);
        (void)alpha_builder.Append(r.alpha);
        (void)source_builder.Append(r.source);
    }

    std::shared_ptr<arrow::Array> sequence_arr;
    std::shared_ptr<arrow::Array> best_bid_price_arr, best_bid_vol_arr;
    std::shared_ptr<arrow::Array> best_ask_price_arr, best_ask_vol_arr;
    std::shared_ptr<arrow::Array> second_bid_price_arr, second_bid_vol_arr;
    std::shared_ptr<arrow::Array> second_ask_price_arr, second_ask_vol_arr;
    std::shared_ptr<arrow::Array> imbalance_arr;
    std::shared_ptr<arrow::Array> mid_arr;
    std::shared_ptr<arrow::Array> timestamp_arr, type_arr, side_arr;
    std::shared_ptr<arrow::Array> price_arr, volume_arr, filled_size_arr, rejected_arr, partial_arr;
    std::shared_ptr<arrow::Array> bias_arr, alpha_arr, source_arr;

    (void)sequence_builder.Finish(&sequence_arr);
    (void)best_bid_price_builder.Finish(&best_bid_price_arr);
    (void)best_bid_vol_builder.Finish(&best_bid_vol_arr);
    (void)best_ask_price_builder.Finish(&best_ask_price_arr);
    (void)best_ask_vol_builder.Finish(&best_ask_vol_arr);
    (void)second_bid_price_builder.Finish(&second_bid_price_arr);
    (void)second_bid_vol_builder.Finish(&second_bid_vol_arr);
    (void)second_ask_price_builder.Finish(&second_ask_price_arr);
    (void)second_ask_vol_builder.Finish(&second_ask_vol_arr);
    (void)imbalance_builder.Finish(&imbalance_arr);
    (void)mid_builder.Finish(&mid_arr);
    (void)timestamp_builder.Finish(&timestamp_arr);
    (void)type_builder.Finish(&type_arr);
    (void)side_builder.Finish(&side_arr);
    (void)price_builder.Finish(&price_arr);
    (void)volume_builder.Finish(&volume_arr);
    (void)filled_size_builder.Finish(&filled_size_arr);
    (void)rejected_builder.Finish(&rejected_arr);
    (void)partial_builder.Finish(&partial_arr);
    (void)bias_builder.Finish(&bias_arr);
    (void)alpha_builder.Finish(&alpha_arr);
    (void)source_builder.Finish(&source_arr);

    auto schema = arrow::schema({
        arrow::field("sequence", arrow::int64()),
        arrow::field("p_-1", arrow::int32()),
        arrow::field("q_-1", arrow::int32()),
        arrow::field("p_1", arrow::int32()),
        arrow::field("q_1", arrow::int32()),
        arrow::field("p_-2", arrow::int32()),
        arrow::field("q_-2", arrow::int32()),
        arrow::field("p_2", arrow::int32()),
        arrow::field("q_2", arrow::int32()),
        arrow::field("imbalance", arrow::float64()),
        arrow::field("mid", arrow::float64()),
        arrow::field("ts_event", arrow::int64()),
        arrow::field("event", arrow::utf8()),
        arrow::field("side", arrow::int8()),
        arrow::field("price", arrow::int32()),
        arrow::field("size", arrow::int32()),
        arrow::field("filled_size", arrow::int32()),
        arrow::field("rejected", arrow::boolean()),
        arrow::field("partial", arrow::boolean()),
        arrow::field("bias", arrow::float64()),
        arrow::field("alpha", arrow::float64()),
        arrow::field("source", arrow::int8())
    });

    auto table = arrow::Table::Make(schema, {
        sequence_arr,
        best_bid_price_arr, best_bid_vol_arr,
        best_ask_price_arr, best_ask_vol_arr,
        second_bid_price_arr, second_bid_vol_arr,
        second_ask_price_arr, second_ask_vol_arr,
        imbalance_arr,
        mid_arr,
        timestamp_arr, type_arr, side_arr,
        price_arr, volume_arr, filled_size_arr, rejected_arr, partial_arr,
        bias_arr, alpha_arr, source_arr
    });

    auto outfile = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    (void)parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, records.size());
}

void AlphaPnL::save_csv(const std::string& path) const {
    std::ofstream file(path);
    file << "lag_sec,quantile,threshold,alpha_tickreturn_cov,alpha_tickreturn_cov_ci\n";
    for (size_t i = 0; i < lag_sec.size(); i++) {
        for (size_t j = 0; j < thresholds.size(); j++){
            file << lag_sec[i] << "," << quantiles[j] << "," << thresholds[j] << "," << alpha_tickreturn_cov[j + i*thresholds.size()] << "," << alpha_tickreturn_cov_ci[j + i*thresholds.size()] << "\n";    
        }
    }
}

AlphaPnL compute_alpha_pnl(const Buffer& buffer, const std::vector<int64_t>& lags_ns, const std::vector<double>& quantiles) {
    AlphaPnL result;
    const auto& records = buffer.records;
    size_t n = records.size();
    size_t m = quantiles.size();

    // Extract arrays
    std::vector<int64_t> timestamps(n);
    std::vector<double> mid(n);
    std::vector<double> alpha(n);
    std::vector<P2Quantile> p2quantiles;
    std::vector<double> thresholds;
    for (size_t j = 0; j < m; j++){
        p2quantiles.push_back(P2Quantile(quantiles[j]));
    }
    
    for (size_t i = 0; i < n; i++) {
        timestamps[i] = records[i].timestamp;
        mid[i] = records[i].mid;
        alpha[i] = records[i].alpha;
        for (size_t j = 0; j < m; j++){
            p2quantiles[j].add(std::abs(alpha[i]));
        }
    }

    for (size_t j = 0; j < m; j++){
        thresholds.push_back(p2quantiles[j].estimate());
        result.quantiles.push_back(quantiles[j]);
        result.thresholds.push_back(thresholds[j]);
    }

    for (int64_t lag : lags_ns) {
        std::vector<double> mean(m);
        std::vector<double> M2(m);
        std::vector<size_t> count(m);
        size_t j = 0;

        for (size_t i = 0; i < n; i++) {
            if (records[i].rejected) continue;
            int64_t target = timestamps[i] + lag;
            while (j < n && timestamps[j] < target) j++;
            if (j < n && mid[i] > 0.0 && mid[j] > 0.0) {
                double sign = (alpha[i] > 0.0) ? 1.0 : -1.0;
                double tick_ret = (mid[j] - mid[i]) * sign;
                for (size_t k = 0; k < m; k++){
                    if (std::abs(alpha[i]) < thresholds[k]) break;
                    count[k]++;
                    double delta = tick_ret - mean[k];
                    mean[k] += delta / static_cast<double>(count[k]);
                    double delta2 = tick_ret - mean[k];
                    M2[k] += delta * delta2;
                }
            }
        }

        result.lag_sec.push_back(static_cast<double>(lag) / 1e9);
        for (size_t k = 0; k < m; k++){
            result.alpha_tickreturn_cov.push_back(mean[k]);
            // Compute 95% CI half-width: 1.96 * SE = 1.96 * sqrt(var / n)
            double ci = 0.0;
            if (count[k] > 1) {
                double variance = M2[k] / static_cast<double>(count[k] - 1);
                double std_err = std::sqrt(variance / static_cast<double>(count[k]));
                ci = 1.96 * std_err;
            }
            result.alpha_tickreturn_cov_ci.push_back(ci);
        }
    }

    return result;
}

Buffer run_simulation(OrderBook& lob, QRModel& model, int64_t duration,
                      Alpha* alpha, MarketImpact* impact) {
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;

    while (time < duration) {
        {
            double alpha_val = alpha ? alpha->value() : 0.0;
            double impact_val = impact ? impact->bias_factor() : 0.0;
            double alpha_scale = alpha ? alpha->scale() : 1.0;
            double total_bias = -alpha_scale * alpha_val + impact_val;
            model.bias(total_bias);
            Order order = model.sample_order(time);

            int64_t dt = model.sample_dt(model.last_event());
            time += dt;
            order.ts = time;
            if (time >= duration) break;

            if (alpha) {
                alpha->step(dt);
            }
            if (impact) {
                impact->step(time);
            }

            // Capture LOB state before processing
            EventRecord base_record;
            base_record.record_lob(lob);
            base_record.bias = total_bias;
            base_record.alpha = alpha_val;

            if (order.type == OrderType::Trade) {
                fills.clear();
                lob.process(order, &fills);

                int64_t order_seq = seq++;
                int32_t filled_size = 0;
                for (const auto& fill : fills) filled_size += fill.size;

                // One trade record with intended size, LOB before
                EventRecord trade_record = base_record;
                trade_record.sequence = order_seq;
                trade_record.timestamp = order.ts;
                trade_record.type = "Trade";
                trade_record.side = order.side;
                trade_record.price = order.price;
                trade_record.volume = order.size;
                trade_record.filled_size = filled_size;
                trade_record.rejected = false;
                trade_record.partial = (filled_size < order.size);
                buffer.records.push_back(trade_record);

                if (impact && filled_size > 0) {
                    impact->add_trade(order.side, filled_size);
                }

                // If partial, record resting limit order (same LOB snapshot)
                if (order.partial) {
                    EventRecord limit_record = base_record;
                    limit_record.sequence = order_seq;
                    limit_record.timestamp = order.ts;
                    limit_record.type = "Add";
                    limit_record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
                    limit_record.price = order.price;
                    limit_record.volume = order.size - filled_size;
                    limit_record.rejected = false;
                    limit_record.partial = true;
                    buffer.records.push_back(limit_record);
                }
            } else {
                lob.process(order);

                EventRecord record = base_record;
                record.sequence = seq++;
                record.record_order(order);
                buffer.records.push_back(record);
            }
        }
    }

    return buffer;
}


void StrategyBuffer::save_parquet(const std::string& path) const {
    arrow::Int64Builder timestamp_builder;
    arrow::DoubleBuilder signal_builder;
    arrow::Int8Builder side_builder;
    arrow::Int32Builder price_builder;
    arrow::Int32Builder size_builder;
    arrow::Int32Builder filled_size_builder;
    arrow::Int32Builder inventory_builder;
    arrow::DoubleBuilder cash_builder;
    arrow::DoubleBuilder pnl_builder;

    for (const auto& r : records) {
        (void)timestamp_builder.Append(r.timestamp);
        (void)signal_builder.Append(r.signal);
        (void)side_builder.Append(r.side);
        (void)price_builder.Append(r.price);
        (void)size_builder.Append(r.size);
        (void)filled_size_builder.Append(r.filled_size);
        (void)inventory_builder.Append(r.inventory);
        (void)cash_builder.Append(r.cash);
        (void)pnl_builder.Append(r.pnl);
    }

    std::shared_ptr<arrow::Array> timestamp_arr, signal_arr, side_arr;
    std::shared_ptr<arrow::Array> price_arr, size_arr, filled_size_arr;
    std::shared_ptr<arrow::Array> inventory_arr, cash_arr, pnl_arr;

    (void)timestamp_builder.Finish(&timestamp_arr);
    (void)signal_builder.Finish(&signal_arr);
    (void)side_builder.Finish(&side_arr);
    (void)price_builder.Finish(&price_arr);
    (void)size_builder.Finish(&size_arr);
    (void)filled_size_builder.Finish(&filled_size_arr);
    (void)inventory_builder.Finish(&inventory_arr);
    (void)cash_builder.Finish(&cash_arr);
    (void)pnl_builder.Finish(&pnl_arr);

    auto schema = arrow::schema({
        arrow::field("ts_event", arrow::int64()),
        arrow::field("signal", arrow::float64()),
        arrow::field("side", arrow::int8()),
        arrow::field("price", arrow::int32()),
        arrow::field("size", arrow::int32()),
        arrow::field("filled_size", arrow::int32()),
        arrow::field("inventory", arrow::int32()),
        arrow::field("cash", arrow::float64()),
        arrow::field("pnl", arrow::float64())
    });

    auto table = arrow::Table::Make(schema, {
        timestamp_arr, signal_arr, side_arr,
        price_arr, size_arr, filled_size_arr,
        inventory_arr, cash_arr, pnl_arr
    });

    auto outfile = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    (void)parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, records.size());
}

void run_simulation_with_strategy(
    OrderBook& lob, QRModel& model, int64_t duration,
    Alpha* alpha, MarketImpact* impact,
    StrategyTrader& strategy)
{
    int64_t time = 0;
    std::vector<Fill> fills;

    while (time < duration) {
        double alpha_val = alpha ? alpha->value() : 0.0;
        double signal = alpha_val;
        bool strategy_wants = strategy.should_trade(signal, lob.spread());

        // Strategy order
        if (strategy_wants) {
            Order strat_order = strategy.generate_order(lob, signal);
            if (strat_order.size > 0) {
                int64_t dt_strat = model.sample_dt(model.last_event());
                time += dt_strat;
                strat_order.ts = time;
                if (time >= duration) break;

                fills.clear();
                lob.process(strat_order, &fills);
                int32_t filled_size = 0;
                for (const auto& f : fills) filled_size += f.size;

                strategy.update(strat_order, filled_size);

                if (impact && filled_size > 0) {
                    impact->add_trade(strat_order.side, filled_size);
                }

                if (alpha) alpha->step(dt_strat);
                if (impact) impact->step(time);
            }
        }

        // QR event
        double alpha_val2 = alpha ? alpha->value() : 0.0;
        double impact_val = impact ? impact->bias_factor() : 0.0;
        double alpha_scale = alpha ? alpha->scale() : 1.0;
        double total_bias = -alpha_scale * alpha_val2 + impact_val;
        model.bias(total_bias);
        Order order = model.sample_order(time);

        int64_t dt = model.sample_dt(model.last_event());
        time += dt;
        order.ts = time;
        if (time >= duration) break;

        if (alpha) alpha->step(dt);
        if (impact) impact->step(time);

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);
            int32_t filled_size = 0;
            for (const auto& f : fills) filled_size += f.size;
            if (impact && filled_size > 0) {
                impact->add_trade(order.side, filled_size);
            }
        } else {
            lob.process(order);
        }
    }
}

// ============================================================================
// DeltaDistrib — truncated normal in log10 space
// ============================================================================

DeltaDistrib::DeltaDistrib(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open delta_distrib CSV: " + csv_path);
    }
    std::string line;
    std::getline(file, line); // header: mu,sigma,lower,upper
    std::getline(file, line);
    std::istringstream ss(line);
    std::string token;
    std::getline(ss, token, ','); mu = std::stod(token);
    std::getline(ss, token, ','); sigma = std::stod(token);
    std::getline(ss, token, ','); lower = std::stod(token);
    std::getline(ss, token, ','); upper = std::stod(token);
}

int64_t DeltaDistrib::sample(std::mt19937_64& rng) const {
    std::normal_distribution<> normal(mu, sigma);
    double x;
    do {
        x = normal(rng);
    } while (x < lower || x > upper);
    return static_cast<int64_t>(std::pow(10.0, x));
}

// ============================================================================
// HFT strategy simulation with race mechanism
// ============================================================================

Buffer run_hft_strategy(
    OrderBook& lob, QRModel& model, int64_t duration,
    MarketImpact* impact,
    StrategyTrader& strategy,
    StrategyBuffer& strat_buffer,
    DeltaDistrib& delta,
    bool strategy_impact)
{
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;
    std::mt19937_64 strat_rng(42);  // strategy latency RNG

    auto process_qr_order = [&](Order& order, double bias_val) {
        EventRecord base_record;
        base_record.record_lob(lob);
        base_record.bias = bias_val;
        base_record.alpha = 0.0;
        base_record.source = SOURCE_QR;

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);

            int64_t order_seq = seq++;
            int32_t filled_size = 0;
            for (const auto& fill : fills) filled_size += fill.size;

            EventRecord trade_record = base_record;
            trade_record.sequence = order_seq;
            trade_record.timestamp = order.ts;
            trade_record.type = "Trade";
            trade_record.side = order.side;
            trade_record.price = order.price;
            trade_record.volume = order.size;
            trade_record.filled_size = filled_size;
            trade_record.rejected = false;
            trade_record.partial = (filled_size < order.size);
            buffer.records.push_back(trade_record);

            if (impact && filled_size > 0) {
                impact->add_trade(order.side, filled_size);
            }

            if (order.partial) {
                EventRecord limit_record = base_record;
                limit_record.sequence = order_seq;
                limit_record.timestamp = order.ts;
                limit_record.type = "Add";
                limit_record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
                limit_record.price = order.price;
                limit_record.volume = order.size - filled_size;
                limit_record.rejected = false;
                limit_record.partial = true;
                limit_record.source = SOURCE_QR;
                buffer.records.push_back(limit_record);
            }
        } else {
            lob.process(order);

            EventRecord record = base_record;
            record.sequence = seq++;
            record.record_order(order);
            buffer.records.push_back(record);
        }
    };

    while (time < duration) {
        double signal = lob.imbalance();
        bool strategy_wants = strategy.should_trade(signal, lob.spread());

        // Pre-sample QR order, then sample dt (needs event from sample_order)
        double impact_val = impact ? impact->bias_factor() : 0.0;
        model.bias(impact_val);
        Order qr_order = model.sample_order(time);
        int64_t dt_qr = model.sample_dt(model.last_event());
        int64_t dt_strat = strategy_wants ? delta.sample(strat_rng) : std::numeric_limits<int64_t>::max();

        if (strategy_wants && dt_strat < dt_qr) {
            // === STRATEGY WINS THE RACE ===
            Order strat_order = strategy.generate_order(lob, signal);
            if (strat_order.size > 0) {
                time += dt_strat;
                strat_order.ts = time;
                if (time >= duration) break;

                // Record LOB state before strategy trade
                EventRecord strat_event;
                strat_event.record_lob(lob);
                strat_event.bias = impact_val;
                strat_event.alpha = 0.0;
                strat_event.source = SOURCE_STRATEGY;

                fills.clear();
                lob.process(strat_order, &fills);
                int32_t filled_size = 0;
                for (const auto& f : fills) filled_size += f.size;

                strategy.update(strat_order, filled_size);

                // Record strategy event in full buffer
                strat_event.sequence = seq++;
                strat_event.timestamp = strat_order.ts;
                strat_event.type = "Trade";
                strat_event.side = strat_order.side;
                strat_event.price = strat_order.price;
                strat_event.volume = strat_order.size;
                strat_event.filled_size = filled_size;
                strat_event.rejected = false;
                strat_event.partial = (filled_size < strat_order.size);
                buffer.records.push_back(strat_event);

                if (strategy_impact && impact && filled_size > 0) {
                    impact->add_trade(strat_order.side, filled_size);
                }

                // Record in strategy buffer
                StrategyRecord sr;
                sr.timestamp = strat_order.ts;
                sr.signal = signal;
                sr.side = static_cast<int8_t>(strat_order.side);
                sr.price = strat_order.price;
                sr.size = strat_order.size;
                sr.filled_size = filled_size;
                sr.inventory = strategy.inventory();
                sr.cash = strategy.cash();
                sr.pnl = strategy.pnl();
                strat_buffer.records.push_back(sr);

                // QR event still happens (remaining time)
                int64_t remaining = dt_qr - dt_strat;
                time += remaining;
                qr_order.ts = time;
                if (time >= duration) break;

                if (impact) impact->step(time);
                process_qr_order(qr_order, impact_val);
            } else {
                // Strategy generated zero-size order, treat as market wins
                time += dt_qr;
                qr_order.ts = time;
                if (time >= duration) break;

                if (impact) impact->step(time);
                process_qr_order(qr_order, impact_val);
            }
        } else {
            // === MARKET WINS (or strategy didn't want to trade) ===
            time += dt_qr;
            qr_order.ts = time;
            if (time >= duration) break;

            if (impact) impact->step(time);
            process_qr_order(qr_order, impact_val);
        }
    }

    return buffer;
}

void run_hft_strategy_stats(
    OrderBook& lob, QRModel& model, int64_t duration,
    MarketImpact* impact,
    StrategyTrader& strategy,
    DeltaDistrib& delta,
    bool strategy_impact)
{
    int64_t time = 0;
    std::vector<Fill> fills;
    std::mt19937_64 strat_rng(42);

    while (time < duration) {
        double signal = lob.imbalance();
        bool strategy_wants = strategy.should_trade(signal, lob.spread());

        double impact_val = impact ? impact->bias_factor() : 0.0;
        model.bias(impact_val);
        Order qr_order = model.sample_order(time);
        int64_t dt_qr = model.sample_dt(model.last_event());
        int64_t dt_strat = strategy_wants ? delta.sample(strat_rng) : std::numeric_limits<int64_t>::max();

        if (strategy_wants && dt_strat < dt_qr) {
            Order strat_order = strategy.generate_order(lob, signal);
            if (strat_order.size > 0) {
                time += dt_strat;
                strat_order.ts = time;
                if (time >= duration) break;

                fills.clear();
                lob.process(strat_order, &fills);
                int32_t filled_size = 0;
                for (const auto& f : fills) filled_size += f.size;

                strategy.update(strat_order, filled_size);

                if (strategy_impact && impact && filled_size > 0)
                    impact->add_trade(strat_order.side, filled_size);

                time += (dt_qr - dt_strat);
                qr_order.ts = time;
                if (time >= duration) break;
                if (impact) impact->step(time);
            } else {
                time += dt_qr;
                qr_order.ts = time;
                if (time >= duration) break;
                if (impact) impact->step(time);
            }
        } else {
            time += dt_qr;
            qr_order.ts = time;
            if (time >= duration) break;
            if (impact) impact->step(time);
        }

        if (qr_order.type == OrderType::Trade) {
            fills.clear();
            lob.process(qr_order, &fills);
            int32_t qr_filled = 0;
            for (const auto& f : fills) qr_filled += f.size;
            if (impact && qr_filled > 0)
                impact->add_trade(qr_order.side, qr_filled);
        } else {
            lob.process(qr_order);
        }
    }
}

}
