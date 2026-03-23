#pragma once
#include "orderbook.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace qr {

struct StrategyParams {
    int32_t q_max;          // max order size
    int32_t max_inventory;  // max absolute inventory
    double threshold;       // |signal| must exceed this to trade
};

struct StrategyRecord {
    int64_t timestamp;
    double signal;          // alpha + imbalance at decision time
    int8_t side;
    int32_t price;
    int32_t size;           // intended
    int32_t filled_size;
    int32_t inventory;      // after trade
    double cash;
    double pnl;             // running sum
};

struct StrategyBuffer {
    std::vector<StrategyRecord> records;
    void save_parquet(const std::string& path) const;
};

class StrategyTrader {
public:
    StrategyTrader(const StrategyParams& params) : params_(params) {}

    // Returns true if |signal| > threshold AND inventory limit not reached
    bool should_trade(double signal, int32_t spread = 1) const {
        if (spread > 1) return false;
        if (std::abs(signal) <= params_.threshold) return false;
        if (signal > 0) {
            return (params_.max_inventory - inventory_) > 0;
        } else {
            return (params_.max_inventory + inventory_) > 0;
        }
    }

    // Side: Ask if signal > 0 (buy at best_ask), Bid if signal < 0 (sell at best_bid)
    // Size: min(Q_max, available_at_best, remaining_inventory_capacity)
    Order generate_order(const OrderBook& lob, double signal) const {
        Side side = (signal > 0) ? Side::Ask : Side::Bid;
        int32_t price, available, remaining;

        if (side == Side::Ask) {
            // Buying: lift the ask
            price = lob.best_ask();
            available = lob.best_ask_vol();
            remaining = params_.max_inventory - inventory_;
        } else {
            // Selling: hit the bid
            price = lob.best_bid();
            available = lob.best_bid_vol();
            remaining = params_.max_inventory + inventory_;
        }

        int32_t size = std::min({params_.q_max, available, remaining});

        Order order;
        order.type = OrderType::Trade;
        order.side = side;
        order.price = price;
        order.size = size;
        return order;
    }

    // Update inventory and cash based on fill, track accumulators
    void update(const Order& order, int32_t filled_size) {
        if (filled_size <= 0) return;
        n_trades_++;
        total_volume_ += filled_size;
        if (order.side == Side::Ask) {
            // Buying (lifted ask)
            inventory_ += filled_size;
            cash_ -= static_cast<double>(order.price) * filled_size;
        } else {
            // Selling (hit bid)
            inventory_ -= filled_size;
            cash_ += static_cast<double>(order.price) * filled_size;
        }
        // Round trip completed when inventory returns to 0
        if (inventory_ == 0) {
            n_round_trips_++;
            cash_at_last_zero_ = cash_;
            volume_at_last_zero_ = total_volume_;
        }
    }

    // Basic state
    double pnl() const { return cash_; }
    int32_t inventory() const { return inventory_; }
    double cash() const { return cash_; }

    // Accumulators
    int64_t n_trades() const { return n_trades_; }
    int64_t total_volume() const { return total_volume_; }
    int64_t n_round_trips() const { return n_round_trips_; }
    double realized_pnl() const { return cash_at_last_zero_; }
    int64_t volume_at_last_zero() const { return volume_at_last_zero_; }
    double edge_per_share() const {
        return volume_at_last_zero_ > 0
            ? cash_at_last_zero_ / static_cast<double>(volume_at_last_zero_) : 0.0;
    }

private:
    StrategyParams params_;
    int32_t inventory_ = 0;
    double cash_ = 0.0;

    // Accumulators
    int64_t n_trades_ = 0;
    int64_t total_volume_ = 0;
    int64_t n_round_trips_ = 0;
    double cash_at_last_zero_ = 0.0;
    int64_t volume_at_last_zero_ = 0;
};

}
