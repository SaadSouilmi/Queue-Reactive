#pragma once
#include "types.h"
#include <algorithm>
#include <array>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace qr {

struct QueueDistributions {
	static constexpr int32_t MAX_Q_SIZE = 100;

	// cum_probs[side][level] - cumulative probabilities for each value
	// side: 0=bid, 1=ask
	// level: 0-3 (q1-q4, where q1 is best)
	std::array<std::array<std::vector<double>, 4>, 2> cum_probs;

	// median event sizes per level (1-4), indexed 0-3
	std::array<int32_t, 4> mes{1, 1, 1, 1};

	QueueDistributions() = default;
	QueueDistributions(const std::string &path);

	void set_mes(const std::array<int32_t, 4> &m) { mes = m; }

	int32_t sample(Side side, int level, std::mt19937_64 &rng) const {
		int s = (side == Side::Bid) ? 0 : 1;
		int lvl = std::min(level, 3); // clamp to max level (q4)
		double u = std::uniform_real_distribution<>(0.0, 1.0)(rng);
		const auto &cp = cum_probs[s][lvl];
		auto it = std::lower_bound(cp.begin(), cp.end(), u);
		return static_cast<int32_t>(it - cp.begin()) * mes[lvl];
	}
};

struct Fill {
	int32_t price;
	int32_t size;
};

struct Order {
	OrderType type;
	Side side;
	int32_t price;
	int32_t size;
	int64_t ts;
	bool rejected = false;
	bool partial = false;

	Order() = default;
	Order(OrderType t, Side s, int64_t p, int32_t sz, int64_t timestamp)
		: type(t), side(s), price(p), size(sz), ts(timestamp) {}
};

class OrderBook {
  public:
	OrderBook(const QueueDistributions &dists, int levels = 4, uint64_t seed = 42);

	void init(const std::vector<int32_t> &bid_prices, const std::vector<int32_t> &bid_vols,
			  const std::vector<int32_t> &ask_prices, const std::vector<int32_t> &ask_vols);

	void process(Order &order, std::vector<Fill> *fills = nullptr);

	int32_t best_bid() const;
	int32_t best_ask() const;
	int32_t best_bid_vol() const;
	int32_t best_ask_vol() const;

	int32_t spread() const { return best_ask() - best_bid(); }

	int32_t volume_at(Side side, int32_t price) const {
		const auto &book = (side == Side::Bid) ? bid_ : ask_;
		auto it = book.find(price);
		if (it == book.end()) {
			throw std::runtime_error("Price level not found: " + std::to_string(price));
		}
		return it->second;
	}

	double imbalance() const {
		double bid_v = best_bid_vol();
		double ask_v = best_ask_vol();
		return (bid_v - ask_v) / (bid_v + ask_v);
	}

  private:
	std::map<int32_t, int32_t> bid_;
	std::map<int32_t, int32_t> ask_;
	int levels_;
	const QueueDistributions &dists_;
	std::mt19937_64 rng_;

	void apply_add(Order &order);
	void apply_cancel(Order &order);
	void apply_trade(Order &order, std::vector<Fill> *fills);
	void apply_create_bid(Order &order);
	void apply_create_ask(Order &order);

	void sweep_bid(Order &order, std::vector<Fill> *fills);
	void sweep_ask(Order &order, std::vector<Fill> *fills);

	void clean_bid();
	void clean_ask();

	int32_t sample_bid_level(int i_from_best);
	int32_t sample_ask_level(int i_from_best);
};

}