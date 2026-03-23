#include "orderbook.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace qr {

QueueDistributions::QueueDistributions(const std::string &path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Cannot open " + path);
	}

	std::array<std::array<std::vector<double>, 4>, 2> probs;

	std::string line;
	std::getline(file, line); // skip header

	while (std::getline(file, line)) {
		std::istringstream ss(line);
		std::string token;

		std::getline(ss, token, ',');
		int l = std::stoi(token) - 1;

		cum_probs[0][l].resize(MAX_Q_SIZE + 1);
		cum_probs[1][l].resize(MAX_Q_SIZE + 1);
		double cumsum = 0.0;
		for (int i = 0; i < MAX_Q_SIZE + 1; i++) {
			std::getline(ss, token, ',');
			cumsum += std::stod(token);
			cum_probs[0][l][i] = cumsum;
			cum_probs[1][l][i] = cumsum;
		}
	}

	file.close();
}

OrderBook::OrderBook(const QueueDistributions &dists, int levels, uint64_t seed)
	: levels_(levels), dists_(dists), rng_(seed) {}

void OrderBook::init(const std::vector<int32_t> &bid_prices, const std::vector<int32_t> &bid_vols,
					 const std::vector<int32_t> &ask_prices, const std::vector<int32_t> &ask_vols) {
	bid_.clear();
	ask_.clear();

	// Insert raw volumes first
	for (size_t i = 0; i < bid_prices.size(); ++i) {
		bid_[bid_prices[i]] = bid_vols[i];
	}
	for (size_t i = 0; i < ask_prices.size(); ++i) {
		ask_[ask_prices[i]] = ask_vols[i];
	}

	// Scale by MES based on distance from best
	int level = 0;
	for (auto it = bid_.rbegin(); it != bid_.rend(); ++it) {
		it->second *= dists_.mes[std::min(level, 3)];
		level++;
	}
	level = 0;
	for (auto it = ask_.begin(); it != ask_.end(); ++it) {
		it->second *= dists_.mes[std::min(level, 3)];
		level++;
	}
}

int32_t OrderBook::best_bid() const {
	if (bid_.empty()) {
		throw std::runtime_error("Empty bid side");
	}
	return bid_.rbegin()->first;
}

int32_t OrderBook::best_ask() const {
	if (ask_.empty()) {
		throw std::runtime_error("Empty ask side");
	}
	return ask_.begin()->first;
}

int32_t OrderBook::best_bid_vol() const {
	if (bid_.empty()) {
		throw std::runtime_error("Empty bid side");
	}
	return bid_.rbegin()->second;
}

int32_t OrderBook::best_ask_vol() const {
	if (ask_.empty()) {
		throw std::runtime_error("Empty ask side");
	}
	return ask_.begin()->second;
}

void OrderBook::process(Order &order, std::vector<Fill> *fills) {
	switch (order.type) {
	case OrderType::Add:
		apply_add(order);
		break;
	case OrderType::Cancel:
		apply_cancel(order);
		break;
	case OrderType::Trade:
		apply_trade(order, fills);
		break;
	case OrderType::CreateBid:
		apply_create_bid(order);
		break;
	case OrderType::CreateAsk:
		apply_create_ask(order);
		break;
	default:
		throw std::runtime_error("Unknown order type");
	}
}

void OrderBook::apply_add(Order &order) {
	if (order.rejected)
		return;

	auto &side = (order.side == Side::Bid) ? bid_ : ask_;

	auto it = side.find(order.price);
	if (it != side.end()) {
		it->second += order.size;
	} else {
		order.rejected = true;
	}
}

void OrderBook::apply_cancel(Order &order) {
	if (order.rejected)
		return;

	auto &side = (order.side == Side::Bid) ? bid_ : ask_;

	auto it = side.find(order.price);
	if (it == side.end()) {
		order.rejected = true;
		return;
	}

	if (it->second < order.size) {
		order.partial = true;
		it->second = 0;
	} else {
		it->second -= order.size;
	}

	if (order.side == Side::Bid) {
		clean_bid();
	} else {
		clean_ask();
	}
}

void OrderBook::apply_trade(Order &order, std::vector<Fill> *fills) {
	if (order.side == Side::Bid) {
		sweep_bid(order, fills);
	} else {
		sweep_ask(order, fills);
	}
}

void OrderBook::apply_create_bid(Order &order) {
	if (order.rejected)
		return;
	if (!ask_.empty() && order.price >= best_ask()) {
		order.rejected = true;
		return;
	}
	bid_[order.price] = order.size;
	clean_bid();
}

void OrderBook::apply_create_ask(Order &order) {
	if (order.rejected)
		return;
	if (!bid_.empty() && order.price <= best_bid()) {
		order.rejected = true;
		return;
	}
	ask_[order.price] = order.size;
	clean_ask();
}

void OrderBook::sweep_bid(Order &order, std::vector<Fill> *fills) {
	int32_t remaining = order.size;

	if (bid_.empty() || order.price > best_bid()) {
		order.rejected = true;
		return;
	}

	while (remaining > 0 && !bid_.empty()) {
		auto it = bid_.rbegin();
		int32_t curr_price = it->first;
		int32_t available = it->second;

		if (curr_price < order.price) {
			break;
		}

		int32_t traded = std::min(remaining, available);

		if (fills) {
			fills->push_back({curr_price, traded});
		}

		remaining -= traded;

		auto fwd_it = std::prev(it.base());
		if (traded >= available) {
			bid_.erase(fwd_it);
		} else {
			fwd_it->second -= traded;
		}
	}

	clean_bid();

	if (remaining > 0) {
		order.partial = true;

		if (ask_.find(order.price) != ask_.end()) {
			ask_[order.price] += remaining;
		} else {
			ask_[order.price] = remaining;
		}
		clean_ask();
	} else if (remaining == order.size) {
		order.rejected = true;
	}
}

void OrderBook::sweep_ask(Order &order, std::vector<Fill> *fills) {
	int32_t remaining = order.size;

	if (ask_.empty() || order.price < best_ask()) {
		order.rejected = true;
		return;
	}

	while (remaining > 0 && !ask_.empty()) {
		auto it = ask_.begin(); // Best ask
		int32_t curr_price = it->first;
		int32_t available = it->second;

		if (curr_price > order.price) {
			break;
		}

		int32_t traded = std::min(remaining, available);

		if (fills) {
			fills->push_back({curr_price, traded});
		}

		remaining -= traded;

		if (traded >= available) {
			ask_.erase(it);
		} else {
			it->second -= traded;
		}
	}

	clean_ask();

	if (remaining > 0) {
		order.partial = true;

		if (bid_.find(order.price) != bid_.end()) {
			bid_[order.price] += remaining;
		} else {
			bid_[order.price] = remaining;
		}
		clean_bid();
	} else if (remaining == order.size) {
		order.rejected = true;
	}
}

void OrderBook::clean_bid() {
	// Step 1: Remove zero-volume levels
	for (auto it = bid_.begin(); it != bid_.end();) {
		if (it->second == 0) {
			it = bid_.erase(it);
		} else {
			++it;
		}
	}

	// Step 2: If bid side is empty, create a new best bid
	if (bid_.empty()) {
		if (ask_.empty()) {
			throw std::runtime_error("Both bid and ask sides are empty");
		}
		int32_t new_price = best_ask() - 5;
		bid_[new_price] = sample_bid_level(0);
	}

	// Step 3: Ensure we have 'levels_' price levels
	int32_t best = best_bid();
	for (int i = 1; i < levels_; ++i) {
		int32_t price = best - i;
		if (bid_.find(price) == bid_.end()) {
			bid_[price] = sample_bid_level(i);
		}
	}

	// Step 4: Remove levels beyond our range
	int32_t min_allowed = best - levels_ + 1;
	for (auto it = bid_.begin(); it != bid_.end();) {
		if (it->first < min_allowed) {
			it = bid_.erase(it);
		} else {
			++it;
		}
	}
}

void OrderBook::clean_ask() {
	// Step 1: Remove zero-volume levels
	for (auto it = ask_.begin(); it != ask_.end();) {
		if (it->second == 0) {
			it = ask_.erase(it);
		} else {
			++it;
		}
	}

	// Step 2: If ask side is empty, create a new best ask
	if (ask_.empty()) {
		if (bid_.empty()) {
			throw std::runtime_error("Both bid and ask sides are empty");
		}
		int32_t new_price = best_bid() + 5;
		ask_[new_price] = sample_ask_level(0);
	}

	// Step 3: Ensure we have 'levels_' price levels
	int32_t best = best_ask();
	for (int i = 1; i < levels_; ++i) {
		int32_t price = best + i;
		if (ask_.find(price) == ask_.end()) {
			ask_[price] = sample_ask_level(i);
		}
	}

	// Step 4: Remove levels beyond our range
	int32_t max_allowed = best + levels_ - 1;
	for (auto it = ask_.begin(); it != ask_.end();) {
		if (it->first > max_allowed) {
			it = ask_.erase(it);
		} else {
			++it;
		}
	}
}

int32_t OrderBook::sample_bid_level(int i_from_best) {
	return dists_.sample(Side::Bid, i_from_best, rng_);
}

int32_t OrderBook::sample_ask_level(int i_from_best) {
	return dists_.sample(Side::Ask, i_from_best, rng_);
}

} // namespace qr
