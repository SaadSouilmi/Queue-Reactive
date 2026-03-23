#include <gtest/gtest.h>
#include "orderbook.h"

using namespace qr;

// ============================================================================
// Fixture: 4-level book with MES={400, 300, 200, 100}
//
// Raw init volumes:
//   Bids: 1516(4) 1517(1) 1518(10) 1519(5)
//   Asks: 1520(3) 1521(17) 1522(22) 1523(23)
//
// After MES scaling (level 0=best, mes[0]=400, mes[1]=300, ...):
//   Bids: 1519(2000) 1518(3000) 1517(200) 1516(400)
//   Asks: 1520(1200) 1521(5100) 1522(4400) 1523(2300)
// ============================================================================

class OrderBookTest : public ::testing::Test {
protected:
    void SetUp() override {
        dists_.set_mes({400, 300, 200, 100});
        lob = std::make_unique<OrderBook>(dists_, 4);
        lob->init({1516, 1517, 1518, 1519},
                  {4, 1, 10, 5},
                  {1520, 1521, 1522, 1523},
                  {3, 17, 22, 23});
    }

    QueueDistributions dists_;
    std::unique_ptr<OrderBook> lob;
};

// ============================================================================
// Initialization
// ============================================================================

TEST_F(OrderBookTest, BestPrices) {
    EXPECT_EQ(lob->best_bid(), 1519);
    EXPECT_EQ(lob->best_ask(), 1520);
}

TEST_F(OrderBookTest, BestVolumes) {
    EXPECT_EQ(lob->best_bid_vol(), 2000);  // 5 * 400
    EXPECT_EQ(lob->best_ask_vol(), 1200);  // 3 * 400
}

TEST_F(OrderBookTest, Spread) {
    EXPECT_EQ(lob->spread(), 1);
}

TEST_F(OrderBookTest, Imbalance) {
    // (2000 - 1200) / (2000 + 1200) = 800/3200 = 0.25
    EXPECT_NEAR(lob->imbalance(), 0.25, 1e-6);
}

// ============================================================================
// Add
// ============================================================================

TEST_F(OrderBookTest, AddToBid) {
    Order o(OrderType::Add, Side::Bid, 1519, 400, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_EQ(lob->best_bid_vol(), 2400);  // 2000 + 400
}

TEST_F(OrderBookTest, AddToAsk) {
    Order o(OrderType::Add, Side::Ask, 1520, 800, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_EQ(lob->best_ask_vol(), 2000);  // 1200 + 800
}

TEST_F(OrderBookTest, AddToDeepLevel) {
    Order o(OrderType::Add, Side::Bid, 1516, 100, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_EQ(lob->best_bid(), 1519);
    EXPECT_EQ(lob->best_bid_vol(), 2000);       // best unchanged
    EXPECT_EQ(lob->volume_at(Side::Bid, 1516), 500);  // 400 + 100
}

TEST_F(OrderBookTest, AddToNonExistentPriceRejected) {
    Order o(OrderType::Add, Side::Bid, 9999, 100, 0);
    lob->process(o);

    EXPECT_TRUE(o.rejected);
}

TEST_F(OrderBookTest, AddToNonExistentAskPriceRejected) {
    Order o(OrderType::Add, Side::Ask, 1500, 100, 0);
    lob->process(o);

    EXPECT_TRUE(o.rejected);
}

// ============================================================================
// Cancel
// ============================================================================

TEST_F(OrderBookTest, CancelPartialFromAsk) {
    Order o(OrderType::Cancel, Side::Ask, 1520, 400, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_FALSE(o.partial);
    EXPECT_EQ(lob->best_ask_vol(), 800);   // 1200 - 400
}

TEST_F(OrderBookTest, CancelEntireBestAsk) {
    Order o(OrderType::Cancel, Side::Ask, 1520, 1200, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_EQ(lob->best_ask(), 1521);
}

TEST_F(OrderBookTest, CancelExceedsVolumeSetsPartial) {
    Order o(OrderType::Cancel, Side::Ask, 1520, 50000, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_TRUE(o.partial);
    EXPECT_EQ(lob->best_ask(), 1521);
}

TEST_F(OrderBookTest, CancelNonExistentPriceRejected) {
    Order o(OrderType::Cancel, Side::Bid, 9999, 100, 0);
    lob->process(o);

    EXPECT_TRUE(o.rejected);
}

TEST_F(OrderBookTest, CancelEntireBestBid) {
    Order o(OrderType::Cancel, Side::Bid, 1519, 2000, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_EQ(lob->best_bid(), 1518);
}

// ============================================================================
// Trade — single level
// ============================================================================

TEST_F(OrderBookTest, SellSingleLevel) {
    Order o(OrderType::Trade, Side::Bid, 1519, 800, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_FALSE(o.partial);
    EXPECT_EQ(lob->best_bid_vol(), 1200);  // 2000 - 800
}

TEST_F(OrderBookTest, BuySingleLevel) {
    Order o(OrderType::Trade, Side::Ask, 1520, 500, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_FALSE(o.partial);
    EXPECT_EQ(lob->best_ask_vol(), 700);   // 1200 - 500
}

// ============================================================================
// Trade — multi-level sweep
// ============================================================================

TEST_F(OrderBookTest, SellMultipleLevels) {
    // Sweeps: 2000@1519 + 3000@1518 + 200@1517 + 400@1516 = 5600
    Order o(OrderType::Trade, Side::Bid, 1516, 5600, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_FALSE(o.partial);
}

TEST_F(OrderBookTest, BuyMultipleLevels) {
    // Sweeps: 1200@1520 + 5100@1521 + 700 from 1522 = 7000
    Order o(OrderType::Trade, Side::Ask, 1522, 7000, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_FALSE(o.partial);
}

TEST_F(OrderBookTest, SellExactMultiLevelFill) {
    // 2000@1519 + 3000@1518 = 5000 exactly
    Order o(OrderType::Trade, Side::Bid, 1518, 5000, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_FALSE(o.partial);
    EXPECT_EQ(lob->best_bid(), 1517);
    EXPECT_EQ(lob->best_ask(), 1520);
}

// ============================================================================
// Trade — rejected
// ============================================================================

TEST_F(OrderBookTest, SellAboveBestBidRejected) {
    Order o(OrderType::Trade, Side::Bid, 1525, 100, 0);
    lob->process(o);

    EXPECT_TRUE(o.rejected);
}

TEST_F(OrderBookTest, BuyBelowBestAskRejected) {
    Order o(OrderType::Trade, Side::Ask, 1515, 100, 0);
    lob->process(o);

    EXPECT_TRUE(o.rejected);
}

// ============================================================================
// Marketable limit — residual posting
// ============================================================================

TEST_F(OrderBookTest, SellWithResidualPostsToAsk) {
    // Sell 3000 @ limit 1519: consumes 2000@1519, remaining 1000 posts as ask@1519
    Order o(OrderType::Trade, Side::Bid, 1519, 3000, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_TRUE(o.partial);
    EXPECT_EQ(lob->best_bid(), 1518);
    EXPECT_EQ(lob->best_ask(), 1519);
}

TEST_F(OrderBookTest, BuyWithResidualPostsToBid) {
    // Buy 2000 @ limit 1520: consumes 1200@1520, remaining 800 posts as bid@1520
    Order o(OrderType::Trade, Side::Ask, 1520, 2000, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_TRUE(o.partial);
    EXPECT_EQ(lob->best_ask(), 1521);
    EXPECT_EQ(lob->best_bid(), 1520);
}

TEST_F(OrderBookTest, SellDeepSweepWithResidual) {
    // Sell 6000 @ limit 1518: consumes 2000@1519 + 3000@1518 = 5000, remaining 1000 posts as ask@1518
    Order o(OrderType::Trade, Side::Bid, 1518, 6000, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_TRUE(o.partial);
    EXPECT_EQ(lob->best_bid(), 1517);
    EXPECT_EQ(lob->best_ask(), 1518);
}

TEST_F(OrderBookTest, BuyDeepSweepWithResidual) {
    // Buy 7000 @ limit 1521: consumes 1200@1520 + 5100@1521 = 6300, remaining 700 posts as bid@1521
    Order o(OrderType::Trade, Side::Ask, 1521, 7000, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_TRUE(o.partial);
    EXPECT_EQ(lob->best_ask(), 1522);
    EXPECT_EQ(lob->best_bid(), 1521);
}

TEST_F(OrderBookTest, ResidualAddsToExistingLevel) {
    // Buy 2000 @ limit 1520: consumes 1200@1520, remaining 800 posts as bid@1520
    Order o1(OrderType::Trade, Side::Ask, 1520, 2000, 0);
    lob->process(o1);
    EXPECT_EQ(lob->best_bid(), 1520);

    // Buy 6000 @ limit 1521: consumes 5100@1521, remaining 900 posts as bid@1521
    Order o2(OrderType::Trade, Side::Ask, 1521, 6000, 0);
    lob->process(o2);
    EXPECT_EQ(lob->best_bid(), 1521);
}

// ============================================================================
// Create
// ============================================================================

TEST_F(OrderBookTest, CreateBidInsideSpread) {
    // Widen spread first: cancel best ask so best_ask becomes 1521
    Order cancel(OrderType::Cancel, Side::Ask, 1520, 1200, 0);
    lob->process(cancel);
    EXPECT_EQ(lob->best_ask(), 1521);  // spread is now 2

    Order o(OrderType::CreateBid, Side::Bid, 1520, 800, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_EQ(lob->best_bid(), 1520);
    EXPECT_EQ(lob->best_bid_vol(), 800);
}

TEST_F(OrderBookTest, CreateAskInsideSpread) {
    // Widen spread first: cancel best bid so best_bid becomes 1518
    Order cancel(OrderType::Cancel, Side::Bid, 1519, 2000, 0);
    lob->process(cancel);
    EXPECT_EQ(lob->best_bid(), 1518);  // spread is now 2

    Order o(OrderType::CreateAsk, Side::Ask, 1519, 600, 0);
    lob->process(o);

    EXPECT_FALSE(o.rejected);
    EXPECT_EQ(lob->best_ask(), 1519);
    EXPECT_EQ(lob->best_ask_vol(), 600);
}

TEST_F(OrderBookTest, CreateBidAtAskPriceRejected) {
    Order o(OrderType::CreateBid, Side::Bid, 1520, 800, 0);
    lob->process(o);

    EXPECT_TRUE(o.rejected);
    EXPECT_EQ(lob->best_bid(), 1519);  // unchanged
}

TEST_F(OrderBookTest, CreateAskAtBidPriceRejected) {
    Order o(OrderType::CreateAsk, Side::Ask, 1519, 600, 0);
    lob->process(o);

    EXPECT_TRUE(o.rejected);
    EXPECT_EQ(lob->best_ask(), 1520);  // unchanged
}

// ============================================================================
// Clean — maintains levels_ levels
// ============================================================================

TEST_F(OrderBookTest, CleanMaintainsLevelCount) {
    Order o(OrderType::Cancel, Side::Bid, 1519, 2000, 0);
    lob->process(o);

    EXPECT_EQ(lob->best_bid(), 1518);
    EXPECT_EQ(lob->spread(), 2);
}

TEST_F(OrderBookTest, CleanAfterSweepEntireSide) {
    // Sweep all 4 bid levels: 2000+3000+200+400 = 5600, send 7000 so there's residual
    Order o(OrderType::Trade, Side::Bid, 1516, 7000, 0);
    lob->process(o);

    EXPECT_NO_THROW(lob->best_bid());
    EXPECT_NO_THROW(lob->best_ask());
    EXPECT_GE(lob->spread(), 1);
}

// ============================================================================
// Fills recording
// ============================================================================

TEST_F(OrderBookTest, FillsSingleLevel) {
    std::vector<Fill> fills;
    Order o(OrderType::Trade, Side::Bid, 1519, 1000, 0);
    lob->process(o, &fills);

    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].price, 1519);
    EXPECT_EQ(fills[0].size, 1000);
}

TEST_F(OrderBookTest, FillsMultipleLevels) {
    std::vector<Fill> fills;
    // Sell: 2000@1519 + 3000@1518 + 200@1517
    Order o(OrderType::Trade, Side::Bid, 1517, 5200, 0);
    lob->process(o, &fills);

    ASSERT_EQ(fills.size(), 3u);
    EXPECT_EQ(fills[0].price, 1519);
    EXPECT_EQ(fills[0].size, 2000);
    EXPECT_EQ(fills[1].price, 1518);
    EXPECT_EQ(fills[1].size, 3000);
    EXPECT_EQ(fills[2].price, 1517);
    EXPECT_EQ(fills[2].size, 200);
}

TEST_F(OrderBookTest, FillsBuySide) {
    std::vector<Fill> fills;
    // Buy: 1200@1520 + 5100@1521
    Order o(OrderType::Trade, Side::Ask, 1521, 6300, 0);
    lob->process(o, &fills);

    ASSERT_EQ(fills.size(), 2u);
    EXPECT_EQ(fills[0].price, 1520);
    EXPECT_EQ(fills[0].size, 1200);
    EXPECT_EQ(fills[1].price, 1521);
    EXPECT_EQ(fills[1].size, 5100);
}

TEST_F(OrderBookTest, FillsWithResidual) {
    std::vector<Fill> fills;
    // Sell 3000 @ 1519: fills 2000, remaining 1000 rests
    Order o(OrderType::Trade, Side::Bid, 1519, 3000, 0);
    lob->process(o, &fills);

    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].price, 1519);
    EXPECT_EQ(fills[0].size, 2000);
    EXPECT_TRUE(o.partial);

    int32_t resting = o.size;
    for (const auto& f : fills) resting -= f.size;
    EXPECT_EQ(resting, 1000);
}

TEST_F(OrderBookTest, FillsNullptrDoesNotCrash) {
    Order o(OrderType::Trade, Side::Bid, 1519, 800, 0);
    lob->process(o, nullptr);

    EXPECT_FALSE(o.rejected);
    EXPECT_EQ(lob->best_bid_vol(), 1200);  // 2000 - 800
}

TEST_F(OrderBookTest, FillsEmptyOnRejected) {
    std::vector<Fill> fills;
    Order o(OrderType::Trade, Side::Bid, 1525, 100, 0);
    lob->process(o, &fills);

    EXPECT_TRUE(o.rejected);
    EXPECT_EQ(fills.size(), 0u);
}

TEST_F(OrderBookTest, FillsEmptyOnNonTrade) {
    std::vector<Fill> fills;
    Order o(OrderType::Add, Side::Bid, 1519, 100, 0);
    lob->process(o, &fills);

    EXPECT_EQ(fills.size(), 0u);
}

// ============================================================================
// Imbalance after operations
// ============================================================================

TEST_F(OrderBookTest, ImbalanceAfterBuy) {
    Order o(OrderType::Trade, Side::Ask, 1520, 800, 0);
    lob->process(o);

    // bid_vol=2000, ask_vol=400 -> (2000-400)/(2000+400) = 1600/2400
    EXPECT_NEAR(lob->imbalance(), 1600.0 / 2400.0, 1e-3);
}

TEST_F(OrderBookTest, ImbalanceAfterAdd) {
    Order o(OrderType::Add, Side::Ask, 1520, 2000, 0);
    lob->process(o);

    // bid_vol=2000, ask_vol=3200 -> (2000-3200)/(2000+3200) = -1200/5200
    EXPECT_NEAR(lob->imbalance(), -1200.0 / 5200.0, 1e-3);
}

// ============================================================================
// Composite sequences
// ============================================================================

TEST_F(OrderBookTest, AddCancelTrade) {
    Order add(OrderType::Add, Side::Bid, 1519, 1000, 0);
    lob->process(add);
    EXPECT_EQ(lob->best_bid_vol(), 3000);  // 2000 + 1000

    Order cancel(OrderType::Cancel, Side::Bid, 1519, 500, 0);
    lob->process(cancel);
    EXPECT_EQ(lob->best_bid_vol(), 2500);  // 3000 - 500

    Order trade(OrderType::Trade, Side::Bid, 1519, 1500, 0);
    lob->process(trade);
    EXPECT_EQ(lob->best_bid_vol(), 1000);  // 2500 - 1500
}

TEST_F(OrderBookTest, SpreadAfterCreateBid) {
    // Widen spread, then create inside it
    Order cancel(OrderType::Cancel, Side::Ask, 1520, 1200, 0);
    lob->process(cancel);

    Order bid(OrderType::CreateBid, Side::Bid, 1520, 800, 0);
    lob->process(bid);

    EXPECT_GE(lob->spread(), 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
