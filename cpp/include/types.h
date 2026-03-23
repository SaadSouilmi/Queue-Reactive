#pragma once
#include <cstdint>

namespace qr {
    enum class Side: int8_t {
        Bid = -1,
        Ask = 1
    };

    enum class OrderType: uint8_t {
        Add = 0,
        Cancel = 1,
        Trade = 2,
        CreateBid = 3,
        CreateAsk = 4
    };

    // QR Model state (imbalance_bin, spread)
    struct QRState {
        uint8_t imbalance_bin;
        int32_t spread;
    };


    };