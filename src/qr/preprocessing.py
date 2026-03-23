from pathlib import Path
from functools import reduce
from itertools import chain
from typing import Any, Final

import polars as pl

lob_cols: Final = list(
    chain.from_iterable(
        [
            [
                f"bid_px_0{i}",
                f"ask_px_0{i}",
                f"bid_sz_0{i}",
                f"ask_sz_0{i}",
                f"bid_ct_0{i}",
                f"ask_ct_0{i}",
            ]
            for i in range(10)
        ]
    )
)


def roll(array: list[Any], shift: int) -> list[Any]:
    """Rotate array elements by a specified shift.

    Args:
        array: Input array to be rotated
        shift: Number of positions to rotate (positive for right rotation,
               negative for left rotation)

    Returns:
        Rotated array where elements are shifted circularly by the specified amount
    """
    return array[-(shift % len(array)):] + array[: -(shift % len(array))]


def pl_select(condlist: list[pl.Expr], choicelist: list[pl.Expr]) -> pl.Expr:
    """Implement numpy's select functionality for Polars expressions.

    This function provides similar functionality to numpy.select() but for Polars
    expressions, allowing conditional selection based on multiple conditions.

    Args:
        condlist (list[pl.Expr]): List of conditions as Polars expressions
        choicelist (list[pl.Expr]): List of values to choose from when conditions are met

    Returns:
        pl.Expr: A Polars expression that evaluates to values from choicelist based on
            the first condition in condlist that evaluates to True

    Note:
        Similar to numpy.select (https://numpy.org/doc/stable/reference/generated/numpy.select.html)
        but implemented for Polars expressions
    """
    return reduce(
        lambda expr, cond_choice: expr.when(cond_choice[0]).then(cond_choice[1]),
        zip(condlist, choicelist),
        pl.when(condlist[0]).then(choicelist[0]),
    )


def preprocess_mbp10(df: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess limit order book data.

    This function performs several preprocessing steps to clean and standardize
    the order book data. The preprocessing includes side adjustments,
    trade corrections, and filtering of specific cancellations.

    Args:
        df (pl.LazyFrame): Raw order book data

    Returns:
        pl.LazyFrame: Preprocessed order book data with the following modifications:
            - Neutral ('N') order sides adjusted based on price levels
            - Trade sides swapped for consistency
            - Trade prices corrected using best bid/ask prices
            - Filtered cancellations that follow trades with same sequence

    Note:
        Input dataframe must contain the following columns:
            - action: Order type ('A'=add, 'C'=cancel, 'T'=trade)
            - side: Order side ('B'=bid, 'A'=ask, 'N'=neutral)
            - price: Order price
            - sequence: Order sequence number
            - bid_px_XX: Bid price levels (XX from 00 to 09)
            - ask_px_XX: Ask price levels (XX from 00 to 09)
    """
    # Adjust side for when it is equal to "N"
    bid_prices = [col for col in lob_cols if col.startswith("bid_px")]
    add_side = (
        pl.when(
            pl.any_horizontal([pl.col(col).eq(pl.col("price")) for col in bid_prices])
        )
        .then(pl.lit("B"))
        .otherwise(pl.lit("A"))
    )
    cancel_side = (
        pl.when(
            pl.any_horizontal(
                [pl.col(col).shift().eq(pl.col("price")) for col in bid_prices]
            )
        )
        .then(pl.lit("B"))
        .otherwise(pl.lit("A"))
    )
    df = df.with_columns(
        side=pl.when(pl.col("action").is_in(["A", "C"]) & pl.col("side").eq("N"))
        .then(pl.when(pl.col("action").eq("A")).then(add_side).otherwise(cancel_side))
        .otherwise(pl.col("side"))
    )
    # Swap trade sides
    df = df.with_columns(
        side=pl.when(pl.col("action").eq("T"))
        .then(pl.col("side").replace({"A": "B", "B": "A", "N": "N"}))
        .otherwise(pl.col("side"))
    )
    # Set trades to the right prices
    df = df.with_columns(
        price=pl.when(pl.col("action").eq("T") & pl.col("side").ne("N"))
        .then(
            pl.when(pl.col("side").eq("B"))
            .then(pl.col("bid_px_00"))
            .otherwise("ask_px_00")
        )
        .otherwise(pl.col("price"))
    )
    # Remove cancels that immediatly follow trades and share same sequence
    trade_sequences = pl.col("sequence").filter(pl.col("action").eq("T"))
    df = df.filter(
        ~(pl.col("action").eq("C") & pl.col("sequence").is_in(trade_sequences))
    )

    return df


def truncate_time(
    df: pl.LazyFrame,
    start_time: pl.Expr = pl.time(9, 30),
    end_time: pl.Expr = pl.time(16, 0),
) -> pl.LazyFrame:
    """Filter order book data to keep only records within specified time bounds.

    This function filters the input dataframe to retain only the records that fall
    within the specified time window, based on the ts_event column.

    Args:
        df (pl.LazyFrame): Input dataframe containing order book data
        start_time (pl.Expr, optional): Start time for filtering. Defaults to 10:00
        end_time (pl.Expr, optional): End time for filtering. Defaults to 15:30

    Returns:
        pl.LazyFrame: Filtered dataframe containing only records between
            start_time and end_time. Returns same type as input (DataFrame or LazyFrame)

    Note:
        Input dataframe must contain a 'ts_event' column with timestamp data
    """
    return df.filter(
        pl.col("ts_event").dt.time().ge(start_time)
        & pl.col("ts_event").dt.time().le(end_time)
    )


def prices_to_ticks(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert price columns from decimal to tick (integer) format.

    This function converts all price-related columns to tick format by multiplying by 100
    and rounding to the nearest integer. This includes both the main price column and all
    limit order book price columns.

    Args:
        df (pl.LazyFrame): Input dataframe containing price columns

    Returns:
        pl.LazyFrame: Dataframe with price columns converted to tick format:
            - Price columns (containing '_px'): multiplied by 100, rounded to integers
            - Other LOB columns: cast to Int32
            - Main 'price' column: multiplied by 100, rounded to integer
    """
    return df.with_columns(
        **{
            col: (
                pl.col(col).mul(100).round().cast(pl.Int32)
                if col.split("_")[1] == "px"
                else pl.col(col).cast(pl.Int32)
            )
            for col in lob_cols
        },
        price=pl.col("price").mul(100).round().cast(pl.Int32),
    )


def aggregate_trades(df: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate trades that occur at the same timestamp and price level.

    This function identifies trade events and aggregates them by timestamp and price level,
    summing up the sizes while keeping other attributes from the first trade in each group.
    Non-trade events are preserved as is.

    Args:
        df (pl.LazyFrame): Input dataframe containing order book events

    Returns:
        pl.LazyFrame: Dataframe with:
            - Trades aggregated by timestamp and price
            - Summed sizes for trades at same timestamp and price
            - Other columns preserved from first trade in each group
            - Non-trade events unchanged
            - All events sorted by timestamp and sequence number
    """
    trades = df.filter(pl.col("action").eq("T"))

    return pl.concat(
        [
            df.filter(pl.col("action").ne("T")),
            trades.group_by(["ts_event", "price", "side"])
            .agg(pl.col("size").sum(), pl.exclude("size").first())
            .select(trades.collect_schema().names()),
        ],
        how="vertical",
    ).sort(["ts_event", "sequence"])


def post_update_lob(df: pl.LazyFrame) -> pl.LazyFrame:
    """Update the limit order book (LOB) state after each event.

    This function adjusts the LOB state to reflect changes after each event by updating
    sizes and shifting queue levels as needed.

    Args:
        df (pl.LazyFrame): Input dataframe containing order book events and current LOB state

    Returns:
        pl.LazyFrame: Dataframe with updated LOB state reflecting:
            - Updated sizes at best bid/ask after trades
            - Shifted queue levels when a level is depleted (size <= 0)
            - Nullified empty levels at the end of the book

    Note:
        Input dataframe must contain the following column groups:
            - ask_px_XX: Ask price levels (XX from 00 to 09)
            - ask_sz_XX: Ask size levels (XX from 00 to 09)
            - ask_ct_XX: Ask count levels (XX from 00 to 09)
            - bid_px_XX: Bid price levels (XX from 00 to 09)
            - bid_sz_XX: Bid size levels (XX from 00 to 09)
            - bid_ct_XX: Bid count levels (XX from 00 to 09)
    """
    # Adjust remaining size at best-bid/ask
    df = df.with_columns(
        pl.when(pl.col("action").eq("T") & pl.col("side").eq("A"))
        .then(pl.col("ask_sz_00").sub(pl.col("size")))
        .otherwise(pl.col("ask_sz_00")),
        pl.when(pl.col("action").eq("T") & pl.col("side").eq("B"))
        .then(pl.col("bid_sz_00").sub(pl.col("size")))
        .otherwise(pl.col("bid_sz_00")),
    )

    ask_px = [col for col in lob_cols if col.startswith("ask_px")]
    ask_sz = [col for col in lob_cols if col.startswith("ask_sz")]
    ask_ct = [col for col in lob_cols if col.startswith("ask_ct")]
    bid_px = [col for col in lob_cols if col.startswith("bid_px")]
    bid_sz = [col for col in lob_cols if col.startswith("bid_sz")]
    bid_ct = [col for col in lob_cols if col.startswith("bid_ct")]

    shifted_ask_px = [
        pl.when(pl.col("ask_sz_00").le(0))
        .then(pl.col(shifted_col).alias(col))
        .otherwise(pl.col(col))
        for col, shifted_col in zip(ask_px, roll(ask_px, -1), strict=False)
    ]
    shifted_ask_sz = [
        pl.when(pl.col("ask_sz_00").le(0))
        .then(pl.col(shifted_col).alias(col))
        .otherwise(pl.col(col))
        for col, shifted_col in zip(ask_sz, roll(ask_sz, -1), strict=False)
    ]
    shifted_ask_ct = [
        pl.when(pl.col("ask_sz_00").le(0))
        .then(pl.col(shifted_col).alias(col))
        .otherwise(pl.col(col))
        for col, shifted_col in zip(ask_ct, roll(ask_ct, -1), strict=False)
    ]

    shifted_bid_px = [
        pl.when(pl.col("bid_sz_00").le(0))
        .then(pl.col(shifted_col).alias(col))
        .otherwise(pl.col(col))
        for col, shifted_col in zip(bid_px, roll(bid_px, -1), strict=False)
    ]
    shifted_bid_sz = [
        pl.when(pl.col("bid_sz_00").le(0))
        .then(pl.col(shifted_col).alias(col))
        .otherwise(pl.col(col))
        for col, shifted_col in zip(bid_sz, roll(bid_sz, -1), strict=False)
    ]
    shifted_bid_ct = [
        pl.when(pl.col("bid_sz_00").le(0))
        .then(pl.col(shifted_col).alias(col))
        .otherwise(pl.col(col))
        for col, shifted_col in zip(bid_ct, roll(bid_ct, -1), strict=False)
    ]

    # Shift depleted queues
    df = df.with_columns(
        *shifted_ask_px,
        *shifted_ask_sz,
        *shifted_ask_ct,
        *shifted_bid_px,
        *shifted_bid_sz,
        *shifted_bid_ct,
    )
    df = df.with_columns(
        *[
            pl.when(pl.col("ask_sz_09").le(0)).then(None).otherwise(col).alias(col)
            for col in ["ask_px_09", "ask_sz_09", "ask_ct_09"]
        ],
        *[
            pl.when(pl.col("bid_sz_09").le(0)).then(None).otherwise(col).alias(col)
            for col in ["bid_px_09", "bid_sz_09", "bid_ct_09"]
        ],
    )

    return df


def pre_update_lob(df: pl.LazyFrame) -> pl.LazyFrame:
    """Reconstruct the limit order book (LOB) state before each event.

    This function reconstructs the LOB state as it was before each event by computing
    the post-update state and applying appropriate shifts based on event types.

    Args:
        df (pl.LazyFrame): Input dataframe containing order book events

    Returns:
        pl.LazyFrame: Dataframe with reconstructed pre-event LOB state where:
            - Trade events maintain current LOB state
            - Events following trades use post-trade state
            - Other events use shifted (previous) state
            - First row is excluded (no previous state)

    Note:
        This function depends on post_update_lob() to compute intermediate states
    """
    df_post_update = post_update_lob(df).collect()
    return df.with_columns(
        **{
            col: pl.when(pl.col("action").eq("T"))
            .then(pl.col(col))
            .when(pl.col("action").shift().eq("T") & pl.col("action").ne("T"))
            .then(df_post_update[col].shift())
            .otherwise(pl.col(col).shift())
            for col in lob_cols
        }
    )[1:]


# === Recording LOB State ===

spread: pl.Expr = pl.col("ask_px_00").sub(pl.col("bid_px_00")).alias("spread")
imbalance: pl.Expr = (
    pl.col("bid_sz_00")
    .sub(pl.col("ask_sz_00"))
    .truediv(pl.col("bid_sz_00").add(pl.col("ask_sz_00")))
    .alias("imbalance")
)

bid_prices: dict[str, pl.Expr] = {
    f"P_{-i}": pl.when(spread.mod(2).eq(0))
    .then(pl.col("bid_px_00").add(spread.floordiv(2) - 1))
    .otherwise(pl.col("bid_px_00").add(spread.floordiv(2)))
    .sub(i - 1)
    for i in range(10, 0, -1)
}
ask_prices: dict[str, pl.Expr] = {
    f"P_{i}": pl.when(spread.mod(2).eq(0))
    .then(pl.col("ask_px_00").sub(spread.floordiv(2) - 1))
    .otherwise(pl.col("ask_px_00").sub(spread.floordiv(2)))
    .add(i - 1)
    for i in range(1, 11)
}
prices: dict[str, pl.Expr] = {
    **bid_prices,
    "P_0": pl.when(spread.mod(2).eq(0))
    .then(pl.col("bid_px_00").add(spread.floordiv(2)))
    .otherwise(None),
    **ask_prices,
}

bid_volumes: dict[str, pl.Expr] = {
    f"Q_{-i}": pl_select(
        condlist=[pl.col(f"bid_px_0{j}").eq(bid_prices[f"P_{-i}"]) for j in range(10)],
        choicelist=[pl.col(f"bid_sz_0{j}") for j in range(10)],
    ).fill_null(0)
    for i in range(10, 0, -1)
}
ask_volumes: dict[str, pl.Expr] = {
    f"Q_{i}": pl_select(
        condlist=[pl.col(f"ask_px_0{j}").eq(ask_prices[f"P_{i}"]) for j in range(10)],
        choicelist=[pl.col(f"ask_sz_0{j}") for j in range(10)],
    ).fill_null(0)
    for i in range(1, 11)
}
volumes: dict[str, pl.Expr] = {
    **bid_volumes,
    "Q_0": pl.lit(0),
    **ask_volumes,
}

best_bid_nbr: pl.Expr = pl.max_horizontal(
    [pl.when(volumes[f"Q_{-i}"].gt(0)).then(-i).otherwise(-11) for i in range(1, 11)]
).alias("best_bid_nbr")
best_ask_nbr: pl.Expr = pl.min_horizontal(
    [pl.when(volumes[f"Q_{i}"].gt(0)).then(i).otherwise(11) for i in range(1, 11)]
).alias("best_ask_nbr")

lob_state: dict[str, pl.Expr] = dict(
    best_bid_nbr=best_bid_nbr,
    **volumes,
    best_ask_nbr=best_ask_nbr,
    **prices,
    spread=spread,
    imbalance=imbalance,
)

# === Recording Events ===

event_queue_nbr: pl.Expr = pl_select(
    condlist=[pl.col("price").eq(prices[f"P_{i}"]) for i in range(-10, 11)],
    choicelist=[pl.lit(i) for i in range(-10, 11)],
)
event_queue_size: pl.Expr = pl_select(
    condlist=[event_queue_nbr.eq(i) for i in range(-10, 11)],
    choicelist=[volumes[f"Q_{i}"] for i in range(-10, 11)],
)

trd_all: pl.Expr = pl.col("action").eq("T") & pl.col("size").eq(event_queue_size)
create_new: pl.Expr = (
    pl.col("action").eq("A")
    & event_queue_nbr.lt(best_ask_nbr)
    & event_queue_nbr.gt(best_bid_nbr)
)
create_ask: pl.Expr = create_new & pl.col("side").eq("A")
create_bid: pl.Expr = create_new & pl.col("side").eq("B")

event: pl.Expr = (
    pl.when(trd_all)
    .then(pl.lit("Trd_All"))
    .when(create_ask)
    .then(pl.lit("Create_Ask"))
    .when(create_bid)
    .then(pl.lit("Create_Bid"))
    .otherwise(pl.col("action").replace({"A": "Add", "C": "Can", "T": "Trd"}))
).alias("event")

event_records: dict[str, pl.Expr] = dict(
    ts_event=pl.col("ts_event"),
    sequence=pl.col("sequence"),
    event=event,
    event_size=pl.col("size"),
    price=pl.col("price"),
    event_side=pl.col("side"),
    event_queue_nbr=event_queue_nbr,
    event_queue_size=event_queue_size,
)


def mbp10_to_qr(file: Path) -> pl.LazyFrame:
    """Transform one day of raw Databento MBP-10 data into queue-reactive format.

    Takes a single parquet file of raw MBP-10 messages and produces one row per
    LOB event with the full book state before the event.

    Args:
        file: Path to one day of raw MBP-10 parquet data.

    Returns:
        A LazyFrame where each row is one event. Columns:

        Event metadata:
            ts_event (datetime)     - Event timestamp (timezone-aware)
            sequence (int)          - Exchange sequence number
            event (str)             - Event type: Add, Can, Trd, Trd_All,
                                      Create_Bid, Create_Ask
            event_size (int)        - Size of the event in raw shares
            price (int)             - Price in ticks (original price * 100)
            event_side (str)        - Side: "A" (ask) or "B" (bid)
            event_queue_nbr (int)   - Queue index relative to mid (-10 to +10)
            event_queue_size (int)  - Volume at the target queue before the event

        Book state (before the event):
            Q_-10 .. Q_-1 (int)     - Bid queue volumes (raw shares), -1 = best bid
            Q_0 (int)               - Always 0 (mid, no queue)
            Q_1 .. Q_10 (int)       - Ask queue volumes (raw shares), 1 = best ask
            P_-10 .. P_-1 (int)     - Bid price levels in ticks
            P_0 (int)               - Mid tick (null if spread is odd)
            P_1 .. P_10 (int)       - Ask price levels in ticks
            best_bid_nbr (int)      - Queue index of best bid (e.g. -1)
            best_ask_nbr (int)      - Queue index of best ask (e.g. 1)
            spread (int)            - Bid-ask spread in ticks
            imbalance (float)       - (best_bid_vol - best_ask_vol) /
                                      (best_bid_vol + best_ask_vol)

        Identifiers:
            symbol (str)            - Ticker symbol
            date (date)             - Trading date

    """
    # Preprocessing
    df = pl.scan_parquet(file)
    df = preprocess_mbp10(df)
    df = truncate_time(df, start_time=pl.time(10, 0), end_time=pl.time(15, 30))
    df = prices_to_ticks(df)
    df = aggregate_trades(df)
    df = df.filter(pl.col("side").ne("N") & pl.col("action").is_in(["T", "A", "C"]))
    df = pre_update_lob(df)

    return df.select(
        symbol=pl.col("symbol"),
        date=pl.col("ts_event").dt.date(),
        **event_records,
        **lob_state,
    ).drop_nulls(subset="event_queue_nbr")