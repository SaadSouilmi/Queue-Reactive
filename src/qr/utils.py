from pathlib import Path
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import polars as pl
from qr.preprocessing import pl_select

from qr import estimations as est

HOURS_PER_DAY = 5.5
DAY_NS = int(5.5 * 3600 * 1e9)  # 5.5 hours in nanoseconds
HOUR_NS = int(3600 * 1e9)
TICK_TO_USD = 0.01
FIVE_MIN_NS = int(5 * 60 * 1e9)


@contextmanager
def ieee():
    with plt.style.context(["science", "ieee"]), \
         mpl.rc_context({"axes.facecolor": "white", "font.serif": ["Times New Roman", "DejaVu Serif"]}):
        yield

@contextmanager
def transparent():
    with mpl.rc_context({"axes.facecolor": "white"}):
        yield

def compute_volatility_sampled_empirical(
    df: pl.LazyFrame, sample_interval_ns: int = FIVE_MIN_NS
) -> pl.DataFrame:
    T = int(HOURS_PER_DAY * 3600e9 / sample_interval_ns)
    return (
        df.filter(pl.col("event") == "Trade")
        .with_columns(
            (pl.col("ts_event").dt.epoch("ns") // sample_interval_ns).alias(
                "time_bucket"
            )
        )
        .group_by(["date", "time_bucket"])
        .agg(pl.col("price").last())
        .sort(["date", "time_bucket"])
        .with_columns(
            (pl.col("price") * TICK_TO_USD).diff().over("date").alias("price_diff")
        )
        .filter(pl.col("price_diff").is_not_null())
        .group_by("date")
        .agg((pl.col("price_diff") ** 2).sum().alias("realized_var"))
        .with_columns(
            (pl.col("realized_var") / (T - 1)).sqrt().alias("volatility_per_hour")
        )
    )


def compute_volatility_sampled_simulation(
    df: pl.LazyFrame, sample_interval_ns: int = FIVE_MIN_NS
) -> pl.DataFrame:
    T = int(HOURS_PER_DAY * 3600e9 / sample_interval_ns)
    return (
        df.filter(pl.col("event") == "Trade")
        .with_columns(
            (pl.col("ts_event") // sample_interval_ns).alias("time_bucket"),
            (pl.col("ts_event") // DAY_NS).alias("date"),
        )
        .group_by(["date", "time_bucket"])
        .agg(pl.col("price").last())
        .sort(["date", "time_bucket"])
        .with_columns(
            (pl.col("price") * TICK_TO_USD).diff().over("date").alias("price_diff")
        )
        .filter(pl.col("price_diff").is_not_null())
        .group_by("date")
        .agg((pl.col("price_diff") ** 2).sum().alias("realized_var"))
        .with_columns(
            (pl.col("realized_var") / (T - 1)).sqrt().alias("volatility_per_hour")
        )
    )


def preprocess_sim(df: pl.DataFrame, mes: dict[int, int] | None = None) -> pl.DataFrame:
    df = df.with_columns((pl.col("ts_event") // DAY_NS).alias("date"))
    df = df.filter(~pl.col("rejected"))
    df = df.filter(~(pl.col("event").eq("Add") & pl.col("partial")))
    df = df.with_columns(pl.col("ts_event").diff().alias("dt"))
    condlist = [
        *[
            pl.col("imbalance").ge(left) & pl.col("imbalance").lt(right)
            for left, right in zip(-est.BINS[1:][::-1], -est.BINS[:-1][::-1])
        ],
        pl.col("imbalance").eq(0),
        *[
            pl.col("imbalance").gt(left) & pl.col("imbalance").le(right)
            for left, right in zip(est.BINS[:-1], est.BINS[1:])
        ],
    ]
    choicelist = [*(-est.BINS[1:][::-1]), 0, *est.BINS[1:]]
    if mes is not None:
        df = df.with_columns(
            (pl.col("q_-1") / mes[1]).ceil().cast(pl.Int32).alias("q_-1"),
            (pl.col("q_1") / mes[1]).ceil().cast(pl.Int32).alias("q_1"),
            (pl.col("q_-2") / mes[2]).ceil().cast(pl.Int32).alias("q_-2"),
            (pl.col("q_2") / mes[2]).ceil().cast(pl.Int32).alias("q_2"),
            (pl.col("size") / mes[1]).ceil().cast(pl.Int32).alias("size"),
        )
        df = df.with_columns(
            imbalance=(pl.col("q_-1") - pl.col("q_1")) / (pl.col("q_-1") + pl.col("q_1"))
        )
    df = df.with_columns(imbalance=pl_select(condlist, choicelist))
    return df.drop_nulls()


def load_data(ticker: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    from lobib import DataLoader
    loader = DataLoader()

    files = list(Path(f"/home/saad.souilmi/dev_cpp/qr/data/{ticker}/daily_estimates").glob("*.parquet"))
    df = pl.scan_parquet(files)
    df = df.collect()
    dates = df["date"].unique().to_list()
    median_event_sizes, total_best_quantiles = est.load_params(
        f"/home/saad.souilmi/dev_cpp/qr/data/{ticker}/daily_estimates/params.json"
    )
    df_emp = pl.concat(
        [
            est.preprocess(
                loader.load(
                    ticker,
                    start_date=date,
                    end_date=date,
                    schema="qr",
                ),
                median_event_sizes,
                est.BINS,
                total_best_quantiles,
            )
            for date in dates
        ]
    )
    df_emp = df_emp.collect().sort("ts_event")

    return df, df_emp


def _acf_fft(signs: np.ndarray, max_lag: int) -> np.ndarray:
    signs = signs - signs.mean()
    n = len(signs)
    fft_size = 1 << (2 * n - 1).bit_length()
    f = np.fft.fft(signs, fft_size)
    acf = np.real(np.fft.ifft(f * np.conj(f)))[:n]
    acf /= acf[0]
    return acf[: max_lag + 1]


def trade_sign_acf_empirical(df: pl.DataFrame, max_lag: int = 500) -> np.ndarray:
    trades = df.filter(pl.col("event") == "Trade")
    acfs = []
    for _, group in trades.group_by("date"):
        signs = group["side"].to_numpy().astype(float)
        if len(signs) < max_lag + 1:
            continue
        acfs.append(_acf_fft(signs, max_lag))
    return np.mean(acfs, axis=0)


def trade_sign_acf_simulation(df: pl.DataFrame, max_lag: int = 500) -> np.ndarray:
    signs = df.filter(pl.col("event") == "Trade")["side"].cast(pl.Float64).to_numpy()
    return _acf_fft(signs, max_lag)


def _queue_survival_vectorized(prices, q_sizes, ts, death_direction):
    change_idx = np.where(np.diff(prices) != 0)[0] + 1
    if len(change_idx) < 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    # Prepend 0 as first life start
    change_idx = np.concatenate([[0], change_idx])

    # Each life spans change_idx[i] to change_idx[i+1]
    # Death if direction matches death_direction
    directions = np.sign(prices[change_idx[1:]] - prices[change_idx[1:] - 1])
    death_mask = directions == death_direction

    q0s = q_sizes[change_idx[:-1]][death_mask]
    survivals = (ts[change_idx[1:]] - ts[change_idx[:-1]])[death_mask]

    return q0s, survivals


def queue_survival_empirical(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("date", "ts_event")

    mid = df["mid"].to_numpy()
    spread = df["spread"].to_numpy()
    best_ask = mid + spread / 2
    best_bid = mid - spread / 2
    q_ask = df["q_1"].to_numpy()
    q_bid = df["q_-1"].to_numpy()
    ts = df["ts_event"].dt.epoch("ns").to_numpy().astype(np.float64)

    # Day boundaries via single extraction + diff
    dates = df["date"].to_numpy()
    day_starts = np.concatenate(
        [[0], np.where(dates[1:] != dates[:-1])[0] + 1, [len(dates)]]
    )

    all_q0, all_surv = [], []
    for i in range(len(day_starts) - 1):
        s, e = day_starts[i], day_starts[i + 1]
        for prices, q, d in [
            (best_ask[s:e], q_ask[s:e], +1),
            (best_bid[s:e], q_bid[s:e], -1),
        ]:
            q0s, survs = _queue_survival_vectorized(prices, q, ts[s:e], d)
            all_q0.append(q0s)
            all_surv.append(survs)

    return pl.DataFrame(
        {"q0": np.concatenate(all_q0), "survival_ns": np.concatenate(all_surv)}
    )


def queue_survival_simulation(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("ts_event")
    ts = df["ts_event"].to_numpy().astype(np.float64)

    all_q0, all_surv = [], []
    for price_col, q_col, d in [("p_1", "q_1", +1), ("p_-1", "q_-1", -1)]:
        prices = df[price_col].to_numpy().astype(np.float64)
        q = df[q_col].to_numpy()
        q0s, survs = _queue_survival_vectorized(prices, q, ts, d)
        all_q0.append(q0s)
        all_surv.append(survs)

    return pl.DataFrame(
        {"q0": np.concatenate(all_q0), "survival_ns": np.concatenate(all_surv)}
    )