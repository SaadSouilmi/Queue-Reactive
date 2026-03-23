import json
from pathlib import Path

import numpy as np
import polars as pl
from datasketches import kll_ints_sketch
from qr.preprocessing import pl_select
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

N_BINS = 11
BINS = np.arange(11, step=1) / 10


def save_params(
    path: Path,
    median_event_sizes: dict[int, int],
    total_best_quantiles: np.ndarray,
) -> None:
    data = {
        "median_event_sizes": median_event_sizes,
        "total_best_quantiles": total_best_quantiles.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_params(path: Path) -> tuple[dict[int, int], np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    median_event_sizes = {int(k): v for k, v in data["median_event_sizes"].items()}
    total_best_quantiles = np.array(data["total_best_quantiles"])
    return median_event_sizes, total_best_quantiles


def compute_median_event_sizes(files: list[str]) -> dict[int, int]:
    sketches = {i: kll_ints_sketch(50000) for i in range(1, 5)}

    for path in tqdm(files, colour="green", leave=True, position=0):
        chunk = (
            pl.scan_parquet(path)
            .filter(
                pl.col("event_side").replace({"A": 1, "B": -1}).cast(int)
                * pl.col("event_queue_nbr")
                >= 0
            )
            .with_columns(
                event_q=pl.when(pl.col("event_queue_nbr").lt(0))
                .then(pl.col("event_queue_nbr") - pl.col("best_bid_nbr") - 1)
                .otherwise(pl.col("event_queue_nbr") - pl.col("best_ask_nbr") + 1)
            )
            .select(
                queue=pl.col("event_q").abs(),
                size=pl.col("event_size"),
            )
            .filter(pl.col("queue").is_between(1, 4))
            .collect()
        )

        for q in range(1, 5):
            vals = (
                chunk.filter(pl.col("queue") == q)["size"].to_numpy().astype(np.int32)
            )
            if len(vals):
                sketches[q].update(vals)
            del vals

        del chunk

    return {i: int(sketches[i].get_quantile(0.5)) for i in range(1, 5)}


def compute_total_best_quantiles(
    files: list[str], median_event_sizes: dict[int, int]
) -> np.ndarray:
    sketch = kll_ints_sketch(50000)

    q1_median = median_event_sizes[1]
    for path in tqdm(files, colour="green", leave=True, position=0):
        chunk = pl.scan_parquet(path).filter(
            pl.col("event_side").replace({"A": 1, "B": -1}).cast(int)
            * pl.col("event_queue_nbr")
            >= 0
        )

        condlist = [pl.col("best_bid_nbr").eq(-i) for i in range(1, 11)]
        choicelist = [pl.col(f"Q_{-i}") for i in range(1, 11)]
        best_bid = pl_select(condlist, choicelist).truediv(q1_median).ceil()

        condlist = [pl.col("best_ask_nbr").eq(i) for i in range(1, 11)]
        choicelist = [pl.col(f"Q_{i}") for i in range(1, 11)]
        best_ask = pl_select(condlist, choicelist).truediv(q1_median).ceil()

        chunk = chunk.select(total_best=best_ask + best_bid).collect()

        vals = chunk["total_best"].to_numpy().astype(np.int32)
        if len(vals):
            sketch.update(vals)
        del vals, chunk

    return np.array([sketch.get_quantile(p) for p in [0.2, 0.4, 0.6, 0.8]])


def compute_queue_levels(
    df: pl.DataFrame, median_event_sizes: dict[int, int]
) -> pl.DataFrame:
    results = {}

    for level in range(4, 0, -1):
        condlist = [pl.col("best_ask_nbr").eq(i) for i in range(1, 11)]
        choicelist = [
            pl.col(f"Q_{i + level - 1}") if i + level - 1 <= 10 else pl.lit(None)
            for i in range(1, 11)
        ]
        results[f"q_{level}"] = (
            pl_select(condlist, choicelist)
            .truediv(median_event_sizes[level])
            .ceil()
            .cast(pl.Int64)
        )

    for level in range(1, 5):
        condlist = [pl.col("best_bid_nbr").eq(i) for i in range(-1, -11, -1)]
        choicelist = [
            pl.col(f"Q_{i - level + 1}") if i - level + 1 >= -10 else pl.lit(None)
            for i in range(-1, -11, -1)
        ]
        results[f"q_{-level}"] = (
            pl_select(condlist, choicelist)
            .truediv(median_event_sizes[level])
            .ceil()
            .cast(pl.Int64)
        )

    return df.with_columns(**results)


def compute_imbalance(df: pl.LazyFrame, bins: np.ndarray) -> pl.LazyFrame:
    condlist = [
        *[
            pl.col("imbalance").ge(left) & pl.col("imbalance").lt(right)
            for left, right in zip(-bins[1:][::-1], -bins[:-1][::-1])
        ],
        pl.col("imbalance").eq(0),
        *[
            pl.col("imbalance").gt(left) & pl.col("imbalance").le(right)
            for left, right in zip(bins[:-1], bins[1:])
        ],
    ]
    choicelist = [*(-bins[1:][::-1]), 0, *bins[1:]]
    df = df.with_columns(
        imbalance=(pl.col("q_-1") - pl.col("q_1")) / (pl.col("q_-1") + pl.col("q_1"))
    )
    df = df.with_columns(imbalance=pl_select(condlist, choicelist))

    return df


def compute_total_best_bin(df: pl.LazyFrame, quantiles: list[float]) -> pl.LazyFrame:
    df = df.with_columns(total_best=pl.col("q_1") + pl.col("q_-1"))

    condlist = []
    choicelist = []

    condlist.append(pl.col("total_best").lt(quantiles[0]))
    choicelist.append(0)

    for i in range(len(quantiles) - 1):
        cond = pl.col("total_best").ge(quantiles[i]) & pl.col("total_best").lt(
            quantiles[i + 1]
        )
        condlist.append(cond)
        choicelist.append(i + 1)

    condlist.append(pl.col("total_best").ge(quantiles[-1]))
    choicelist.append(len(quantiles))

    return df.with_columns(total_best=pl_select(condlist, choicelist))


def preprocess(
    df: pl.LazyFrame,
    median_event_sizes: dict[int, int],
    bins: np.ndarray,
    total_best_quantiles: list[float],
) -> pl.LazyFrame:
    df = df.filter(
        (
            pl.col("event_side")
            .replace({"A": 1, "B": -1})
            .cast(int)
            .mul(pl.col("event_queue_nbr"))
            >= 0
        )
    )
    df = df.with_columns(
        pl.when(pl.col("event_queue_nbr").lt(0))
        .then(pl.col("event_queue_nbr").sub(pl.col("best_bid_nbr")).sub(1))
        .otherwise(pl.col("event_queue_nbr").sub(pl.col("best_ask_nbr")).add(1))
        .alias("event_q")
    )
    df = df.filter(pl.col("event_q").abs().le(2))
    df = df.with_columns(
        event_side=pl.col("event_side").replace({"A": 1, "B": -1}).cast(int)
    )
    df = df.with_columns(
        pl.col("event").replace({"Can": "Cancel", "Trd": "Trade", "Trd_All": "Trade"})
    )
    df = compute_queue_levels(df, median_event_sizes)
    df = compute_imbalance(df, bins)
    df = compute_total_best_bin(df, quantiles=total_best_quantiles)
    df = df.with_columns(delta_t=pl.col("ts_event").diff().cast(int))

    # Aggregate split trades: same ts_event and side, tack sizes onto first Trade
    is_trade = pl.col("event").eq("Trade")
    df = df.with_columns(
        is_dup_trade=is_trade
        & (pl.col("ts_event").eq(pl.col("ts_event").shift(1)))
        & (pl.col("event_side").eq(pl.col("event_side").shift(1)))
        & pl.col("event").shift(1).eq("Trade"),
    )
    df = df.with_columns(
        trade_group_id=((is_trade & ~pl.col("is_dup_trade")) | ~is_trade).cum_sum()
    )
    df = df.with_columns(
        event_size=pl.when(is_trade)
        .then(pl.col("event_size").sum().over("trade_group_id"))
        .otherwise(pl.col("event_size")),
    )
    df = df.filter(~pl.col("is_dup_trade"))

    # True add in spread sizes
    df = df.with_columns(
        is_create=pl.col("event").is_in(["Create_Bid", "Create_Ask"]),
        not_add=~pl.col("event").is_in(["Add", "Create_Bid", "Create_Ask"]),
        price_changed=pl.col("price").ne(pl.col("price").shift(1)),
    )
    df = df.with_columns(
        (pl.col("is_create") | pl.col("price_changed") | pl.col("not_add"))
        .cum_sum()
        .alias("group_id")
    )
    df = df.with_columns(
        event_size=pl.when(pl.col("is_create"))
        .then(pl.col("event_size").sum().over("group_id"))
        .otherwise(pl.col("event_size")),
        is_followup_add=pl.col("event").eq("Add")
        & pl.col("is_create").any().over("group_id"),
    )
    df = df.filter(~pl.col("is_followup_add"))

    df = df.with_columns(
        event_size=pl.when(pl.col("event_q").abs().eq(1))
        .then(pl.col("event_size").truediv(median_event_sizes[1]).ceil().cast(int))
        .otherwise(pl.col("event_size").truediv(median_event_sizes[2]).ceil().cast(int))
    )

    df = df.with_columns(
        delta_t=pl.when(pl.col("delta_t").eq(0))
        .then(None)
        .otherwise(pl.col("delta_t"))
        .forward_fill()
    )
    df = df.with_columns(mid=pl.col("P_1").add(pl.col("P_-1"))/2)

    return df.select(
        pl.exclude(
            "^Q.*$",
            "^P.*$",
            "best_ask_nbr",
            "best_bid_nbr",
            "symbol",
            "event_queue_size",
            "event_queue_nbr",
            "is_create",
            "not_add",
            "price_changed",
            "group_id",
            "is_followup_add",
            "is_dup_trade",
            "trade_group_id",
        )
    ).rename(
        {
            "event_side": "side",
            "event_q": "queue",
            "event_size": "size",
        }
    )


def daily_estimates(df: pl.LazyFrame) -> pl.LazyFrame:
    df = df.with_columns(
        pl.when(pl.col("spread").ge(2))
        .then(2)
        .otherwise(pl.col("spread"))
        .alias("spread")
    )
    is_create = (pl.col("event").eq("Create_Bid") | pl.col("event").eq("Create_Ask")) & pl.col("queue").eq(0)
    df = df.filter(pl.col("spread").eq(1) | is_create)
    stats = df.group_by(
        "date",
        "imbalance",
        "spread",
        "total_best",
        "event",
        "side",
        "queue",
    ).agg(
        pl.len(),
        pl.col("size"),
        pl.col("delta_t"),
        delta_t_sum=pl.col("delta_t").sum(),
        *(
            pl.concat([pl.col(f"q_{i}"), pl.col(f"q_-{i}")]).alias(f"q_{i}")
            for i in range(1, 5)
        ),
    )

    return stats


#######################################
### Delta_t fit
#######################################

def exp_delta_t(df: pl.LazyFrame) -> pl.DataFrame:
    df = (
        df.group_by(pl.col("imbalance").abs(), "spread")
        .agg(average_dt=pl.col("delta_t_sum").sum()/pl.col("len").sum())
        .collect()
        .sort("imbalance", "spread")
    )

    return df

def group_delta_t(df: pl.LazyFrame) -> pl.DataFrame:
    df = df.group_by("imbalance", "spread", "event", "queue", "side").agg(pl.col("delta_t").flatten()).collect()
    pos = df.filter(pl.col("imbalance") >= 0)
    neg = df.filter(pl.col("imbalance") <= 0).with_columns(
        imbalance=-pl.col("imbalance"),
        side=-pl.col("side"),
        queue=-pl.col("queue"),
        event=pl.col("event").replace(
            {"Create_Ask": "Create_Bid", "Create_Bid": "Create_Ask"}
        ),
    )

    df = (
        pl.concat([pos, neg])
        .group_by("imbalance", "spread", "event", "queue", "side")
        .agg(
            delta_t=pl.col("delta_t").flatten(),
        )
    )

    return df
    
def fit_gmm(df: pl.DataFrame, k: int = 5, floor: float = 0, random_state: int = 1337) -> pl.DataFrame:
    rows = []
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Fitting GMMs"):
        x = np.log10(np.array(row["delta_t"], dtype=float))
        x = x[x>=floor].reshape(-1, 1)
        gmm = GaussianMixture(n_components=k, random_state=random_state).fit(x)
        entry = {
            "imbalance": row["imbalance"],
            "spread": row["spread"],
            "event": row["event"],
            "queue": row["queue"],
            "side": row["side"],
        }
        for i in range(k):
            entry[f"w_{i+1}"] = gmm.weights_[i]
            entry[f"mu_{i+1}"] = gmm.means_[i, 0]
            entry[f"sig_{i+1}"] = np.sqrt(gmm.covariances_[i, 0, 0])
        rows.append(entry)
    return pl.DataFrame(rows)

#######################################
### Event probabilities
#######################################

def event_probabilities(df: pl.LazyFrame, include_total_best: bool=False) -> pl.DataFrame:
    state = ("imbalance", "spread", "total_best") if include_total_best else ("imbalance", "spread")
    stats = df.group_by(*state, "queue", "side", "event").agg(
        pl.col("len").sum(), pl.col("delta_t_sum").sum()
    )
    stats = stats.with_columns(
        total_len_cat=pl.col("len").sum().over(*state)
    )
    
    stats = stats.collect()
    pos = stats.filter(pl.col("imbalance") >= 0)
    neg = stats.filter(pl.col("imbalance") <= 0).with_columns(
        imbalance=-pl.col("imbalance"),
        side=-pl.col("side"),
        queue=-pl.col("queue"),
        event=pl.col("event").replace(
            {"Create_Ask": "Create_Bid", "Create_Bid": "Create_Ask"}
        ),
    )
    
    probabilities = (
        pl.concat([pos, neg])
        .group_by(*state, "event", "queue", "side")
        .agg(probability=pl.col("len").sum().truediv(pl.col("total_len_cat").sum()))
        .sort(*state, "event", "queue")
    )
    return probabilities

#######################################
### Volumes
#######################################

MAX_SIZE = 50


def volumes_dist(df: pl.LazyFrame) -> pl.DataFrame:
    stats = (
        df.group_by("imbalance", "spread", "event", "queue", "side")
        .agg(pl.col("size").flatten())
        .collect()
    )

    pos = stats.filter(pl.col("imbalance") >= 0)
    neg = stats.filter(pl.col("imbalance") <= 0).with_columns(
        imbalance=-pl.col("imbalance"),
        side=-pl.col("side"),
        queue=-pl.col("queue"),
        event=pl.col("event").replace(
            {"Create_Ask": "Create_Bid", "Create_Bid": "Create_Ask"}
        ),
    )

    sizes = (
        pl.concat([pos, neg])
        .group_by("imbalance", "spread", "event", "queue", "side")
        .agg(pl.col("size").flatten())
    )

    size_hist = (
        sizes.explode("size")
        .with_columns(pl.col("size").clip(1, MAX_SIZE))
        .group_by("imbalance", "spread", "event", "queue", "side", "size")
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum()
             .over("imbalance", "spread", "event", "queue", "side"))
            .alias("prob")
        )
        .drop("len")
        .pivot(on="size", index=["imbalance", "spread", "event", "queue", "side"], values="prob")
        .fill_null(0.0)
    )

    # Ensure all size columns 1..MAX_SIZE exist
    for i in range(1, MAX_SIZE + 1):
        if str(i) not in size_hist.columns:
            size_hist = size_hist.with_columns(pl.lit(0.0).alias(str(i)))

    key_cols = ["imbalance", "spread", "event", "queue", "side"]
    size_cols = [str(i) for i in range(1, MAX_SIZE + 1)]
    return size_hist.select(key_cols + size_cols).sort("imbalance", "spread", "event", "queue")

MAX_Q_SIZE = 100

def queue_size_dist(df : pl.LazyFrame) -> pl.DataFrame:
    df = df.lazy()
    q_dist = (
        df.select(pl.col("q_1", "q_2", "q_3", "q_4").explode())
        .unpivot(variable_name="queue_level", value_name="size")
        .with_columns(pl.col("queue_level").str.extract(r"(\d+)").cast(pl.Int32))
    )
    
    q_dist = q_dist.filter(pl.col("size").le(MAX_Q_SIZE))
    
    q_dist = (
        q_dist.group_by("queue_level", "size")
        .agg(probability=pl.len())
        .with_columns(
            pl.col("probability").truediv(pl.col("probability").sum().over("queue_level"))
        )
    )
    
    q_dist = q_dist.sort("queue_level", "size").collect().pivot(on="size", index="queue_level", values="probability").fill_null(0)

    for i in range(MAX_Q_SIZE + 1):
        if str(i) not in q_dist.columns:
            q_dist = q_dist.with_columns(pl.lit(0.0).alias(str(i)))

    q_dist = q_dist.select("queue_level", *[str(i) for i in range(MAX_Q_SIZE + 1)])

    return q_dist