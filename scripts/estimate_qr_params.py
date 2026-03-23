import argparse
from pathlib import Path

import polars as pl
from lobib import DataLoader

import qr.estimations as est

FLOOR_THRESHOLD = 4.39
GMM_K = 5

loader = DataLoader()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate QR delta_t parameters")
    parser.add_argument("--ticker", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ticker = args.ticker

    daily_dir = Path(f"data/{ticker}/daily_estimates")
    output_dir = Path(f"data/{ticker}/qr_params")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(daily_dir.glob("*.parquet"))
    print(f"Loading {len(files)} daily estimate files...")
    df = pl.scan_parquet(files)

    print("Sampling LOB configurations...")
    dates = df.select(pl.col("date").unique()).collect()["date"].to_list()
    median_event_sizes, total_best_quantiels = est.load_params(
        f"data/{ticker}/daily_estimates/params.json"
    )
    df2 = pl.concat(
        [
            est.preprocess(
                loader.load(ticker, start_date=date, end_date=date, schema="qr"),
                median_event_sizes,
                est.BINS,
                total_best_quantiels,
            )
            for date in dates
        ]
    )
    df2 = df2.select(pl.col("^q_.*$"))
    step = 200
    sampled = df2.with_row_index().filter(pl.col("index") % step == 0).head(10000).drop("index")
    sampled.collect().write_csv(output_dir / "random_lob.csv")
    
    # Exponential: lambda = 1 / mean(dt)
    print("Estimating exponential delta_t...")
    exp_dt = est.exp_delta_t(df)
    exp_dt.write_csv(output_dir / "delta_t_exponential.csv")
    print(f"  Saved {len(exp_dt)} rows")

    # Symmetrise and group delta_t lists
    print("Grouping and symmetrising delta_t...")
    grouped = est.group_delta_t(df)
    print(f"  {len(grouped)} groups")

    # GMM
    print(f"Fitting GMM (k={GMM_K}, no floor)...")
    gmm = est.fit_gmm(grouped, k=GMM_K, floor=0)
    gmm.write_csv(output_dir / "delta_t_gmm.csv")
    print(f"  Saved {len(gmm)} rows")
    
    # Event Probabilities
    print(f"Fitting Event probabilities")
    probabilities = est.event_probabilities(df)
    probabilities.write_csv(output_dir / "event_probabilities.csv")
    print(f"  Saved {len(probabilities)} rows")

    # Event Probabilities
    print(f"Fitting Event probabilities 3D")
    probabilities_3D = est.event_probabilities(df, include_total_best=True)
    probabilities_3D.write_csv(output_dir / "event_probabilities_3D.csv")
    print(f"  Saved {len(probabilities_3D)} rows")

    # Volumes
    print(f"Fitting volumes distribution")
    size_distrib = est.volumes_dist(df)
    size_distrib.write_csv(output_dir / "size_distrib.csv")
    print(f"  Saved {len(size_distrib)} rows")

    # Invariant queue distribution 
    print(f"Fitting queue size distribution")
    queue_distrib = est.queue_size_dist(df)
    queue_distrib.write_csv(output_dir / f"invariant_distributions_qmax{est.MAX_Q_SIZE}.csv")
    print(f"  Saved {len(queue_distrib)} rows")

    print(f"\nAll saved to {output_dir}/")


if __name__ == "__main__":
    main()
