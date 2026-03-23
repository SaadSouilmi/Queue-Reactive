import argparse
from pathlib import Path

import polars as pl
from lobib import DataLoader
from tqdm import tqdm

import qr.estimations as est

loader = DataLoader()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate relevant qr statistics per day")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., PFE)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    info = loader.ticker_info(args.ticker).filter(pl.col("schema").eq("qr"))
    files = info["file"].to_list()

    median_event_sizes = est.compute_median_event_sizes(files)
    total_best_quantiles = est.compute_total_best_quantiles(files, median_event_sizes)

    output_dir = Path(f"data/{args.ticker}/daily_estimates")
    output_dir.mkdir(parents=True, exist_ok=True)

    est.save_params(output_dir / "params.json", median_event_sizes, total_best_quantiles)

    for file in tqdm(files, colour="green", leave=True, position=0):
        df = pl.scan_parquet(file)
        df = est.preprocess(
            df,
            median_event_sizes=median_event_sizes,
            bins=est.BINS,
            total_best_quantiles=total_best_quantiles,
        ).drop_nulls()
        stats = est.daily_estimates(df)
        
        date = file.split("/")[-1].split("_")[1]
        stats.sink_parquet(output_dir / f"{date}.parquet")
    

if __name__ == "__main__":
    main()