#!/usr/bin/env python3
"""Generate empirical vs QR comparison plots for a given ticker and simulation hash."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scienceplots
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import qr.estimations as est
from qr.utils import (
    HOUR_NS,
    compute_volatility_sampled_empirical,
    compute_volatility_sampled_simulation,
    load_data,
    preprocess_sim,
    trade_sign_acf_empirical,
    trade_sign_acf_simulation,
)

sns.set_style("whitegrid")
plt.style.use(["science", "grid", "no-latex"])

BASE_PATH = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / "paper"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot empirical vs QR comparison")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--hash", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def load_simulation(ticker: str, sim_hash: str, mes: dict[int, int]) -> pl.DataFrame:
    registry_path = BASE_PATH / "results" / ticker / "samples" / "registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    if sim_hash not in registry:
        available = list(registry.keys())
        raise ValueError(f"Hash {sim_hash} not found. Available: {available}")

    print(f"Config: {json.dumps(registry[sim_hash], indent=2)}")

    parquet_path = BASE_PATH / "results" / ticker / "samples" / f"{sim_hash}.parquet"
    df_sim = pl.read_parquet(parquet_path)
    return preprocess_sim(df_sim, mes)


def plot_delta_t(df, df_sim, ticker, outdir):
    with mpl.rc_context({"axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        ax.hist(df["delta_t"].explode().log10(), bins=80, density=True, label="Empirical", alpha=0.85)
        ax.hist(df_sim.filter(pl.col("dt") > 0)["dt"].log10(), bins=80, density=True, label="QR", alpha=0.6)
        ax.set_xlabel(r"$\Delta t$ in nanoseconds")
        ax.set_ylabel("Density")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$10^{{{int(x)}}}$"))
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"delta_t_exp_fit_{ticker}.pdf", bbox_inches="tight")
        plt.close()


def plot_imbalance_before_trade(df, df_sim, ticker, outdir):
    imb_dist = (
        df.filter(pl.col("event").eq("Trade"))
        .group_by("imbalance")
        .agg(pl.col("len").sum())
        .with_columns(proportion=pl.col("len") / pl.col("len").sum())
        .sort("imbalance")
    )
    imb_sim = (
        df_sim.filter(pl.col("event").eq("Trade"))["imbalance"]
        .value_counts(normalize=True)
        .sort("imbalance")
    )

    with mpl.rc_context({"axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        x = imb_dist["imbalance"]
        width = 0.05
        ax.bar(x, imb_dist["proportion"], width, label="Empirical", alpha=0.85)
        ax.bar(x + width * 3 / 4, imb_sim["proportion"], width, label="QR", alpha=0.85)
        ax.set_xlabel("Imbalance Bin")
        ax.set_ylabel("Proportion")
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / f"imb_distrib_before_trade_{ticker}.pdf", bbox_inches="tight")
        plt.close()


def plot_event_distribution(df, df_sim, ticker, outdir):
    event_dist = (
        df.group_by("event")
        .agg(pl.col("len").sum())
        .with_columns(proportion=pl.col("len") / pl.col("len").sum())
        .sort("event")
    )
    event_dist_sim = df_sim["event"].value_counts(normalize=True).sort("event")

    labels = event_dist["event"]
    x = np.arange(len(labels))
    width = 0.25

    with mpl.rc_context({"axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        ax.bar(x, event_dist["proportion"], width, label="Empirical", alpha=0.7)
        ax.bar(x + width / 2, event_dist_sim["proportion"], width, label="QR", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Event")
        ax.set_ylabel("Proportion")

        create_idx = [i for i, l in enumerate(labels) if "Create" in l]
        if create_idx:
            x1 = min(create_idx) - 0.3
            x2 = max(create_idx) + 0.5
            y1 = 0
            y2 = (
                max(
                    max(event_dist["proportion"][i] for i in create_idx),
                    max(event_dist_sim["proportion"][i] for i in create_idx),
                )
                * 1.3
            )
            axins = ax.inset_axes([0.45, 0.35, 0.4, 0.55])
            axins.bar(x[create_idx], [event_dist["proportion"][i] for i in create_idx], width, alpha=0.7)
            axins.bar(x[create_idx] + width / 2, [event_dist_sim["proportion"][i] for i in create_idx], width, alpha=0.7)
            axins.set_xticks([])
            axins.tick_params(labelsize=6)
            axins.grid(True, alpha=0.3)
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            ax.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.5)

        plt.tight_layout()
        fig.savefig(outdir / f"emp_vs_qr_event_distrib_{ticker}.pdf", bbox_inches="tight")
        plt.close()


def plot_returns_5m(df_emp, df_sim, ticker, outdir):
    bin_ns = int(5 * 60 * 1e9)

    returns_emp = (
        df_emp.with_columns((pl.col("ts_event").cast(pl.Int64) // bin_ns).alias("bin"))
        .group_by("date", "bin")
        .agg(pl.col("mid").last())
        .sort("date", "bin")
        .with_columns((pl.col("mid") - pl.col("mid").shift(1)).over("date").alias("ret_5m"))
        .drop_nulls("ret_5m")
    )
    returns_sim = (
        df_sim.with_columns((pl.col("ts_event") // bin_ns).alias("bin"))
        .group_by("date", "bin")
        .agg(pl.col("mid").last())
        .sort("date", "bin")
        .with_columns((pl.col("mid") - pl.col("mid").shift(1)).over("date").alias("ret_5m"))
        .drop_nulls("ret_5m")
    )

    with mpl.rc_context({"axes.facecolor": "white"}):
        # Histogram
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        bins = np.linspace(-20, 20, 80)
        ax.hist(returns_emp["ret_5m"], bins=bins, density=True, alpha=0.5, color="C0", label="Empirical")
        ax.hist(returns_sim["ret_5m"], bins=bins, density=True, alpha=0.5, color="C1", label="QR")
        ax.set_xlim(-20, 20)
        ax.set_xlabel("5-min return (ticks)")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"returns_5m_hist_{ticker}.pdf", bbox_inches="tight", dpi=300)
        plt.close()

        # QQ plot
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        q = np.linspace(0.001, 0.999, 500)
        emp_q = np.quantile(returns_emp["ret_5m"].to_numpy(), q)
        sim_q = np.quantile(returns_sim["ret_5m"].to_numpy(), q)
        ax.scatter(emp_q, sim_q, s=2, color="black", alpha=0.6)
        lim = max(abs(emp_q.min()), abs(emp_q.max()), abs(sim_q.min()), abs(sim_q.max()))
        ax.plot([-lim, lim], [-lim, lim], "--", lw=0.8, color="#B5121B")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Empirical quantiles")
        ax.set_ylabel("QR quantiles")
        fig.tight_layout()
        fig.savefig(outdir / f"returns_5m_qq_{ticker}.pdf", bbox_inches="tight", dpi=300)
        plt.close()


def plot_volatility(df_emp, df_sim, ticker, outdir):
    vol_emp = compute_volatility_sampled_empirical(df_emp)
    vol_sim = compute_volatility_sampled_simulation(df_sim)

    with mpl.rc_context({"axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        labels = ["Empirical", "QR"]
        colors = ["tab:blue", "green"]
        vol_data = [vol_emp["volatility_per_hour"].to_numpy(), vol_sim["volatility_per_hour"].to_numpy()]
        bp = ax.boxplot(vol_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(r"Volatility")
        plt.tight_layout()
        fig.savefig(outdir / f"volatility_5m_{ticker}.pdf", bbox_inches="tight")
        plt.close()


def plot_hourly_volume(df_emp, df_sim, ticker, outdir):
    sz = (
        df_emp.filter(pl.col("event").eq("Trade"))
        .group_by(pl.col("date"), pl.col("ts_event").dt.hour())
        .agg(pl.col("size").sum())["size"]
    )
    sze = (
        df_sim.filter(pl.col("event").eq("Trade"))
        .group_by(pl.col("date"), pl.col("ts_event") // HOUR_NS)
        .agg(pl.col("size").sum())["size"]
    )

    with mpl.rc_context({"axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        labels = ["Empirical", "QR"]
        colors = ["tab:blue", "tab:green"]
        bp = ax.boxplot([sz, sze], labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(r"Volume")
        plt.tight_layout()
        fig.savefig(outdir / f"hourly_traded_volume_{ticker}.pdf", bbox_inches="tight")
        plt.close()


def plot_trade_sign_acf(df_emp, df_sim, ticker, outdir, max_lag=100):
    acf_emp = trade_sign_acf_empirical(df_emp, max_lag)
    acf_sim = trade_sign_acf_simulation(df_sim, max_lag)

    with mpl.rc_context({"axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        lags = np.arange(1, max_lag + 1)
        ax.plot(lags, acf_emp[1:], lw=1.5, color="#1B9E77", label="Empirical")
        ax.plot(lags, acf_sim[1:], lw=1.5, color="#D95F02", label="Simulation")
        ax.set_xlabel("Lag (trades)")
        ax.set_ylabel("Autocorrelation")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"orderflow_acf_{ticker}.pdf", bbox_inches="tight")
        plt.close()


def plot_daily_events(df_emp, df_sim, ticker, outdir):
    daily_emp = df_emp.group_by("date").len()["len"]
    daily_sim = df_sim.group_by("date").len()["len"]

    with mpl.rc_context({"axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        fig.patch.set_alpha(0)
        bp = ax.boxplot([daily_emp, daily_sim], labels=["Empirical", "QR"], patch_artist=True)
        for patch, color in zip(bp["boxes"], ["tab:blue", "tab:green"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Daily events")
        plt.tight_layout()
        fig.savefig(outdir / f"daily_events_{ticker}.pdf", bbox_inches="tight")
        plt.close()


def main():
    args = parse_args()
    ticker = args.ticker
    sim_hash = args.hash
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {ticker}...")
    df, df_emp = load_data(ticker)
    mes, _ = est.load_params(BASE_PATH / ticker / "daily_estimates" / "params.json")

    print(f"Loading simulation {sim_hash}...")
    df_sim = load_simulation(ticker, sim_hash, mes)

    print("Plotting delta_t...")
    plot_delta_t(df, df_sim, ticker, outdir)

    print("Plotting imbalance before trade...")
    plot_imbalance_before_trade(df, df_sim, ticker, outdir)

    print("Plotting event distribution...")
    plot_event_distribution(df, df_sim, ticker, outdir)

    print("Plotting 5-min returns...")
    plot_returns_5m(df_emp, df_sim, ticker, outdir)

    print("Plotting volatility...")
    plot_volatility(df_emp, df_sim, ticker, outdir)

    print("Plotting hourly traded volume...")
    plot_hourly_volume(df_emp, df_sim, ticker, outdir)

    print("Plotting daily events...")
    plot_daily_events(df_emp, df_sim, ticker, outdir)

    print("Plotting trade sign ACF...")
    plot_trade_sign_acf(df_emp, df_sim, ticker, outdir)

    print(f"All plots saved to {outdir}/")


if __name__ == "__main__":
    main()
