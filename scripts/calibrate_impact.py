"""
Calibrate the impact multiplier m for a given (tau, beta) by binary-searching
for the value that matches the theoretical sqrt impact curve.

For a given (tau, beta):
  1. Binary search on m: run_metaorder handles the kernel internally
  2. Compare terminal impact to theoretical terminal
  3. Log all runs — user decides best m

Usage:
    uv run python scripts/calibrate_impact.py --tau 50 --beta 1.5 --ticker PFE --num-sims 50000
"""

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def target_curve(t_norm: np.ndarray) -> np.ndarray:
    """t_norm: time normalized so t=1 is end of execution."""
    return np.where(
        t_norm <= 1.0,
        np.sqrt(np.maximum(t_norm, 0.0)),
        np.sqrt(t_norm) - np.sqrt(np.maximum(t_norm - 1.0, 0.0)),
    )


def run_metaorder(binary: str, config: dict, tmp_dir: Path) -> str | None:
    """Run the binary, return the output CSV path or None on failure."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=tmp_dir, delete=False
    ) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        result = subprocess.run(
            [binary, config_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"    Error: {result.stderr[:200]}")
            return None

        for line in result.stdout.split("\n"):
            if "Output:" in line:
                return line.split("Output:")[-1].strip()
    except subprocess.TimeoutExpired:
        print("    Timeout!")
        return None
    finally:
        Path(config_path).unlink(missing_ok=True)

    return None


def read_impact_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read timestamp, avg_mid_price_change, and avg_meta_vol from output CSV."""
    ts, mid, meta_vol = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row["timestamp"]))
            mid.append(float(row["avg_mid_price_change"]))
            meta_vol.append(float(row["avg_meta_vol"]))
    return np.array(ts), np.array(mid), np.array(meta_vol)


def analyze(
    ts_ns: np.ndarray, impact: np.ndarray, meta_vol: np.ndarray,
    exec_duration_min: float,
) -> tuple[float, float, float]:
    """Return (mse, terminal_impact, gamma) for a simulated impact curve."""
    t_min = ts_ns / (1e9 * 60)
    t_norm = t_min / exec_duration_min

    exec_idx = np.argmin(np.abs(t_min - exec_duration_min))
    peak = impact[exec_idx]
    if peak <= 0:
        return 1.0, -1.0, float("nan")

    normalized = impact / peak
    terminal = normalized[-1]
    target = target_curve(t_norm)
    mse = float(np.mean((normalized - target) ** 2))

    exec_mask = t_min <= exec_duration_min
    vol_exec = meta_vol[exec_mask]
    imp_exec = impact[exec_mask]
    pos = (vol_exec > 0) & (imp_exec > 0)
    if pos.sum() >= 2:
        log_v = np.log(vol_exec[pos])
        log_i = np.log(imp_exec[pos])
        A = np.vstack([log_v, np.ones(len(log_v))]).T
        gamma = float(np.linalg.lstsq(A, log_i, rcond=None)[0][0])
    else:
        gamma = float("nan")

    return mse, terminal, gamma


def _eval_m(binary, base_config, tau, beta, K, exec_duration_min, tmp_dir, log_rows, m, theoretical_terminal):
    """Run one evaluation at a given m. Returns (mse, terminal) or None."""
    config = json.loads(json.dumps(base_config))
    config["impact"] = {
        "type": "power_law",
        "tau": tau,
        "beta": beta,
        "m": m,
        "K": K,
    }

    csv_path = run_metaorder(binary, config, tmp_dir)
    if csv_path is None:
        return None

    ts_ns, impact, meta_vol = read_impact_csv(csv_path)
    mse, terminal, gamma = analyze(ts_ns, impact, meta_vol, exec_duration_min)

    log_rows.append({
        "tau": tau,
        "beta": beta,
        "m": m,
        "mse": mse,
        "terminal_impact": terminal,
        "theoretical_terminal": theoretical_terminal,
        "gamma": gamma,
        "csv_path": csv_path,
    })

    return mse, terminal


def main():
    parser = argparse.ArgumentParser(description="Calibrate impact parameter m for given (tau, beta)")
    parser.add_argument("--tau", type=float, required=True, help="Kernel decay time constant (seconds)")
    parser.add_argument("--beta", type=float, default=1.5, help="Kernel power-law exponent")
    parser.add_argument("--ticker", type=str, default="PFE")
    parser.add_argument("--num-sims", type=int, default=50000)
    parser.add_argument("--duration-min", type=int, default=60)
    parser.add_argument("--exec-duration-min", type=int, default=10)
    parser.add_argument("--hourly-vol", type=int, default=1000)
    parser.add_argument("--metaorder-pct", type=float, default=10.0)
    parser.add_argument("--max-order-size", type=int, default=2)
    parser.add_argument("--binary", type=str, default="build/run_metaorder")
    parser.add_argument("--K", type=int, default=12, help="Number of exponential components in kernel")
    parser.add_argument("--m-start", type=float, default=0.1)
    parser.add_argument("--m-min", type=float, default=0.0, help="Lower bound for m")
    parser.add_argument("--max-iter", type=int, default=10)
    args = parser.parse_args()

    tau = args.tau
    beta = args.beta

    base_config = {
        "ticker": args.ticker,
        "use_mixture": False,
        "use_total_lvl": False,
        "duration_min": args.duration_min,
        "exec_duration_min": args.exec_duration_min,
        "num_sims": args.num_sims,
        "grid_step_ms": 500,
        "metaorder_pcts": [args.metaorder_pct],
        "hourly_vol": args.hourly_vol,
        "max_order_size": args.max_order_size,
    }

    output_dir = Path(f"data/{args.ticker}/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    t_norm_final = args.duration_min / args.exec_duration_min
    theoretical_terminal = float(target_curve(np.array([t_norm_final]))[0])

    log_rows = []

    print(f"Calibrating m for tau={tau}s, beta={beta}, ticker={args.ticker}")
    print(f"  theoretical terminal: {theoretical_terminal:.6f}")
    print(f"  m range: starting at {args.m_start}")
    print(f"  max iterations: {args.max_iter}")
    print(f"  num_sims: {args.num_sims}")
    print()

    m_low = args.m_min
    m = args.m_start
    m_high = None
    print(f"  Phase 1: stepping up from m={args.m_start}")

    while True:
        print(f"  probe m={m:.6f}")
        result = _eval_m(
            binary=args.binary,
            base_config=base_config,
            tau=tau,
            beta=beta,
            K=args.K,
            exec_duration_min=args.exec_duration_min,
            tmp_dir=output_dir,
            log_rows=log_rows,
            m=m,
            theoretical_terminal=theoretical_terminal,
        )
        if result is None:
            print(f"    run failed, treating as too high")
            m_high = m
            break
        mse, terminal = result
        gamma = log_rows[-1]["gamma"]
        print(f"    terminal={terminal:.6f}, theoretical={theoretical_terminal:.6f}, gamma={gamma:.4f}")
        if terminal < theoretical_terminal:
            print(f"    terminal below target, bracket found")
            m_high = m
            break
        m_low = m
        m *= 2

    print(f"  Phase 2: binary search in [{m_low:.6f}, {m_high:.6f}]")

    for it in range(args.max_iter):
        m = (m_low + m_high) / 2.0
        print(f"  iter {it}: m={m:.6f} [{m_low:.6f}, {m_high:.6f}]")

        result = _eval_m(
            binary=args.binary,
            base_config=base_config,
            tau=tau,
            beta=beta,
            K=args.K,
            exec_duration_min=args.exec_duration_min,
            tmp_dir=output_dir,
            log_rows=log_rows,
            m=m,
            theoretical_terminal=theoretical_terminal,
        )

        if result is None:
            print(f"    run failed, treating as m too high")
            m_high = m
            continue

        mse, terminal = result
        gamma = log_rows[-1]["gamma"]
        print(f"    terminal={terminal:.6f}, theoretical={theoretical_terminal:.6f}, mse={mse:.6f}, gamma={gamma:.4f}")

        if terminal < theoretical_terminal:
            print(f"    terminal below target -> m too high, lowering m_high")
            m_high = m
        else:
            print(f"    terminal above target -> m too low, raising m_low")
            m_low = m

    print()

    log_path = output_dir / f"calibration_log_tau{int(tau)}_beta{beta}.csv"
    fieldnames = ["tau", "beta", "m", "mse", "terminal_impact", "theoretical_terminal", "gamma", "csv_path"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log saved to {log_path}")

    print()
    print(f"{'iter':>4}  {'m':>10}  {'terminal':>10}  {'theoretical':>12}  {'mse':>10}  {'gamma':>8}")
    print("-" * 70)
    for i, row in enumerate(log_rows):
        print(
            f"{i:4d}  {row['m']:10.6f}  {row['terminal_impact']:10.6f}  "
            f"{row['theoretical_terminal']:12.6f}  {row['mse']:10.6f}  {row['gamma']:8.4f}"
        )


if __name__ == "__main__":
    main()
