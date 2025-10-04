# experiments/aggregate_and_plot.py
"""
Robust aggregator + plotter for experiment CSVs.

Features:
- tolerant CSV reading (skips malformed lines but reports counts)
- ensures output CSV columns exist and have consistent types
- aggregates mean/std and trial counts to experiments/bench_agg.csv
- if baseline_results.csv is missing, creates a baseline automatically
  from the (size,digits) min mean_time_s in bench_agg.csv
- computes speedups and saves bench_agg_with_speedup.csv
- produces comparison barplots (one per digits value) saved to experiments/plots/
- defensive against missing columns, NaNs, and type errors

Usage:
    python experiments/aggregate_and_plot.py --plot
"""

from pathlib import Path
import argparse
import sys
import math
import csv
import warnings
import textwrap
import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

ROOT = Path("experiments")
ROOT.mkdir(exist_ok=True)
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Input files (producers: bench_strategies.py, numba_results.csv, large_experiments.csv, baseline_results.csv)
BENCH_STRAT_FILE = ROOT / "bench_strategies.csv"
NUMBA_FILE = ROOT / "numba_results.csv"        # optional, left for compatibility
LARGE_FILE = ROOT / "large_experiments.csv"    # optional
BASELINE_FILE = ROOT / "baseline_results.csv"

# Outputs
AGG_OUT = ROOT / "bench_agg.csv"
AGG_WITH_SPEEDUP_OUT = ROOT / "bench_agg_with_speedup.csv"

# expected fields from bench_strategies.py
EXPECTED_FIELDS = ["size", "digits", "algo", "trial", "elapsed", "muls", "adds"]

def read_csv_if_exists(path: Path, expected_cols=None):
    """
    Read CSV with tolerant parsing. If parsing fails due to C engine tokenization issues,
    retry with engine='python' and on_bad_lines='skip'. Reports counts and returns DataFrame.
    """
    if not path.exists():
        print(f"Read {path}: not found.")
        return None

    print(f"Read {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        # Attempt tolerant read with python engine and skip bad lines
        print(f"  pd.read_csv failed (C engine). Falling back to engine='python', on_bad_lines='skip'.")
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            print("  tolerant read produced DataFrame (skipped malformed rows).")
        except Exception as e2:
            print(f"  FATAL: fallback read also failed: {e2}")
            raise

    # Ensure expected columns exist
    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"  read_csv_if_exists: missing expected columns {missing} -> adding empty columns")
            for c in missing:
                df[c] = np.nan

    return df

def clean_bench_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize bench_strategies DataFrame:
    - rename 'elapsed' if other names present
    - coerce size/digits to ints when possible
    - ensure elapsed column is numeric (NaN for invalid)
    - drop rows that clearly have no useful measurement (elapsed NaN and other counters missing)
    - return cleaned frame
    """
    if df is None:
        return None

    df = df.copy()

    # canonicalize column names lower()
    df.columns = [str(c) for c in df.columns]

    # possible alternate elapsed column names -> keep 'elapsed'
    alt_elapsed = None
    for cand in ["elapsed", "elapsed_seconds", "time", "mean_time_s"]:
        if cand in df.columns:
            alt_elapsed = cand
            break
    if alt_elapsed and alt_elapsed != "elapsed":
        df["elapsed"] = df[alt_elapsed]

    # ensure expected columns exist
    for col in EXPECTED_FIELDS:
        if col not in df.columns:
            df[col] = np.nan

    # coerce types for size/digits/trial
    def to_int_series(s):
        return pd.to_numeric(s, errors="coerce").dropna().astype(int)

    # try to coerce size/digits/trial and keep as ints where possible
    for col in ("size", "digits", "trial"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # elapsed must be numeric seconds
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")

    # muls/adds may be provided as ints
    df["muls"] = pd.to_numeric(df["muls"], errors="coerce")
    df["adds"] = pd.to_numeric(df["adds"], errors="coerce")

    # Ensure algo is string
    if "algo" in df.columns:
        df["algo"] = df["algo"].astype(str).replace("nan", np.nan)

    # Report counts
    total_rows = len(df)
    invalid_elapsed = df["elapsed"].isna().sum()
    print(f"  clean_bench_df: total rows={total_rows}, elapsed_missing_or_invalid={invalid_elapsed}")

    # Drop rows that have totally missing experiment metadata (size/digits/algo)
    meta_missing = df[["size", "digits", "algo"]].isna().all(axis=1).sum()
    if meta_missing > 0:
        print(f"  clean_bench_df: dropping {meta_missing} rows with all of size,digits,algo missing")
        df = df[~df[["size", "digits", "algo"]].isna().all(axis=1)]

    # Keep rows where elapsed is present (we need elapsed to compute means/speedups).
    # But we will keep rows with elapsed missing if muls/adds present (some counters-only runs).
    has_counters = (~df["muls"].isna()) | (~df["adds"].isna())
    keep_mask = (~df["elapsed"].isna()) | has_counters
    dropped = (~keep_mask).sum()
    if dropped > 0:
        print(f"  clean_bench_df: dropping {dropped} rows where elapsed and counters are missing")
        df = df[keep_mask]

    # If size/digits are floats representing ints, round them to int
    for col in ("size", "digits", "trial"):
        if col in df.columns:
            # convert floats that are integral to ints
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # if not NaN and close to integer, cast
            df.loc[~df[col].isna(), col] = df.loc[~df[col].isna(), col].round().astype("Int64")

    return df

def aggregate_bench(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bench_strategies-style DataFrame to mean/std/n_trials per (size,digits,algo).
    Returns DataFrame with columns:
       size, digits, algo, mean_time_s, std_time_s, n_trials, muls_mean, adds_mean
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["size", "digits", "algo", "mean_time_s", "std_time_s", "n_trials"])

    # Work on elapsed column
    agg_list = []
    # group by size,digits,algo
    gb = df.groupby(["size", "digits", "algo"], dropna=False)
    for (size, digits, algo), group in gb:
        # collect numeric elapsed values
        elapsed_vals = pd.to_numeric(group["elapsed"], errors="coerce").dropna().values
        n_trials = len(elapsed_vals)
        mean_time = float(np.nan) if n_trials == 0 else float(np.mean(elapsed_vals))
        std_time = float(np.nan) if n_trials == 0 else float(np.std(elapsed_vals, ddof=1)) if n_trials > 1 else 0.0

        muls_mean = None
        adds_mean = None
        try:
            muls_vals = pd.to_numeric(group["muls"], errors="coerce").dropna().values
            adds_vals = pd.to_numeric(group["adds"], errors="coerce").dropna().values
            muls_mean = float(np.mean(muls_vals)) if len(muls_vals) > 0 else None
            adds_mean = float(np.mean(adds_vals)) if len(adds_vals) > 0 else None
        except Exception:
            pass

        agg_list.append({
            "size": int(size) if not pd.isna(size) else pd.NA,
            "digits": int(digits) if not pd.isna(digits) else pd.NA,
            "algo": algo if pd.notna(algo) else pd.NA,
            "mean_time_s": mean_time,
            "std_time_s": std_time,
            "n_trials": int(n_trials),
            "muls": muls_mean,
            "adds": adds_mean,
        })

    agg_df = pd.DataFrame(agg_list)
    # sort for readability
    agg_df = agg_df.sort_values(["size", "digits", "algo"], ignore_index=True)
    return agg_df

def ensure_baseline_file(agg_df: pd.DataFrame, baseline_path: Path):
    """
    If baseline file exists and readable, return baseline_df.
    Otherwise generate baseline from agg_df by taking min mean_time_s per (size,digits).
    Saves baseline_path.
    """
    if baseline_path.exists():
        try:
            b = pd.read_csv(baseline_path)
            print(f"Found baseline_results.csv ({len(b)} rows). Using it.")
            # ensure expected columns
            if "size" in b.columns and "digits" in b.columns and ("mean_time_s" in b.columns or "elapsed" in b.columns):
                # unify column
                if "elapsed" in b.columns and "mean_time_s" not in b.columns:
                    b = b.rename(columns={"elapsed": "mean_time_s"})
                return b
            else:
                print("  baseline_results.csv missing expected columns -> regenerating from agg")
        except Exception as e:
            print("  failed to read baseline_results.csv -> regenerating from agg", e)

    # create baseline from agg: min mean_time_s per (size,digits)
    if agg_df is None or agg_df.empty:
        print("No agg data to build baseline from.")
        return None

    # Drop rows where mean_time_s is NaN
    tmp = agg_df[~pd.isna(agg_df["mean_time_s"])].copy()
    if tmp.empty:
        print("No valid mean_time_s in agg to build baseline from.")
        return None

    baseline = tmp.groupby(["size", "digits"], dropna=False, as_index=False).agg({"mean_time_s": "min"})
    baseline.to_csv(baseline_path, index=False)
    print(f"WROTE baseline to {baseline_path} ({len(baseline)} rows)")
    return baseline

def compute_speedups(agg_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join baseline time with aggregated df and compute speedup = baseline / mean_time_s
    baseline_df must contain size,digits,mean_time_s
    """
    if agg_df is None or agg_df.empty:
        return agg_df

    df = agg_df.copy()
    # name baseline mean_time column to baseline_time_s
    baseline = baseline_df.rename(columns={"mean_time_s": "baseline_time_s"})[["size", "digits", "baseline_time_s"]]
    merged = pd.merge(df, baseline, on=["size", "digits"], how="left")
    # compute speedup
    def safe_speedup(row):
        b = row["baseline_time_s"]
        a = row["mean_time_s"]
        if pd.isna(b) or pd.isna(a) or a == 0:
            return pd.NA
        try:
            return float(b) / float(a)
        except Exception:
            return pd.NA

    merged["speedup"] = merged.apply(safe_speedup, axis=1)
    return merged

def plot_by_digits(speed_df: pd.DataFrame, outdir: Path):
    """
    For each digits value produce a grouped bar chart comparing algos across sizes.
    Saves files: compare_digits_{digits}.png
    """
    if speed_df is None or speed_df.empty:
        print("No data to plot.")
        return

    # Only consider rows with numeric size and digits
    plot_df = speed_df.copy()
    plot_df = plot_df[~plot_df["size"].isna() & ~plot_df["digits"].isna()]
    if plot_df.empty:
        print("No numeric size/digits rows to plot.")
        return

    digits_values = sorted(plot_df["digits"].dropna().unique().astype(int).tolist())
    algos = sorted(plot_df["algo"].dropna().unique().tolist())

    for d in digits_values:
        df_d = plot_df[plot_df["digits"] == d].copy()
        if df_d.empty:
            continue
        # pivot: rows are sizes, columns are algos, values mean_time_s
        pivot = df_d.pivot_table(index="size", columns="algo", values="mean_time_s", aggfunc="first")
        # We'll plot mean_time_s bars (log scale y) and include errorbars (std_time_s) where available
        pivot_std = df_d.pivot_table(index="size", columns="algo", values="std_time_s", aggfunc="first").reindex_like(pivot)

        sizes = list(pivot.index)
        if len(sizes) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        # plot group bars
        n_groups = len(sizes)
        n_algos = len(pivot.columns)
        indices = np.arange(n_groups)
        width = 0.8 / max(1, n_algos)

        for i, algo in enumerate(pivot.columns):
            vals = pivot[algo].values
            errs = pivot_std[algo].values if pivot_std is not None else None
            # replace NaNs with zeros for plotting off-axis (we'll mask them)
            mask = ~np.isnan(vals.astype(np.float64))
            x = indices - 0.4 + (i + 0.5) * width
            if mask.any():
                ax.bar(x[mask], vals[mask], width=width, label=str(algo))
                # errorbars
                if errs is not None:
                    try:
                        errvals = np.array([e if not pd.isna(e) else 0.0 for e in errs], dtype=float)
                        ax.errorbar(x[mask], vals[mask], yerr=errvals[mask], fmt="none", ecolor="k", capsize=2)
                    except Exception:
                        pass

        ax.set_yscale("log")
        ax.set_xlabel("matrix size (n)")
        ax.set_ylabel("mean time (s)")
        ax.set_title(f"Mean time (s) by algo — digits={d}")
        ax.set_xticks(indices)
        ax.set_xticklabels([str(int(s)) for s in sizes])
        ax.legend(title="algo", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", linestyle="--", linewidth=0.5)

        out = outdir / f"compare_digits_{d}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved plot {out}")

def main(plot=False):
    # 1) read raw bench strategies CSV (tolerant)
    raw = read_csv_if_exists(BENCH_STRAT_FILE, expected_cols=EXPECTED_FIELDS)
    if raw is None:
        print("No bench_strategies.csv found -> nothing to aggregate.")
        return

    # 2) clean dataframe
    clean = clean_bench_df(raw)

    # 3) aggregate
    agg = aggregate_bench(clean)
    # unify column names; ensure numeric columns exist
    for col in ["mean_time_s", "std_time_s", "n_trials", "muls", "adds"]:
        if col not in agg.columns:
            agg[col] = pd.NA

    # Save aggregated results
    agg.to_csv(AGG_OUT, index=False)
    print(f"Saved aggregated results to {AGG_OUT}")

    # 4) create baseline if needed (or read)
    baseline = ensure_baseline_file(agg, BASELINE_FILE)
    if baseline is None:
        print("compute_speedup: baseline missing -> will skip speedup computation")
        # write empty speedup file with baseline missing info
        agg_with_speed = agg.copy()
        agg_with_speed["baseline_time_s"] = pd.NA
        agg_with_speed["speedup"] = pd.NA
        agg_with_speed.to_csv(AGG_WITH_SPEEDUP_OUT, index=False)
        print(f"Saved aggregated+speed results to {AGG_WITH_SPEEDUP_OUT}")
    else:
        # 5) compute speedups and save
        agg_with_speed = compute_speedups(agg, baseline)
        agg_with_speed.to_csv(AGG_WITH_SPEEDUP_OUT, index=False)
        print(f"Saved aggregated+speed results to {AGG_WITH_SPEEDUP_OUT}")

    # 6) Plot if requested
    if plot:
        if baseline is None:
            print("Plotting with no baseline (speedups will be empty).")
        plot_by_digits(agg_with_speed if 'agg_with_speed' in locals() else agg, PLOTS_DIR)
        print(f"Plots saved to {PLOTS_DIR}")

    # Print a short sample for quick inspection
    print("\nSample aggregated (first 20 rows):\n")
    with pd.option_context("display.max_rows", 20, "display.width", 120):
        print(agg_with_speed.head(20) if 'agg_with_speed' in locals() else agg.head(20))

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate experiment CSVs and optionally plot.")
    parser.add_argument("--plot", action="store_true", help="Also produce plots in experiments/plots")
    args = parser.parse_args()
    main(plot=args.plot)
