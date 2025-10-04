# experiments/plot_benchmark_vs_numpy.py
"""
Plot NumPy vs Decimal benchmark results saved in experiments/benchmark_vs_numpy.csv.

Saves:
 - experiments/plots/benchmark_vs_numpy_linear.png
 - experiments/plots/benchmark_vs_numpy_log.png

Run:
    python experiments/plot_benchmark_vs_numpy.py
"""
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("experiments")
CSV = ROOT / "benchmark_vs_numpy.csv"
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(exist_ok=True)

def safe_read_csv(path: Path):
    if not path.exists():
        print(f"File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Failed to read CSV {path}: {e}")
        return None

def prepare_df(df: pd.DataFrame):
    # Expected minimal columns:
    # for numpy baseline rows: 'n', 'mean_s', 'notes' maybe 'numpy baseline'
    # for decimal rows: 'n', 'digits', 'mean_s', 'notes' maybe 'decimal ...'
    if df is None or df.empty:
        return None

    # normalize column names to str
    df.columns = [str(c) for c in df.columns]

    # ensure n exists
    if "n" not in df.columns:
        # maybe column named 'size' in some logs
        if "size" in df.columns:
            df = df.rename(columns={"size": "n"})
        else:
            print("CSV missing 'n' column")
            return None

    # ensure mean column exists
    for cand in ("mean_s", "mean"):
        if cand in df.columns:
            if "mean_s" not in df.columns:
                df = df.rename(columns={cand: "mean_s"})
            break
    if "mean_s" not in df.columns:
        print("CSV missing mean column (mean_s or mean).")
        return None

    # fill missing std_s with 0
    if "std_s" not in df.columns:
        df["std_s"] = np.nan

    # mark row types: numpy vs decimal vs other (by notes)
    def detect_kind(row):
        notes = str(row.get("notes", "")).lower()
        if "numpy" in notes:
            return "numpy"
        if "decimal" in notes:
            return "decimal"
        # fallback: look for 'mean' rows with digits column
        if "digits" in row and not pd.isna(row["digits"]):
            return "decimal"
        return "other"

    df["kind"] = df.apply(detect_kind, axis=1)
    # convert numeric types
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["mean_s"] = pd.to_numeric(df["mean_s"], errors="coerce")
    df["std_s"] = pd.to_numeric(df["std_s"], errors="coerce")
    return df

def plot_comparison(df: pd.DataFrame):
    # pivot by kind and size using mean_s
    plot_df = df.dropna(subset=["n"]).copy()
    sizes = sorted(plot_df["n"].unique().tolist())
    kinds = ["numpy", "decimal", "other"]
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(sizes))
    width = 0.25
    offsets = {
        "numpy": -width,
        "decimal": 0,
        "other": width
    }
    for kind in kinds:
        vals = []
        errs = []
        for s in sizes:
            sel = plot_df[(plot_df["n"] == s) & (plot_df["kind"] == kind)]
            if sel.empty:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                # if multiple rows, take mean of mean_s and pooled std if present
                mean_val = float(sel["mean_s"].dropna().mean())
                vals.append(mean_val)
                # choose first std if present else 0
                std = float(sel["std_s"].dropna().mean()) if sel["std_s"].dropna().shape[0] > 0 else 0.0
                errs.append(std)
        vals = np.array(vals, dtype=float)
        mask = ~np.isnan(vals)
        if mask.any():
            ax.bar(x + offsets[kind], vals, width=width, label=kind, yerr=np.array(errs), capsize=3)
    ax.set_xlabel("matrix size (n)")
    ax.set_ylabel("mean time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in sizes])
    ax.set_title("Benchmark: NumPy vs Decimal (mean time)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    out_linear = OUTDIR / "benchmark_vs_numpy_linear.png"
    fig.tight_layout()
    fig.savefig(out_linear, dpi=150)
    plt.close(fig)
    print("Saved", out_linear)

    # log-scale version
    fig2, ax2 = plt.subplots(figsize=(10,5))
    for kind in kinds:
        vals = []
        for s in sizes:
            sel = plot_df[(plot_df["n"] == s) & (plot_df["kind"] == kind)]
            if sel.empty:
                vals.append(np.nan)
            else:
                vals.append(float(sel["mean_s"].dropna().mean()))
        vals = np.array(vals, dtype=float)
        mask = ~np.isnan(vals)
        ax2.plot(x[mask], vals[mask], marker='o', label=kind)
    ax2.set_yscale("log")
    ax2.set_xlabel("matrix size (n)")
    ax2.set_ylabel("mean time (s) [log scale]")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(s)) for s in sizes])
    ax2.set_title("Benchmark: NumPy vs Decimal (log scale)")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", linewidth=0.5)
    out_log = OUTDIR / "benchmark_vs_numpy_log.png"
    fig2.tight_layout()
    fig2.savefig(out_log, dpi=150)
    plt.close(fig2)
    print("Saved", out_log)

def main():
    df = safe_read_csv(CSV)
    if df is None or df.empty:
        print("No data to plot.")
        return
    df = prepare_df(df)
    if df is None:
        print("Failed to prepare data.")
        return
    plot_comparison(df)

if __name__ == "__main__":
    main()
