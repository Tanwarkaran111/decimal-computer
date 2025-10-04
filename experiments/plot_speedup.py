#!/usr/bin/env python3
"""
Plot speedup: NumPy_time / Decimal_time

Usage:
  python experiments/plot_speedup.py --input experiments/benchmark_vs_numpy_long.csv --out experiments/plots/speedup.png
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_prepare(path):
    df = pd.read_csv(path)
    # Expect columns 'algo' ('numpy' or 'decimal'), 'mean_s', 'n', 'digits'
    # Pivot so we have numpy_mean_s and decimal_mean_s per (n,digits)
    if 'algo' not in df.columns or 'mean_s' not in df.columns:
        raise RuntimeError("Input CSV missing required columns ('algo' or 'mean_s').")
    # keep only baseline rows with algo in {'numpy','decimal'}
    df = df[df['algo'].isin(['numpy','decimal'])]
    pivot = df.pivot_table(index=['n','digits'], columns='algo', values='mean_s', aggfunc='mean')
    pivot = pivot.reset_index()
    pivot.columns.name = None
    # rename for clarity
    pivot = pivot.rename(columns={'numpy': 'numpy_mean_s', 'decimal': 'decimal_mean_s'})
    return pivot

def compute_speedup(df):
    # safe divide: if decimal time missing or zero -> set speedup = NaN or large
    m = df.copy()
    m['speedup'] = np.nan
    mask = (m['decimal_mean_s'].notna()) & (m['decimal_mean_s'] > 0)
    m.loc[mask, 'speedup'] = m.loc[mask, 'numpy_mean_s'] / m.loc[mask, 'decimal_mean_s']
    return m

def plot_speedup(df, outpath):
    # group by digits, plot speedup vs n for each digits value
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    digits_values = sorted(df['digits'].unique())
    for d in digits_values:
        sub = df[df['digits'] == d].sort_values('n')
        if sub['speedup'].notna().any():
            plt.plot(sub['n'], sub['speedup'], marker='o', label=f"digits={d}")
    plt.xscale('linear')
    plt.yscale('log')   # speedups often span orders of magnitude
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Speedup (NumPy time / Decimal time) [log scale]")
    plt.title("NumPy vs Decimal: Speedup (higher -> NumPy faster)")
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved speedup plot -> {outpath}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Prepared long CSV (benchmark_vs_numpy_long.csv)")
    p.add_argument("--out", default="experiments/plots/speedup.png")
    args = p.parse_args()

    df = load_and_prepare(args.input)
    df2 = compute_speedup(df)

    # quick console summary
    print("Summary (first rows):")
    print(df2[['n','digits','numpy_mean_s','decimal_mean_s','speedup']].head().to_string(index=False))

    plot_speedup(df2, args.out)

if __name__ == "__main__":
    main()
