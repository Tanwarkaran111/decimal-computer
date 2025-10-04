# phase3/plot_results.py
"""
Simple plotting for phase3_benchmarks_aggregated.csv

Produces:
 - phase3_plot_mean_vs_size.png   (mean runtime vs size per algorithm)
 - phase3_plot_bar_by_digits.png  (bar chart of mean runtime grouped by digits & algo)

Usage:
  python -m phase3.plot_results --agg-file ./phase3_benchmarks_aggregated.csv --out-dir ./phase3_results
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agg-file", required=True)
    p.add_argument("--out-dir", default=".")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.agg_file)
    # ensure numeric
    df['n'] = df['n'].astype(int)
    df['mean_seconds'] = df['mean_seconds'].astype(float)

    # plot: mean vs size for each algo (one subplot per digits value)
    digits_values = sorted(df['digits'].unique())
    for digits in digits_values:
        sub = df[df['digits'] == digits]
        plt.figure(figsize=(8,5))
        for algo in sorted(sub['algo'].unique()):
            s = sub[sub['algo'] == algo].sort_values('n')
            plt.plot(s['n'], s['mean_seconds'], marker='o', label=algo)
        plt.xlabel("matrix size (n)")
        plt.ylabel("mean runtime (s)")
        plt.title(f"Mean runtime vs size — digits={digits}")
        plt.legend()
        out = os.path.join(args.out_dir, f"phase3_mean_vs_size_d{digits}.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print("Wrote", out)

    # bar chart: group by (algo, digits) at largest n
    max_n = df['n'].max()
    top = df[df['n'] == max_n]
    plt.figure(figsize=(10,5))
    labels = []
    heights = []
    for _, row in top.sort_values(['digits','algo']).iterrows():
        labels.append(f"{row['algo']}\nd={row['digits']}")
        heights.append(row['mean_seconds'])
    plt.bar(range(len(heights)), heights)
    plt.xticks(range(len(heights)), labels, rotation=45, ha='right')
    plt.ylabel("mean runtime (s)")
    plt.title(f"Comparison at n={max_n}")
    out2 = os.path.join(args.out_dir, f"phase3_bar_n{max_n}.png")
    plt.tight_layout()
    plt.savefig(out2)
    plt.close()
    print("Wrote", out2)

if __name__ == "__main__":
    main()
