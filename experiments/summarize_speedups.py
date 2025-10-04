# experiments/summarize_speedups.py
import pandas as pd
import numpy as np
from pathlib import Path

IN = Path("experiments/bench_agg_with_speedup.csv")
ALT_IN = Path("experiments/bench_agg.csv")

def safe_load():
    if IN.exists():
        df = pd.read_csv(IN)
        print("Loaded", IN)
    elif ALT_IN.exists():
        df = pd.read_csv(ALT_IN)
        print("Loaded", ALT_IN)
    else:
        raise SystemExit("No aggregated CSV found (bench_agg_with_speedup.csv or bench_agg.csv).")
    return df

def compute_speedups(df):
    # If bench_agg_with_speedup already has speedup columns, use them.
    if "speedup" in df.columns or "baseline_time_s_x" in df.columns:
        # try to normalize: ensure there is a single numeric 'speedup' column if present
        if "speedup" in df.columns:
            return df
    # otherwise try to compute baseline by grouping on size,digits and choosing baseline algo
    # We will treat algorithm named 'NA' or '<NA>' as baseline if present, else take min mean_time (fastest)
    out = df.copy()
    out["mean_time_s"] = pd.to_numeric(out["mean_time_s"], errors="coerce")
    grp = out.groupby(["size", "digits"])
    baseline_rows = []
    for (size, digs), g in grp:
        base = None
        # prefer named baseline algos if present:
        for cand in ["NA", "<NA>", "baseline", "numba", "auto"]:
            if cand in g["algo"].values:
                base = g[g["algo"] == cand]
                break
        if base is None:
            # fallback: choose smallest mean_time (fastest) as baseline
            base = g.loc[g["mean_time_s"].idxmin()]
            base = pd.DataFrame([base])
        # attach baseline mean_time for the group
        base_time = float(base["mean_time_s"].iloc[0]) if not np.isnan(base["mean_time_s"].iloc[0]) else np.nan
        g = g.copy()
        g["baseline_time_s"] = base_time
        g["speedup"] = g["baseline_time_s"] / g["mean_time_s"]
        baseline_rows.append(g)
    out2 = pd.concat(baseline_rows, ignore_index=True)
    return out2

def summary(df):
    # drop non-numeric speedups
    s = pd.to_numeric(df["speedup"], errors="coerce")
    print("\nSpeedup stats (ignoring NaN):")
    print("count:", int(s.notna().sum()))
    print("mean:", float(s.mean()) if s.notna().any() else None)
    print("median:", float(s.median()) if s.notna().any() else None)
    print("min:", float(s.min()) if s.notna().any() else None)
    print("max:", float(s.max()) if s.notna().any() else None)
    # show top improvements and top slowdowns
    if s.notna().any():
        df_valid = df[s.notna()].copy()
        df_valid["speedup"] = pd.to_numeric(df_valid["speedup"])
        print("\nTop 10 fastest vs baseline (largest speedup):")
        print(df_valid.sort_values("speedup", ascending=False).head(10)[["size","digits","algo","mean_time_s","baseline_time_s","speedup"]])
        print("\nTop 10 slowest vs baseline (speedup < 1):")
        print(df_valid.sort_values("speedup", ascending=True).head(10)[["size","digits","algo","mean_time_s","baseline_time_s","speedup"]])

if __name__ == "__main__":
    df = safe_load()
    df2 = compute_speedups(df)
    summary(df2)
