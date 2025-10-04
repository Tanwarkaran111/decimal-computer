# experiments/prepare_benchmark_vs_numpy.py
from pathlib import Path
import json
import csv
import pandas as pd

IN = Path("experiments/benchmark_vs_numpy.csv")
OUT = Path("experiments/benchmark_vs_numpy_long.csv")

if not IN.exists():
    raise SystemExit(f"Input not found: {IN}")

df = pd.read_csv(IN)

rows = []
for _, r in df.iterrows():
    common = {
        "timestamp": float(r.get("timestamp") or 0),
        "machine": r.get("machine", ""),
        "n": int(r.get("n")) if pd.notna(r.get("n")) else "",
        "digits": int(r.get("digits")) if pd.notna(r.get("digits")) else "",
    }

    # numpy row (if present)
    nm = r.get("numpy_mean_s")
    ns = r.get("numpy_std_s")
    nr = r.get("numpy_repeat_times", "")
    if pd.notna(nm):
        rows.append({
            **common,
            "algo": "numpy",
            "mean_s": float(nm),
            "std_s": float(ns) if pd.notna(ns) else None,
            "repeat_times": nr if not pd.isna(nr) else "",
            "notes": "numpy baseline",
            "source_row_index": int(_)
        })

    # decimal row (if present)
    dm = r.get("decimal_mean_s")
    ds = r.get("decimal_std_s")
    dr = r.get("decimal_repeat_times", "")
    derr = r.get("decimal_error", "")
    if pd.notna(dm):
        rows.append({
            **common,
            "algo": "decimal",
            "mean_s": float(dm),
            "std_s": float(ds) if pd.notna(ds) else None,
            "repeat_times": dr if not pd.isna(dr) else "",
            "notes": ("decimal error: " + str(derr)) if (not pd.isna(derr) and str(derr).strip()) else "decimal run",
            "source_row_index": int(_)
        })

# write CSV
fieldnames = ["timestamp","machine","n","digits","algo","mean_s","std_s","repeat_times","notes","source_row_index"]
with OUT.open("w", newline="", encoding="utf8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"WROTE transformed CSV -> {OUT} ({len(rows)} rows)")
