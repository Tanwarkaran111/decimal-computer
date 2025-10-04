# compute_cutoff_per_digit.py
import pandas as pd
from pathlib import Path
import json

CSV = Path("benchmarks_compare_time.csv")
if not CSV.exists():
    raise SystemExit("benchmarks_compare_time.csv not found. Run benchmarks_compare_time.py first.")

df = pd.read_csv(CSV)

# Expect columns: algo, cutoff, digit_len, size, trial, time_sec, muls, adds
# Aggregate mean time across sizes and trials for each (algo, cutoff, digit_len)
agg = df.groupby(["algo","cutoff","digit_len"])["time_sec"].mean().reset_index()

# Build dictionary: digit_len -> best_choice
best_map = {}
for d in sorted(agg["digit_len"].unique()):
    sub = agg[agg["digit_len"]==d].copy()
    if sub.empty:
        continue
    # find row with minimum mean time for this digit length
    best_row = sub.loc[sub["time_sec"].idxmin()]
    algo = str(best_row["algo"])
    cutoff = best_row["cutoff"]
    # For schoolbook, cutoff may be NaN or empty; represent cleanly
    if algo == "schoolbook":
        best_map[int(d)] = {"algo": "schoolbook"}
    else:
        # cutoff might be numeric; ensure int
        best_map[int(d)] = {"algo": "karatsuba", "cutoff": int(cutoff)}

# Save JSON
OUT = Path("cutoff_table.json")
OUT.write_text(json.dumps(best_map, indent=2))
print("Saved cutoff table to", OUT)
print("Entries:", len(best_map))
print(json.dumps(best_map, indent=2))
