# experiments/auto_select_best_algo.py
"""
Select the best algorithm per (size,digits) from aggregated experiment results.

Produces:
 - experiments/best_algos.csv  (summary table)
 - experiments/best_algos.json (mapping for use in production)

Selection logic (rank candidates):
  1) smallest mean_time_s
  2) smallest std_time_s
  3) largest n_trials
  4) algorithm preference list (configurable)
Marks selections as low-confidence when n_trials < min_trials OR std_time_s/mean_time_s > max_rel_std.

Usage:
    python experiments/auto_select_best_algo.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import math

ROOT = Path("experiments")
AGG_WITH_SPEED = ROOT / "bench_agg_with_speedup.csv"
AGG = ROOT / "bench_agg.csv"

OUT_CSV = ROOT / "best_algos.csv"
OUT_JSON = ROOT / "best_algos.json"

# Configuration: preference order for tie-breaking (lower index preferred)
PREFERRED_ORDER = ["numba", "schoolbook", "karatsuba", "strassen", "auto"]
MIN_TRIALS = 2               # require at least this many measured trials for good confidence
MAX_REL_STD = 0.5            # if std/mean > this, mark low confidence (50% default)

def read_agg():
    # prefer agg with speed if available (contains baseline_time_s and speedup)
    if AGG_WITH_SPEED.exists():
        try:
            df = pd.read_csv(AGG_WITH_SPEED)
            print(f"Read {AGG_WITH_SPEED} ({len(df)} rows).")
            return df
        except Exception as e:
            print(f"Failed to read {AGG_WITH_SPEED}: {e}")
    if AGG.exists():
        try:
            df = pd.read_csv(AGG)
            print(f"Read {AGG} ({len(df)} rows).")
            return df
        except Exception as e:
            print(f"Failed to read {AGG}: {e}")
    print("No aggregated input found (bench_agg_with_speedup.csv or bench_agg.csv). Exiting.")
    return None

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns exist
    for col in ("size", "digits", "algo", "mean_time_s", "std_time_s", "n_trials", "baseline_time_s"):
        if col not in df.columns:
            df[col] = pd.NA

    # Normalize types
    df = df.copy()
    df["size"] = pd.to_numeric(df["size"], errors="coerce").astype("Int64")
    df["digits"] = pd.to_numeric(df["digits"], errors="coerce").astype("Int64")
    df["algo"] = df["algo"].astype(str).replace("nan", pd.NA)
    df["mean_time_s"] = pd.to_numeric(df["mean_time_s"], errors="coerce")
    df["std_time_s"] = pd.to_numeric(df["std_time_s"], errors="coerce").fillna(0.0)
    df["n_trials"] = pd.to_numeric(df["n_trials"], errors="coerce").fillna(0).astype(int)
    if "baseline_time_s" in df.columns:
        df["baseline_time_s"] = pd.to_numeric(df["baseline_time_s"], errors="coerce")
    return df

def algo_pref_index(algo_name: str):
    if pd.isna(algo_name):
        return len(PREFERRED_ORDER) + 1
    algo = str(algo_name).lower()
    try:
        return PREFERRED_ORDER.index(algo)
    except ValueError:
        return len(PREFERRED_ORDER)  # unknown algos get lowest preference

def choose_best_for_group(group: pd.DataFrame):
    """
    Given subset for a particular (size,digits), choose best algo row.
    Returns chosen row (as dict) plus metadata fields confidence and reason.
    """
    # drop rows with NaN mean_time_s (can't choose from missing measurement)
    cand = group.copy()
    cand = cand[~cand["mean_time_s"].isna()].copy()
    if cand.empty:
        return None  # no valid candidates

    # compute relative std (guard against zero)
    cand["rel_std"] = cand.apply(lambda r: (r["std_time_s"] / r["mean_time_s"]) if (not pd.isna(r["std_time_s"]) and not pd.isna(r["mean_time_s"]) and r["mean_time_s"] != 0) else 0.0, axis=1)

    # ranking columns: mean_time_s asc, rel_std asc, -n_trials desc, pref_index asc
    cand["pref_idx"] = cand["algo"].apply(algo_pref_index)
    # fill NaNs in std_time_s and n_trials to keep sorting stable
    cand["std_time_s"] = cand["std_time_s"].fillna(0.0)
    cand["n_trials"] = cand["n_trials"].fillna(0).astype(int)

    cand_sorted = cand.sort_values(
        by=["mean_time_s", "rel_std", "n_trials", "pref_idx"],
        ascending=[True, True, False, True],
        ignore_index=True
    )

    best = cand_sorted.iloc[0].to_dict()

    # confidence heuristics
    confidence = True
    reasons = []
    if int(best.get("n_trials", 0)) < MIN_TRIALS:
        confidence = False
        reasons.append(f"n_trials<{MIN_TRIALS}")
    if best.get("mean_time_s") is None or pd.isna(best.get("mean_time_s")):
        confidence = False
        reasons.append("no_mean_time")
    else:
        rel_std = best.get("rel_std", 0.0)
        if rel_std is None:
            rel_std = 0.0
        if rel_std > MAX_REL_STD:
            confidence = False
            reasons.append(f"rel_std>{MAX_REL_STD:.2f}")
    # also compare against baseline_time_s if present: if chosen mean_time > baseline and baseline is from different algo,
    # include a note (this usually won't happen because baseline is min mean).
    baseline_time = best.get("baseline_time_s", None)
    if baseline_time is not None and not pd.isna(baseline_time):
        if best.get("mean_time_s") is not None and best["mean_time_s"] > baseline_time * 1.01:
            reasons.append("mean > baseline")
    reason_str = ";".join(reasons) if reasons else "ok"

    out = {
        "size": int(best.get("size")) if not pd.isna(best.get("size")) else None,
        "digits": int(best.get("digits")) if not pd.isna(best.get("digits")) else None,
        "algo": best.get("algo"),
        "mean_time_s": float(best.get("mean_time_s")) if best.get("mean_time_s") is not None and not pd.isna(best.get("mean_time_s")) else None,
        "std_time_s": float(best.get("std_time_s")) if best.get("std_time_s") is not None and not pd.isna(best.get("std_time_s")) else None,
        "n_trials": int(best.get("n_trials")) if best.get("n_trials") is not None else 0,
        "baseline_time_s": float(best.get("baseline_time_s")) if best.get("baseline_time_s") is not None and not pd.isna(best.get("baseline_time_s")) else None,
        "speedup": float(best.get("speedup")) if best.get("speedup") is not None and not pd.isna(best.get("speedup")) else None,
        "confidence": bool(confidence),
        "reason": reason_str
    }
    return out

def run_selection():
    df = read_agg()
    if df is None:
        return 1
    df = prepare_df(df)

    # group by (size,digits)
    group_keys = ["size", "digits"]
    # drop rows missing keys
    df_keys = df.dropna(subset=group_keys)
    if df_keys.empty:
        print("No valid size/digits rows in aggregated table. Exiting.")
        return 1

    results = []
    mapping = {}

    gb = df_keys.groupby(group_keys, dropna=False)
    for (size, digits), group in gb:
        chosen = choose_best_for_group(group)
        if chosen is None:
            # no valid measured candidates: record NA entry
            rec = {
                "size": int(size) if not pd.isna(size) else None,
                "digits": int(digits) if not pd.isna(digits) else None,
                "algo": None,
                "mean_time_s": None,
                "std_time_s": None,
                "n_trials": 0,
                "baseline_time_s": None,
                "speedup": None,
                "confidence": False,
                "reason": "no_valid_candidates"
            }
        else:
            rec = chosen
            if rec["algo"] is None:
                rec["confidence"] = False
                rec["reason"] = "algo_missing"
        results.append(rec)
        # mapping key for JSON: "size_digits" or tuple; JSON requires string keys, use "size:digits"
        key = f"{rec['size']}:{rec['digits']}"
        mapping[key] = {
            "algo": rec["algo"],
            "mean_time_s": rec["mean_time_s"],
            "n_trials": rec["n_trials"],
            "confidence": bool(rec["confidence"]),
            "reason": rec["reason"]
        }

    # write CSV
    out_df = pd.DataFrame(results).sort_values(by=["size", "digits"], ignore_index=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"WROTE best algos to {OUT_CSV} ({len(out_df)} rows)")

    # write JSON mapping
    with open(OUT_JSON, "w", encoding="utf8") as fh:
        json.dump(mapping, fh, indent=2)
    print(f"WROTE mapping to {OUT_JSON}")

    # print summary
    print("\nSummary (first 20):\n")
    with pd.option_context("display.max_rows", 20, "display.width", 120):
        display_df = out_df[["size", "digits", "algo", "mean_time_s", "speedup", "n_trials", "confidence", "reason"]].copy()
        print(display_df.head(20))

    return 0

if __name__ == "__main__":
    exit(run_selection())
