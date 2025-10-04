#!/usr/bin/env python3
# experiments/bench_strategies.py
"""
Numpy/Numba-free benchmark harness for decimal_computer.decimal_gemm.

Usage:
    python experiments/bench_strategies.py --sizes 4 8 --digits 4 8 --trials 2
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OUT_CSV = Path(__file__).resolve().parent / "bench_strategies.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = [
    "size", "digits", "algo", "trial",
    "elapsed", "muls", "adds",
    "error", "rc", "checksum",
    "sample", "stdout", "stderr"
]

def int_to_digits_fallback(x: int, ndigits: int) -> List[int]:
    s = str(abs(int(x))).rjust(ndigits, "0")[-ndigits:]
    return [int(ch) for ch in s]

def to_digits_safe(x: int, ndigits: int) -> List[int]:
    return int_to_digits_fallback(x, ndigits)

def random_digit_matrix(n: int, digits: int, rng: random.Random) -> List[List[List[int]]]:
    maxv = 10 ** digits - 1
    return [
        [to_digits_safe(rng.randint(0, maxv), digits) for _ in range(n)]
        for _ in range(n)
    ]

RUNNER_TEMPLATE = r'''#!/usr/bin/env python3
import json, sys, time, traceback
if len(sys.argv) < 2:
    print(json.dumps({"ok": False, "error": "no_input"})); sys.exit(2)
p = json.load(open(sys.argv[1],"r"))
A = p.get("A"); B = p.get("B"); algo = p.get("algo", "auto"); cutoff = p.get("cutoff", None)
t0 = time.perf_counter()
try:
    from decimal_computer.decimal_gemm import decimal_gemm_naive
except Exception:
    import traceback as _tb
    print(json.dumps({"ok": False, "error": "import_error", "trace": _tb.format_exc()}))
    sys.exit(3)
try:
    res = decimal_gemm_naive(A, B, mul_algo=algo, cutoff=cutoff, return_counters=True)
except TypeError:
    try:
        res = decimal_gemm_naive(A, B, mul_algo=algo, cutoff=cutoff)
    except Exception:
        print(json.dumps({"ok": False, "error": "call_failed", "trace": traceback.format_exc()})); sys.exit(4)
except Exception:
    print(json.dumps({"ok": False, "error": "call_failed", "trace": traceback.format_exc()})); sys.exit(4)
t1 = time.perf_counter()
out = {"ok": True, "elapsed": None, "muls": None, "adds": None, "sample": None, "checksum": None}
try:
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        C, counters = res
        out["muls"] = int(counters.get("muls", 0)) if counters else None
        out["adds"] = int(counters.get("adds", 0)) if counters else None
        out["elapsed"] = float(counters.get("time_s", t1 - t0)) if counters else (t1 - t0)
    else:
        C = res
        out["elapsed"] = (t1 - t0)
    try:
        if isinstance(C, list) and len(C) > 0 and isinstance(C[0], list):
            out["sample"] = [row[:2] for row in C[:2]]
            out["checksum"] = sum(sum(int(v) for v in row) for row in C)
    except Exception:
        out["sample"] = None
except Exception:
    print(json.dumps({"ok": False, "error": "pack_failed", "trace": traceback.format_exc()})); sys.exit(5)
print(json.dumps(out))
'''

def safe_call_decimal_gemm_subprocess(A: List[List[List[int]]], B: List[List[List[int]]], algo: str = "auto", cutoff: Optional[int] = None, timeout_seconds: float = 30.0) -> Tuple[Optional[Any], Optional[int], Optional[int], Optional[float], Optional[str], Optional[int], str, str, int]:
    with tempfile.TemporaryDirectory(prefix="bench_runner_") as td:
        td_path = Path(td)
        input_path = td_path / "input.json"
        runner_path = td_path / "runner.py"
        payload = {"A": A, "B": B, "algo": algo, "cutoff": cutoff}
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        runner_path.write_text(RUNNER_TEMPLATE, encoding="utf-8")
        cmd = [sys.executable, str(runner_path), str(input_path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as te:
            stdout = (te.stdout or "")[:2000]; stderr = (te.stderr or "")[:2000]
            return None, None, None, None, "timeout", None, stdout, stderr, -1
        except Exception as e:
            return None, None, None, None, f"spawn_error:{type(e).__name__}", None, "", str(e), -1
        stdout = (proc.stdout or ""); stderr = (proc.stderr or "")
        rc = proc.returncode
        if not stdout.strip():
            err = stderr.strip() or "no_output"
            return None, None, None, None, f"no_output: {err}", None, stdout, stderr, rc
        try:
            j = json.loads(stdout.strip().splitlines()[-1])
        except Exception:
            return None, None, None, None, "invalid_json", None, stdout, stderr, rc
        if not j.get("ok", False):
            trace = j.get("trace") or stderr or j.get("error") or "call_failed"
            return None, None, None, float(j.get("elapsed") or 0.0), f"runner_error: {trace}", j.get("checksum"), stdout, stderr, rc
        sample = j.get("sample"); muls = j.get("muls"); adds = j.get("adds"); elapsed = float(j.get("elapsed")) if j.get("elapsed") is not None else None; checksum = j.get("checksum")
        return sample, muls, adds, elapsed, None, checksum, stdout, stderr, rc


def bench_one(n: int, digits: int, algo: str, cutoff: Optional[int], trials: int, rng_seed: Optional[int], timeout_seconds: float) -> List[Dict[str, Any]]:
    rng = random.Random(rng_seed)
    rows: List[Dict[str, Any]] = []
    for trial in range(trials):
        A = random_digit_matrix(n, digits, rng)
        B = random_digit_matrix(n, digits, rng)
        sample, muls, adds, elapsed, error, checksum, stdout, stderr, rc = safe_call_decimal_gemm_subprocess(A, B, algo=algo, cutoff=cutoff, timeout_seconds=timeout_seconds)
        if error:
            print(f"[bench] n={n} d={digits} trial={trial} algo={algo} -> ERROR {error}")
        else:
            print(f"[bench] n={n} d={digits} trial={trial} algo={algo} -> elapsed={elapsed}")
        row = {
            "size": n, "digits": digits, "algo": algo, "trial": trial, "elapsed": elapsed,
            "muls": muls, "adds": adds, "error": error, "rc": rc, "checksum": checksum,
            "sample": sample, "stdout": stdout, "stderr": stderr,
        }
        rows.append(row)
    return rows


def write_header_if_needed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_results_to_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        for r in rows:
            out = {k: r.get(k, None) for k in CSV_FIELDS}
            writer.writerow(out)


def parse_args():
    p = argparse.ArgumentParser(description="Bench decimal_gemm (numpy-free harness)")
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--digits", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--algos", type=str, nargs="+", default=["auto", "schoolbook", "karatsuba", "strassen"])
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--out", type=str, default=str(OUT_CSV))
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    write_header_if_needed(out_path)
    total = 0
    start_all = time.perf_counter()
    for n in args.sizes:
        for d in args.digits:
            for algo in args.algos:
                rows = bench_one(n, d, algo, cutoff=None, trials=args.trials, rng_seed=args.seed, timeout_seconds=args.timeout)
                append_results_to_csv(out_path, rows)
                total += len(rows)
                elapsed_all = time.perf_counter() - start_all
                print(f"[bench orchestrator] progress: {total} rows appended  elapsed={elapsed_all:.3f}s")
    print(f"Saved results to {out_path} (rows appended: {total})")


if __name__ == "__main__":
    main()
