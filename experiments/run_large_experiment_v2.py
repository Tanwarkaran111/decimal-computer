# experiments/run_large_experiment_v2.py
"""
Run large experiments (NumPy baseline + optional decimal auto-runtime).
Safer v2: captures decimal-run logs, parses numeric output (JSON first,
then human text), and records everything to CSV.

Usage examples:
  # baseline only (NumPy)
  python experiments/run_large_experiment_v2.py --sizes 2048 4096 --repeats 3

  # baseline + decimal auto-runtime attempts for sizes
  python experiments/run_large_experiment_v2.py --decimal --sizes 512 1024 --digits 8 --repeats 3

  # decimal-only (skip NumPy baseline)
  python experiments/run_large_experiment_v2.py --decimal-only --sizes 1024 --digits 8 --repeats 5
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import time
import subprocess
import re
import sys
from datetime import datetime
import numpy as np
from typing import Optional

ROOT = Path("experiments")
ROOT.mkdir(exist_ok=True)
OUT = ROOT / "large_experiments.csv"
META = ROOT / "last_run_meta.json"
LOG_DIR = ROOT / "decimal_logs"
LOG_DIR.mkdir(exist_ok=True)

# Helpful regexes for parsing printed formats and JSON
RE_JSON_OBJ = re.compile(r"\{.*?\"mean.*?\}", re.DOTALL)
RE_LINE_JSON = re.compile(r"(\{.*\})")
RE_MEAN_S = re.compile(
    r"->\s*mean\s*([0-9.eE+\-]+)s|mean[:=]?\s*([0-9.eE+\-]+)s|\"mean_s\"\s*:\s*([0-9.eE+\-]+)"
)
RE_STD_S = re.compile(r"std[:=]?\s*([0-9.eE+\-]+)s|\"std_s\"\s*:\s*([0-9.eE+\-]+)")
RE_SAMPLES = re.compile(r"sample[s]?\s*[:=]?\s*([0-9]+)|\"samples\"\s*:\s*([0-9]+)")

def run_numpy_gemm(n, dtype=np.float32, repeats=3):
    """Run a small numpy baseline for a square matrix multiply."""
    A = np.random.randn(n, n).astype(dtype)
    B = np.random.randn(n, n).astype(dtype)
    # warmup
    _ = np.dot(A, B)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = np.dot(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean = sum(times) / len(times)
    gflops = (2.0 * n * n * n) / (mean * 1e9)
    return mean, gflops, times

def _safe_write_header(path: Path, fieldnames):
    first = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if first:
            writer.writeheader()
        fh.flush()

def _append_rows_to_csv(path: Path, rows, fieldnames):
    """
    Append rows to CSV. This writer will:
      - write header if file did not exist
      - ignore extra keys in row dicts (extrasaction='ignore')
      - fill missing keys with empty string (handled by DictWriter)
    """
    first = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if first:
            writer.writeheader()
        for r in rows:
            # ensure all expected columns exist (coerce None -> "")
            safe_row = {k: ("" if r.get(k, None) is None else r.get(k)) for k in fieldnames}
            writer.writerow(safe_row)
        fh.flush()

def _extract_json_candidates(text: str):
    """Return list of JSON-like substrings found in text (greedy & line-based attempts)."""
    candidates = []
    # first try non-greedy search for JSON objects
    for m in RE_JSON_OBJ.finditer(text):
        candidates.append(m.group(0))
    # also try line-by-line JSON extraction if nothing found
    if not candidates:
        for line in text.splitlines():
            lm = RE_LINE_JSON.search(line)
            if lm:
                candidates.append(lm.group(1))
    return candidates

def call_decimal_auto_module(n: int, digits: int, repeats: int, timeout: int = 600, python_exec: Optional[str] = None):
    """
    Invoke the decimal auto runtime via 'python -m decimal_computer.auto_runtime'
    Capture stdout/stderr, save a log file, and attempt to parse numeric results.

    Return dict: {"mean_s": float|None, "std_s": float|None, "samples": int|None, "logfile": str, "raw_output": str, "returncode": int}
    """
    timestamp = int(time.time())
    logf = LOG_DIR / f"decimal_n{n}_d{digits}_{timestamp}.log"
    python_exec = python_exec or sys.executable or "python"

    # Try multiple command shapes
    tried_cmds = [
        [python_exec, "-m", "decimal_computer.auto_runtime", "--n", str(n), "--digits", str(digits), "--trials", str(repeats)],
        [python_exec, "-m", "decimal_computer.auto_runtime", "--size", str(n), "--digits", str(digits), "--trials", str(repeats)],
        [python_exec, "-m", "decimal_computer.auto_runtime", "--size", str(n), "--digits", str(digits)],
        [python_exec, "-m", "decimal_computer.auto_runtime"],
    ]

    last_proc = None
    output_text = ""
    for cmd in tried_cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            last_proc = proc
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            output_text = stdout + ("\n" + stderr if stderr else "")
            # write log with contextual header
            header = f"RUN: {' '.join(cmd)}\nRET: {proc.returncode}\nTIME: {datetime.utcnow().isoformat()}Z\n\n"
            try:
                logf.write_text(header + output_text)
            except Exception:
                # best-effort write; if fails, continue
                pass
            # if returncode 0 and we have content, break; if output present, break too
            if proc.returncode == 0 or output_text.strip():
                break
        except Exception as e:
            txt = f"Exception while running {' '.join(cmd)}: {e}\n"
            if last_proc is not None:
                txt += f"last returncode: {last_proc.returncode}\n"
            try:
                logf.write_text(txt)
            except Exception:
                pass
            output_text = txt
            # continue to next shape

    mean_s = None
    std_s = None
    samples = None

    # Try to find JSON candidates anywhere in output and parse them
    candidates = _extract_json_candidates(output_text)
    if candidates:
        for c in candidates:
            try:
                parsed = json.loads(c)
                # support various key names
                mean_s = parsed.get("mean_s") or parsed.get("mean") or parsed.get("mean_seconds") or parsed.get("mean_time_s")
                std_s = parsed.get("std_s") or parsed.get("std") or parsed.get("std_seconds") or parsed.get("std_time_s")
                samples = parsed.get("samples") or parsed.get("sample_count") or parsed.get("samples_count")
                # coerce
                if mean_s is not None:
                    mean_s = float(mean_s)
                if std_s is not None:
                    std_s = float(std_s)
                if samples is not None:
                    samples = int(samples)
                # if we parsed numbers, prefer JSON result and stop
                if mean_s is not None:
                    break
            except Exception:
                continue

    # Text fallback parsing if JSON parse did not produce mean
    if mean_s is None:
        m = RE_MEAN_S.search(output_text)
        if m:
            for g in m.groups():
                if g:
                    try:
                        mean_s = float(g)
                        break
                    except Exception:
                        continue

    if std_s is None:
        m = RE_STD_S.search(output_text)
        if m:
            for g in m.groups():
                if g:
                    try:
                        std_s = float(g)
                        break
                    except Exception:
                        continue

    if samples is None:
        m = RE_SAMPLES.search(output_text)
        if m:
            for g in m.groups():
                if g:
                    try:
                        samples = int(g)
                        break
                    except Exception:
                        continue

    return {
        "mean_s": mean_s,
        "std_s": std_s,
        "samples": samples,
        "logfile": str(logf),
        "raw_output": output_text,
        "returncode": (last_proc.returncode if last_proc is not None else None)
    }

def run_decimal_experiment(n: int, digits: int, repeats: int = 3, timeout: int = 600, python_exec: Optional[str] = None):
    """
    Attempt decimal auto-runtime and return structured row(s) to append to CSV.
    We return a list with a single row dict.
    """
    print(f"Attempting decimal auto runtime experiments for n={n} digits={digits} ...")
    res = call_decimal_auto_module(n, digits, repeats, timeout=timeout, python_exec=python_exec)
    mean = res["mean_s"]
    std = res["std_s"]
    samples = res["samples"]
    logfile = res["logfile"]

    if mean is None:
        print(f"  decimal run did not parse a numeric result. See log: {logfile}")
        notes = "decimal run parse-failed; see log"
    else:
        notes = "decimal auto runtime result"

    row = {
        "timestamp": time.time(),
        "machine": platform.node(),
        "n": n,
        "digits": digits,
        "mean_s": mean,
        "std_s": std,
        "samples": samples,
        "repeat_times": "",  # unknown for decimal-auto-run parsing
        "notes": notes,
        "logfile": logfile
    }
    return [row]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[2048, 4096])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decimal", action="store_true", help="Attempt decimal auto-runtime experiments in addition to baseline")
    p.add_argument("--decimal-only", action="store_true", help="Skip NumPy baseline; only attempt decimal runs")
    p.add_argument("--digits", type=int, default=8, help="Digits parameter for decimal runs")
    p.add_argument("--timeout", type=int, default=600, help="Timeout (s) for decimal runs")
    p.add_argument("--python-exec", type=str, default=None, help="Path to python executable to use when running decimal runner")
    args = p.parse_args()

    rows_to_write = []

    # Run baseline (NumPy) unless decimal-only requested
    if not args.decimal_only:
        for n in args.sizes:
            print(f"Running baseline n={n} ...")
            mean, gflops, times = run_numpy_gemm(n, repeats=args.repeats)
            print(f"n={n}: mean={mean:.4f}s, gflops={gflops:.2f}")
            rows_to_write.append({
                "timestamp": time.time(),
                "machine": platform.node(),
                "n": n,
                "digits": "",
                "mean_s": mean,
                "std_s": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                "samples": None,
                "repeat_times": json.dumps(times),
                "notes": "numpy baseline",
                "logfile": ""
            })

    # Decimal runs
    if args.decimal:
        for n in args.sizes:
            try:
                dec_rows = run_decimal_experiment(n, args.digits, repeats=args.repeats, timeout=args.timeout, python_exec=args.python_exec)
                rows_to_write.extend(dec_rows)
                # small sleep to avoid race in log filenames
                time.sleep(0.1)
            except Exception as e:
                print(f"[decimal] ERROR while running decimal experiment n={n}: {e}")
                rows_to_write.append({
                    "timestamp": time.time(),
                    "machine": platform.node(),
                    "n": n,
                    "digits": args.digits,
                    "mean_s": None,
                    "std_s": None,
                    "samples": None,
                    "repeat_times": "",
                    "notes": f"decimal run exception: {e}",
                    "logfile": ""
                })

    if rows_to_write:
        # deterministic header
        fieldnames = ["timestamp", "machine", "n", "digits", "mean_s", "std_s", "samples", "repeat_times", "notes", "logfile"]
        _append_rows_to_csv(OUT, rows_to_write, fieldnames)
        print("Saved results to", OUT)

    # Save metadata
    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.node(),
        "sizes": args.sizes,
        "repeats": args.repeats,
        "digits": args.digits,
        "decimal_attempted": args.decimal,
        "decimal_only": args.decimal_only,
        "timestamp": time.time()
    }
    try:
        META.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass
    print("Saved meta to", META)

if __name__ == "__main__":
    main()
