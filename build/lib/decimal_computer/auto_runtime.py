# decimal_computer/auto_runtime.py
"""
Runtime helper to pick and call the best multiplication algorithm using
experiments/best_algos.json mapping produced by experiments/auto_select_best_algo.py.

This upgraded version:
 - selects an algorithm (from mapping or fallback list)
 - times repeated runs of decimal_gemm_naive (or alternative API shapes)
 - prints a JSON summary to stdout with keys:
     { "algo": ..., "mean_s": ..., "std_s": ..., "samples": ..., "trials": ..., "success": true/false, "error": null or str }
 - also logs helpful human-readable diagnostics to stderr
 - CLI accepts --size, --digits, --trials, --verbose etc.

Notes:
 - The wrapper is defensive about your project's API; it will try several call styles.
 - If decimal_gemm_naive itself returns timing/counters, we prefer those values;
   otherwise we time the calls here.
"""
from pathlib import Path
import json
import logging
import importlib
import time
import statistics
from typing import Any, Dict, Optional, Tuple, List

logger = logging.getLogger("decimal_computer.auto_runtime")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[auto_runtime] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

DEFAULT_MAPPING_PATH = Path("experiments/best_algos.json")
DEFAULT_FALLBACK_ORDER = ["numba", "schoolbook", "karatsuba", "strassen", "auto"]

# ---------- mapping utilities ----------

def load_best_algos(path: Path = DEFAULT_MAPPING_PATH) -> Dict[str, Dict[str, str]]:
    """Load best_algos.json and normalize to mapping[size_str][digits_str] -> algo."""
    if not Path(path).exists():
        logger.info(f"best_algos mapping not found at {path} — proceeding without mapping")
        return {}
    try:
        with open(path, "r", encoding="utf8") as fh:
            raw = json.load(fh)
    except Exception as e:
        logger.warning(f"failed to load mapping {path}: {e}")
        return {}

    mapping: Dict[str, Dict[str, str]] = {}
    if isinstance(raw, dict):
        # support either "size:digits" keys or nested {size: {digits: algo}}
        sample_key = next(iter(raw.keys())) if raw else None
        if sample_key and isinstance(sample_key, str) and ":" in sample_key:
            for k, v in raw.items():
                try:
                    s, d = str(k).split(":", 1)
                    algo = v.get("algo") if isinstance(v, dict) else v
                    mapping.setdefault(str(int(s)), {})[str(int(d))] = str(algo) if algo is not None else None
                except Exception:
                    continue
        else:
            for s, inner in raw.items():
                try:
                    if isinstance(inner, dict):
                        for d, algo in inner.items():
                            mapping.setdefault(str(int(s)), {})[str(int(d))] = str(algo) if algo is not None else None
                except Exception:
                    continue
    return mapping

def choose_best_algo(size: Optional[int], digits: Optional[int],
                     mapping: Optional[Dict[str, Dict[str, str]]] = None,
                     default: Optional[str] = "auto") -> Optional[str]:
    if mapping is None:
        mapping = load_best_algos()
    s = str(int(size)) if size is not None else None
    d = str(int(digits)) if digits is not None else None
    try:
        if s is not None and d is not None:
            algo = mapping.get(s, {}).get(d)
            if algo:
                logger.info(f"Mapped (size={s}, digits={d}) -> algo='{algo}'")
                return algo
    except Exception:
        pass
    logger.info(f"No mapping for (size={s}, digits={d}); using default='{default}'")
    return default

# ---------- import decimal_gemm function ----------

def _import_decimal_gemm():
    try:
        mod = importlib.import_module("decimal_computer.decimal_gemm")
    except Exception as e:
        raise ImportError(f"Failed to import decimal_computer.decimal_gemm: {e}") from e
    # Choose likely function names (many variants in your project)
    candidates = ["decimal_gemm_naive", "decimal_gemm", "decimal_gemm_auto", "decimal_gemm_wrapper"]
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name)
    # fallback: try any callable in module (risky)
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if callable(obj):
            return obj
    raise AttributeError("module decimal_computer.decimal_gemm has no usable callable")

# ---------- runtime wrapper and timing ----------

def _prepare_test_matrices(size: int, digits: Optional[int]) -> Tuple[Any, Any]:
    """
    Build small test matrices in the representation expected by decimal_gemm functions.
    Try int_to_digits helper; otherwise fall back to plain integers.
    """
    try:
        dd = importlib.import_module("decimal_computer.decimal_digit_starter")
        if hasattr(dd, "int_to_digits"):
            int_to_digits = getattr(dd, "int_to_digits")
            s = size
            d = digits if digits is not None else 4
            A = [[int_to_digits(1, d) for _ in range(s)] for _ in range(s)]
            B = [[int_to_digits(2, d) for _ in range(s)] for _ in range(s)]
            return A, B
    except Exception:
        pass
    # fallback: integer matrices
    s = size
    A = [[1 for _ in range(s)] for _ in range(s)]
    B = [[2 for _ in range(s)] for _ in range(s)]
    return A, B

def _call_decimal_gemm_with_algo(fn, A, B, algo: Optional[str], kwargs: dict):
    """
    Try calling decimal_gemm in multiple styles to support varying APIs:
     - decimal_gemm(A, B, mul_algo=algo, cutoff=..., return_counters=True)
     - decimal_gemm(A, B, mul_algo=algo)
     - decimal_gemm(A, B)
    Returns (result, counters, elapsed_from_fn_or_None)
    Counters may be None or (muls, adds, elapsed) depending on API.
    """
    # build kw copy
    kw = dict(kwargs or {})
    if algo is not None:
        kw["mul_algo"] = algo

    # 1) try with return_counters=True (common in our project)
    try:
        res = fn(A, B, **kw, return_counters=True)
        # res could be (C, muls, adds, elapsed) or similar
        return res
    except TypeError:
        # maybe function doesn't accept return_counters kw
        pass
    except Exception as e:
        # propagate to caller as failure
        raise

    # 2) try without return_counters but with mul_algo
    try:
        res = fn(A, B, **kw)
        return res
    except TypeError:
        pass
    except Exception as e:
        raise

    # 3) try simple positional call fn(A, B)
    try:
        res = fn(A, B)
        return res
    except Exception as e:
        raise

def time_fn_call(fn, A, B, algo: Optional[str], kwargs: dict):
    """
    Execute a single call and try to extract elapsed if returned by fn.
    Otherwise measure elapsed locally.
    Returns tuple (success_bool, elapsed_seconds_or_None, returned_result, exception_or_None).
    """
    start = time.perf_counter()
    try:
        res = _call_decimal_gemm_with_algo(fn, A, B, algo, kwargs)
        # If function returned a tuple containing timing, try to extract numeric elapsed
        elapsed = None
        # Case heuristics:
        # - If res is a tuple and last element is numeric and small -> treat as elapsed
        if isinstance(res, tuple) and len(res) >= 2:
            # find numeric element that looks like elapsed (float, int)
            for el in reversed(res):
                if isinstance(el, (float, int)):
                    # sanity: elapsed not absurdly large
                    if 0 <= float(el) < 1e6:
                        elapsed = float(el)
                        break
        # If not found, measure local
        if elapsed is None:
            elapsed = time.perf_counter() - start
        return True, float(elapsed), res, None
    except Exception as e:
        return False, None, None, e

def run_and_time(fn, size: int, digits: Optional[int], algo: Optional[str], trials: int = 3, kwargs: dict = None, verbose: bool = False):
    """Run decimal gemm `trials` times and return summary dict."""
    A, B = _prepare_test_matrices(size, digits)
    times: List[float] = []
    last_exc = None
    results_samples = None

    for i in range(trials):
        ok, elapsed, res, exc = time_fn_call(fn, A, B, algo, kwargs or {})
        if not ok:
            last_exc = exc
            if verbose:
                logger.warning(f"attempt {i+1}/{trials} failed for algo='{algo}': {exc}")
            # small backoff
            time.sleep(0.01)
            continue
        times.append(elapsed)
        # if function returned a 'samples' count inside result/counters, attempt to parse it
        if isinstance(res, tuple):
            # try to find integer sample-like value
            for v in res:
                if isinstance(v, int) and v > 1:
                    results_samples = int(v)
                    break

    if not times:
        # all runs failed
        return {
            "success": False,
            "algo": algo,
            "mean_s": None,
            "std_s": None,
            "samples": results_samples,
            "trials": trials,
            "error": str(last_exc) if last_exc is not None else "All attempts failed"
        }

    mean_s = float(statistics.mean(times))
    std_s = float(statistics.pstdev(times) if len(times) >= 1 else 0.0)
    return {
        "success": True,
        "algo": algo,
        "mean_s": mean_s,
        "std_s": std_s,
        "samples": results_samples,
        "trials": trials,
        "error": None
    }

# ---------- high-level API ----------

def decimal_gemm_auto(A, B, size: Optional[int] = None, digits: Optional[int] = None,
                      mapping_path: Optional[Path] = None, fallback_order: Optional[List[str]] = None,
                      trials: int = 3, kwargs: Optional[dict] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Try chosen algo (via mapping) and fallbacks; perform multiple trials and return summary dict.
    The returned dict contains keys: success, algo, mean_s, std_s, samples, trials, error.
    """
    if mapping_path is None:
        mapping_path = DEFAULT_MAPPING_PATH
    mapping = load_best_algos(Path(mapping_path))
    chosen = choose_best_algo(size, digits, mapping=mapping, default=None)

    if fallback_order is None:
        fallback_order = DEFAULT_FALLBACK_ORDER

    try_list = []
    if chosen:
        try_list.append(str(chosen))
    for a in fallback_order:
        if a not in try_list:
            try_list.append(a)

    # import function
    try:
        fn = _import_decimal_gemm()
    except Exception as e:
        logger.error(f"Failed to import decimal_gemm function: {e}")
        return {"success": False, "algo": None, "mean_s": None, "std_s": None, "samples": None, "trials": trials, "error": str(e)}

    last_err = None
    for algo in try_list:
        if verbose:
            logger.info(f"Trying algo='{algo}' for size={size} digits={digits}")
        try:
            summary = run_and_time(fn, size if size is not None else (len(A) if hasattr(A, "__len__") else 4),
                                   digits, algo, trials=trials, kwargs=kwargs or {}, verbose=verbose)
            if summary.get("success"):
                return summary
            else:
                last_err = summary.get("error")
                logger.warning(f"Algo '{algo}' completed but reported failure: {last_err}")
        except Exception as e:
            last_err = e
            logger.warning(f"Failed with algo='{algo}': {e}")

    return {"success": False, "algo": None, "mean_s": None, "std_s": None, "samples": None, "trials": trials, "error": str(last_err)}

# ---------- CLI for local testing ----------

def _print_json_stdout(d: Dict):
    # pretty json to stdout (for machine parsing)
    print(json.dumps(d), flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Runtime auto-selection wrapper for decimal GEMM (with timing & JSON output)")
    parser.add_argument("--size", "--n", type=int, dest="size", default=4, help="matrix size n")
    parser.add_argument("--digits", type=int, default=4, help="digit width (if using digit lists)")
    parser.add_argument("--trials", type=int, default=3, help="number of trials to time")
    parser.add_argument("--mapping", type=str, default=str(DEFAULT_MAPPING_PATH), help="path to best_algos.json")
    parser.add_argument("--fallback", type=str, default=",".join(DEFAULT_FALLBACK_ORDER), help="comma-separated fallback algos")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # configure logger verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Prepare tiny test matrices for CLI-run
    A, B = _prepare_test_matrices(args.size, args.digits)

    # Run selection & timing
    try:
        mapping_path = Path(args.mapping)
        fallback_order = [s.strip() for s in args.fallback.split(",") if s.strip()]
        summary = decimal_gemm_auto(A, B, size=args.size, digits=args.digits,
                                    mapping_path=mapping_path, fallback_order=fallback_order,
                                    trials=args.trials, kwargs=None, verbose=args.verbose)
        # attach metadata
        summary_out = {
            "timestamp": time.time(),
            "algo": summary.get("algo"),
            "mean_s": summary.get("mean_s"),
            "std_s": summary.get("std_s"),
            "samples": summary.get("samples"),
            "trials": summary.get("trials"),
            "success": bool(summary.get("success")),
            "error": summary.get("error")
        }
        # Print JSON to stdout (machine-readable) and a short human log to stderr
        _print_json_stdout(summary_out)
        if summary_out["success"]:
            logger.info(f"Completed: algo={summary_out['algo']} mean_s={summary_out['mean_s']} std_s={summary_out['std_s']} trials={summary_out['trials']}")
        else:
            logger.warning(f"Failed to produce numeric result: {summary_out.get('error')}")
    except Exception as exc:
        err = {"timestamp": time.time(), "success": False, "error": str(exc)}
        print(json.dumps(err), flush=True)
        logger.error(f"Fatal error in auto runtime: {exc}")
        raise
