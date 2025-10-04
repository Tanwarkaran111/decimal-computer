# decimal_computer/auto_runtime.py
"""
Runtime auto-selection wrapper for decimal GEMM.

Behavior:
 - Loads experiments/best_algos.json (if present) to map (size,digits) -> algo.
 - choose_best_algo picks mapped algo or falls back to default.
 - decimal_gemm_auto tries chosen algo then fallback list until success.
 - Very defensive: tolerates missing modules, signature differences, and reports helpful logs.

Usage (example):
    python -m decimal_computer.auto_runtime --size 64 --digits 8 --verbose
"""
from pathlib import Path
import json
import logging
from typing import Any, Dict, Optional, Tuple, List

import importlib

# logging
logger = logging.getLogger("decimal_computer.auto_runtime")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[auto_runtime] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

DEFAULT_MAPPING_PATH = Path("experiments/best_algos.json")
DEFAULT_FALLBACK_ORDER = ["numba", "karatsuba", "schoolbook", "strassen", "auto"]
# Number of attempts per algo (useful for flaky implementations)
DEFAULT_ATTEMPTS = 3


def load_best_algos(path: Path = DEFAULT_MAPPING_PATH) -> Dict[str, Dict[str, str]]:
    """Load mapping file; normalize different shapes into mapping[str(size)][str(digits)] -> algo."""
    path = Path(path)
    if not path.exists():
        logger.info(f"Mapping not found at {path}; proceeding without mapping")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf8"))
    except Exception as e:
        logger.warning(f"Failed to load mapping {path}: {e}")
        return {}

    mapping: Dict[str, Dict[str, str]] = {}
    if not isinstance(raw, dict):
        return mapping

    # Two common shapes:
    # 1) nested: {"4": {"8": "numba", ...}, ...}
    # 2) flattened keys: {"4:8": {"algo": "numba", ...}, ...}
    sample_key = next(iter(raw.keys()), None)
    if sample_key is None:
        return mapping

    try:
        if isinstance(sample_key, str) and ":" in sample_key:
            # flatten case
            for k, v in raw.items():
                try:
                    s, d = k.split(":", 1)
                    algo = v.get("algo") if isinstance(v, dict) else v
                    mapping.setdefault(str(int(s)), {})[str(int(d))] = str(algo) if algo is not None else None
                except Exception:
                    continue
        else:
            # nested case
            for s, inner in raw.items():
                try:
                    if isinstance(inner, dict):
                        for d, algo in inner.items():
                            mapping.setdefault(str(int(s)), {})[str(int(d))] = str(algo) if algo is not None else None
                except Exception:
                    continue
    except Exception:
        # best-effort; return whatever we have
        return mapping

    return mapping


def choose_best_algo(size: Optional[int], digits: Optional[int],
                     mapping: Optional[Dict[str, Dict[str, str]]] = None,
                     default: Optional[str] = "auto") -> Optional[str]:
    """Return mapping[size][digits] if present, else default."""
    if mapping is None:
        mapping = load_best_algos(DEFAULT_MAPPING_PATH)
    try:
        s = str(int(size)) if size is not None else None
        d = str(int(digits)) if digits is not None else None
    except Exception:
        s, d = None, None

    algo = None
    if s is not None and d is not None:
        algo = mapping.get(s, {}).get(d)
    if algo is None:
        logger.info(f"No mapping for (size={s}, digits={d}); using default='{default}'")
        return default
    logger.info(f"Mapped (size={s}, digits={d}) -> algo='{algo}'")
    return algo


def _import_decimal_gemm():
    """Return decimal_gemm_naive callable or raise."""
    try:
        mod = importlib.import_module("decimal_computer.decimal_gemm")
    except Exception as e:
        raise ImportError(f"Failed to import decimal_computer.decimal_gemm: {e}") from e
    if not hasattr(mod, "decimal_gemm_naive"):
        raise AttributeError("decimal_computer.decimal_gemm has no 'decimal_gemm_naive'")
    return getattr(mod, "decimal_gemm_naive")


def _is_numba_available() -> bool:
    """Quick probe whether the numba_impl backend exists and reports availability."""
    try:
        mod = importlib.import_module("decimal_computer.numba_impl")
        return bool(getattr(mod, "_has_numba", True))
    except Exception:
        return False


def decimal_gemm_auto(A, B, size: Optional[int] = None, digits: Optional[int] = None,
                      mapping_path: Optional[Path] = None, fallback_order: Optional[List[str]] = None,
                      attempts: int = DEFAULT_ATTEMPTS, verbose: bool = False, **kwargs) -> Any:
    """
    Choose algorithm and call decimal_gemm_naive. Tries chosen then fallbacks until success.

    Returns whatever decimal_gemm_naive returns. If all attempts fail, re-raises last exception.
    """
    # infer size/digits if possible
    if size is None:
        try:
            size = int(len(A))
        except Exception:
            size = None
    if digits is None:
        try:
            first = A[0][0]
            if isinstance(first, (list, tuple)):
                digits = int(len(first))
        except Exception:
            digits = None

    # mapping
    mapping = load_best_algos(mapping_path or DEFAULT_MAPPING_PATH)

    chosen = choose_best_algo(size, digits, mapping=mapping, default=None)

    # compose fallback list
    if fallback_order is None:
        fallback_order = DEFAULT_FALLBACK_ORDER.copy()

    # If chosen present and not in fallback list, prefer it first
    try_list = []
    if chosen:
        try_list.append(str(chosen))
    for a in fallback_order:
        if a not in try_list:
            try_list.append(a)

    # prefer to skip numba attempts if numba isn't available
    if "numba" in try_list and not _is_numba_available():
        logger.info("Numba backend not available in environment -> skipping 'numba' in try list")
        try_list = [a for a in try_list if a != "numba"]

    # import function
    try:
        decimal_gemm_fn = _import_decimal_gemm()
    except Exception as e:
        logger.error(e)
        raise

    last_exc = None
    for algo in try_list:
        # attempts loop for flaky impls
        for attempt in range(1, max(1, attempts) + 1):
            # prepare kwargs to pass
            kw = dict(kwargs)
            # try common keyword name
            kw.update({"mul_algo": algo})
            try:
                if verbose:
                    logger.info(f"Attempting decimal_gemm_naive with algo='{algo}' (size={size}, digits={digits}) attempt={attempt}")
                res = decimal_gemm_fn(A, B, **kw)
                # if function returned successfully, return its result
                if verbose:
                    logger.info(f"Success with algo='{algo}' on attempt {attempt}")
                return res
            except TypeError as te:
                # maybe function doesn't accept mul_algo kw; try positional call (best-effort)
                try:
                    if verbose:
                        logger.info(f"TypeError calling with mul_algo kw -> retrying positional call: attempt {attempt}")
                    res = decimal_gemm_fn(A, B)
                    return res
                except Exception as e2:
                    last_exc = e2
                    logger.warning(f"attempt {attempt}/{attempts} failed for algo='{algo}': {e2}")
            except Exception as e:
                last_exc = e
                logger.warning(f"attempt {attempt}/{attempts} failed for algo='{algo}': {e}")

        # After attempts exhausted for this algo, log and continue to next
        logger.warning(f"Algo '{algo}' completed but reported failure after {attempts} attempts")

    # all attempts failed
    logger.error(f"All algorithm attempts failed for size={size},digits={digits}. Raising last exception.")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No algorithms attempted (unexpected) -- check environment")


# Small CLI for quick local testing
# Small CLI for quick local testing
if __name__ == "__main__":
    import argparse, pprint
    parser = argparse.ArgumentParser(description="Runtime auto-selection wrapper for decimal GEMM")
    parser.add_argument("--size", type=int, default=None, help="matrix size")
    parser.add_argument("--digits", type=int, default=None, help="digit width (if representing digits)")
    parser.add_argument("--verbose", action="store_true", help="more logs")
    args = parser.parse_args()

    s = args.size or 4
    try:
        from decimal_computer.decimal_digit_starter import int_to_digits  # type: ignore
        A = [[int_to_digits(12, args.digits or 4) for _ in range(s)] for _ in range(s)]
        B = [[int_to_digits(34, args.digits or 4) for _ in range(s)] for _ in range(s)]
    except Exception:
        A = [[1 for _ in range(s)] for _ in range(s)]
        B = [[2 for _ in range(s)] for _ in range(s)]

    try:
        res = decimal_gemm_auto(A, B, size=args.size, digits=args.digits, verbose=args.verbose)
        print("decimal_gemm_auto result (summary):")
        if isinstance(res, tuple) and isinstance(res[0], list):
            # only show matrix shape and small preview
            rows = len(res[0])
            cols = len(res[0][0]) if rows > 0 else 0
            print(f"Matrix shape: {rows}x{cols}")
            preview = [row[:2] for row in res[0][:2]]  # top-left 2x2 block
            print("Top-left corner (2x2):", preview)
            if len(res) > 1:
                print("Counters:", res[1])
        else:
            pprint.pprint(res)
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        raise


