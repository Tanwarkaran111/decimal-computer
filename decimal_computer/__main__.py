# decimal_computer/__main__.py
"""
Simple CLI demo for decimal_computer.

Run with:
    python -m decimal_computer

This prints a few example operations and a tiny self-check using the test-style asserts.
Designed to be safe to run from developer machines.
"""
from __future__ import annotations
from decimal import Decimal
import sys

from .context import DecimalContext, get_context
from .fastdecimal import FastDecimal


def demo_examples() -> None:
    print("decimal_computer demo\n---------------------")
    # show current default context
    ctx = get_context()
    print(f"Active context: precision={ctx.precision}, scale={ctx.scale}, rounding={ctx.rounding}\n")

    # Example 1: construction & printing
    with DecimalContext(precision=10, scale=3, rounding="ROUND_HALF_EVEN"):
        a = FastDecimal.from_str("12.34567")
        b = FastDecimal.from_str("1.234")
        print("Example 1: from_str + str")
        print("  a = FastDecimal.from_str('12.34567')  ->", a)
        print("  b = FastDecimal.from_str('1.234')     ->", b)
        print()

    # Example 2: arithmetic with preserved input precision for multiplication
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        a = FastDecimal.from_str("1.23456")
        b = FastDecimal.from_str("2.5")
        prod = a * b
        print("Example 2: multiplication (uses original input precision during op, result quantized to context)")
        print("  1.23456 * 2.5 ->", prod)
        print()

    # Example 3: division and integer printing
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_EVEN"):
        x = FastDecimal.from_str("12.0")
        y = FastDecimal.from_str("4.0")
        print("Example 3: division")
        print("  12.0 / 4.0 ->", x / y)
        print()

    # Quick self-checks (mirrors core tests)
    print("Running quick self-checks...")
    with DecimalContext(precision=10, scale=3, rounding="ROUND_HALF_EVEN"):
        a = FastDecimal.from_str("12.34567")
        assert str(a) == "12.346", "rounding/self-check failed"
    with DecimalContext(precision=12, scale=4, rounding="ROUND_HALF_UP"):
        a = FastDecimal.from_str("1.23456")
        b = FastDecimal.from_str("2.5")
        prod = a * b
        expected = (Decimal("1.23456") * Decimal("2.5")).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
        assert str(prod) == format(expected, "f"), "multiply/self-check failed"
    print("All quick checks passed ✅\n")


def main(argv: list[str] | None = None) -> int:
    try:
        demo_examples()
        return 0
    except AssertionError as exc:
        print("Self-check failed:", exc, file=sys.stderr)
        return 2
    except Exception as exc:  # show unexpected errors
        print("Unexpected error:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
