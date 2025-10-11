# decimal_computer/context.py
"""
decimal_computer.context
------------------------

Small context manager and helper types for controlling precision, scale, rounding,
and computation mode used by decimal_computer.FastDecimal.

This version adds a `mode` field to globally control arithmetic behavior:
    - "safe": pure Python fallback path (maximum safety, slower)
    - "fast": use compiled C / zero-copy paths where possible
    - "auto": prefer fast path, fallback to safe if unsupported

You can also override mode globally via:
    export DECIMAL_COMPUTER_MODE=fast

Usage example:
--------------
from decimal_computer.context import DecimalContext

with DecimalContext(precision=10, scale=4, rounding='ROUND_HALF_EVEN', mode='auto'):
    ...
"""

from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from typing import Iterator, Optional
import os

# Thread-local storage so contexts are local to the thread.
_thread_ctx = threading.local()


@dataclass
class DecimalContext:
    """
    Lightweight decimal context controlling precision, scale, rounding, and mode.

    Attributes
    ----------
    precision : int
        Total number of significant digits to maintain for operations.
    scale : int
        Number of digits after the decimal point for storage/quantization.
    rounding : str
        Rounding rule name (e.g. 'ROUND_HALF_UP', 'ROUND_HALF_EVEN', etc.)
    mode : str
        Computation mode:
            'safe'  → always use pure Python path
            'fast'  → always use compiled C / zero-copy path
            'auto'  → try fast path, fallback to safe if needed
    """
    precision: int = 28
    scale: int = 2
    rounding: str = "ROUND_HALF_EVEN"
    mode: str = "safe"

    def __post_init__(self) -> None:
        if self.precision < 1:
            raise ValueError("precision must be >= 1")
        if self.scale < 0:
            raise ValueError("scale must be >= 0")
        if not isinstance(self.rounding, str):
            raise TypeError("rounding must be a string naming a decimal rounding mode")
        if self.mode not in ("safe", "fast", "auto"):
            raise ValueError("mode must be one of: 'safe', 'fast', 'auto'")

    def __enter__(self) -> "DecimalContext":
        # set this context as the current one (thread-local)
        prev = getattr(_thread_ctx, "current", None)
        _thread_ctx.current = self
        # store previous so nested contexts can restore
        _thread_ctx._prev = prev
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # restore previous context (or clear)
        prev = getattr(_thread_ctx, "_prev", None)
        if prev is not None:
            _thread_ctx.current = prev
            _thread_ctx._prev = getattr(prev, "_prev", None)
        else:
            if hasattr(_thread_ctx, "current"):
                del _thread_ctx.current
            if hasattr(_thread_ctx, "_prev"):
                del _thread_ctx._prev

    @staticmethod
    @contextmanager
    def use(precision: Optional[int] = None,
            scale: Optional[int] = None,
            rounding: Optional[str] = None,
            mode: Optional[str] = None) -> Iterator["DecimalContext"]:
        """
        Temporarily override the current context.

        Example:
        >>> with DecimalContext.use(scale=6, mode='fast'):
        ...     ...
        """
        cur = get_context()
        new = DecimalContext(
            precision=(precision if precision is not None else (cur.precision if cur else 28)),
            scale=(scale if scale is not None else (cur.scale if cur else 2)),
            rounding=(rounding if rounding is not None else (cur.rounding if cur else "ROUND_HALF_EVEN")),
            mode=(mode if mode is not None else (cur.mode if cur else "safe"))
        )
        with new:
            yield new

    def __repr__(self):
        return f"<DecimalContext precision={self.precision} scale={self.scale} rounding={self.rounding} mode={self.mode}>"


def get_context(default: Optional[DecimalContext] = None) -> DecimalContext:
    """
    Return the currently active DecimalContext for this thread.

    If no context is active, returns default if provided, otherwise returns
    a safe default (precision=28, scale=2, ROUND_HALF_EVEN).
    Also respects DECIMAL_COMPUTER_MODE environment variable.
    """
    cur = getattr(_thread_ctx, "current", None)
    if cur is not None:
        return cur
    if default is not None:
        return default

    # Allow global mode override via environment variable
    env_mode = os.getenv("DECIMAL_COMPUTER_MODE", "safe").lower()
    if env_mode not in ("safe", "fast", "auto"):
        env_mode = "safe"

    return DecimalContext(mode=env_mode)
