# decimal_computer/fastdecimal.py
"""
decimal_computer.fastdecimal

FastDecimal implementation with a small 'freeze' API to convert values to
pure integer-backed representation (drop _orig_decimal) to enable faster
integer-only arithmetic in hot paths.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional, Sequence, Union, List

from .context import get_context
from .rounding import quantize_int

# --- paste into decimal_computer/fastdecimal.py (near top-level helpers) ---

from array import array as _array
from typing import Iterable, Iterator

class FastDecimalView:
    """
    A lazy view over an array('q') of quantized integer values (int_value)
    with a fixed scale. Acts like a read-only sequence of FastDecimal-like objects.

    - Does NOT allocate a list of FastDecimal objects.
    - __getitem__ returns a lightweight FastDecimal constructed on-demand.
    - __iter__ yields FastDecimal objects lazily.
    - Use .to_list() to eagerly materialize a list of FastDecimal (if needed).

    Example:
        arr = array('q', [1234, 5678])
        view = FastDecimalView(arr, scale=3)
        print(view[0])            # FastDecimal created on access
        for fd in view: ...
    """
    __slots__ = ("_arr", "scale")

    def __init__(self, int_array: _array, scale: int):
        if int_array.typecode != "q":
            raise TypeError("FastDecimalView expects an array('q') of int64")
        self._arr = int_array
        self.scale = int(scale)

    def __len__(self) -> int:
        return len(self._arr)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            sub = self._arr[idx]
            return FastDecimalView(sub, self.scale)
        iv = self._arr[idx]
        # Create a FastDecimal on demand. Use internal constructor for speed.
        return FastDecimal(int_value=int(iv), scale=self.scale, _orig_decimal=None)

    def __iter__(self) -> Iterator["FastDecimal"]:
        for iv in self._arr:
            yield FastDecimal(int_value=int(iv), scale=self.scale, _orig_decimal=None)

    def to_list(self) -> list:
        """Return an eagerly-allocated list of FastDecimal objects."""
        return [FastDecimal(int_value=int(iv), scale=self.scale, _orig_decimal=None) for iv in self._arr]

    def to_decimal_list(self):
        """Return a list of `decimal.Decimal` values (useful for printing/comparison)."""
        from decimal import Decimal
        return [Decimal(int(iv)).scaleb(-self.scale) for iv in self._arr]

    def __repr__(self) -> str:
        return f"FastDecimalView(len={len(self._arr)}, scale={self.scale})"


@dataclass
class FastDecimal:
    int_value: int
    scale: int
    _orig_decimal: Optional[Decimal] = field(default=None, repr=False, compare=False)

    @classmethod
    def _decimal_input_scale(cls, dec: Decimal) -> int:
        exp = -dec.as_tuple().exponent
        return int(exp) if exp > 0 else 0

    @classmethod
    def from_str(cls, s: str) -> "FastDecimal":
        ctx = get_context()
        dec = Decimal(s)
        orig = dec
        quant = Decimal(1).scaleb(-ctx.scale)
        q = dec.quantize(quant, rounding=ctx.rounding)
        int_value = int(q.scaleb(ctx.scale).to_integral_value(rounding=ctx.rounding))
        return cls(int_value=int_value, scale=ctx.scale, _orig_decimal=orig)

    @classmethod
    def from_int(cls, i: int) -> "FastDecimal":
        ctx = get_context()
        return cls(int_value=int(i) * 10 ** ctx.scale, scale=ctx.scale, _orig_decimal=None)

    @classmethod
    def from_float(cls, f: float) -> "FastDecimal":
        ctx = get_context()
        dec = Decimal(str(f))
        orig = dec
        quant = Decimal(1).scaleb(-ctx.scale)
        q = dec.quantize(quant, rounding=ctx.rounding)
        int_value = int(q.scaleb(ctx.scale).to_integral_value(rounding=ctx.rounding))
        return cls(int_value=int_value, scale=ctx.scale, _orig_decimal=orig)

    def to_decimal(self) -> Decimal:
        return Decimal(self.int_value).scaleb(-self.scale)

    def to_float(self) -> float:
        return float(self.to_decimal())

    def __str__(self) -> str:
        s = format(self.to_decimal(), 'f')
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    def __repr__(self) -> str:
        return f"FastDecimal({self.int_value}, scale={self.scale})"

    def __neg__(self) -> "FastDecimal":
        orig = -self._orig_decimal if self._orig_decimal is not None else None
        return FastDecimal(int_value=-self.int_value, scale=self.scale, _orig_decimal=orig)

    def __abs__(self) -> "FastDecimal":
        orig = abs(self._orig_decimal) if self._orig_decimal is not None else None
        return FastDecimal(int_value=abs(self.int_value), scale=self.scale, _orig_decimal=orig)

    def __hash__(self) -> int:
        return hash((self.int_value, self.scale))

    def to_json(self) -> dict:
        return {"int": self.int_value, "scale": self.scale}

    @classmethod
    def from_json(cls, d: dict) -> "FastDecimal":
        if not isinstance(d, dict) or "int" not in d or "scale" not in d:
            raise ValueError("Invalid serialization for FastDecimal")
        return cls(int_value=int(d["int"]), scale=int(d["scale"]), _orig_decimal=None)

    # ---- New: freeze API ----
    def freeze(self) -> "FastDecimal":
        """
        Return a FastDecimal that is quantized to the current context.scale and
        does NOT preserve _orig_decimal (i.e., pure integer-backed).

        Use this to convert values to the fast integer-backed representation
        when you don't need the original Decimal for future exactness.
        """
        ctx = get_context()
        # If already pure and at context.scale, return self
        if (self._orig_decimal is None) and (self.scale == ctx.scale):
            return self
        # Quantize using integer engine when possible
        if self._orig_decimal is None:
            new_int = quantize_int(self.int_value, orig_scale=self.scale, target_scale=ctx.scale, rounding=ctx.rounding)
            return FastDecimal(int_value=new_int, scale=ctx.scale, _orig_decimal=None)
        # If original decimal was present, quantize using Decimal and drop orig
        quant = Decimal(1).scaleb(-ctx.scale)
        q = (self._orig_decimal).quantize(quant, rounding=ctx.rounding)
        int_value = int(q.scaleb(ctx.scale).to_integral_value(rounding=ctx.rounding))
        return FastDecimal(int_value=int_value, scale=ctx.scale, _orig_decimal=None)

    @classmethod
    def freeze_list(cls, seq: Sequence[Union[int, float, str, "FastDecimal"]]) -> List["FastDecimal"]:
        """
        Convert a sequence of scalars/FastDecimal into a list of pure integer-backed
        FastDecimal instances (calling freeze() on each). Useful before batch ops.
        """
        out: List[FastDecimal] = []
        for item in seq:
            if isinstance(item, FastDecimal):
                out.append(item.freeze())
            else:
                # convert via ensure-like construction then freeze (from_str/from_int/from_float will set orig)
                fd = cls.from_str(str(item))
                out.append(fd.freeze())
        return out

    # ---- Existing internal helpers & arithmetic (unchanged) ----
    def _ensure_same_scale(self, other: "FastDecimal") -> tuple[int, int, int]:
        ctx = get_context()
        if self.scale == ctx.scale and other.scale == ctx.scale:
            return self.int_value, other.int_value, ctx.scale

        if (self._orig_decimal is None) and (other._orig_decimal is None):
            a_int_res = quantize_int(self.int_value, orig_scale=self.scale, target_scale=ctx.scale, rounding=ctx.rounding)
            b_int_res = quantize_int(other.int_value, orig_scale=other.scale, target_scale=ctx.scale, rounding=ctx.rounding)
            return a_int_res, b_int_res, ctx.scale

        a_dec = (self._orig_decimal if self._orig_decimal is not None else Decimal(self.int_value).scaleb(-self.scale))
        b_dec = (other._orig_decimal if other._orig_decimal is not None else Decimal(other.int_value).scaleb(-other.scale))
        quant = Decimal(1).scaleb(-ctx.scale)
        a_q = a_dec.quantize(quant, rounding=ctx.rounding)
        b_q = b_dec.quantize(quant, rounding=ctx.rounding)
        a_int = int(a_q.scaleb(ctx.scale).to_integral_value(rounding=ctx.rounding))
        b_int = int(b_q.scaleb(ctx.scale).to_integral_value(rounding=ctx.rounding))
        return a_int, b_int, ctx.scale

    def _from_decimal_with_ctx(self, dec: Decimal) -> "FastDecimal":
        ctx = get_context()
        quant = Decimal(1).scaleb(-ctx.scale)
        q = dec.quantize(quant, rounding=ctx.rounding)
        int_value = int(q.scaleb(ctx.scale).to_integral_value(rounding=ctx.rounding))
        return FastDecimal(int_value=int_value, scale=ctx.scale, _orig_decimal=None)

    def __add__(self, other: Any) -> "FastDecimal":
        if not isinstance(other, FastDecimal):
            other = FastDecimal.from_str(str(other))
        a_int, b_int, scale = self._ensure_same_scale(other)
        res_int = a_int + b_int
        return FastDecimal(int_value=res_int, scale=scale, _orig_decimal=None)

    def __sub__(self, other: Any) -> "FastDecimal":
        if not isinstance(other, FastDecimal):
            other = FastDecimal.from_str(str(other))
        a_int, b_int, scale = self._ensure_same_scale(other)
        res_int = a_int - b_int
        return FastDecimal(int_value=res_int, scale=scale, _orig_decimal=None)

    def __mul__(self, other: Any) -> "FastDecimal":
        if not isinstance(other, FastDecimal):
            other = FastDecimal.from_str(str(other))

        ctx = get_context()
        if (self._orig_decimal is None) and (other._orig_decimal is None):
            prod_int = self.int_value * other.int_value
            prod_scale = self.scale + other.scale
            res_int = quantize_int(prod_int, orig_scale=prod_scale, target_scale=ctx.scale, rounding=ctx.rounding)
            return FastDecimal(int_value=res_int, scale=ctx.scale, _orig_decimal=None)

        a_dec = self._orig_decimal if self._orig_decimal is not None else Decimal(self.int_value).scaleb(-self.scale)
        b_dec = other._orig_decimal if other._orig_decimal is not None else Decimal(other.int_value).scaleb(-other.scale)
        dec_res = a_dec * b_dec
        return self._from_decimal_with_ctx(dec_res)

    def __truediv__(self, other: Any) -> "FastDecimal":
        if not isinstance(other, FastDecimal):
            other = FastDecimal.from_str(str(other))
        if other.int_value == 0 and (other._orig_decimal is None or other._orig_decimal == Decimal(0)):
            raise ZeroDivisionError("division by zero FastDecimal")
        a_dec = self._orig_decimal if self._orig_decimal is not None else Decimal(self.int_value).scaleb(-self.scale)
        b_dec = other._orig_decimal if other._orig_decimal is not None else Decimal(other.int_value).scaleb(-other.scale)
        dec_res = a_dec / b_dec
        return self._from_decimal_with_ctx(dec_res)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FastDecimal):
            try:
                other = FastDecimal.from_str(str(other))
            except Exception:
                return False
        a_int, b_int, _ = self._ensure_same_scale(other)
        return a_int == b_int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, FastDecimal):
            other = FastDecimal.from_str(str(other))
        a_int, b_int, _ = self._ensure_same_scale(other)
        return a_int < b_int

    def __le__(self, other: Any) -> bool:
        return self == other or self < other

    def quantize(self, scale: int) -> "FastDecimal":
        ctx = get_context()
        if self._orig_decimal is None:
            res_int = quantize_int(self.int_value, orig_scale=self.scale, target_scale=scale, rounding=ctx.rounding)
            return FastDecimal(int_value=res_int, scale=scale, _orig_decimal=None)
        dec = self._orig_decimal if self._orig_decimal is not None else Decimal(self.int_value).scaleb(-self.scale)
        quant = Decimal(1).scaleb(-scale)
        q = dec.quantize(quant, rounding=ctx.rounding)
        int_value = int(q.scaleb(scale).to_integral_value(rounding=ctx.rounding))
        return FastDecimal(int_value=int_value, scale=scale, _orig_decimal=None)

    @classmethod
    def vector_add(cls, a_list: Sequence[Union[int, float, str, "FastDecimal"]], b: Union[int, float, str, "FastDecimal"]) -> list["FastDecimal"]:
        from . import vector_ops
        return vector_ops.add_vectors(a_list, b)

    @classmethod
    def vector_sub(cls, a_list: Sequence[Union[int, float, str, "FastDecimal"]], b: Union[int, float, str, "FastDecimal"]) -> list["FastDecimal"]:
        from . import vector_ops
        return vector_ops.sub_vectors(a_list, b)

    @classmethod
    def vector_mul(cls, a_list: Sequence[Union[int, float, str, "FastDecimal"]], b: Union[int, float, str, "FastDecimal"]) -> list["FastDecimal"]:
        from . import vector_ops
        return vector_ops.mul_vectors(a_list, b)
