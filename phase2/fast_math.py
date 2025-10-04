# --- compatibility exports for tests ---
# If your internal names differ, adapt these wrappers to call them.
try:
    # If digit_mul/digit_add are already defined, this does nothing.
    digit_mul  # type: ignore
    digit_add  # type: ignore
except NameError:
    # provide shims if underlying implementation uses different names
    # Replace `my_digit_mul_impl` / `my_digit_add_impl` with real names if needed.
    def digit_mul(a: int, b: int) -> int:
        """Multiply single decimal digits (compat shim)."""
        # try to use a verified internal function if present
        impl = globals().get("digit_mul_impl") or globals().get("digit_mul_internal")
        if callable(impl):
            return impl(a, b)
        # fallback naive impl (safe)
        return a * b

    def digit_add(a: int, b: int, carry: int = 0) -> (int, int):
        """Add two digits plus carry -> (result_digit, new_carry)."""
        impl = globals().get("digit_add_impl") or globals().get("digit_add_internal")
        if callable(impl):
            return impl(a, b, carry)
        s = a + b + (carry or 0)
        return s % 10, s // 10
