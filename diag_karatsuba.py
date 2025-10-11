# diag_karatsuba.py
# Save and run with: python diag_karatsuba.py

import random
import sys

from phase8_karatsuba_nogil import multiply

MASK32 = (1 << 32) - 1

def to_limbs(x):
    if x == 0:
        return [0]
    limbs = []
    while x:
        limbs.append(x & MASK32)
        x >>= 32
    return limbs

def fmt_hex(x, max_chars=200):
    h = hex(x)[2:]
    if len(h) > max_chars:
        return "0x..." + h[-max_chars:]
    return "0x" + h

def compare(a, b, bits, show_limit=10):
    r1 = multiply(a, b)
    r2 = a * b
    ok = (r1 == r2)
    print(f"\n{bits} bits test — OK = {ok}")
    if ok:
        return True
    # else print diagnostics
    print(" builtin (python) len(bits) =", r2.bit_length())
    print(" kar  (c extension) len(bits) =", r1.bit_length())
    print(" builtin (hex, tail) :", fmt_hex(r2))
    print(" kar     (hex, tail) :", fmt_hex(r1))
    la = to_limbs(r1)
    lb = to_limbs(r2)
    print(" limbs builtin length:", len(lb))
    print(" limbs kar      length:", len(la))
    # find first differing limb
    L = max(len(la), len(lb))
    for i in range(L):
        va = la[i] if i < len(la) else 0
        vb = lb[i] if i < len(lb) else 0
        if va != vb:
            print(f"First differing limb index (0 = least significant): {i}")
            start = max(0, i - 4)
            end = min(L, i + 5)
            print(" index | kar limb (hex) | builtin limb (hex)")
            for j in range(start, end):
                ka = la[j] if j < len(la) else 0
                kb = lb[j] if j < len(lb) else 0
                mark = "<--" if j == i else ""
                print(f"{j:5d} | {hex(ka):18} | {hex(kb):18} {mark}")
            break
    else:
        # no differing limb but numbers differ? print lengths and low/high words
        print("No limb-level difference found (weird).")
        return False
    return False

def run_one(bits, tries=3):
    for t in range(tries):
        a = random.getrandbits(bits)
        b = random.getrandbits(bits)
        print(f"\nRunning trial {t+1}/{tries} for {bits} bits...")
        ok = compare(a, b, bits)
        if not ok:
            return False
    return True

def main():
    random.seed(0xC0FFEE)
    sizes = [16384, 65536]   # failing sizes you reported
    all_ok = True
    for s in sizes:
        ok = run_one(s, tries=3)
        all_ok = all_ok and ok
    if all_ok:
        print("\nAll diagnostic checks PASSED.")
    else:
        print("\nOne or more diagnostic checks FAILED. Please copy the output and paste here.")

if __name__ == "__main__":
    main()
