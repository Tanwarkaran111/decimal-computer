# phase5/__main__.py
"""
Entry point for Phase 5.
Running `python -m phase5` will execute NTT multiplication self-test.
"""

from . import ntt_multiply

def main():
    print("=== Phase 5: NTT (Number Theoretic Transform) Multiplication ===")
    ntt_multiply._self_test()

if __name__ == "__main__":
    main()
