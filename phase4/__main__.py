# phase4/__main__.py

"""
Entry point for Phase 4.
Running `python -m phase4` will execute fft_multiply self-test.
"""

from . import fft_multiply


def main():
    print("=== Phase 4: FFT Multiplication ===")
    fft_multiply._self_test()


if __name__ == "__main__":
    main()
