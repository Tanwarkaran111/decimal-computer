# bench_tune.py
# Usage: python bench_tune.py
# This script edits src/karatsuba.pyx to set the basecase cutoff, rebuilds, and benchmarks.
# It runs the timing/import in a separate Python process to avoid Windows locking issues.

import subprocess, time, os, statistics, shutil, sys, json
from time import sleep

ROOT = os.path.abspath(os.path.dirname(__file__))
PKG_DIR = os.path.join(ROOT, "phase8_karatsuba_nogil")
PYX = os.path.join(PKG_DIR, "src", "karatsuba.pyx")
BACKUP = PYX + ".bak"

# sizes to benchmark (bits) and reps per step
SIZES = [16384, 32768, 65536, 131072]  # tuneable; increase if you want bigger
REPEATS = 5
WARMUP = 1

# cutoffs to try (in 32-bit limbs)
CUTOFFS = [8, 16, 32, 48, 64, 96, 128, 192, 256]

def run_cmd(cmd, cwd=PKG_DIR, timeout=None):
    print(">", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return p.returncode, p.stdout

def modify_cutoff(newcut):
    # backup original if not yet backed up
    if not os.path.exists(BACKUP):
        shutil.copyfile(PYX, BACKUP)
    txt = open(BACKUP, "r", encoding="utf8").read()
    import re
    if re.search(r"if\s+n\s+<=\s+\d+\s*:", txt):
        txt2 = re.sub(r"if\s+n\s+<=\s+\d+\s*:", f"if n <= {newcut}:", txt)
    else:
        txt2 = txt.replace("if n <= 64:", f"if n <= {newcut}:")
    open(PYX, "w", encoding="utf8").write(txt2)

def restore_original():
    if os.path.exists(BACKUP):
        shutil.copyfile(BACKUP, PYX)
        os.remove(BACKUP)

# Worker mode: when invoked with --worker the script runs the timings (in a fresh process)
def worker_main():
    import random, time, statistics
    # constants should match parent
    SIZES_LOCAL = SIZES
    REPEATS_LOCAL = REPEATS
    WARMUP_LOCAL = WARMUP

    # import after build — fresh process so import lock not a problem
    try:
        from phase8_karatsuba_nogil import multiply
    except Exception as e:
        # return JSON error
        print(json.dumps({"error": f"import_failed: {e}"}))
        return 1

    def bench_once(bits):
        a = random.getrandbits(bits); b = random.getrandbits(bits)
        for _ in range(WARMUP_LOCAL):
            multiply(a, b)
            _ = a * b
        kar = []
        builtin = []
        for _ in range(REPEATS_LOCAL):
            t0 = time.perf_counter(); multiply(a, b); kar.append(time.perf_counter()-t0)
            t0 = time.perf_counter(); _ = a * b; builtin.append(time.perf_counter()-t0)
        return statistics.mean(kar), statistics.mean(builtin)

    results = []
    for bits in SIZES_LOCAL:
        try:
            km, bm = bench_once(bits)
        except Exception as e:
            print(json.dumps({"error": f"bench_failed: {e}"}))
            return 1
        results.append({"bits": bits, "kar": km, "builtin": bm, "ratio": (km/bm if bm>0 else None)})
    print(json.dumps({"results": results}))
    return 0

def main():
    if "--worker" in sys.argv:
        # named entry: run worker_main and exit
        rc = worker_main()
        sys.exit(rc)

    results = {}
    for cut in CUTOFFS:
        print("\n=== Testing cutoff", cut, "===")
        modify_cutoff(cut)

        # rebuild
        rc, out = run_cmd([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=PKG_DIR)
        print(out)
        if rc != 0:
            print("Build failed for cutoff", cut, " — skipping")
            continue

        # Now run timings in a separate Python process (worker)
        try:
            # run worker; current script file used as worker; ensure cwd is package dir so imports work
            worker_cmd = [sys.executable, os.path.abspath(__file__), "--worker"]
            rc2 = subprocess.run(worker_cmd, cwd=PKG_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
            out2 = rc2.stdout
        except subprocess.TimeoutExpired:
            print("Worker timed out — skipping cutoff", cut)
            continue

        # parse worker output JSON
        try:
            payload = json.loads(out2.strip().splitlines()[-1])
            if "error" in payload:
                print("Worker reported error:", payload["error"])
                continue
            size_results = []
            for rec in payload["results"]:
                bits = rec["bits"]
                km = rec["kar"]
                bm = rec["builtin"]
                ratio = rec["ratio"] if rec["ratio"] is not None else float("inf")
                size_results.append((bits, km, bm, ratio))
                print(f"{bits} bits: kar={km:.6f}s  builtin={bm:.6f}s  ratio={ratio:.3f}")
            results[cut] = size_results
        except Exception as e:
            print("Failed to parse worker output. raw output:")
            print(out2)
            print("parse error:", e)
            continue

    # restore file
    restore_original()

    # summarize: pick cutoff with lowest mean ratio across sizes
    best = None
    bestscore = float("inf")
    for cut, vals in results.items():
        ratios = [r for (_,_,_,r) in vals]
        score = sum(ratios) / len(ratios)
        print("cutoff", cut, "avg ratio", score)
        if score < bestscore:
            bestscore = score
            best = cut
    print("\nBEST cutoff:", best, "avg ratio:", bestscore)

if __name__ == "__main__":
    main()
