# run_bench_cmul.py
import timeit
import inspect
import decimal_computer._c_mul as cm

def find_candidate_callable(mod):
    # return first public callable that looks like an elementwise multiply
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if callable(obj):
            # skip classes, keep functions
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                return name, obj
    return None, None

name, fn = find_candidate_callable(cm)
print("Module:", cm)
print("Found candidate callable:", name)
if name is None:
    print("No public callable found in module. Available names:\n", dir(cm))
    raise SystemExit(1)

# Prepare test data (1e6 ints) — adjust size if memory/time is an issue
N = 1_000_000
data1 = list(range(N))
data2 = list(range(N))

# Basic sanity run once to ensure it works
print("Running a quick sanity call to ensure signature matches...")
try:
    res = fn(data1[:1000], data2[:1000])
    print("Sanity call OK. Returned type:", type(res))
except Exception as e:
    print("Sanity call failed with exception:", e)
    print("You might need to adapt input types. Available signature:", inspect.signature(fn))
    raise SystemExit(1)

# Timeit: run a moderate number of repeats
timer = timeit.Timer(lambda: fn(data1, data2))
num, total = timer.autorange()
print(f"Autorange decided: number={num}, total_time={total:.4f}s")
# Run a proper timing with fewer repeats if autorange yields huge number
repeats = min(max(3, int(num//2)), 10)
t = timer.timeit(number=repeats)
print(f"Timed {repeats} runs -> {t:.6f}s total, avg {t/repeats:.6f}s per run")
