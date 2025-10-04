# run.py - convenience CLI for demo, benchmarks, compare, and plotting
import subprocess
import argparse
import sys

SCRIPTS = {
    "demo": ["python", "demo.py"],
    "benchmarks": ["python", "benchmarks.py"],
    "plot_benchmarks": ["python", "plot_benchmarks.py"],
    "compare": ["python", "benchmarks_compare.py"],
    "compare_plot": ["python", "plot_compare.py"],
    "compare_time": ["python", "benchmarks_compare_time.py"],
    "plot_compare_time": ["python", "plot_compare_time.py"],
    "compute_cutoff": ["python", "compute_best_cutoff.py"]
}

def run_cmd(cmd):
    try:
        rc = subprocess.run(cmd, check=True)
        return rc.returncode
    except subprocess.CalledProcessError as e:
        print("Command failed:", e)
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="Run Decimal_Computer prototype tasks")
    parser.add_argument("action", choices=list(SCRIPTS.keys()) + ["all"], help="action to run")
    args = parser.parse_args()

    if args.action == "all":
        # recommended sequence
        sequence = ["demo","benchmarks","plot_benchmarks","compare","compare_plot","compare_time","plot_compare_time","compute_cutoff"]
    else:
        sequence = [args.action]

    for act in sequence:
        print("\n=== RUN:", act, "===")
        cmd = SCRIPTS.get(act)
        if not cmd:
            print("Unknown action", act)
            continue
        rc = run_cmd(cmd)
        if rc != 0:
            print("Stopped due to error.")
            sys.exit(rc)

if __name__ == "__main__":
    main()
