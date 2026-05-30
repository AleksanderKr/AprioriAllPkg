import subprocess
import sys
import argparse
import os
import itertools
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--supports", nargs="+", type=int, required=True)
    ap.add_argument("--algos", nargs="+", default=["apriori_all", "apriori_all_parallel", "prefixspan", "spade", "gsp"])
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--clear-results", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    csv_path = os.path.join(args.out_dir, "benchmarks", "benchmark_results.csv")

    if args.clear_results and os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except OSError:
            pass

    print(f"\n=== STARTING BENCHMARK SUITE ===", flush=True)
    print(f"Dataset: {args.input}", flush=True)
    print(f"Algorithms: {args.algos}", flush=True)
    print(f"Support thresholds: {args.supports}\n", flush=True)

    tasks = list(itertools.product(args.supports, args.algos))

    with tqdm(total=len(tasks), desc="Benchmark Progress", unit="run") as pbar:
        for sup, algo in tasks:
            pbar.set_postfix_str(f"Current: {algo} @ sup={sup}")

            cmd = [
                py, os.path.join("utils", "evaluation", "run_pipeline.py"),
                "--algo", algo,
                "--input", args.input,
                "--min-sup-count", str(sup),
                "--out-dir", args.out_dir
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                tqdm.write(f"Error running {algo}: {e}")

            pbar.update(1)

    print("\n=== BENCHMARK SUITE FINISHED ===", flush=True)
    print(f"Results saved to {csv_path}", flush=True)


if __name__ == "__main__":
    main()