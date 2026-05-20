import subprocess
import sys
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/seq_sign.csv", help="Path to the input file")
    ap.add_argument("--supports", nargs="+", type=int, default=[350, 300, 250, 200, 150],
                    help="List of min_sup_count thresholds separated by spaces")
    ap.add_argument("--algos", nargs="+", default=["apriori_all", "apriori_all_parallel", "prefixspan", "spade", "gsp"],
                    help="Algorithms to include in the benchmark")
    ap.add_argument("--out-dir", default="output", help="Output directory for results")
    args = ap.parse_args()

    py = sys.executable

    csv_path = os.path.join(args.out_dir, "benchmarks", "benchmark_results.csv")
    if os.path.exists(csv_path):
        print(f"[CLEANUP] Removing old results from: {csv_path}")
        try:
            os.remove(csv_path)
        except OSError as e:
            print(f"[WARNING] Could not remove old CSV file ({e}). Results might be skewed!")

    print(f"\n=== STARTING BENCHMARK SUITE ===")
    print(f"Dataset: {args.input}")
    print(f"Algorithms: {args.algos}")
    print(f"Support thresholds: {args.supports}\n")

    for sup in args.supports:
        for algo in args.algos:
            print(f"\n>>> Running: {algo} | min_sup_count: {sup}")
            cmd = [
                py, os.path.join("utils", "run_pipeline.py"),
                "--algo", algo,
                "--input", args.input,
                "--min-sup-count", str(sup),
                "--out-dir", args.out_dir
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"[ERROR] Execution failed for {algo} with support {sup}. Skipping.")

    print("\n=== BENCHMARK SUITE FINISHED ===")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()