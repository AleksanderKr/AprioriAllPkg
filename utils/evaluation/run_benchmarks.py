import subprocess
import sys
import argparse
import os

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

    print(f"\n=== STARTING BENCHMARK SUITE ===")
    print(f"Dataset: {args.input}")
    print(f"Algorithms: {args.algos}")
    print(f"Support thresholds: {args.supports}\n")

    for sup in args.supports:
        for algo in args.algos:
            print(f"\n>>> Running: {algo} | min_sup_count: {sup}")

            target_input = args.input

            if algo in ["prefixspan", "spade", "gsp"]:
                if args.input.endswith(".csv"):
                    base_name = os.path.basename(args.input)
                    if base_name.startswith("seq_"):
                        base_name = base_name[4:]
                    base_name = base_name.replace(".csv", ".txt")
                    target_input = os.path.join("data", "raw", base_name)
                else:
                    target_input = args.input

            cmd = [
                py, os.path.join("utils", "run_pipeline.py"),
                "--algo", algo,
                "--input", target_input,
                "--min-sup-count", str(sup),
                "--out-dir", args.out_dir
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                pass

    print("\n=== BENCHMARK SUITE FINISHED ===")
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()