import subprocess
import sys
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/seq_mini_sign.txt")
    ap.add_argument("--supports", nargs="+", type=int, default=[350, 300, 250, 200, 150])
    ap.add_argument("--algos", nargs="+", default=["apriori_all", "apriori_all_parallel", "prefixspan", "spade", "gsp"])
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    py = sys.executable

    csv_path = os.path.join(args.out_dir, "benchmarks", "benchmark_results.csv")
    if os.path.exists(csv_path):
        print(f"[CLEANUP] Removing old results from: {csv_path}")
        try:
            os.remove(csv_path)
        except OSError as e:
            print(f"[WARNING] Could not remove old CSV file ({e}).")

    print(f"\n=== STARTING BENCHMARK SUITE ===")
    print(f"Dataset: {args.input}")
    print(f"Algorithms: {args.algos}")
    print(f"Support thresholds: {args.supports}\n")

    for sup in args.supports:
        for algo in args.algos:
            print(f"\n>>> Running: {algo} | min_sup_count: {sup}")

            target_input = args.input
            if algo in ["prefixspan", "spade", "gsp"]:
                base_name = os.path.basename(args.input)
                if base_name.startswith("seq_"):
                    base_name = base_name[4:]
                target_input = os.path.join("data", "raw", base_name)

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
                print(f"[ERROR] Execution failed for {algo} with support {sup}. Skipping.")

    print("\n=== BENCHMARK SUITE FINISHED ===")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()