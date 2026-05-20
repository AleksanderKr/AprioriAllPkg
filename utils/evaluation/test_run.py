import subprocess
import sys
import os


def run_cmd(cmd):
    print(f"\n[TEST EXECUTION] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[TEST ERROR] Command failed: {e}")


def main():
    py = sys.executable
    benchmark_script = os.path.join("utils", "evaluation", "run_benchmarks.py")
    results_csv = os.path.join("output", "benchmarks", "benchmark_results.csv")

    if os.path.exists(results_csv):
        print(f"[TEST CLEANUP] Removing old results from: {results_csv}")
        try:
            os.remove(results_csv)
        except OSError:
            pass

    cmd_native = [
        py, benchmark_script,
        "--input", "data/seq_mini_sign.txt",
        "--algos", "apriori_all", "apriori_all_parallel",
        "--supports", "350", "250", "150", "50", "20"
    ]
    run_cmd(cmd_native)

    if os.path.exists(benchmark_script):
        with open(benchmark_script, "r", encoding="utf-8") as f:
            code = f.read()

        modified_code = code.replace("os.remove(csv_path)", "# os.remove(csv_path)")

        with open(benchmark_script, "w", encoding="utf-8") as f:
            f.write(modified_code)

    cmd_spmf = [
        py, benchmark_script,
        "--input", "data/raw/mini_sign.txt",
        "--algos", "prefixspan", "spade", "gsp",
        "--supports", "350", "250", "150", "50", "20"
    ]
    run_cmd(cmd_spmf)

    if os.path.exists(benchmark_script):
        with open(benchmark_script, "w", encoding="utf-8") as f:
            f.write(code)

    print(f"\n[TEST FINISHED] All benchmarks executed successfully. Results combined in: {results_csv}")


if __name__ == "__main__":
    main()