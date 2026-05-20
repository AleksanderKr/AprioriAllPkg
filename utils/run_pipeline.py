import argparse
import subprocess
import sys
import os
import time
import csv
import psutil


def get_process_tree_metrics(parent_pid, process_cache):
    total_rss = 0
    current_cpu_total = 0.0

    try:
        parent = psutil.Process(parent_pid)
        all_procs = [parent] + parent.children(recursive=True)

        for p in all_procs:
            if p.pid not in process_cache:
                process_cache[p.pid] = p
                p.cpu_percent(interval=None)

            cached_p = process_cache[p.pid]
            try:
                if cached_p.is_running():
                    total_rss += cached_p.memory_info().rss
                    current_cpu_total += cached_p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return total_rss, current_cpu_total


def count_input_sequences(file_path):
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".csv"):
                reader = csv.reader(f)
                next(reader, None)
                sids = set(row[0] for row in reader if row)
                return len(sids)
            else:
                return sum(1 for line in f if line.strip() and not line.startswith("#"))
    except Exception:
        return 0


def count_output_patterns(out_dir, algo):
    csv_path = os.path.join(out_dir, "frequent_sequences.csv")
    if algo == "apriori":
        csv_path = os.path.join(out_dir, "frequent_itemsets.csv")

    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except Exception:
        return 0


def run_and_profile(cmd):
    print(">", " ".join(cmd))
    start_time = time.perf_counter()

    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
        print(f"Error: Script or executable not found for command: {' '.join(cmd)}")
        return {"duration_sec": 0, "max_memory_mb": 0, "avg_cpu_percent": 0}

    max_memory_bytes = 0
    cpu_samples = []
    process_cache = {}

    try:
        while proc.poll() is None:
            current_rss, current_cpu = get_process_tree_metrics(proc.pid, process_cache)

            if current_rss > max_memory_bytes:
                max_memory_bytes = current_rss
            if current_cpu > 0.0:
                cpu_samples.append(current_cpu)

            time.sleep(0.05)
    except Exception as e:
        proc.terminate()
        proc.wait()
        raise e

    exit_code = proc.wait()
    end_time = time.perf_counter()

    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, cmd)

    duration = end_time - start_time
    max_mem_mb = max_memory_bytes / (1024 * 1024)

    logical_cores = psutil.cpu_count(logical=True) or 1
    avg_cpu = (sum(cpu_samples) / len(cpu_samples) / logical_cores) if cpu_samples else 0.0

    print(f"\n[METRICS] Time: {duration:.4f}s | Peak RAM: {max_mem_mb:.2f} MB | Avg CPU: {avg_cpu:.1f}%")

    return {
        "duration_sec": duration,
        "max_memory_mb": max_mem_mb,
        "avg_cpu_percent": avg_cpu
    }


def save_metrics(out_dir, algo, input_file, min_sup, metrics):
    os.makedirs(out_dir, exist_ok=True)
    benchmarks_dir = os.path.join(out_dir, "benchmarks")
    os.makedirs(benchmarks_dir, exist_ok=True)

    input_size = count_input_sequences(input_file)
    pattern_count = count_output_patterns(out_dir, algo)

    summary_data = {
        "algo": algo,
        "input": os.path.basename(input_file),
        "input_sequences": input_size,
        "min_sup_count": min_sup,
        "duration_sec": round(metrics["duration_sec"], 4),
        "max_memory_mb": round(metrics["max_memory_mb"], 2),
        "avg_cpu_percent": round(metrics["avg_cpu_percent"], 1),
        "pattern_count": pattern_count
    }

    csv_path = os.path.join(benchmarks_dir, "benchmark_results.csv")
    file_exists = os.path.exists(csv_path)

    headers = list(summary_data.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_data)
    print(f"[METRICS] Results appended to: {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["apriori", "apriori_all", "apriori_all_parallel", "prefixspan", "spade", "gsp"],
                    required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--min-sup-count", type=int, default=2)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--spmf", action="store_true")

    args = ap.parse_args()
    py = sys.executable

    target_input = args.input

    if args.spmf and (args.algo in ["apriori_all", "apriori_all_parallel"]):
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        target_input = os.path.join("data", f"seq_{base_name}.csv")
        print(f"--- SPMF Conversion for Sequences: {args.input} -> {target_input} ---")

        converter_path = os.path.join("utils", "convert_spmf_to_seq.py")
        subprocess.check_call([py, converter_path, "--input", args.input, "--out", target_input])

    elif args.spmf and args.algo == "apriori":
        print(f"--- Info: Apriori handles SPMF format directly. Skipping conversion. ---")

    print(f"--- Running algorithm: {args.algo} ---")

    algo_scripts = {
        "apriori": os.path.join("src", "apriori.py"),
        "apriori_all": os.path.join("src", "apriori_all.py"),
        "apriori_all_parallel": os.path.join("src", "apriori_all_parallel.py"),
        "prefixspan": os.path.join("utils", "prefixspan_wrapper.py"),
        "spade": os.path.join("utils", "spade_wrapper.py"),
        "gsp": os.path.join("utils", "gsp_wrapper.py")
    }

    script = algo_scripts[args.algo]
    input_flag = "--transactions" if args.algo == "apriori" else "--sequences"

    cmd = [
        py, script,
        input_flag, target_input,
        "--out-dir", args.out_dir,
        "--min-sup-count", str(args.min_sup_count)
    ]

    if args.mapping and os.path.exists(args.mapping):
        cmd.extend(["--mapping", args.mapping])

    metrics = run_and_profile(cmd)
    save_metrics(args.out_dir, args.algo, args.input, args.min_sup_count, metrics)
    print(f"\nOK: Pipeline finished. Results in '{args.out_dir}/'")


if __name__ == "__main__":
    main()

r"""
python utils/run_pipeline.py --algo apriori_all --input data/seq_sign.csv --min-sup-count 200
python utils/run_pipeline.py --algo apriori_all_parallel --input data/seq_sign.csv --min-sup-count 200
"""