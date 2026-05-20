import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-file", default="out/benchmark_results.csv")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: {args.csv_file} not found. Run run_experiments.py first.")
        return

    try:
        df = pd.read_csv(args.csv_file, on_bad_lines="skip")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    df = df.sort_values(by="min_sup_count", ascending=False)
    datasets = df["input"].unique()

    for dataset in datasets:
        df_subset = df[df["input"] == dataset]
        clean_ds_name = os.path.basename(dataset).replace(".csv", "")

        # 1. Execution Time Plot
        plt.figure(figsize=(8, 6))
        for algo in df_subset["algo"].unique():
            df_algo = df_subset[df_subset["algo"] == algo]
            plt.plot(df_algo["min_sup_count"], df_algo["duration_sec"], marker='o', label=algo, linewidth=2)

        plt.gca().invert_xaxis()
        plt.title(f"Scalability: Execution Time ({clean_ds_name})", fontsize=14)
        plt.xlabel("Minimum Support Count", fontsize=12)
        plt.ylabel("Execution Time (seconds)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Algorithm")
        plt.tight_layout()
        time_chart_path = os.path.join(args.out_dir, f"time_chart_{clean_ds_name}.png")
        plt.savefig(time_chart_path, dpi=300)
        plt.close()

        # 2. Peak Memory Plot
        plt.figure(figsize=(8, 6))
        for algo in df_subset["algo"].unique():
            df_algo = df_subset[df_subset["algo"] == algo]
            plt.plot(df_algo["min_sup_count"], df_algo["max_memory_mb"], marker='s', label=algo, linewidth=2)

        plt.gca().invert_xaxis()
        plt.title(f"Scalability: Memory Consumption ({clean_ds_name})", fontsize=14)
        plt.xlabel("Minimum Support Count", fontsize=12)
        plt.ylabel("Peak Memory (MB)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Algorithm")
        plt.tight_layout()
        mem_chart_path = os.path.join(args.out_dir, f"memory_chart_{clean_ds_name}.png")
        plt.savefig(mem_chart_path, dpi=300)
        plt.close()

        print(f"Charts successfully saved for {clean_ds_name} in {args.out_dir}/")


if __name__ == "__main__":
    main()