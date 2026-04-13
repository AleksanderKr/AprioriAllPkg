import argparse
import subprocess
import sys
import os


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["apriori", "apriori_all"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--min-sup-count", type=int, default=2)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--spmf", action="store_true")

    args = ap.parse_args()
    py = sys.executable

    target_input = args.input

    if args.spmf and args.algo == "apriori_all":
        # Wyciąga nazwę bez rozszerzenia i dodaje przedrostek seq_
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        target_input = os.path.join("data", f"seq_{base_name}.csv")

        print(f"--- SPMF Conversion for Sequences: {args.input} -> {target_input} ---")
        run([py, "convert_spmf_to_seq.py", "--input", args.input, "--out", target_input])
    elif args.spmf and args.algo == "apriori":
        print(f"--- Info: Apriori handles SPMF format directly. Skipping conversion. ---")

    print(f"--- Running algorithm: {args.algo} ---")

    script = "apriori.py" if args.algo == "apriori" else "apriori_all.py"
    input_flag = "--transactions" if args.algo == "apriori" else "--sequences"

    cmd = [
        py, script,
        input_flag, target_input,
        "--out-dir", args.out_dir,
        "--min-sup-count", str(args.min_sup_count)
    ]

    if args.mapping and os.path.exists(args.mapping):
        cmd.extend(["--mapping", args.mapping])

    run(cmd)
    print(f"\nOK: Pipeline finished. Results in '{args.out_dir}/'")


if __name__ == "__main__":
    main()

r"""
Przykładowe komendy:
python run_pipeline.py --algo apriori --input data/raw/mushrooms.csv --min-sup-count 300
python run_pipeline.py --algo apriori_all --input data/raw/mini_sign.csv --spmf --min-sup-count 10
python run_pipeline.py --algo apriori_all --input data/sequences_test.csv --min-sup-count 2
"""