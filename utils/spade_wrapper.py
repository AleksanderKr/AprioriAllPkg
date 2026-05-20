import argparse
import csv
import os
import subprocess
import sys

def convert_csv_to_spmf(csv_path, spmf_path):
    sequences = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row: continue
            sid = row[0]
            eid = int(row[1]) if len(row) > 1 else 0
            items = [x.strip() for x in row[2:] if x.strip()] if len(row) > 2 else []
            if sid not in sequences:
                sequences[sid] = {}
            if eid not in sequences[sid]:
                sequences[sid][eid] = []
            sequences[sid][eid].extend(items)

    with open(spmf_path, "w", encoding="utf-8") as f:
        for sid in sorted(sequences.keys()):
            line = []
            for eid in sorted(sequences[sid].keys()):
                if sequences[sid][eid]:
                    line.extend(sequences[sid][eid])
                    line.append("-1")
            line.append("-2")
            f.write(" ".join(line) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-sup-count", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    spmf_jar = "spmf.jar"
    if not os.path.exists(spmf_jar):
        print(f"Error: spmf.jar not found. Download from http://www.philippe-fournier-viger.com/spmf/ and place in {os.getcwd()}")
        sys.exit(1)

    temp_input = os.path.join(args.out_dir, "temp_spmf_in_spade.txt")
    temp_output = os.path.join(args.out_dir, "temp_spmf_out_spade.txt")

    convert_csv_to_spmf(args.sequences, temp_input)

    total_sequences = 0
    if os.path.exists(temp_input):
        with open(temp_input, "r", encoding="utf-8") as count_f:
            total_sequences = sum(1 for line in count_f if line.strip())

    if total_sequences == 0:
        print("Error: Converted SPMF input file is empty.")
        sys.exit(1)

    min_sup_relative = args.min_sup_count / total_sequences

    cmd = ["java", "-jar", spmf_jar, "run", "SPADE", temp_input, temp_output, str(min_sup_relative)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Error: SPMF Execution failed. Make sure Java is installed and in PATH.")
        sys.exit(1)

    out_file = os.path.join(args.out_dir, "frequent_sequences.csv")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Support", "Sequence"])

        if os.path.exists(temp_output):
            with open(temp_output, "r", encoding="utf-8") as out_f:
                for line in out_f:
                    parts = line.strip().split(" #SUP: ")
                    if len(parts) == 2:
                        seq_raw = parts[0].split("-1")
                        count = parts[1]

                        formatted_seq = []
                        for ev in seq_raw:
                            ev_items = ev.strip().split()
                            if ev_items:
                                formatted_seq.append(f"{{{','.join(ev_items)}}}")

                        writer.writerow([count, " -> ".join(formatted_seq)])

if __name__ == "__main__":
    main()