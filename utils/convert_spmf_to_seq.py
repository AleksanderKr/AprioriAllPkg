import argparse
import csv
import os
from collections import defaultdict

DATA_DIR = "data"


def convert_csv_to_spmf(input_path, out_path, limit=None):
    sequences = defaultdict(lambda: defaultdict(list))

    with open(input_path, "r", encoding="utf-8") as f_in:
        reader = csv.reader(f_in)
        next(reader)

        for row in reader:
            if not row:
                continue
            seq_id, pos, item = row[0], int(row[1]), row[2]
            sequences[seq_id][pos].append(item)

    sid = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for seq_id in sorted(sequences.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
            sid += 1
            if limit is not None and sid > limit:
                sid -= 1
                break

            line_parts = []
            itemsets = sequences[seq_id]

            for pos in sorted(itemsets.keys()):
                items = itemsets[pos]
                line_parts.extend(items)
                line_parts.append("-1")

            line_parts.append("-2")
            f_out.write(" ".join(line_parts) + "\n")
    print(f"OK: Saved to {out_path} (sequences={sid})")


def convert_spmf_to_csv(input_path, out_path, limit=None):
    sid = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["sequence_id", "pos", "item"])

        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            sid += 1
            if limit is not None and sid > limit:
                sid -= 1
                break

            tokens = line.split()
            pos_idx = 1

            for token in tokens:
                if token == "-1":
                    pos_idx += 1
                elif token == "-2":
                    break
                else:
                    writer.writerow([f"s{sid}", pos_idx, token])
    print(f"OK: Saved to {out_path} (sequences={sid})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction", choices=["csv2spmf", "spmf2csv"], default="csv2spmf")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.direction == "csv2spmf":
        convert_csv_to_spmf(args.input, args.out, args.limit)
    else:
        convert_spmf_to_csv(args.input, args.out, args.limit)


if __name__ == "__main__":
    main()