import argparse
import csv
import os
from collections import defaultdict

DATA_DIR = "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "sequences.txt"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sequences = defaultdict(lambda: defaultdict(list))

    with open(args.input, "r", encoding="utf-8") as f_in:
        reader = csv.reader(f_in)
        next(reader)

        for row in reader:
            if not row:
                continue
            seq_id, pos, item = row[0], int(row[1]), row[2]
            sequences[seq_id][pos].append(item)

    sid = 0
    with open(args.out, "w", encoding="utf-8") as f_out:
        for seq_id in sorted(sequences.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
            sid += 1
            if args.limit is not None and sid > args.limit:
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

    print(f"OK: Saved to {args.out} (sequences={sid})")


if __name__ == "__main__":
    main()