import csv
import os
import argparse
from collections import defaultdict
from itertools import combinations

DATA_DIR = "data"
OUT_DIR = "output"

POS_KEYS = ("pos", "position", "event_idx", "idx", "order", "time", "t")


def item_key(x: str):
    if x.startswith("i") and x[1:].isdigit():
        return int(x[1:])
    return x


def read_mapping(path: str) -> dict:
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mapping[row["item_id"]] = row["item_name"]
    return mapping


def read_sequences_long_itemsets(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = [c.strip() for c in (r.fieldnames or [])]

        if "sequence_id" not in fieldnames:
            raise RuntimeError("Missing sequence_id column in data/sequences.csv")

        pos_key = None
        for k in POS_KEYS:
            if k in fieldnames:
                pos_key = k
                break
        if pos_key is None:
            raise RuntimeError("Missing position column (e.g., pos / order / time) in data/sequences.csv")

        if "item" not in fieldnames:
            raise RuntimeError("Missing item column in data/sequences.csv")

        by_sid_pos = defaultdict(lambda: defaultdict(set))
        for row in r:
            sid = row["sequence_id"]
            pos = int(row[pos_key])
            it = row["item"].strip()
            if it:
                by_sid_pos[sid][pos].add(it)

    sequences = []
    for sid, pos_map in by_sid_pos.items():
        events = []
        for pos in sorted(pos_map.keys()):
            items = pos_map[pos]
            if items:
                events.append(frozenset(items))
        if events:
            sequences.append(events)
    return sequences


def mine_frequent_itemsets(sequences, min_sup_count):
    item_counts = defaultdict(int)
    for seq in sequences:
        seen = set()
        for ev in seq:
            seen |= ev
        for it in seen:
            item_counts[frozenset([it])] += 1

    freq_itemsets = []
    current_l = {it: c for it, c in item_counts.items() if c >= min_sup_count}
    freq_itemsets.extend(current_l.keys())

    k = 2
    while current_l:
        prev_items = list(current_l.keys())
        candidates = set()
        for i in range(len(prev_items)):
            for j in range(i + 1, len(prev_items)):
                cand = prev_items[i] | prev_items[j]
                if len(cand) == k:
                    valid = True
                    for sub in combinations(cand, k - 1):
                        if frozenset(sub) not in current_l:
                            valid = False
                            break
                    if valid:
                        candidates.add(cand)

        current_counts = defaultdict(int)
        for seq in sequences:
            for cand in candidates:
                if any(cand.issubset(ev) for ev in seq):
                    current_counts[cand] += 1

        current_l = {cand: c for cand, c in current_counts.items() if c >= min_sup_count}
        freq_itemsets.extend(current_l.keys())
        k += 1

    return freq_itemsets


def transformed_subsequence_check(trans_seq, cand_tuple):
    curr = 0
    for cand_id in cand_tuple:
        found = False
        while curr < len(trans_seq):
            if cand_id in trans_seq[curr]:
                found = True
                curr += 1
                break
            curr += 1
        if not found:
            return False
    return True


def count_sequence_support(candidates, transformed_db):
    counts = defaultdict(int)
    for cand in candidates:
        for trans_seq in transformed_db:
            if transformed_subsequence_check(trans_seq, cand):
                counts[cand] += 1
    return counts


def apriori_generate_sequences(prev_ls, k):
    prev_list = list(prev_ls)
    candidates = set()
    for i in range(len(prev_list)):
        for j in range(len(prev_list)):
            a = prev_list[i]
            b = prev_list[j]
            if a[:-1] == b[:-1]:
                cand = a + (b[-1],)
                valid = True
                for drop in range(k):
                    sub = cand[:drop] + cand[drop + 1:]
                    if sub not in prev_ls:
                        valid = False
                        break
                if valid:
                    candidates.add(cand)
    return candidates


def filter_maximal_sequences(seq_counts: dict) -> dict:
    seqs = sorted(seq_counts.keys(), key=lambda s: (len(s), sum(len(ev) for ev in s)), reverse=True)

    maximal = {}
    kept = []

    for s in seqs:
        is_sub = False
        for t in kept:
            i = 0
            for ev_s in s:
                found = False
                while i < len(t):
                    if ev_s.issubset(t[i]):
                        found = True
                        i += 1
                        break
                    i += 1
                if not found:
                    break
            else:
                is_sub = True
                break

        if not is_sub:
            maximal[s] = seq_counts[s]
            kept.append(s)

    return maximal


def seq_to_string(seq) -> str:
    parts = []
    for ev in seq:
        inner = ",".join(sorted(ev, key=item_key))
        parts.append("{" + inner + "}")
    return "<" + ",".join(parts) + ">"


def mapped_seq_to_string(seq_tuple) -> str:
    return "<" + ",".join(str(i) for i in seq_tuple) + ">"


def itemset_to_string(ev) -> str:
    return "{" + ",".join(sorted(ev, key=item_key)) + "}"


def event_tuple(ev) -> tuple:
    return tuple(sorted(ev, key=item_key))


def seq_sort_key(s) -> tuple:
    return (len(s), [event_tuple(e) for e in s])


def write_debug_file(title: str, data, debug: bool = False):
    if not debug:
        return

    total = len(data)
    print(f"\n=== {title} (Total: {total}) ===")

    if total == 0:
        print("(empty)")
        return

    if "Step 1" in title:
        for seq in data:
            print(seq_to_string(seq))

    elif "Step 2" in title:
        sorted_data = sorted(data, key=lambda x: (len(x), sorted(list(x), key=item_key)))
        for itemset in sorted_data:
            print(itemset_to_string(itemset))

    elif "Mapping" in title:
        for itemset, int_id in data.items():
            print(f"{itemset_to_string(itemset)} -> {int_id}")

    elif "Mapped Database" in title:
        for seq in data:
            parts = []
            for ev in seq:
                inner = ",".join(str(i) for i in sorted(ev))
                parts.append("{" + inner + "}")
            print("<" + ",".join(parts) + ">")

    elif "Patterns" in title:
        for seq in sorted(data.keys(), key=seq_sort_key):
            print(f"{seq_to_string(seq)} (sup: {data[seq]})")

    else:
        # Obsługa Step 4: dict (LS) lub set (CS)
        items = data.keys() if isinstance(data, dict) else data
        for t in sorted(items):
            count_str = f" (sup: {data[t]})" if isinstance(data, dict) else ""
            print(mapped_seq_to_string(t) + count_str)


def write_sequences(path: str, seq_counts: dict, n_sequences: int):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "support_count", "support"])
        for s in sorted(seq_counts.keys(), key=seq_sort_key):
            sc = seq_counts[s]
            sup = sc / n_sequences
            w.writerow([seq_to_string(s), sc, f"{sup:.6f}"])


def write_sequences_human(path: str, seq_counts: dict, n_sequences: int, mapping: dict):
    def human_event(ev):
        return frozenset(mapping.get(it, it) for it in ev)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence_human", "support_count", "support"])
        for s in sorted(seq_counts.keys(), key=seq_sort_key):
            sc = seq_counts[s]
            sup = sc / n_sequences
            human_seq = tuple(human_event(ev) for ev in s)
            w.writerow([seq_to_string(human_seq), sc, f"{sup:.6f}"])


def apriori_all(sequences, min_sup_count: int, debug: bool = False) -> tuple:
    write_debug_file("Step 1: Sorted and Grouped Sequences Database (Ds)", sequences, debug)

    freq_itemsets = mine_frequent_itemsets(sequences, min_sup_count)
    write_debug_file("Step 2: Discovered Frequent Itemsets (L)", freq_itemsets, debug)

    sorted_freq_itemsets = sorted(freq_itemsets, key=lambda x: (len(x), sorted(list(x), key=item_key)))
    mapping_to_id = {itemset: i + 1 for i, itemset in enumerate(sorted_freq_itemsets)}
    mapping_from_id = {i + 1: itemset for i, itemset in enumerate(sorted_freq_itemsets)}

    write_debug_file("Step 3: Frequent Itemsets Mapping to Integers", mapping_to_id, debug)

    transformed_db = []
    for seq in sequences:
        transformed_seq = []
        for ev in seq:
            ev_mapped_ids = set()
            for itemset, itemset_id in mapping_to_id.items():
                if itemset.issubset(ev):
                    ev_mapped_ids.add(itemset_id)
            if ev_mapped_ids:
                transformed_seq.append(frozenset(ev_mapped_ids))
        if transformed_seq:
            transformed_db.append(transformed_seq)

    write_debug_file("Step 3: Transformed and Mapped Database (Dts)", transformed_db, debug)

    all_mapped_sequences = {}

    ls_k = {(it_id,) for it_id in mapping_from_id.keys()}
    current_ls_counts = count_sequence_support(ls_k, transformed_db)
    current_ls = {cand: c for cand, c in current_ls_counts.items() if c >= min_sup_count}

    write_debug_file("Step 4: Frequent 1-Sequences (LS1)", current_ls, debug)

    all_mapped_sequences.update(current_ls)

    k = 2
    prev_ls = set(current_ls.keys())

    while prev_ls:
        ck = apriori_generate_sequences(prev_ls, k)
        write_debug_file(f"Step 4: Candidate {k}-Sequences (CS{k})", ck, debug)

        if not ck:
            break

        counts = count_sequence_support(ck, transformed_db)
        current_ls = {cand: c for cand, c in counts.items() if c >= min_sup_count}

        write_debug_file(f"Step 4: Frequent {k}-Sequences (LS{k})", current_ls, debug)

        if not current_ls:
            break

        all_mapped_sequences.update(current_ls)
        prev_ls = set(current_ls.keys())
        k += 1

    frequent_sequences_unmapped = {}
    for mapped_tuple, count in all_mapped_sequences.items():
        unmapped_seq = tuple(mapping_from_id[it_id] for it_id in mapped_tuple)
        frequent_sequences_unmapped[unmapped_seq] = count

    write_debug_file("Step 5: All Frequent Sequential Patterns Before Maximization", frequent_sequences_unmapped, debug)

    maximal_sequences = filter_maximal_sequences(frequent_sequences_unmapped)
    write_debug_file("Step 5: Final Maximal Sequential Patterns", maximal_sequences, debug)

    return frequent_sequences_unmapped, maximal_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default=os.path.join(DATA_DIR, "sequences.csv"))
    ap.add_argument("--mapping", default=os.path.join(DATA_DIR, "mapping.csv"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--min-sup-count", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="Print step-by-step execution trace")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sequences = read_sequences_long_itemsets(args.sequences)
    mapping = read_mapping(args.mapping)

    total_seqs = len(sequences)
    sup_percent = (args.min_sup_count / total_seqs) * 100 if total_seqs > 0 else 0

    print("=== RUNNING APRIORI ALL ALGORITHM ===")
    print(f"Total sequences loaded: {total_seqs}")
    print(f"Minimum Support Count threshold: {args.min_sup_count} ({sup_percent:.2f}%)\n")

    all_seq_counts, max_seq_counts = apriori_all(
        sequences,
        min_sup_count=args.min_sup_count,
        debug=args.debug
    )

    out_all = os.path.join(args.out_dir, "frequent_sequences.csv")
    write_sequences(out_all, all_seq_counts, total_seqs)

    out_max = os.path.join(args.out_dir, "maximal_sequences.csv")
    write_sequences(out_max, max_seq_counts, total_seqs)

    if mapping:
        out_human = os.path.join(args.out_dir, "maximal_sequences_human.csv")
        write_sequences_human(out_human, max_seq_counts, total_seqs, mapping)

    print(
        f"\nOK: Execution finished. Discovered {len(all_seq_counts)} ALL frequent patterns and {len(max_seq_counts)} MAXIMAL sequential patterns.")
    print(f"All frequent sequences saved to: {out_all}")
    print(f"Maximal sequences saved to: {out_max}")


if __name__ == "__main__":
    main()