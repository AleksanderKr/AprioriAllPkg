import csv
import itertools
import os
import argparse
import io
from collections import defaultdict

DATA_DIR = "data"
OUT_DIR = "out"

def ensure_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)

def read_transactions(path):
    transactions = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return []

        first_line = lines[0].lower()
        if "id" in first_line or "item" in first_line:
            f_mem = io.StringIO("".join(lines))
            r = csv.DictReader(f_mem)
            by_tid = defaultdict(set)
            tid_col = r.fieldnames[0]
            for row in r:
                by_tid[row[tid_col]].add(row["item"])
            return [frozenset(items) for items in by_tid.values()]
        else:
            for line in lines:
                items = [it for it in line.strip().split() if it not in ("-1", "-2")]
                if items:
                    transactions.append(frozenset(items))
            return transactions

def read_mapping(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mapping[row["item_id"]] = row["item_name"]
    return mapping

def support_count(itemset, transactions):
    c = 0
    for t in transactions:
        if itemset.issubset(t):
            c += 1
    return c

def join_step(prev_frequents, k):
    prev_list = sorted(prev_frequents)
    candidates = set()
    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            a = prev_list[i]
            b = prev_list[j]
            if a[:k - 2] == b[:k - 2]:
                cand = tuple(sorted(set(a) | set(b)))
                if len(cand) == k:
                    candidates.add(cand)
            else:
                break
    return candidates

def prune_step(candidates, prev_frequents_set, k):
    pruned = set()
    for cand in candidates:
        ok = True
        for subset in itertools.combinations(cand, k - 1):
            if subset not in prev_frequents_set:
                ok = False
                break
        if ok:
            pruned.add(cand)
    return pruned

def apriori(transactions, min_sup_count):
    item_counts = defaultdict(int)
    for t in transactions:
        for it in t:
            item_counts[(it,)] += 1

    L = {}
    L1 = {k: v for k, v in item_counts.items() if v >= min_sup_count}
    L.update(L1)

    k = 2
    prev = sorted(L1.keys())
    prev_set = set(prev)

    while prev:
        Ck = join_step(prev, k)
        Ck = prune_step(Ck, prev_set, k)

        counts = {}
        for cand in Ck:
            sc = support_count(set(cand), transactions)
            if sc >= min_sup_count:
                counts[cand] = sc

        if not counts:
            break

        L.update(counts)
        prev = sorted(counts.keys())
        prev_set = set(prev)
        k += 1

    return L

def generate_rules(itemsets, total_transactions, min_lift, min_conf):
    rules = []
    for itemset, support_count_i in itemsets.items():
        if len(itemset) < 2:
            continue

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                support_a = itemsets.get(antecedent, 0)
                support_b = itemsets.get(consequent, 0)

                if support_a == 0 or support_b == 0:
                    continue

                supp = support_count_i / total_transactions
                conf = support_count_i / support_a
                lift = (support_count_i * total_transactions) / (support_a * support_b)

                if lift >= min_lift and conf >= min_conf:
                    rules.append({
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": supp,
                        "confidence": conf,
                        "lift": lift
                    })

    rules.sort(key=lambda x: x["lift"], reverse=True)
    return rules

def write_itemsets(path, itemsets, n_transactions):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["itemset", "support_count", "support"])
        for items in sorted(itemsets.keys(), key=lambda x: (len(x), x)):
            sc = itemsets[items]
            sup = sc / n_transactions
            w.writerow([" ".join(items), sc, f"{sup:.6f}"])

def write_itemsets_human(path, itemsets, n_transactions, mapping):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["itemset_human", "support_count", "support"])
        for items in sorted(itemsets.keys(), key=lambda x: (len(x), x)):
            sc = itemsets[items]
            sup = sc / n_transactions
            human = [mapping.get(it, it) for it in items]
            w.writerow([", ".join(human), sc, f"{sup:.6f}"])

def write_rules(path, rules):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["antecedent", "consequent", "support", "confidence", "lift"])
        for r in rules:
            ant = " ".join(r["antecedent"])
            con = " ".join(r["consequent"])
            w.writerow([ant, con, f"{r['support']:.6f}", f"{r['confidence']:.6f}", f"{r['lift']:.6f}"])

def write_rules_human(path, rules, mapping):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["antecedent_human", "consequent_human", "support", "confidence", "lift"])
        for r in rules:
            ant = ", ".join([mapping.get(it, it) for it in r["antecedent"]])
            con = ", ".join([mapping.get(it, it) for it in r["consequent"]])
            w.writerow([ant, con, f"{r['support']:.6f}", f"{r['confidence']:.6f}", f"{r['lift']:.6f}"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default=os.path.join(DATA_DIR, "transactions.csv"))
    ap.add_argument("--mapping", default=os.path.join(DATA_DIR, "mapping.csv"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--min-sup-count", type=int, default=8)
    ap.add_argument("--min-lift", type=float, default=1.0)
    ap.add_argument("--min-conf", type=float, default=0.0)
    args = ap.parse_args()

    ensure_dirs(args.out_dir)

    transactions = read_transactions(args.transactions)
    mapping = read_mapping(args.mapping)

    itemsets = apriori(transactions, min_sup_count=args.min_sup_count)
    rules = generate_rules(itemsets, len(transactions), args.min_lift, args.min_conf)

    out1 = os.path.join(args.out_dir, "frequent_itemsets.csv")
    write_itemsets(out1, itemsets, len(transactions))

    out_rules = os.path.join(args.out_dir, "association_rules.csv")
    write_rules(out_rules, rules)

    if mapping:
        out2 = os.path.join(args.out_dir, "frequent_itemsets_human.csv")
        write_itemsets_human(out2, itemsets, len(transactions), mapping)

        out_rules_human = os.path.join(args.out_dir, "association_rules_human.csv")
        write_rules_human(out_rules_human, rules, mapping)

    print(f"OK: Found {len(itemsets)} frequent itemsets and {len(rules)} association rules.")

if __name__ == "__main__":
    main()