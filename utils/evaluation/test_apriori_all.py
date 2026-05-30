import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

USE_PARALLEL = False
if "--parallel" in sys.argv:
    USE_PARALLEL = True
    sys.argv.remove("--parallel")

if USE_PARALLEL:
    from src import apriori_all_parallel as apriori_lib
else:
    from src import apriori_all as apriori_lib


def print_title(text):
    print(f"\n{'=' * 20} {text} {'=' * 20}")


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    fmt = " | ".join([f"{{:<{w}}}" for w in widths])
    sep = "-+-".join(["-" * w for w in widths])

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


class TestAprioriAllVerbose(unittest.TestCase):

    def setUp(self):
        self.dataset = [
            [frozenset(["1", "2"]), frozenset(["3"]), frozenset(["4"])],
            [frozenset(["1"]), frozenset(["4", "5"])],
            [frozenset(["1", "2"]), frozenset(["4"])],
            [frozenset(["2"]), frozenset(["3"]), frozenset(["4"])]
        ]

    def test_complete_pipeline_with_visualization(self):
        mode_text = "PARALLEL" if USE_PARALLEL else "SEQUENTIAL"
        print_title(f"STEP 1: Original Sequence Database (Ds) [{mode_text} MODE]")

        rows_ds = [[i + 1, " -> ".join([f"{{{','.join(sorted(list(ev)))}}}" for ev in seq])] for i, seq in
                   enumerate(self.dataset)]
        print_table(["Sequence ID", "Sequence Layout"], rows_ds)

        print_title("STEP 2: Frequent Itemsets Mining (L)")
        freq_itemsets = apriori_lib.mine_frequent_itemsets(self.dataset, min_sup_count=2)
        sorted_freq = sorted(list(freq_itemsets), key=lambda x: (len(x), sorted(list(x))))
        rows_l = [[i + 1, f"{{{','.join(sorted(list(itemset)))}}}"] for i, itemset in enumerate(sorted_freq)]
        print_table(["Itemset ID", "Frequent Itemset Layout"], rows_l)

        mapping_to_id = {itemset: i + 1 for i, itemset in enumerate(sorted_freq)}

        print_title("STEP 3: Transformed Mapped Database (Dts)")
        transformed_db = []
        rows_dts = []
        for idx, seq in enumerate(self.dataset):
            trans_seq = []
            for ev in seq:
                mapped_ids = set()
                for itemset, itemset_id in mapping_to_id.items():
                    if itemset.issubset(ev):
                        mapped_ids.add(itemset_id)
                if mapped_ids:
                    trans_seq.append(frozenset(mapped_ids))
            if trans_seq:
                transformed_db.append(trans_seq)
                layout = " -> ".join([f"{{{','.join(sorted(str(x) for x in ev))}}}" for ev in seq])
                rows_dts.append([idx + 1, layout])
        print_table(["Sequence ID", "Transformed (Mapped IDs)"], rows_dts)

        print_title("STEP 4: Sequential Patterns Generation Loop")
        results = apriori_lib.apriori_all(self.dataset, min_sup_count=2)

        print_title("STEP 5: Final Maximal Sequential Patterns Output")
        rows_final = []
        extracted_sequences = []

        if isinstance(results, (list, tuple)) and len(results) == 2 and isinstance(results[0], dict):
            maximal_patterns = results[1]
        elif isinstance(results, dict):
            maximal_patterns = results
        else:
            maximal_patterns = {}

        for idx, (seq, count) in enumerate(maximal_patterns.items()):
            extracted_sequences.append(seq)

            parts = []
            for ev in seq:
                if isinstance(ev, (frozenset, set)):
                    parts.append(f"{{{','.join(sorted(list(ev)))}}}")
                else:
                    parts.append(str(ev))
            layout = " -> ".join(parts)

            rows_final.append([idx + 1, layout, count])

        print_table(["Pattern ID", "Maximal Sequence", "Support Count"], rows_final)

        self.assertEqual(len(extracted_sequences), 2)
        path_a = (frozenset(["1", "2"]), frozenset(["4"]))
        path_b = (frozenset(["2"]), frozenset(["3"]), frozenset(["4"]))
        self.assertIn(path_a, extracted_sequences)
        self.assertIn(path_b, extracted_sequences)


if __name__ == "__main__":
    print("\n" + "=" * 50 + "\n")
    runner = unittest.TextTestRunner(stream=sys.stdout)
    unittest.main(argv=['first-arg-is-ignored'], exit=False, testRunner=runner)