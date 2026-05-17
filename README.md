# Association and Sequential Pattern Mining Toolkit

A toolkit for frequent itemset mining (Apriori) and sequential pattern discovery (AprioriAll). Supports CSV and SPMF formats.

## Installation

1. Clone the repository:
   `git clone https://github.com/username/AprioriAll.git`
   `cd AprioriAll`

2. Requirements:
   Python 3.10+. No external dependencies required.

## Available Commands

Use `run_pipeline.py` to execute the algorithms.

### AprioriAll (Sequential Patterns)
Used for discovering patterns in chronological sequences.
**Input data:** Requires `sequence_id` and `pos` (position/time) columns.

* **Example from article:**
  `python run_pipeline.py --algo apriori_all --input data/sequences_test.csv --min-sup-count 2`

* **Unconverted SPMF data:**
  `python run_pipeline.py --algo apriori_all --input data/raw/mini_sign.csv --spmf --min-sup-count 100`

### Apriori (Frequent Itemsets)
Used for classic market basket analysis.
**Input data:** Requires transaction-based format (rows of items).

* **Standard command:**
  `python run_pipeline.py --algo apriori --input data/raw/mini_retail.csv --min-sup-count 20`

## Command Options

| Option | Description |
| :--- | :--- |
| `--algo` | Algorithm choice: `apriori` or `apriori_all`. |
| `--input` | Path to the input file. |
| `--min-sup-count` | Minimum support count (threshold). |
| `--spmf` | Enable for SPMF formatted input files. |
| `--out-dir` | Output directory (default: `out`). |
| `--mapping` | Path to a JSON item mapping file. |

## Results
Outputs are generated in the `out/` directory:
* `frequent_itemsets.csv` - Found itemsets.
* `sequences.csv` - Found sequences.
* `association_rules.csv` - Generated rules with Confidence and Lift.
