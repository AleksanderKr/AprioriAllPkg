# AprioriAllPkg: Association and Sequential Pattern Mining Toolkit

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AleksanderKr/AprioriAllPkg/HEAD?urlpath=%2Fdoc%2Ftree%2FAprioriAllDemo.ipynb)
AprioriAllPkg is a Python toolkit for sequential pattern discovery (AprioriAll) and frequent itemset mining (Apriori). It natively supports both standard CSV (long format) and SPMF sequence formats.

## Table of Contents
- [Installation](#installation)
- [Input Data Format Specifications](#input-data-format-specifications)
- [Command Line Usage](#command-line-usage)
- [Minimal Working Example (Python API)](#minimal-working-example-python-api)
- [Testing and Reproducibility](#testing-and-reproducibility)
- [Outputs](#outputs)

## Installation

1. Clone the repository:
   `git clone https://github.com/username/AprioriAll.git`
   `cd AprioriAll`

2. Ensure you have Python 3.10 or higher installed:
   `python --version`

*(Note: Pandas and Matplotlib are only required if executing the benchmarking suite or Jupyter notebooks).*

## Input Data Format Specifications

The package supports two primary data structures:

### 1. CSV Format (Long Format)
When mining sequential patterns (`apriori_all`), the software expects a CSV file representing data in a "long format". The file must contain a header with the following columns:
* `sequence_id` (integer or string): Uniquely identifies a specific sequence.
* `pos` (integer or timestamp): Denotes the chronological position or order of the event within the sequence.
* `item` (integer or string): The specific item or event occurring at that position.

*Example structure (`data/sequences_test.csv`):*
sequence_id,pos,item
1,10,1
1,10,2
1,20,3
1,30,4
2,10,1

*(Items occurring at the same `pos` within the same `sequence_id` are treated as occurring simultaneously in a single itemset).*

### 2. SPMF Format
The package can parse standard `.txt` sequence files formatted according to the SPMF specification (items separated by spaces, itemsets terminated by `-1`, and sequences terminated by `-2`).

## Command Line Usage

Use the execution script located in `utils/evaluation/run_pipeline.py` to run the algorithms.

### AprioriAll (Sequential Patterns)
Used for discovering frequent chronological sequences.

* **Using CSV input:**
  `python utils/evaluation/run_pipeline.py --algo apriori_all --input data/sequences_test.csv --min-sup-count 2 --out-dir output`

* **Using Parallel implementation:**
  `python utils/evaluation/run_pipeline.py --algo apriori_all_parallel --input data/sequences_test.csv --min-sup-count 2 --out-dir output`

### Apriori (Frequent Itemsets & Association Rules)
Used for classic market basket analysis (non-sequential).

* **Standard command:**
  `python utils/evaluation/run_pipeline.py --algo apriori --input data/raw/mini_retail.csv --min-sup-count 20 --out-dir output`

### Command Options

| Option | Description |
| :--- | :--- |
| `--algo` | Algorithm choice: `apriori`, `apriori_all`, or `apriori_all_parallel`. |
| `--input` | Path to the input file (CSV or TXT). |
| `--min-sup-count` | Minimum support count threshold (absolute integer). |
| `--spmf` | Enable explicitly if parsing an SPMF formatted input file. |
| `--out-dir` | Output directory (default: `output`). |
| `--mapping` | Path to a JSON item mapping file. |

## Minimal Working Example (Python API)

You can import and use the package directly in your Python code.

from src.apriori_all import AprioriAll
from src.data_structures import SequenceDatabase

# 1. Initialize an empty Sequence Database
db = SequenceDatabase()

# 2. Add sequences programmatically
# Format: add_sequence(sequence_id, [(pos1, item1), (pos2, item2), ...])
db.add_sequence(1, [(10, 'A'), (10, 'B'), (20, 'C'), (30, 'D')])
db.add_sequence(2, [(10, 'A'), (20, 'D'), (20, 'E')])
db.add_sequence(3, [(10, 'A'), (10, 'B'), (20, 'D')])
db.add_sequence(4, [(10, 'B'), (20, 'C'), (30, 'D')])

# 3. Initialize the AprioriAll algorithm
miner = AprioriAll(min_sup_count=2)

# 4. Execute the mining process
maximal_patterns = miner.mine(db)

# 5. Output the results
print("Found Maximal Sequential Patterns:")
for pattern, support in maximal_patterns:
    print(f"Pattern: {pattern} | Support: {support}")


## Testing and Reproducibility

To verify the functional correctness and trace the algorithmic execution pipeline, run the unit test script:
`python utils/evaluation/test_apriori_all.py`

For an interactive demonstration including empirical benchmarking against SPMF, launch the Reproducible Capsule using the Binder badge at the top of this document.

## Outputs
Execution via `run_pipeline.py` generates results in the specified `--out-dir` (default: `output/`):
* `frequent_sequences.csv` - Maximal sequential patterns extracted by AprioriAll.
* `frequent_itemsets.csv` - Itemsets found by Apriori.
* `association_rules.csv` - Generated rules with Confidence and Lift (Apriori only).
