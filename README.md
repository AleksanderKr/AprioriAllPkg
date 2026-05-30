# AprioriAllPkg: Association and Sequential Pattern Mining Toolkit

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AleksanderKr/AprioriAllPkg/HEAD?urlpath=%2Fdoc%2Ftree%2FAprioriAllDemo.ipynb)

AprioriAllPkg is a Python toolkit for sequential pattern discovery (AprioriAll) and frequent itemset mining (Apriori). It natively supports both standard CSV (long format) and SPMF sequence formats.

## Table of Contents
* [Interactive Demonstration (Zero Setup)](#interactive-demonstration-zero-setup)
* [Installation](#installation)
* [Input Data Format Specifications](#input-data-format-specifications)
* [Command Line Usage](#command-line-usage)
* [Benchmarking Suite](#benchmarking-suite)
* [Data Conversion Utility](#data-conversion-utility)
* [Minimal Working Example (Python API)](#minimal-working-example-python-api)
* [Testing and Reproducibility](#testing-and-reproducibility)
* [Outputs](#outputs)

## Interactive Demonstration (Zero Setup)

You can run and evaluate the entire toolkit directly in your browser using our preconfigured Jupyter Notebook environment. This interactive capsule reproduces the core algorithmic examples, functional verification against SPMF, and the performance benchmarking suite described in the software article.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AleksanderKr/AprioriAllPkg/HEAD?urlpath=%2Fdoc%2Ftree%2FAprioriAllDemo.ipynb)

*(Click the badge above to launch the reproducible capsule. The initial environment build may take a few minutes).*

> Performance Note: Binder environments operate under strict resource constraints (limited RAM and CPU). For extensive empirical benchmarking, large datasets, or optimal execution speed, it is highly recommended to clone the repository and run the Jupyter notebook locally.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/AleksanderKr/AprioriAllPkg.git](https://github.com/AleksanderKr/AprioriAllPkg.git)
   cd AprioriAllPkg
   ```

2. Ensure you have Python 3.10 or higher installed:
   ```bash
   python --version
   ```

*(Note: Pandas and Matplotlib are only required if executing the benchmarking suite or Jupyter notebooks).*

## Input Data Format Specifications

The package supports two primary data structures:

### 1. CSV Format (Long Format)
When mining sequential patterns (`apriori_all`), the software expects a CSV file representing data in a "long format". The file must contain a header with the following columns:
* `sequence_id` (integer or string): Uniquely identifies a specific sequence.
* `pos` (integer or timestamp): Denotes the chronological position or order of the event within the sequence.
* `item` (integer or string): The specific item or event occurring at that position.

*Example structure (`data/sequences_test.csv`):*
```csv
sequence_id,pos,item
105,1,30
105,2,90
106,1,10
106,1,20
```

*(Items occurring at the same `pos` within the same `sequence_id` are treated as occurring simultaneously in a single itemset).*

### 2. SPMF Format
The package can parse standard `.txt` sequence files formatted according to the SPMF specification (items separated by spaces, itemsets terminated by `-1`, and sequences terminated by `-2`).

## Command Line Usage

Use the execution script located in `utils/evaluation/run_pipeline.py` to run the algorithms.

### AprioriAll (Sequential Patterns)
Used for discovering frequent chronological sequences.

* Using CSV input:
  ```bash
  python utils/evaluation/run_pipeline.py --algo apriori_all --input data/sequences_test.csv --min-sup-count 2 --out-dir output
  ```

* Using Parallel implementation:
  ```bash
  python utils/evaluation/run_pipeline.py --algo apriori_all_parallel --input data/sequences_test.csv --min-sup-count 2 --out-dir output
  ```

### Apriori (Frequent Itemsets & Association Rules)
Used for classic market basket analysis (non-sequential).

* Standard command:
  ```bash
  python utils/evaluation/run_pipeline.py --algo apriori --input data/raw/mini_retail.csv --min-sup-count 20 --out-dir output
  ```

### Pipeline Command Options (`run_pipeline.py`)

* **`--algo`**: Algorithm choice: `apriori`, `apriori_all`, `apriori_all_parallel`, or SPMF native: `prefixspan`, `spade`, `gsp`.
* **`--input`**: Path to the input file (CSV or TXT).
* **`--min-sup-count`**: Minimum support count threshold (absolute integer).
* **`--spmf`**: Enable explicitly if parsing an SPMF formatted input file to trigger auto-conversion.
* **`--out-dir`**: Output directory (default: `output`).
* **`--mapping`**: Path to an optional JSON item mapping file.

## Benchmarking Suite

The toolkit includes a dedicated script (`run_benchmarks.py`) to systematically evaluate performance across multiple algorithms and support thresholds.

* Example Execution:
  ```bash
  python utils/evaluation/run_benchmarks.py --input data/seq_sign.csv --supports 120 160 200 --algos apriori_all prefixspan --clear-results
  ```

### Benchmark Command Options (`run_benchmarks.py`)

* **`--input`**: (Required) Path to the input dataset.
* **`--supports`**: (Required) Space-separated list of absolute support thresholds (e.g., `120 160 200`).
* **`--algos`**: Space-separated list of algorithms to test. Default: `apriori_all apriori_all_parallel prefixspan spade gsp`.
* **`--out-dir`**: Output directory for metrics (default: `output`).
* **`--clear-results`**: If flagged, deletes any existing `benchmark_results.csv` file before starting the run.

## Data Conversion Utility

The package provides a standalone utility to bidirectionally convert datasets between the standard CSV (long format) and the SPMF format.

* Convert SPMF to CSV:
  ```bash
  python utils/convert_spmf_to_seq.py --input data/raw/sign.txt --out data/seq_sign.csv --direction spmf2csv
  ```

* Convert CSV to SPMF:
  ```bash
  python utils/convert_spmf_to_seq.py --input data/seq_sign.csv --out data/raw/sign.txt --direction csv2spmf
  ```

## Minimal Working Example

Since this package is not distributed publicly via PyPI, you must execute your scripts from the root directory of the cloned repository so that the local `src` directory is accessible.

```python
# Note: Execute this script from the repository root directory
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
```

## Testing and Reproducibility

To verify the functional correctness and trace the algorithmic execution pipeline, run the unit test script:

```bash
python utils/evaluation/test_apriori_all.py
```


## Testing and Reproducibility

To verify the functional correctness and trace the algorithmic execution pipeline, run the unit test script:
```bash
python utils/evaluation/test_apriori_all.py
```

For an interactive, cloud-based demonstration including empirical benchmarking against native SPMF tools, launch the Reproducible Capsule using the Binder badge at the top of this document. Note that for large workload validation, executing the notebook locally within a native Jupyter installation is preferred due to cloud instance performance ceilings.

## Outputs
Execution via `run_pipeline.py` generates results in the specified `--out-dir` (default: `output/`):
* `frequent_sequences.csv`: Maximal sequential patterns extracted by AprioriAll.
* `frequent_itemsets.csv`: Itemsets found by Apriori.
* `association_rules.csv`: Generated rules with Confidence and Lift (Apriori only).
* `benchmarks/benchmark_results.csv`: Appended performance metrics (Time, RAM, CPU) if running benchmarks.
```