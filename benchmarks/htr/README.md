# MPHD: Handwriting Text Recognition (HTR) Dataset Preparation
This directory contains the dataset preparation script for the **Handwriting Text Recognition (HTR)** task on the MPHD dataset.

The provided script reproduces the official writer-level train/validation/test splits used in the MPHD paper (Section 4.1, Table 7).

---

## 🛠 Script Overview ([`mphd_htr_prepare_splits.py`](mphd_htr_prepare_splits.py))

The extraction and splitting process consists of two steps:

1. **Extraction (`step1_extract_lines`):**
   * Walks every writer directory (`[ID]-[CODE]/`) in the MPHD dataset.
   * Reads line-level ground-truth transcriptions from each writer's `.json` file.
   * Copies all Text 1 (T1) and Text 2 (T2) line images into a flat output directory (`MPHD_Lines/`).
   * Writes a matching `.txt` transcription file for every line image.
   * Records line categories (`T1` vs. `T2`) in `lines_type.txt`.

2. **Writer-Level Splitting (`step2_split_dataset`):**
   * Performs a **70% / 15% / 15%** split at the **writer level** using `seed = 42` to ensure no lines from the same writer overlap across splits.
   * Generates `.ln` split files containing line identifiers for `train`, `val`, and `test`.
   * Further separates the test split into disjoint `test-t1` and `test-t2` subsets by text type.

---

## Data Split Statistics (Official Baseline)

The generated splits correspond to Table 7 in the MPHD paper:

| Split | Number of Lines |
| :--- | :--- |
| **Train** | 3,515 |
| **Validation** | 767 |
| **Test (Combined)** | 739 |
| ↳ *Test-T1 (Fixed Text)* | 403 |
| ↳ *Test-T2 (Variable Text)* | 336 |

---

## How to Run

1. Open `mphd_htr_prepare_splits.py` and update the `CONFIG` variables if necessary:
   ```python
   DATASET_ROOT = r"../MPHD"     # Path to root MPHD directory containing writer folders
   OUTPUT_ROOT  = r"./"          # Path where MPHD_Lines/ and split files will be saved
   


## Output Files Generated
* MPHD_Lines/: Directory containing all extracted line images (.png) and their ground-truth text (.txt).
* train.ln, val.ln, test.ln: List of line IDs assigned to each split.
* test-t1.ln, test-t2.ln: Disjoint test sets evaluated separately by text type.
* [optional] lines_type.txt: Mapping of line IDs to their text type (T1 or T2).
