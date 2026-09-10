# MPHD: Line Segmentation Benchmark
This directory contains the line segmentation benchmark script for the MPHD dataset.

The provided script reproduces the official line segmentation baseline reported in the MPHD paper (Section: Line Segmentation, Table 8), using EasyOCR as an off-the-shelf detector, so any researcher can obtain comparable, reproducible results.

---

## Script Overview ([`line_segmentation_easyocr_benchmark.py`](line_segmentation_easyocr_benchmark.py))

Since MPHD provides only full-form and cropped text-region images (not pre-segmented lines), the script performs the full evaluation pipeline in three steps:

1. **Ground-truth recovery:**
   * Since MPHD ships ground-truth lines as cropped image files rather than box coordinates, each writer's `[ID]-T*-Lines/` crop is matched back onto its source text-region image via SIFT keypoint matching and RANSAC homography estimation.
   * The recovered corners give a precise ground-truth bounding box for every line.

2. **Baseline prediction:**
   * Runs EasyOCR (`fa` language model) as a word/box detector on the source text image.
   * Groups the detected word boxes into lines via a simple vertical (y-coordinate) clustering strategy, no training on MPHD is required.

3. **Matching and metrics:**
   * Predicted and ground-truth lines are matched greedily by descending IoU (one-to-one).
   * Reports mIoU(all), mIoU(m), F1@0.50, F1@0.75, Line Count Accuracy (LCA), relative line-count error (RelCntErr), and mean SSIM, following standard document-analysis evaluation practice.

---

## Evaluation Protocols

The script supports both protocols used in the paper (set via the `PROTOCOL` variable):

* **Protocol 1 (all 500 writers):** evaluates the baseline on the entire dataset (1,000 text images) to assess the upper-bound performance of general-purpose, off-the-shelf tools under ideal conditions. Intended for unsupervised or pre-trained methods.
* **Protocol 2 (official test split, 15%):** follows the official 70% / 15% / 15% writer-level split; the test set comprises writers 426–500 (75 writers, 150 images). Intended as the primary, reproducible benchmark for comparing newly developed supervised segmentation methods. All hyperparameters are kept identical to Protocol 1 and are not tuned on the test set.

---

## Results (Official Baseline)

The generated results correspond to Table 8 in the MPHD paper:

| Protocol | mIoU(all) | mIoU(m) | F1@0.50 | F1@0.75 | LCA (%) | RelCntErr | MeanSSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Protocol 1** (All 500 Writers) | 0.4393 | 0.6724 | 0.6757 | 0.3813 | 18.20 | 0.5376 | 0.9103 |
| **Protocol 2** (Test Set, 15%) | 0.4171 | 0.6455 | 0.6694 | 0.2968 | 20.67 | 0.5510 | 0.9056 |

The gap between mIoU(all) and mIoU(m) indicates that while detected lines are localized reasonably well, the baseline frequently misestimates the number of lines per document, the primary challenge on MPHD is accurate line separation/counting rather than raw spatial localization.

> **Note:** Protocol 1 metrics may exhibit minor run-to-run variation (≲0.2% absolute) due to the inherent non-determinism of RANSAC-based homography estimation in ground-truth box recovery and GPU-based inference; Protocol 2 (the primary held-out benchmark) was verified to be fully reproducible.
---

## How to Run

1. Open `line_segmentation_easyocr_benchmark.py` and update the config variables if necessary:
   ```python
   DATASET_ROOT = "../MPHD/"   # Path to root MPHD directory containing writer folders
   PROTOCOL     = 2            # 1 = all 500 writers, 2 = official held-out test split (writers 426-500)
   ```
2. Run:
   ```bash
   python line_segmentation_easyocr_benchmark.py
   ```

---

## Output Files Generated
* `evaluation_official_test.json`: per-image results (mIoU, F1, LCA, etc.) plus final aggregated metrics.
* `visualization_results/`: predicted line bounding boxes overlaid on each evaluated text image.
