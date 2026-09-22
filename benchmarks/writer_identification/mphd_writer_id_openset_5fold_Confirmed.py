"""
mphd_writer_id_openset_5fold.py

5-Fold Open-Set Writer Identification Benchmark on the MPHD Dataset.

OVERVIEW
--------
This script implements a rigorous 5-Fold Open-Set Writer Identification protocol
on the MPHD dataset. The 500 writers are partitioned into 5 disjoint folds of 100 writers
each. In every fold, the model is trained exclusively on 400 known writers and evaluated
on the remaining 100 unknown (open-set impostor) writers.

EXPERIMENTAL PROTOCOL
---------------------
For each of the 5 folds:

  1. Writer Partitioning:
     - 400 Known Writers: Used for training the feature extractor / classifier (num_classes=400).
     - 100 Unknown Writers: Completely held out from training.

  2. Model Training:
     - The model is trained from scratch using only the T1 (fixed-text) documents
       of the 400 known writers.

  3. Threshold Calibration:
     - Decision thresholds (for target FAR levels) are calibrated using the T1
       documents of the 100 unknown writers alongside T1 documents of known writers.

  4. Evaluation & Testing:
     - Known Evaluation: Detection and Identification Rate (DIR) is evaluated
       on the T2 (variable-text) documents of the 400 known writers.
     - Unknown Evaluation: False Accept Rate (FAR) is measured on the T2
       documents of the 100 held-out unknown writers.

  5. Cross-Fold Pooling:
     - Results are aggregated across all 5 folds. Over the full cross-validation cycle,
       every writer in the MPHD dataset is evaluated exactly once as an unknown
       impostor while keeping a balanced 400-writer training set per fold.

EXPECTED DIRECTORY LAYOUT
-------------------------
    ./MPHD/[Writer_ID_Code]/[ID]-T1.png
    ./MPHD/[Writer_ID_Code]/[ID]-T2-*.png

Run:
    python mphd_writer_id_openset_5fold.py
"""

import os
import re
import glob
import json
import random
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

from PIL import Image

# =========================================================================
# 1. CONFIGURATION
# =========================================================================

DATA_ROOT = "../MPHD"                                   # root folder containing [Writer_ID_Code]/*.png

CHECKPOINT_DIR = "./checkpoints_openset_5fold"          # one checkpoint saved per fold
RESULTS_PATH   = "./openset_5fold_results.json"         # incrementally updated after each fold

NUM_WRITERS = 500
NUM_FOLDS   = 5
FOLD_SEED   = 42                                        # controls the random writer->fold assignment (reproducible)
RANDOM_SEED = 42
NUM_WORKERS = 4

PATCH_SIZE   = 128
PATCH_STRIDE = 64
INK_AREA_MIN_RATIO  = 0.05
INK_PIXEL_THRESHOLD = 128
MODEL_NAME = "resnet18"                                 # "resnet18" or "densenet121"
PRETRAINED = True

BATCH_SIZE    = 64
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
EARLY_STOPPING_PATIENCE = 10
TRAIN_VAL_SPLIT_RATIO   = 0.80          # 80% train / 20% val, split over T1 patches of KNOWN writers only

TARGET_FARS = [0.01, 0.05, 0.10]        # target False Alarm Rates to calibrate/report, as in the original protocol
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 2. REPRODUCIBILITY
# =========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# 3. DATASET INDEXING (writer name -> T1 / T2 file lists)
# =========================================================================

def index_writer_documents_by_name(data_root: str, num_writers: int):
    """
    Recursively scan data_root for writer folders. Unlike the closed-set
    script, documents are keyed by writer NAME (not a fixed 0..499 label),
    since the known/unknown label space is redefined per fold.

    Returns
    -------
    writer_names : sorted list[str]
    t1_by_writer : dict[str, list[filepath]]
    t2_by_writer : dict[str, list[filepath]]
    """
    writer_dirs = sorted(
        d for d in glob.glob(os.path.join(data_root, "*"))
        if os.path.isdir(d)
    )
    if len(writer_dirs) == 0:
        raise RuntimeError(f"No writer subfolders found under '{data_root}'.")

    writer_names = [os.path.basename(d) for d in writer_dirs]

    t1_re = re.compile(r"-T1\.png$", re.IGNORECASE)
    t2_re = re.compile(r"-T2(-.*)?\.png$", re.IGNORECASE)

    t1_by_writer = defaultdict(list)
    t2_by_writer = defaultdict(list)

    for wdir, wname in zip(writer_dirs, writer_names):
        for f in glob.glob(os.path.join(wdir, "*.png")):
            base = os.path.basename(f)
            if t1_re.search(base):
                t1_by_writer[wname].append(f)
            elif t2_re.search(base):
                t2_by_writer[wname].append(f)

    n_t1 = sum(len(v) for v in t1_by_writer.values())
    n_t2 = sum(len(v) for v in t2_by_writer.values())
    if n_t1 == 0:
        raise RuntimeError("No T1 (fixed text) documents found. Check filename pattern '*-T1.png'.")
    if n_t2 == 0:
        raise RuntimeError("No T2 (variable text) documents found. Check filename pattern '*-T2*.png'.")

    print(f"[index_writer_documents_by_name] Writers found: {len(writer_names)}")
    print(f"[index_writer_documents_by_name] T1 documents : {n_t1}")
    print(f"[index_writer_documents_by_name] T2 documents : {n_t2}")

    if len(writer_names) != num_writers:
        print(f"[index_writer_documents_by_name] WARNING: expected {num_writers} writers, "
              f"found {len(writer_names)}. Continuing with actual count.")

    return writer_names, dict(t1_by_writer), dict(t2_by_writer)


def build_writer_folds(writer_names, num_folds, seed):
    """
    Randomly (reproducibly) partition writer_names into num_folds
    approximately-equal, disjoint folds. Each fold plays the role of the
    "unknown" set exactly once across the full protocol.
    """
    rng = random.Random(seed)
    shuffled = sorted(writer_names)   # deterministic base order before shuffling
    rng.shuffle(shuffled)

    n = len(shuffled)
    base_size = n // num_folds
    remainder = n % num_folds
    if remainder != 0:
        print(f"[build_writer_folds] WARNING: {n} writers not evenly divisible by "
              f"{num_folds} folds; the first {remainder} fold(s) will have one extra writer.")

    folds = []
    start = 0
    for i in range(num_folds):
        size = base_size + (1 if i < remainder else 0)
        folds.append(shuffled[start:start + size])
        start += size
    return folds


# =========================================================================
# 4. PATCH EXTRACTION (unchanged from the closed-set script)
# =========================================================================

def extract_patches_from_image(image: Image.Image, patch_size: int, stride: int,
                                 ink_area_min_ratio: float, ink_pixel_threshold: int):
    gray = image.convert("L")
    width, height = gray.size
    np_img = np.array(gray)

    patches = []

    if width < patch_size or height < patch_size:
        pad_w = max(0, patch_size - width)
        pad_h = max(0, patch_size - height)
        padded = Image.new("L", (width + pad_w, height + pad_h), color=255)
        padded.paste(gray, (0, 0))
        gray = padded
        np_img = np.array(gray)
        width, height = gray.size

    y_positions = list(range(0, height - patch_size + 1, stride))
    if len(y_positions) == 0 or y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)

    x_positions = list(range(0, width - patch_size + 1, stride))
    if len(x_positions) == 0 or x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)

    for y in sorted(set(y_positions)):
        for x in sorted(set(x_positions)):
            patch_np = np_img[y:y + patch_size, x:x + patch_size]
            ink_ratio = np.mean(patch_np < ink_pixel_threshold)
            if ink_ratio < ink_area_min_ratio:
                continue
            patches.append(Image.fromarray(patch_np, mode="L"))

    return patches


def stratified_patch_split(patch_records, split_ratio, seed):
    """
    Per-writer (stratified) train/val split over patches, so every known
    writer's class is represented on both sides even though each writer
    contributes only a single T1 document.
    """
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for rec in patch_records:
        _, label, _ = rec
        by_label[label].append(rec)

    train_records, val_records = [], []
    empty_val_writers = []

    for label, records in by_label.items():
        records = records.copy()
        rng.shuffle(records)
        split_idx = int(len(records) * split_ratio)

        if len(records) >= 2:
            split_idx = min(max(split_idx, 1), len(records) - 1)
        else:
            empty_val_writers.append(label)

        train_records.extend(records[:split_idx])
        val_records.extend(records[split_idx:])

    if empty_val_writers:
        print(f"[stratified_patch_split] WARNING: {len(empty_val_writers)} classes had only 1 patch "
              f"and could not be represented in both train and val.")

    n_train_writers = len(set(r[1] for r in train_records))
    n_val_writers = len(set(r[1] for r in val_records))
    print(f"[stratified_patch_split] Train patches: {len(train_records)} ({n_train_writers} writers) | "
          f"Val patches: {len(val_records)} ({n_val_writers} writers)")

    return train_records, val_records


def build_patch_index(documents, patch_size, stride, ink_ratio, ink_thresh, desc=""):
    """documents: list of (filepath, label). Returns list of (filepath, label, PIL_patch)."""
    patch_records = []
    for fpath, label in documents:
        try:
            image = Image.open(fpath)
        except Exception as e:
            print(f"[build_patch_index] WARNING: failed to open '{fpath}': {e}")
            continue
        for p in extract_patches_from_image(image, patch_size, stride, ink_ratio, ink_thresh):
            patch_records.append((fpath, label, p))

    print(f"[build_patch_index] {desc}: {len(documents)} documents -> {len(patch_records)} valid patches.")
    return patch_records


# =========================================================================
# 5. TRANSFORMS & DATASETS
# =========================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PatchDataset(Dataset):
    """(filepath, label, PIL_patch) records for training/validation."""

    def __init__(self, patch_records, transform):
        self.records = patch_records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        _, label, patch_img = self.records[idx]
        return self.transform(patch_img), label


class DocPatchDataset(Dataset):
    """
    (doc_id, PIL_patch) records for document-level confidence scoring.
    No label is required here -- this is used both for known-writer T2
    documents (where DIR needs a true label, tracked separately by doc_id)
    and for unknown-writer T1/T2 documents (where only the confidence
    score matters, since no true class exists in the trained model).
    """

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        doc_id, patch_img = self.records[idx]
        return self.transform(patch_img), doc_id


def build_doc_patch_records(doc_list, patch_size, stride, ink_ratio, ink_thresh):
    """doc_list: list of filepaths. Returns [(doc_id, PIL_patch), ...], doc_id -> filepath."""
    records = []
    doc_id_to_path = {}
    for doc_id, fpath in enumerate(doc_list):
        doc_id_to_path[doc_id] = fpath
        try:
            image = Image.open(fpath)
        except Exception as e:
            print(f"[build_doc_patch_records] WARNING: failed to open '{fpath}': {e}")
            continue
        for p in extract_patches_from_image(image, patch_size, stride, ink_ratio, ink_thresh):
            records.append((doc_id, p))
    return records, doc_id_to_path


@torch.no_grad()
def compute_doc_level_avg_probs(model, doc_list, patch_size, stride, ink_ratio, ink_thresh,
                                 transform, device, batch_size, num_workers):
    """
    Returns dict: doc_id (index into doc_list) -> averaged softmax probability
    vector across that document's patches. A document with zero valid patches
    (e.g. failed to open) will simply be absent from the returned dict.
    """
    records, doc_id_to_path = build_doc_patch_records(doc_list, patch_size, stride, ink_ratio, ink_thresh)
    if len(records) == 0:
        return {}, doc_id_to_path

    loader = DataLoader(DocPatchDataset(records, transform), batch_size=batch_size,
                         shuffle=False, num_workers=num_workers, pin_memory=True)

    model.eval()
    prob_sum = {}
    prob_count = defaultdict(int)

    for images, doc_ids in loader:
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        doc_ids = doc_ids.numpy()
        for i in range(len(doc_ids)):
            did = int(doc_ids[i])
            if did not in prob_sum:
                prob_sum[did] = np.zeros(probs.shape[1], dtype=np.float64)
            prob_sum[did] += probs[i]
            prob_count[did] += 1

    avg_probs = {did: prob_sum[did] / prob_count[did] for did in prob_sum}
    return avg_probs, doc_id_to_path


# =========================================================================
# 6. MODEL FACTORY & TRAINING LOOP (unchanged from the closed-set script)
# =========================================================================

def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: '{model_name}'. Use 'resnet18' or 'densenet121'.")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, num_epochs, lr, weight_decay,
                 patience, device, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"  Epoch [{epoch:03d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_model_state, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping after {epoch} epochs "
                      f"(no val_loss improvement for {patience} epochs).")
                break

    model.load_state_dict(best_model_state)
    return model


# =========================================================================
# 7. PER-FOLD OPEN-SET EVALUATION
# =========================================================================

def run_fold(fold_idx, known_writers, unknown_writers, t1_by_writer, t2_by_writer, device):
    print(f"\n{'=' * 70}\nFOLD {fold_idx + 1}/{NUM_FOLDS}  |  "
          f"known={len(known_writers)}  unknown={len(unknown_writers)}\n{'=' * 70}")

    known_writers_sorted = sorted(known_writers)
    unknown_writers_sorted = sorted(unknown_writers)
    writer_to_newlabel = {w: i for i, w in enumerate(known_writers_sorted)}
    num_classes = len(known_writers_sorted)

    # ---- Train the model from scratch on KNOWN writers' T1 only -----------
    known_t1_docs = [
        (fpath, writer_to_newlabel[w])
        for w in known_writers_sorted
        for fpath in t1_by_writer.get(w, [])
    ]
    t1_patch_records = build_patch_index(
        known_t1_docs, PATCH_SIZE, PATCH_STRIDE, INK_AREA_MIN_RATIO, INK_PIXEL_THRESHOLD,
        desc=f"Fold {fold_idx + 1} T1 (known, pre-split)"
    )
    train_records, val_records = stratified_patch_split(t1_patch_records, TRAIN_VAL_SPLIT_RATIO, RANDOM_SEED)

    train_transform = build_transforms(train=True)
    eval_transform = build_transforms(train=False)

    train_loader = DataLoader(PatchDataset(train_records, train_transform), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(PatchDataset(val_records, eval_transform), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold{fold_idx + 1}_best.pt")

    model = build_model(MODEL_NAME, num_classes, PRETRAINED).to(device)
    model = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                         EARLY_STOPPING_PATIENCE, device, ckpt_path)

    # ---- Calibrate threshold on UNKNOWN writers' T1 (not the reported T2) --
    calib_docs = [fpath for w in unknown_writers_sorted for fpath in t1_by_writer.get(w, [])]
    calib_avg_probs, _ = compute_doc_level_avg_probs(
        model, calib_docs, PATCH_SIZE, PATCH_STRIDE, INK_AREA_MIN_RATIO, INK_PIXEL_THRESHOLD,
        eval_transform, device, BATCH_SIZE, NUM_WORKERS
    )
    calib_max_conf = np.array([p.max() for p in calib_avg_probs.values()])
    print(f"[Fold {fold_idx + 1}] Calibration set (unknown T1): {len(calib_max_conf)} documents")

    # Note: with only ~100 unknown writers per fold, this per-fold calibration
    # set has coarse resolution at low target FARs (e.g. ~1% per document).
    # Pooling the *reported* FAR/DIR across all 5 folds (500 unknown docs total)
    # substantially improves the resolution of the final numbers even though
    # each fold's calibration set alone remains coarse.
    thresholds_per_far = {
        target_far: float(np.percentile(calib_max_conf, 100 * (1 - target_far)))
        for target_far in TARGET_FARS
    }

    # ---- Reported DIR: KNOWN writers' T2 -----------------------------------
    known_t2_docs, known_t2_true_labels = [], []
    for w in known_writers_sorted:
        for fpath in t2_by_writer.get(w, []):
            known_t2_docs.append(fpath)
            known_t2_true_labels.append(writer_to_newlabel[w])

    known_avg_probs, _ = compute_doc_level_avg_probs(
        model, known_t2_docs, PATCH_SIZE, PATCH_STRIDE, INK_AREA_MIN_RATIO, INK_PIXEL_THRESHOLD,
        eval_transform, device, BATCH_SIZE, NUM_WORKERS
    )
    known_doc_ids = sorted(known_avg_probs.keys())
    known_max_conf = np.array([known_avg_probs[d].max() for d in known_doc_ids])
    known_argmax = np.array([known_avg_probs[d].argmax() for d in known_doc_ids])
    known_true = np.array([known_t2_true_labels[d] for d in known_doc_ids])

    # ---- Reported actual FAR: UNKNOWN writers' T2 --------------------------
    unknown_t2_docs = [fpath for w in unknown_writers_sorted for fpath in t2_by_writer.get(w, [])]
    unknown_avg_probs, _ = compute_doc_level_avg_probs(
        model, unknown_t2_docs, PATCH_SIZE, PATCH_STRIDE, INK_AREA_MIN_RATIO, INK_PIXEL_THRESHOLD,
        eval_transform, device, BATCH_SIZE, NUM_WORKERS
    )
    unknown_doc_ids = sorted(unknown_avg_probs.keys())
    unknown_max_conf = np.array([unknown_avg_probs[d].max() for d in unknown_doc_ids])

    fold_result = {
        "fold": fold_idx + 1,
        "num_known_writers": len(known_writers_sorted),
        "num_unknown_writers": len(unknown_writers_sorted),
        "num_calib_docs": int(len(calib_max_conf)),
        "num_known_t2_docs": int(len(known_doc_ids)),
        "num_unknown_t2_docs": int(len(unknown_doc_ids)),
        "per_far": {},
    }

    for target_far in TARGET_FARS:
        thr = thresholds_per_far[target_far]

        accepted_and_correct = (known_max_conf >= thr) & (known_argmax == known_true)
        dir_value = float(np.mean(accepted_and_correct)) if len(known_true) > 0 else float("nan")

        false_accepts = unknown_max_conf >= thr
        far_actual = float(np.mean(false_accepts)) if len(unknown_max_conf) > 0 else float("nan")

        fold_result["per_far"][str(target_far)] = {
            "threshold": thr,
            "DIR": dir_value,
            "FAR_actual": far_actual,
            "n_known_correct_accepted": int(accepted_and_correct.sum()),
            "n_known_total": int(len(known_true)),
            "n_unknown_false_accepted": int(false_accepts.sum()),
            "n_unknown_total": int(len(unknown_max_conf)),
        }

        print(f"  [target FAR={target_far * 100:.0f}%] threshold={thr:.4f}  "
              f"DIR={dir_value * 100:.2f}%  actual FAR={far_actual * 100:.2f}%")

    return fold_result


# =========================================================================
# 8. MAIN
# =========================================================================

def main():
    set_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    writer_names, t1_by_writer, t2_by_writer = index_writer_documents_by_name(DATA_ROOT, NUM_WRITERS)

    folds = build_writer_folds(writer_names, NUM_FOLDS, FOLD_SEED)
    for i, f in enumerate(folds):
        print(f"Fold {i + 1}: {len(f)} writers assigned as 'unknown'")

    all_fold_results = []
    for fold_idx in range(NUM_FOLDS):
        unknown_writers = set(folds[fold_idx])
        known_writers = set(writer_names) - unknown_writers

        result = run_fold(fold_idx, known_writers, unknown_writers, t1_by_writer, t2_by_writer, DEVICE)
        all_fold_results.append(result)

        # Persist incrementally after every fold, so a crash mid-run doesn't
        # lose already-completed folds (5 full training runs is expensive).
        with open(RESULTS_PATH, "w", encoding="utf-8") as f_out:
            json.dump({"folds": all_fold_results, "fold_assignments": folds}, f_out, indent=2, ensure_ascii=False)

    # ---- Pool DIR/FAR across all 5 folds (per target FAR) ------------------
    print(f"\n{'=' * 70}\nPOOLED RESULTS ACROSS {NUM_FOLDS} FOLDS "
          f"({sum(len(f) for f in folds)} writers total)\n{'=' * 70}")

    pooled = {}
    for target_far in TARGET_FARS:
        key = str(target_far)
        total_known_correct = sum(r["per_far"][key]["n_known_correct_accepted"] for r in all_fold_results)
        total_known = sum(r["per_far"][key]["n_known_total"] for r in all_fold_results)
        total_unknown_fa = sum(r["per_far"][key]["n_unknown_false_accepted"] for r in all_fold_results)
        total_unknown = sum(r["per_far"][key]["n_unknown_total"] for r in all_fold_results)

        pooled_dir = total_known_correct / total_known if total_known > 0 else float("nan")
        pooled_far = total_unknown_fa / total_unknown if total_unknown > 0 else float("nan")

        pooled[key] = {
            "pooled_DIR": pooled_dir,
            "pooled_FAR": pooled_far,
            "n_known_total": total_known,
            "n_unknown_total": total_unknown,
        }
        print(f"[target FAR={target_far * 100:.0f}%] Pooled DIR={pooled_dir * 100:.2f}%  "
              f"Pooled actual FAR={pooled_far * 100:.2f}%  "
              f"(n_known={total_known}, n_unknown={total_unknown})")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f_out:
        json.dump({"folds": all_fold_results, "fold_assignments": folds, "pooled": pooled},
                   f_out, indent=2, ensure_ascii=False)

    print(f"\n[main] Full results (per-fold + pooled + fold assignments) saved to '{RESULTS_PATH}'")
    print("[main] Done.")


if __name__ == "__main__":
    main()
