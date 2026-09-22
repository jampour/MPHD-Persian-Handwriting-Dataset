"""
mphd_writer_id_benchmark.py

Patch-based Text-Independent Writer Identification benchmark on the MPHD Dataset, across 500 writers.

Protocol
--------
- Train/Val: patches extracted from fixed-text images  [ID]-T1.png        (80/20 split)
- Test     : patches extracted from variable-text images [ID]-T2-*.png
             (unseen text content -> tests true writer-style generalization)

Directory layout expected:
    ../MPHD/[Writer_ID_Code]/[ID]-T1.png
    ../MPHD/[Writer_ID_Code]/[ID]-T2-*.png

Run:
    python mphd_writer_id_benchmark.py
"""

import os
import re
import glob
import copy
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. CONFIGURATION
# =========================================================================

DATA_ROOT = "../MPHD"            # root folder containing [Writer_ID_Code]/*.png
NUM_WRITERS  = 500
PATCH_SIZE   = 128                # square patch side length
PATCH_STRIDE = 64                 # 128 = non-overlapping; smaller = sliding window overlap
INK_AREA_MIN_RATIO = 0.05         # discard patch if dark/ink pixels < 10% of patch area
INK_PIXEL_THRESHOLD = 128         # grayscale value below which a pixel counts as "ink"

MODEL_NAME = "resnet18"           # "resnet18" or "densenet121"
PRETRAINED = True
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
EARLY_STOPPING_PATIENCE = 10
TRAIN_VAL_SPLIT_RATIO = 0.80      # 80% train / 20% val, split over T1 patches

RANDOM_SEED = 42
NUM_WORKERS = 4
CHECKPOINT_PATH = "./writer_id_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 2. REPRODUCIBILITY
# =========================================================================

def set_seed(seed: int) -> None:
    """Set deterministic seeds across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# 3. DATASET INDEXING (writer folders -> T1 / T2 file lists)
# =========================================================================

def index_writer_documents(data_root: str, num_writers: int):
    """
    Recursively scan data_root for writer folders, each containing one T1
    (fixed text) image and one T2 (variable text) images.

    Returns
    -------
    writer_to_label : dict[str, int]   writer folder name -> class label (0..num_writers-1)
    t1_docs : list[(filepath, label)]  one entry per writer's T1 document
    t2_docs : list[(filepath, label)]  one entry per T2 document
    """
    writer_dirs = sorted(
        d for d in glob.glob(os.path.join(data_root, "*"))
        if os.path.isdir(d)
    )
    if len(writer_dirs) == 0:
        raise RuntimeError(f"No writer subfolders found under '{data_root}'.")

    writer_to_label = {os.path.basename(d): idx for idx, d in enumerate(writer_dirs)}

    t1_re = re.compile(r"-T1\.png$", re.IGNORECASE)
    t2_re = re.compile(r"-T2(-.*)?\.png$", re.IGNORECASE)

    t1_docs = []
    t2_docs = []

    for wdir in writer_dirs:
        wname = os.path.basename(wdir)
        label = writer_to_label[wname]

        all_pngs = glob.glob(os.path.join(wdir, "*.png"))
        t1_files = [f for f in all_pngs if t1_re.search(os.path.basename(f))]
        t2_files = [f for f in all_pngs if t2_re.search(os.path.basename(f))]

        for f in t1_files:
            t1_docs.append((f, label))
        for f in t2_files:
            t2_docs.append((f, label))

    if len(t1_docs) == 0:
        raise RuntimeError("No T1 (fixed text) documents found. Check filename pattern '*-T1.png'.")
    if len(t2_docs) == 0:
        raise RuntimeError("No T2 (variable text) documents found. Check filename pattern '*-T2*.png'.")

    print(f"[index_writer_documents] Writers found: {len(writer_to_label)}")
    print(f"[index_writer_documents] T1 (train/val) documents: {len(t1_docs)}")
    print(f"[index_writer_documents] T2 (test) documents: {len(t2_docs)}")

    if len(writer_to_label) != num_writers:
        print(f"[index_writer_documents] WARNING: expected {num_writers} writers, "
              f"found {len(writer_to_label)}. Continuing with actual count.")

    return writer_to_label, t1_docs, t2_docs


# =========================================================================
# 4. PATCH EXTRACTION (sliding window + background filtering)
# =========================================================================

def extract_patches_from_image(image: Image.Image, patch_size: int, stride: int,
                                 ink_area_min_ratio: float, ink_pixel_threshold: int):
    """
    Slide a (patch_size x patch_size) window across a grayscale document image
    and return a list of PIL.Image patches that contain enough ink (foreground)
    to be considered informative (i.e. not mostly blank background).

    Handles the last row/column by clamping the window so the full image area
    is covered even when dimensions are not exact multiples of patch_size.
    """
    gray = image.convert("L")
    width, height = gray.size
    np_img = np.array(gray)

    patches = []

    if width < patch_size or height < patch_size:
        # Document smaller than a patch: pad up to patch_size with white background.
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
                continue  # mostly blank background, discard
            patch_img = Image.fromarray(patch_np, mode="L")
            patches.append(patch_img)

    return patches


def stratified_patch_split(patch_records, split_ratio, seed):
    """
    Split a list of (filepath, label, PIL_patch) records into train/val sets
    using a PER-WRITER (stratified) split over patches.

    This is required when each writer contributes only a single T1 document:
    splitting at the document level would exclude entire writer classes from
    either train or val. Instead, patches belonging to each writer's T1
    document are grouped by label, shuffled, and split individually so every
    writer (class) is guaranteed to have patches on both sides of the split.

    Note: because all patches for a given writer come from that writer's one
    T1 document, some structural leakage (pen thickness, paper texture)
    between a writer's train and val patches is unavoidable here -- this is
    an accepted trade-off of the 1-document-per-writer T1 protocol. True
    generalization is still measured on the held-out T2 (unseen text) set.
    """
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for rec in patch_records:
        _, label, _ = rec
        by_label[label].append(rec)

    train_records = []
    val_records = []
    empty_val_writers = []

    for label, records in by_label.items():
        records = records.copy()
        rng.shuffle(records)
        split_idx = int(len(records) * split_ratio)

        # Guarantee at least one patch on each side when a writer has >= 2 patches.
        if len(records) >= 2:
            split_idx = min(max(split_idx, 1), len(records) - 1)
        else:
            empty_val_writers.append(label)

        train_records.extend(records[:split_idx])
        val_records.extend(records[split_idx:])

    if empty_val_writers:
        print(f"[stratified_patch_split] WARNING: {len(empty_val_writers)} writers had only 1 patch "
              f"and could not be represented in both train and val: {empty_val_writers[:10]}"
              f"{' ...' if len(empty_val_writers) > 10 else ''}")

    n_train_writers = len(set(r[1] for r in train_records))
    n_val_writers = len(set(r[1] for r in val_records))
    print(f"[stratified_patch_split] Train patches: {len(train_records)} ({n_train_writers} writers) | "
          f"Val patches: {len(val_records)} ({n_val_writers} writers)")

    return train_records, val_records


def build_patch_index(documents, patch_size, stride, ink_ratio, ink_thresh, desc=""):
    """
    Given a list of (filepath, label) documents, extract all valid patches
    from every document and return:
        patch_records: list of (filepath, label, patch_index_within_doc)
    Patches are re-extracted lazily at __getitem__ time by the Dataset
    (only coordinates/metadata are stored here to keep memory usage low);
    however for simplicity and speed at moderate dataset sizes, this function
    extracts patches eagerly and returns them alongside metadata.
    """
    patch_records = []  # (filepath, label, patch_pil_image)
    for fpath, label in documents:
        try:
            image = Image.open(fpath)
        except Exception as e:
            print(f"[build_patch_index] WARNING: failed to open '{fpath}': {e}")
            continue
        patches = extract_patches_from_image(image, patch_size, stride, ink_ratio, ink_thresh)
        for p in patches:
            patch_records.append((fpath, label, p))

    print(f"[build_patch_index] {desc}: {len(documents)} documents -> {len(patch_records)} valid patches.")
    return patch_records


# =========================================================================
# 5. TRANSFORMS
# =========================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    """
    Build the torchvision transform pipeline. Grayscale patches are converted
    to 3-channel (RGB) and normalized with standard ImageNet statistics so
    that ImageNet-pretrained backbones can be used directly.
    """
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# =========================================================================
# 6. DATASETS
# =========================================================================

class PatchDataset(Dataset):
    """
    Wraps a list of (filepath, label, PIL_patch) records for training/validation.
    Patches are already extracted in memory; only the transform is applied
    on-the-fly (so training augmentation differs every epoch).
    """

    def __init__(self, patch_records, transform):
        self.records = patch_records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        _, label, patch_img = self.records[idx]
        patch_tensor = self.transform(patch_img)
        return patch_tensor, label


class DocumentPatchDataset(Dataset):
    """
    Wraps patches for TEST documents, additionally tracking which document
    (index into `documents`) each patch belongs to, so predictions can later
    be aggregated at the document level.
    """

    def __init__(self, patch_records_with_doc_id, transform):
        # patch_records_with_doc_id: list of (doc_id, label, PIL_patch)
        self.records = patch_records_with_doc_id
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        doc_id, label, patch_img = self.records[idx]
        patch_tensor = self.transform(patch_img)
        return patch_tensor, label, doc_id


def build_test_patch_records(documents, patch_size, stride, ink_ratio, ink_thresh):
    """
    Extract patches for the test (T2) documents, tagging each patch with its
    source document index (doc_id) for later document-level aggregation.
    Also returns doc_id -> label mapping and doc_id -> filepath mapping.
    """
    records = []
    doc_id_to_label = {}
    doc_id_to_path = {}

    for doc_id, (fpath, label) in enumerate(documents):
        doc_id_to_label[doc_id] = label
        doc_id_to_path[doc_id] = fpath
        try:
            image = Image.open(fpath)
        except Exception as e:
            print(f"[build_test_patch_records] WARNING: failed to open '{fpath}': {e}")
            continue
        patches = extract_patches_from_image(image, patch_size, stride, ink_ratio, ink_thresh)
        for p in patches:
            records.append((doc_id, label, p))

    print(f"[build_test_patch_records] {len(documents)} test documents -> {len(records)} valid patches.")
    return records, doc_id_to_label, doc_id_to_path


# =========================================================================
# 7. MODEL FACTORY
# =========================================================================

def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision CNN backbone with its head replaced for num_classes."""
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


# =========================================================================
# 8. TRAINING & VALIDATION LOOP (with early stopping + checkpointing)
# =========================================================================

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
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
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
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, num_epochs, lr, weight_decay,
                 patience, device, checkpoint_path):
    """Full training loop with AdamW, early stopping, and best-checkpoint saving on min val loss."""
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

        print(f"Epoch [{epoch:03d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_model_state, checkpoint_path)
            print(f"  -> New best model saved (val_loss={val_loss:.4f}) to '{checkpoint_path}'")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs "
                      f"(no val_loss improvement for {patience} epochs).")
                break

    model.load_state_dict(best_model_state)
    return model


# =========================================================================
# 9. TEST-SET EVALUATION: PATCH-LEVEL + DOCUMENT-LEVEL AGGREGATION
# =========================================================================

@torch.no_grad()
def run_inference_on_test_patches(model, loader, device, num_classes):
    """
    Run inference over all test patches, returning per-patch softmax
    probabilities, predicted labels, true labels, and the doc_id each patch
    belongs to.
    """
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_doc_ids = []

    for images, labels, doc_ids in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_doc_ids.append(doc_ids.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_doc_ids = np.concatenate(all_doc_ids, axis=0)

    return all_probs, all_preds, all_labels, all_doc_ids


def compute_patch_level_accuracy(all_preds, all_labels):
    return float(np.mean(all_preds == all_labels))


def aggregate_document_predictions(all_probs, all_preds, all_labels, all_doc_ids, doc_id_to_label):
    """
    Aggregate patch-level predictions to document-level using:
      (a) Softmax probability averaging (mean prob per class across patches -> argmax)
      (b) Majority voting (most frequent predicted class among patches)

    Returns dicts: doc_id -> (avg_prob_vector, majority_vote_label, true_label)
    """
    doc_probs_sum = defaultdict(lambda: None)
    doc_probs_count = defaultdict(int)
    doc_votes = defaultdict(list)

    num_classes = all_probs.shape[1]

    for i in range(len(all_doc_ids)):
        doc_id = int(all_doc_ids[i])
        prob_vec = all_probs[i]
        pred_label = int(all_preds[i])

        if doc_probs_sum[doc_id] is None:
            doc_probs_sum[doc_id] = np.zeros(num_classes, dtype=np.float64)
        doc_probs_sum[doc_id] += prob_vec
        doc_probs_count[doc_id] += 1
        doc_votes[doc_id].append(pred_label)

    doc_results = {}
    for doc_id in doc_probs_sum.keys():
        avg_prob = doc_probs_sum[doc_id] / doc_probs_count[doc_id]
        majority_label = Counter(doc_votes[doc_id]).most_common(1)[0][0]
        true_label = doc_id_to_label[doc_id]
        doc_results[doc_id] = {
            "avg_prob": avg_prob,
            "majority_vote_label": majority_label,
            "true_label": true_label,
        }

    return doc_results


def compute_document_level_accuracy(doc_results, top_k=1, method="avg_prob"):
    """
    Compute document-level Top-K accuracy using either:
      method="avg_prob"      -> ranks classes by averaged softmax probability
      method="majority_vote" -> only supports top-1 (single predicted class)
    """
    correct = 0
    total = 0

    for doc_id, res in doc_results.items():
        true_label = res["true_label"]
        total += 1

        if method == "avg_prob":
            top_k_labels = np.argsort(-res["avg_prob"])[:top_k]
            if true_label in top_k_labels:
                correct += 1
        elif method == "majority_vote":
            if top_k != 1:
                raise ValueError("majority_vote aggregation only supports top_k=1.")
            if res["majority_vote_label"] == true_label:
                correct += 1
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return correct / total if total > 0 else 0.0


def compute_patch_level_topk_accuracy(all_probs, all_labels, top_k=1):
    """
    Compute Top-K accuracy at the patch level using softmax probabilities.
    """
    correct = 0
    total = len(all_labels)

    for i in range(total):
        probs = all_probs[i]
        true_label = all_labels[i]
        top_k_labels = np.argsort(-probs)[:top_k]  # indices of top K probabilities
        if true_label in top_k_labels:
            correct += 1

    return correct / total if total > 0 else 0.0

# =========================================================================
# 10. MAIN
# =========================================================================

def main():
    set_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    # ---- Index writer documents (T1 = train/val, T2 = test) --------------
    writer_to_label, t1_docs, t2_docs = index_writer_documents(DATA_ROOT, NUM_WRITERS)
    num_classes = len(writer_to_label)

    # ---- Extract ALL T1 patches, then split per-writer (stratified) 80/20 ----
    # Each writer contributes exactly ONE T1 document in this protocol, so a
    # document-level split would exclude entire writer classes from train or
    # val (as observed: 100/500 writers missing -> Val Acc 0.0). Instead we
    # extract all patches from every T1 document first, then split patches
    # per writer so all 500 classes are represented in both train and val.
    t1_patch_records = build_patch_index(
        t1_docs, PATCH_SIZE, PATCH_STRIDE, INK_AREA_MIN_RATIO, INK_PIXEL_THRESHOLD,
        desc="T1 (all, pre-split)"
    )
    train_records, val_records = stratified_patch_split(
        t1_patch_records, TRAIN_VAL_SPLIT_RATIO, RANDOM_SEED
    )

    train_transform = build_transforms(train=True)
    eval_transform = build_transforms(train=False)

    train_dataset = PatchDataset(train_records, train_transform)
    val_dataset = PatchDataset(val_records, eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ---- Build & train model ----------------------------------------------
    model = build_model(MODEL_NAME, num_classes, PRETRAINED).to(DEVICE)

    model = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        patience=EARLY_STOPPING_PATIENCE, device=DEVICE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # ---- Extract patches from T2 (test) documents -------------------------
    test_records, doc_id_to_label, doc_id_to_path = build_test_patch_records(
        t2_docs, PATCH_SIZE, PATCH_STRIDE, INK_AREA_MIN_RATIO, INK_PIXEL_THRESHOLD
    )

    test_dataset = DocumentPatchDataset(test_records, eval_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ---- Run inference & aggregate -----------------------------------------
    all_probs, all_preds, all_labels, all_doc_ids = run_inference_on_test_patches(
        model, test_loader, DEVICE, num_classes
    )

    patch_top1_acc = compute_patch_level_accuracy(all_preds, all_labels)
    patch_top5_acc = compute_patch_level_topk_accuracy(all_probs, all_labels, top_k=5)

    doc_results = aggregate_document_predictions(
        all_probs, all_preds, all_labels, all_doc_ids, doc_id_to_label
    )

    doc_top1_avgprob_acc = compute_document_level_accuracy(doc_results, top_k=1, method="avg_prob")
    doc_top5_avgprob_acc = compute_document_level_accuracy(doc_results, top_k=5, method="avg_prob")
    # doc_top1_majority_acc = compute_document_level_accuracy(doc_results, top_k=1, method="majority_vote")

    # بعد از محاسبه‌ی doc_top1_avgprob_acc و doc_top5_avgprob_acc:
    doc_top10_avgprob_acc = compute_document_level_accuracy(doc_results, top_k=10, method="avg_prob")
    doc_top20_avgprob_acc = compute_document_level_accuracy(doc_results, top_k=20, method="avg_prob")

    # ---- Report -------------------------------------------------------------
    print("\n===== WRITER IDENTIFICATION TEST RESULTS (T2, unseen text) =====")
    print(f"Patch-level Top-1 Accuracy                 : {patch_top1_acc:.4f}")
    print(f"Patch-level Top-5 Accuracy                 : {patch_top5_acc:.4f}")
    # print(f"Document-level Top-1 Accuracy (majority vote): {doc_top1_majority_acc:.4f}")
    print(f"Document-level Top-1 Accuracy (avg prob)    : {doc_top1_avgprob_acc:.4f}")
    print(f"Document-level Top-5 Accuracy (avg prob)    : {doc_top5_avgprob_acc:.4f}")
    print(f"Document-level Top-10 Accuracy (avg prob)   : {doc_top10_avgprob_acc:.4f}")
    print(f"Document-level Top-20 Accuracy (avg prob)   : {doc_top20_avgprob_acc:.4f}")
    print("==================================================================\n")

    # ---- Generate CMC Curve -------------------------------------------------
    def compute_cmc_curve(doc_results, max_k=50):
        """
        Compute CMC curve values up to max_k.
        doc_results: dict from aggregate_document_predictions
        returns: list of identification rates for k=1..max_k
        """
        ranks = []
        for doc_id, res in doc_results.items():
            true_label = res["true_label"]
            avg_prob = res["avg_prob"]
            sorted_indices = np.argsort(-avg_prob)  # descending
            rank = np.where(sorted_indices == true_label)[0][0] + 1  # 1-indexed
            ranks.append(rank)

        cmc = []
        for k in range(1, max_k + 1):
            acc = np.mean(np.array(ranks) <= k)
            cmc.append(acc)
        return cmc

    max_k = 50  # could be 100 or even up to 500
    cmc_values = compute_cmc_curve(doc_results, max_k)

    # ---- Print selected CMC points for the paper ----------------------------
    print("\n===== CMC SELECTED POINTS =====")
    for k in [1, 5, 10, 20, 30, 50]:
        if k <= max_k:
            print(f"Rank {k:2d}: {cmc_values[k - 1]:.4f}")

    # ---- Plot and save CMC curve --------------------------------------------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), cmc_values, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel('Rank (Top-K)')
    plt.ylabel('Identification Rate')
    plt.title('CMC Curve - Writer Identification on MPHD Dataset')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max_k + 5)
    plt.ylim(0, 1.05)
    plt.xticks(range(0, max_k + 1, 5))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig('cmc_curve.png', dpi=300, bbox_inches='tight')
    print(f"\n[main] CMC curve saved to 'cmc_curve.png'")

    print("[main] Done.")


if __name__ == "__main__":
    main()
