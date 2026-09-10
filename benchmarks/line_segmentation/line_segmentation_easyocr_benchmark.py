"""
MPHD Line Segmentation Benchmark (EasyOCR baseline)

Reproduces the line segmentation results reported in the MPHD dataset paper
(Section: Line Segmentation). Given the full-form / cropped text-region
images and their per-writer "-Lines" ground-truth crop folders, this script:

  1. Recovers ground-truth line bounding boxes by SIFT feature matching each
     ground-truth line crop back onto the source image (MPHD ships line crops,
     not line boxes, so this step re-derives the box geometry needed for IoU).
  2. Runs EasyOCR as an off-the-shelf line-detection baseline and groups its
     word/box detections into lines via simple y-coordinate clustering.
  3. Matches predicted lines to ground-truth lines (greedy, IoU-ranked) and
     reports mIoU(all), mIoU(matched), F1@0.50, F1@0.75, Line Count Accuracy
     (LCA), relative line-count error, and mean SSIM.

Supports the two evaluation protocols described in the paper:
  - Protocol 1: all 500 writers (upper-bound / general-purpose baseline).
  - Protocol 2: official held-out test split (writers 426-500, 70/15/15 split).

Usage:
    python line_segmentation_easyocr_benchmark.py
"""

import os
import cv2
import easyocr
import numpy as np
import json
from shapely.geometry import Polygon
from skimage.metrics import structural_similarity as ssim

# ==================== Config ====================
DATASET_ROOT = "../MPHD/"
PROTOCOL = 2  # 1 = all 500 writers, 2 = official held-out test split (writers 426-500)

OUTPUT_VISUALIZATION_DIR = "visualization_results"
OUTPUT_RESULTS_FILE = "evaluation_official_test.json"
Y_TOLERANCE = 10
USE_GPU = True

# ==================== 1. Load EasyOCR ====================
print("Loading EasyOCR...")
reader = easyocr.Reader(['fa'], gpu=USE_GPU)
os.makedirs(OUTPUT_VISUALIZATION_DIR, exist_ok=True)


# ==================== 2. Detection / matching / metric helpers ====================
def group_boxes_into_lines(ocr_results, y_tolerance=10):
    """Cluster EasyOCR word boxes into text lines using running y-center distance."""
    if not ocr_results:
        return []
    boxes_with_y = []
    for bbox, _, _ in ocr_results:
        ys = [point[1] for point in bbox]
        center_y = sum(ys) / len(ys)
        boxes_with_y.append((center_y, bbox))
    boxes_with_y.sort(key=lambda x: x[0])
    lines = []
    current_line = []
    current_y_avg = None
    for center_y, bbox in boxes_with_y:
        if current_y_avg is None:
            current_line = [bbox]
            current_y_avg = center_y
        elif abs(center_y - current_y_avg) <= y_tolerance:
            current_line.append(bbox)
            current_y_avg = (current_y_avg + center_y) / 2
        else:
            lines.append(current_line)
            current_line = [bbox]
            current_y_avg = center_y
    if current_line:
        lines.append(current_line)
    return lines


def get_ground_truth_boxes_feature(original_img_path, crops_folder):
    """
    Recover ground-truth line bounding boxes on the source image.

    MPHD provides ground-truth lines as pre-cropped image files rather than
    box coordinates. To obtain a box usable for IoU, each ground-truth crop is
    located back on the source image via SIFT keypoint matching + RANSAC
    homography, and the crop's corners are projected into source-image space.
    """
    original = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        return []
    detector = cv2.SIFT_create()
    kp_orig, des_orig = detector.detectAndCompute(original, None)
    if des_orig is None:
        return []
    crop_files = [f for f in os.listdir(crops_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    crop_files.sort()
    gt_boxes = []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    for crop_file in crop_files:
        crop_path = os.path.join(crops_folder, crop_file)
        crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if crop is None:
            continue
        h, w = crop.shape
        kp_crop, des_crop = detector.detectAndCompute(crop, None)
        if des_crop is None or len(kp_crop) < 4:
            continue
        matches = bf.knnMatch(des_crop, des_orig, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        if len(good_matches) < 4:
            continue
        src_pts = np.float32([kp_crop[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_orig[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            continue
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)
        pts = transformed_corners.reshape(-1, 2)
        x_min, y_min = np.min(pts, axis=0).astype(int)
        x_max, y_max = np.max(pts, axis=0).astype(int)
        bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        gt_boxes.append(bbox)
    return gt_boxes


def get_line_bbox(line_boxes):
    all_x = [point[0] for box in line_boxes for point in box]
    all_y = [point[1] for box in line_boxes for point in box]
    if not all_x or not all_y:
        return None
    return [
        [min(all_x), min(all_y)],
        [max(all_x), min(all_y)],
        [max(all_x), max(all_y)],
        [min(all_x), max(all_y)]
    ]


def compute_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0


def compute_ssim(mask1, mask2):
    h = max(mask1.shape[0], mask2.shape[0])
    w = max(mask1.shape[1], mask2.shape[1])
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    m1[:mask1.shape[0], :mask1.shape[1]] = mask1
    m2[:mask2.shape[0], :mask2.shape[1]] = mask2
    return ssim(m1, m2, data_range=255)


def create_line_mask(img_shape, bbox):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if bbox is None:
        return mask
    pts = np.array(bbox, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def visualize_segmentation(img, gt_boxes, pred_lines, img_name):
    vis = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)]
    for idx, line in enumerate(pred_lines):
        color = colors[idx % len(colors)]
        bbox = get_line_bbox(line)
        if bbox is None:
            continue
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], True, color, 2)
        x, y = int(bbox[0][0]), int(bbox[0][1]) - 5
        if y < 0:
            y = 5
        cv2.putText(vis, f"L{idx + 1}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    output_path = os.path.join(OUTPUT_VISUALIZATION_DIR, f"{img_name}_seg.png")
    cv2.imwrite(output_path, vis)


# ==================== 3. Dataset split builders for the two protocols ====================
def create_or_load_split_protocol1(root_path):
    """
    Protocol 1: evaluate on all 500 writers (no train/test separation).
    Returns every writer folder as the test set.
    """
    print("\n=== Running Protocol 1: Evaluating on ALL 500 Writers ===")
    all_folders = [d for d in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, d)) and d[0].isdigit()]
    all_folders.sort()

    split = {
        "train": [],  # not used in this protocol
        "val": [],  # not used in this protocol
        "test": all_folders
    }
    print(f"Total writers loaded for Protocol 1: {len(split['test'])}")
    return split


def create_or_load_split_protocol2(root_path):
    """
    Protocol 2: evaluate on the official 15% held-out test split
    (writers 426-500), following the 70/15/15 split ordered by writer ID.
    """
    split_file = os.path.join(root_path, "official_split_protocol2.json")

    if os.path.exists(split_file):
        print(f"Loading existing split for Protocol 2 from {split_file}")
        with open(split_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    print("Creating official dataset split for Protocol 2 (70% Train, 15% Val, 15% Test)...")
    all_folders = [d for d in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, d)) and d[0].isdigit()]

    # Sort by numeric writer ID so writers 426-500 land in the test split
    def extract_writer_id(folder_name):
        return int(folder_name.split('-')[0]) if '-' in folder_name else int(folder_name)

    all_folders.sort(key=extract_writer_id)

    total = len(all_folders)
    train_end = int(0.7 * total)   # 350
    val_end = int(0.85 * total)    # 425
    val_end = min(val_end, total)

    split = {
        "train": all_folders[:train_end],
        "val": all_folders[train_end:val_end],
        "test": all_folders[val_end:]   # writers 426-500
    }

    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2, ensure_ascii=False)

    print(f"Split created: Train={len(split['train'])}, Val={len(split['val'])}, Test={len(split['test'])}")
    print(f"Test set writers: {split['test'][0]} to {split['test'][-1]}")
    return split


# ==================== 4. Main evaluation loop ====================
def evaluate_dataset(root_path, y_tolerance, split):
    all_image_results = []
    all_line_ious = []
    matched_line_ious = []
    all_f1_50 = []
    all_f1_75 = []
    all_ssims = []
    count_accuracy = 0
    rel_count_errors = []
    total_images = 0
    test_writers = split["test"]

    # Sanity check: flag writers with an unexpected number of text images before running
    missing_files = []
    for writer_folder in test_writers:
        writer_path = os.path.join(root_path, writer_folder)
        image_files = [f for f in os.listdir(writer_path)
                       if f.endswith('.png') or f.endswith('.jpg')]
        text_images = [f for f in image_files if 'T' in f and not f.endswith('-Lines')]
        if len(text_images) != 2:
            missing_files.append((writer_folder, text_images))

    if missing_files:
        print("\n⚠️ WARNING: Some writers have missing text images:")
        for writer, files in missing_files:
            print(f"   {writer}: {files}")
        print("Please fix these before proceeding.\n")

    print(f"\n=== Evaluating on Test Set ({len(test_writers)} writers) ===\n")

    for writer_folder in test_writers:
        writer_path = os.path.join(root_path, writer_folder)
        image_files = [f for f in os.listdir(writer_path)
                       if f.endswith('.png') or f.endswith('.jpg')]
        text_images = [f for f in image_files if 'T' in f and not f.endswith('-Lines')]

        for img_name in text_images:
            base_name = os.path.splitext(img_name)[0]
            lines_folder = os.path.join(writer_path, base_name + "-Lines")

            if not os.path.exists(lines_folder):
                continue

            img_path = os.path.join(writer_path, img_name)
            print(f"\nProcessing: {writer_folder}/{img_name}")
            total_images += 1

            # --- Ground truth ---
            gt_boxes = get_ground_truth_boxes_feature(img_path, lines_folder)
            if not gt_boxes:
                print(f"  No GT boxes found. Skipping.")
                continue
            n_gt = len(gt_boxes)

            # --- Prediction ---
            img = cv2.imread(img_path)
            ocr_results = reader.readtext(img)
            pred_lines = group_boxes_into_lines(ocr_results, y_tolerance)
            n_pred = len(pred_lines)

            # --- IoU matrix ---
            iou_matrix = np.zeros((n_gt, n_pred))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_line in enumerate(pred_lines):
                    pred_bbox = get_line_bbox(pred_line)
                    iou_matrix[i, j] = compute_iou(gt_box, pred_bbox)

            # --- Greedy one-to-one matching, highest IoU first ---
            matched_pairs = []
            used_gt = set()
            used_pred = set()
            pairs = [(i, j, iou_matrix[i, j]) for i in range(n_gt) for j in range(n_pred)]
            pairs.sort(key=lambda x: x[2], reverse=True)
            for i, j, iou in pairs:
                if i not in used_gt and j not in used_pred:
                    matched_pairs.append((i, j, iou))
                    used_gt.add(i)
                    used_pred.add(j)

            # --- Metrics ---
            all_ious_for_img = []
            for i in range(n_gt):
                best_iou = max(iou_matrix[i, :]) if n_pred > 0 else 0.0
                all_ious_for_img.append(best_iou)
            for j in range(n_pred):
                if j not in used_pred:
                    all_ious_for_img.append(0.0)
            mean_iou_all = np.mean(all_ious_for_img) if all_ious_for_img else 0.0
            all_line_ious.extend(all_ious_for_img)

            matched_ious = [iou for _, _, iou in matched_pairs]
            mean_iou_matched = np.mean(matched_ious) if matched_ious else 0.0
            matched_line_ious.extend(matched_ious)

            tp_50 = sum(1 for _, _, iou in matched_pairs if iou >= 0.50)
            tp_75 = sum(1 for _, _, iou in matched_pairs if iou >= 0.75)
            fp_50 = n_pred - tp_50
            fn_50 = n_gt - tp_50
            precision_50 = tp_50 / (tp_50 + fp_50) if (tp_50 + fp_50) > 0 else 0.0
            recall_50 = tp_50 / (tp_50 + fn_50) if (tp_50 + fn_50) > 0 else 0.0
            f1_50 = 2 * precision_50 * recall_50 / (precision_50 + recall_50) if (precision_50 + recall_50) > 0 else 0.0
            all_f1_50.append(f1_50)

            fp_75 = n_pred - tp_75
            fn_75 = n_gt - tp_75
            precision_75 = tp_75 / (tp_75 + fp_75) if (tp_75 + fp_75) > 0 else 0.0
            recall_75 = tp_75 / (tp_75 + fn_75) if (tp_75 + fn_75) > 0 else 0.0
            f1_75 = 2 * precision_75 * recall_75 / (precision_75 + recall_75) if (precision_75 + recall_75) > 0 else 0.0
            all_f1_75.append(f1_75)

            if n_pred == n_gt:
                count_accuracy += 1

            rel_err = abs(n_pred - n_gt) / n_gt if n_gt > 0 else 0.0
            rel_count_errors.append(rel_err)

            ssims = []
            for i, j, _ in matched_pairs:
                gt_mask = create_line_mask(img.shape, gt_boxes[i])
                pred_bbox = get_line_bbox(pred_lines[j])
                pred_mask = create_line_mask(img.shape, pred_bbox)
                ssims.append(compute_ssim(gt_mask, pred_mask))
            mean_ssim_img = np.mean(ssims) if ssims else 0.0
            all_ssims.extend(ssims)

            image_result = {
                "image": f"{writer_folder}/{img_name}",
                "num_gt": n_gt,
                "num_pred": n_pred,
                "mIoU_all": mean_iou_all,
                "mIoU_matched": mean_iou_matched,
                "F1@0.50": f1_50,
                "F1@0.75": f1_75,
                "LCA": 1 if n_pred == n_gt else 0,
                "RelCntErr": rel_err,
                "MeanSSIM": mean_ssim_img,
                "per_line_iou": matched_ious
            }
            all_image_results.append(image_result)

            print(f"  GT: {n_gt}, Pred: {n_pred}, mIoU: {mean_iou_all:.4f}, F1@0.5: {f1_50:.2f}")

            visualize_segmentation(img, gt_boxes, pred_lines, f"{writer_folder}_{base_name}")

    # --- Aggregate metrics ---
    final_metrics = {
        "mIoU(all)": np.mean(all_line_ious) if all_line_ious else 0.0,
        "mIoU(m)": np.mean(matched_line_ious) if matched_line_ious else 0.0,
        "F1@0.50": np.mean(all_f1_50) if all_f1_50 else 0.0,
        "F1@0.75": np.mean(all_f1_75) if all_f1_75 else 0.0,
        "LCA (%)": (count_accuracy / total_images * 100) if total_images > 0 else 0.0,
        "RelCntErr": np.mean(rel_count_errors) if rel_count_errors else 0.0,
        "MeanSSIM": np.mean(all_ssims) if all_ssims else 0.0
    }

    print(f"\n{'=' * 50}")
    print("FINAL RESULTS (OFFICIAL TEST SET)")
    print(f"{'=' * 50}")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")

    return all_image_results, final_metrics


# ==================== 5. Entry point ====================
if __name__ == "__main__":
    if PROTOCOL == 1:
        split = create_or_load_split_protocol1(DATASET_ROOT)   # all 500 writers used as test
    else:
        split = create_or_load_split_protocol2(DATASET_ROOT)   # only writers 426-500 (15%) used as test

    results, metrics = evaluate_dataset(DATASET_ROOT, Y_TOLERANCE, split)

    output = {
        "split_info": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"])
        },
        "per_image": results,
        "final_metrics": metrics
    }
    with open(OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {OUTPUT_RESULTS_FILE}")
    print(f"✅ Visualizations saved to {OUTPUT_VISUALIZATION_DIR}/")
