"""
mphd_htr_prepare_splits.py

Prepares MPHD line-level data for HTR training and reproduces the official
train/val/test split used in the MPHD paper (Section 4.1, Table 7).

STEP 1: Walk every writer directory [ID]-[CODE]/, read the writer's JSON file,
        copy every T1/T2 line image into a single flat folder (MPHD_Lines),
        write a matching .txt file with the transcription for every line image,
        and record whether each line belongs to T1 or T2 in lines_type.txt.

STEP 2: Using the flat MPHD_Lines folder, split all lines into train/val/test
        (70/15/15) at the WRITER level (seed=42), so lines from the same
        writer never end up in two different splits. Also produces
        test-t1.ln and test-t2.ln (disjoint subsets of test.ln, by text type).

Data split:
train=3,515 / val=767 / test=739 (test-t1=403 test-t2=336, see. Table 7 in the
dataset paper).


No command-line arguments. Just edit the CONFIG section below and run.
"""

import os
import json
import random
import shutil

# ============================== CONFIG ==============================
DATASET_ROOT = r"../../MPHD"       # root folder containing [ID]-[CODE]/ writer folders
OUTPUT_ROOT  = r"./"            # where MPHD_Lines/ and .ln files will be written
IMAGE_EXT    = ".png"           # extension of the line images on disk
RANDOM_SEED  = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ======================================================================
ALL_LINES_DIR = os.path.join(OUTPUT_ROOT, "MPHD_Lines")
TYPE_FILE = os.path.join(OUTPUT_ROOT, "lines_type.txt")


def find_writer_dirs(dataset_root):
    """Return sorted list of writer directory names directly under dataset_root."""
    entries = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]
    entries.sort()
    return entries


def find_writer_json(writer_dir_path):
    """Find the single .json file inside a writer directory."""
    json_files = [f for f in os.listdir(writer_dir_path) if f.lower().endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {writer_dir_path}")
    return os.path.join(writer_dir_path, json_files[0])


def step1_extract_lines():
    """Copy all T1/T2 line images into MPHD_Lines/, write transcription .txt files,
    and record the T1/T2 type of every line image."""
    os.makedirs(ALL_LINES_DIR, exist_ok=True)

    writer_dirs = find_writer_dirs(DATASET_ROOT)

    global_idx = 0
    type_records = []  # list of (name, "T1"/"T2")

    for writer_dir_name in writer_dirs:
        writer_dir_path = os.path.join(DATASET_ROOT, writer_dir_name)
        json_path = find_writer_json(writer_dir_path)

        # The real writer ID (e.g. "050" from "050-173010_3") is preserved in
        # every output filename, rather than a sequential counter, so lines
        # can always be traced back to metadata.csv / the writer's folder.
        writer_id = writer_dir_name.split("-")[0]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process text1 (T1) lines then text2 (T2) lines, in that order
        for text_key, type_label in (("text1", "T1"), ("text2", "T2")):
            text_block = data.get(text_key)
            if not text_block:
                print(f"WARNING: {writer_dir_name} has no '{text_key}' block, skipped")
                continue

            lines_block = text_block.get("lines", {})
            line_images = lines_block.get("images")
            line_texts = lines_block.get("text")

            if line_images is None:
                print(f"WARNING: {writer_dir_name}/{text_key} has no 'lines.images' field, "
                      f"skipped. VERIFY this key name against your actual JSON schema "
                      f"before running on the full dataset.")
                continue
            if line_texts is None:
                print(f"WARNING: {writer_dir_name}/{text_key} has no 'lines.text' field, skipped")
                continue
            if len(line_images) != len(line_texts):
                print(f"WARNING: {writer_dir_name}/{text_key} images/text count mismatch "
                      f"({len(line_images)} images vs {len(line_texts)} texts)")

            for img_rel_path, line_text in zip(line_images, line_texts):
                name = f"{writer_id}_{global_idx:06d}"

                src_img = os.path.join(writer_dir_path, img_rel_path + IMAGE_EXT)
                dst_img = os.path.join(ALL_LINES_DIR, name + IMAGE_EXT)
                dst_txt = os.path.join(ALL_LINES_DIR, name + ".txt")

                if not os.path.isfile(src_img):
                    print(f"WARNING: missing image, skipped: {src_img}")
                    continue

                shutil.copyfile(src_img, dst_img)
                with open(dst_txt, "w", encoding="utf-8") as tf:
                    tf.write(line_text)

                type_records.append((name, type_label))
                global_idx += 1

    with open(TYPE_FILE, "w", encoding="utf-8") as f:
        for name, type_label in type_records:
            f.write(f"{name}\t{type_label}\n")

    print(f"Step 1 done: {global_idx} line images extracted into {ALL_LINES_DIR}")
    print(f"(Any writer/text/line skipped above via WARNING explains a gap "
          f"between this number and the total line count reported in the JSON files.)")
    return type_records


def load_type_records():
    """Load (name, type) records from TYPE_FILE (used if step1 was already run before)."""
    records = []
    with open(TYPE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, type_label = line.split("\t")
            records.append((name, type_label))
    return records


def step2_split_dataset(type_records):
    """Split lines into train/val/test by writer, and split test into test-t1/test-t2 by type."""
    # group line names by writer id (the prefix of "050_000000" before the "_")
    writer_to_names = {}
    name_to_type = {}
    for name, type_label in type_records:
        writer_id = name.split("_")[0]
        writer_to_names.setdefault(writer_id, []).append(name)
        name_to_type[name] = type_label

    writer_ids = list(writer_to_names.keys())
    random.Random(RANDOM_SEED).shuffle(writer_ids)

    n_writers = len(writer_ids)
    n_train = int(round(n_writers * TRAIN_RATIO))
    n_val = int(round(n_writers * VAL_RATIO))
    # remaining writers go to test (avoids rounding losing/duplicating writers)
    train_writers = writer_ids[:n_train]
    val_writers = writer_ids[n_train:n_train + n_val]
    test_writers = writer_ids[n_train + n_val:]

    def names_for(writers):
        names = []
        for w in writers:
            names.extend(writer_to_names[w])
        return names

    train_names = names_for(train_writers)
    val_names = names_for(val_writers)
    test_names = names_for(test_writers)

    def write_ln(path, names):
        with open(path, "w", encoding="utf-8") as f:
            for n in names:
                f.write(n + "\n")

    write_ln(os.path.join(OUTPUT_ROOT, "train.ln"), train_names)
    write_ln(os.path.join(OUTPUT_ROOT, "val.ln"), val_names)
    write_ln(os.path.join(OUTPUT_ROOT, "test.ln"), test_names)

    test_t1_names = [n for n in test_names if name_to_type[n] == "T1"]
    test_t2_names = [n for n in test_names if name_to_type[n] == "T2"]

    write_ln(os.path.join(OUTPUT_ROOT, "test-t1.ln"), test_t1_names)
    write_ln(os.path.join(OUTPUT_ROOT, "test-t2.ln"), test_t2_names)

    print(f"Step 2 done: {n_writers} writers -> "
          f"{len(train_writers)} train / {len(val_writers)} val / {len(test_writers)} test")
    print(f"Lines: {len(train_names)} train / {len(val_names)} val / {len(test_names)} test "
          f"({len(test_t1_names)} test-t1 / {len(test_t2_names)} test-t2)")
    print("Verify against the paper (Table 7): train=3,515 / val=767 / test=739 "
          "(test-t1=403 / test-t2=336).")


if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    type_records = step1_extract_lines()
    step2_split_dataset(type_records)
