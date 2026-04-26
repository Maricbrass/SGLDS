from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import pandas as pd


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".fits", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build train/val/test CSV splits for lens_detection from labeled Euclid cutouts."
        )
    )
    parser.add_argument(
        "--annotations",
        default="data/euclid_q1/labels.csv",
        help="CSV with columns image_path,label. image_path can be absolute or relative to --source-dir.",
    )
    parser.add_argument(
        "--source-dir",
        default="data/euclid_q1/cutouts",
        help="Directory used for relative image_path values.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/euclid_q1_dataset",
        help="Output dataset directory with images/ and split CSV files.",
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into out-dir/images. If not set, images are referenced in-place.",
    )
    parser.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Allow building splits even if labels only contain one class.",
    )
    return parser.parse_args()


def validate_fractions(train_frac: float, val_frac: float):
    if not (0 < train_frac < 1):
        raise ValueError("--train-frac must be between 0 and 1")
    if not (0 <= val_frac < 1):
        raise ValueError("--val-frac must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("--train-frac + --val-frac must be < 1")


def resolve_image_path(raw_path: str, source_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return source_dir / p


def stratified_split(rows: list[tuple[str, int]], train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)

    class_to_rows: dict[int, list[tuple[str, int]]] = {}
    for row in rows:
        class_to_rows.setdefault(int(row[1]), []).append(row)

    train_rows, val_rows, test_rows = [], [], []
    for label, cls_rows in class_to_rows.items():
        rng.shuffle(cls_rows)
        n = len(cls_rows)

        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        cls_train = cls_rows[:n_train]
        cls_val = cls_rows[n_train : n_train + n_val]
        cls_test = cls_rows[n_train + n_val :]

        train_rows.extend(cls_train)
        val_rows.extend(cls_val)
        test_rows.extend(cls_test)

        print(
            f"Label {label}: total={n} train={len(cls_train)} val={len(cls_val)} test={len(cls_test)}"
        )

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_csv(path: Path, rows: list[tuple[str, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)


def labels_in_rows(rows: list[tuple[str, int]]) -> set[int]:
    return {int(label) for _, label in rows}


def validate_training_compatibility(
    train_rows: list[tuple[str, int]],
    val_rows: list[tuple[str, int]],
    test_rows: list[tuple[str, int]],
    allow_single_class: bool,
):
    if allow_single_class:
        return

    if not train_rows:
        raise ValueError("Train split is empty. Add more labeled samples.")
    if not val_rows:
        raise ValueError(
            "Validation split is empty. Increase data size or reduce --train-frac/--val-frac."
        )
    if not test_rows:
        raise ValueError(
            "Test split is empty. Increase data size or adjust split fractions."
        )

    train_labels = labels_in_rows(train_rows)
    val_labels = labels_in_rows(val_rows)
    test_labels = labels_in_rows(test_rows)

    if train_labels != {0, 1}:
        raise ValueError(
            "Train split must contain both classes {0,1} for meaningful lens training."
        )
    if val_labels != {0, 1}:
        raise ValueError(
            "Validation split must contain both classes {0,1}; ROC-AUC requires both classes."
        )
    if test_labels != {0, 1}:
        raise ValueError(
            "Test split must contain both classes {0,1}; evaluation metrics require both classes."
        )


def main():
    args = parse_args()
    validate_fractions(args.train_frac, args.val_frac)

    annotations_path = Path(args.annotations)
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_images_dir = out_dir / "images"

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotations_path}. "
            "Create it with columns image_path,label (label: 0 non_lens, 1 lens)."
        )

    df = pd.read_csv(annotations_path)
    required = {"image_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{annotations_path} missing columns: {sorted(missing)}")

    resolved_rows: list[tuple[Path, int]] = []
    for _, row in df.iterrows():
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"Invalid label {label}. Use 0 for non_lens and 1 for lens.")

        src = resolve_image_path(str(row["image_path"]), source_dir)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        if src.suffix.lower() not in VALID_EXTS:
            raise ValueError(f"Unsupported image extension: {src.suffix}")

        resolved_rows.append((src, label))

    if not resolved_rows:
        raise ValueError("No labeled rows found in annotations file.")

    labels = sorted({label for _, label in resolved_rows})
    if len(labels) < 2 and not args.allow_single_class:
        raise ValueError(
            "Only one class found in labels. Add both classes (0 and 1) or use --allow-single-class for a dry run."
        )

    # Prepare destination image paths.
    rows_for_split: list[tuple[str, int]] = []
    out_images_dir.mkdir(parents=True, exist_ok=True)

    for src, label in resolved_rows:
        if args.copy_images:
            dst = out_images_dir / src.name
            if dst.exists() and dst.resolve() != src.resolve():
                stem = dst.stem
                suffix = dst.suffix
                k = 1
                while True:
                    candidate = out_images_dir / f"{stem}_{k}{suffix}"
                    if not candidate.exists():
                        dst = candidate
                        break
                    k += 1
            shutil.copy2(src, dst)
            rel_path = dst.name
        else:
            # Keep references in-place by absolute path.
            rel_path = str(src.resolve())

        rows_for_split.append((rel_path, label))

    train_rows, val_rows, test_rows = stratified_split(
        rows_for_split,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    validate_training_compatibility(
        train_rows,
        val_rows,
        test_rows,
        allow_single_class=args.allow_single_class,
    )

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)
    write_csv(out_dir / "test.csv", test_rows)

    if args.copy_images:
        root_dir = out_images_dir.resolve()
    else:
        # CSV stores absolute paths in this mode; root_dir can be current directory.
        root_dir = Path(".").resolve()

    print("\nDataset build complete.")
    print(f"annotations: {annotations_path}")
    print(f"root_dir:    {root_dir}")
    print(f"train.csv:   {out_dir / 'train.csv'} ({len(train_rows)} rows)")
    print(f"val.csv:     {out_dir / 'val.csv'} ({len(val_rows)} rows)")
    print(f"test.csv:    {out_dir / 'test.csv'} ({len(test_rows)} rows)")
    print("\nUse these files in your training config data section.")


if __name__ == "__main__":
    main()
