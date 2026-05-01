from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".fits", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/val/test CSV splits from a folder-labeled dataset."
    )
    parser.add_argument(
        "--source-dir",
        default="Training_data",
        help="Root directory containing class subfolders named 0 and 1.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/training_data_dataset",
        help="Output directory for split CSV files.",
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_fractions(train_frac: float, val_frac: float):
    if not (0 < train_frac < 1):
        raise ValueError("--train-frac must be between 0 and 1")
    if not (0 <= val_frac < 1):
        raise ValueError("--val-frac must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("--train-frac + --val-frac must be < 1")


def collect_rows(source_dir: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for label_dir in sorted(source_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        if label_dir.name not in {"0", "1"}:
            continue

        label = int(label_dir.name)
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in VALID_EXTS:
                continue
            rows.append((str(image_path.resolve()), label))

    return rows


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


def main():
    args = parse_args()
    validate_fractions(args.train_frac, args.val_frac)

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    rows = collect_rows(source_dir)
    if not rows:
        raise ValueError(
            f"No labeled images found in {source_dir}. Expected class folders named 0 and 1."
        )

    labels = sorted({label for _, label in rows})
    if labels != [0, 1]:
        raise ValueError(f"Expected both labels 0 and 1, got {labels}")

    train_rows, val_rows, test_rows = stratified_split(
        rows,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)
    write_csv(out_dir / "test.csv", test_rows)

    print("\nDataset build complete.")
    print(f"source_dir: {source_dir.resolve()}")
    print(f"train.csv:  {out_dir / 'train.csv'} ({len(train_rows)} rows)")
    print(f"val.csv:    {out_dir / 'val.csv'} ({len(val_rows)} rows)")
    print(f"test.csv:   {out_dir / 'test.csv'} ({len(test_rows)} rows)")


if __name__ == "__main__":
    main()