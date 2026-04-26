from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image


def draw_ring(img: np.ndarray):
    h, w = img.shape
    cy, cx = random.randint(h // 3, 2 * h // 3), random.randint(w // 3, 2 * w // 3)
    radius = random.randint(min(h, w) // 8, min(h, w) // 4)
    thickness = random.randint(2, 4)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (dist >= radius - thickness) & (dist <= radius + thickness)
    img[mask] += random.uniform(0.6, 1.0)


def draw_non_lens_blob(img: np.ndarray):
    h, w = img.shape
    cy, cx = random.randint(h // 4, 3 * h // 4), random.randint(w // 4, 3 * w // 4)
    sy, sx = random.randint(6, 20), random.randint(6, 20)

    yy, xx = np.ogrid[:h, :w]
    blob = np.exp(-(((yy - cy) ** 2) / (2 * sy * sy) + ((xx - cx) ** 2) / (2 * sx * sx)))
    img += blob * random.uniform(0.5, 1.0)


def save_png(arr: np.ndarray, out_path: Path):
    arr = np.clip(arr, 0.0, 1.0)
    rgb = np.stack([arr, arr, arr], axis=-1)
    img = (rgb * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def write_split_csv(csv_path: Path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)


def main():
    random.seed(42)
    np.random.seed(42)

    root = Path("data/euclid_like")
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    n_per_class = 240
    size = 96

    for i in range(n_per_class):
        arr = np.random.normal(0.0, 0.08, size=(size, size)).astype(np.float32)
        draw_non_lens_blob(arr)
        name = f"non_lens_{i:04d}.png"
        save_png(arr, image_dir / name)
        all_rows.append((name, 0))

    for i in range(n_per_class):
        arr = np.random.normal(0.0, 0.08, size=(size, size)).astype(np.float32)
        draw_ring(arr)
        name = f"lens_{i:04d}.png"
        save_png(arr, image_dir / name)
        all_rows.append((name, 1))

    random.shuffle(all_rows)
    n = len(all_rows)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_rows = all_rows[:n_train]
    val_rows = all_rows[n_train : n_train + n_val]
    test_rows = all_rows[n_train + n_val :]

    write_split_csv(root / "train.csv", train_rows)
    write_split_csv(root / "val.csv", val_rows)
    write_split_csv(root / "test.csv", test_rows)

    print(f"Created dataset in {root}")
    print(f"Train/Val/Test sizes: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")


if __name__ == "__main__":
    main()
