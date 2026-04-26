from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive CLI labeler for Euclid cutout PNG images. "
            "Keys: 1=lens, 0=non_lens, s=skip, q=quit"
        )
    )
    parser.add_argument("--images-dir", default="data/euclid_q1/cutouts")
    parser.add_argument("--labels-csv", default="data/euclid_q1/labels.csv")
    parser.add_argument(
        "--queue-csv",
        default="data/euclid_q1/label_queue.csv",
        help="Optional queue CSV from fetch_euclid_targets_batch.py; if present, only labels those images.",
    )
    parser.add_argument("--start-from", default="", help="Start labeling from this image filename.")
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Open each image before prompting. Disable if running headless.",
    )
    return parser.parse_args()


def load_labels(labels_csv: Path) -> pd.DataFrame:
    if labels_csv.exists():
        df = pd.read_csv(labels_csv)
    else:
        df = pd.DataFrame(columns=["image_path", "label"])

    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{labels_csv} must have columns image_path,label")

    # Keep latest label if duplicates exist.
    df = df.drop_duplicates(subset=["image_path"], keep="last").copy()
    return df


def gather_image_list(images_dir: Path, queue_csv: Path) -> list[str]:
    if queue_csv.exists():
        qdf = pd.read_csv(queue_csv)
        if "image_path" not in qdf.columns:
            raise ValueError(f"{queue_csv} must include image_path column")
        names = [str(x).strip() for x in qdf["image_path"].tolist() if str(x).strip()]
        return sorted(set(names))

    names = []
    for p in sorted(images_dir.glob("*.png")):
        names.append(p.name)
    return names


def maybe_show_image(image_path: Path, enable: bool):
    if not enable:
        return
    try:
        img = Image.open(image_path)
        img.show()
    except Exception as exc:
        print(f"Could not open image {image_path}: {exc}")


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_csv = Path(args.labels_csv)
    queue_csv = Path(args.queue_csv)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    image_names = gather_image_list(images_dir, queue_csv)
    if not image_names:
        raise ValueError("No images found for labeling.")

    labels_df = load_labels(labels_csv)
    label_map = {str(r["image_path"]): int(r["label"]) for _, r in labels_df.iterrows()}

    if args.start_from:
        if args.start_from in image_names:
            start_idx = image_names.index(args.start_from)
            image_names = image_names[start_idx:]
        else:
            print(f"start-from image not found in queue: {args.start_from}")

    print("Interactive labeling started.")
    print("Commands: 1=lens, 0=non_lens, s=skip, q=quit")

    n_labeled_now = 0
    for idx, name in enumerate(image_names, start=1):
        image_path = images_dir / name
        if not image_path.exists():
            print(f"[{idx}/{len(image_names)}] Missing image, skipping: {name}")
            continue

        current = label_map.get(name)
        if current is not None:
            prompt = f"[{idx}/{len(image_names)}] {name} (existing={current}) -> "
        else:
            prompt = f"[{idx}/{len(image_names)}] {name} -> "

        maybe_show_image(image_path, args.show_image)

        while True:
            cmd = input(prompt).strip().lower()
            if cmd in {"q", "quit"}:
                out_df = pd.DataFrame(
                    sorted(label_map.items(), key=lambda x: x[0]),
                    columns=["image_path", "label"],
                )
                labels_csv.parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(labels_csv, index=False)
                print(f"Saved labels to {labels_csv}. Labeled now: {n_labeled_now}")
                return

            if cmd in {"s", "skip", ""}:
                break

            if cmd in {"0", "1"}:
                label_map[name] = int(cmd)
                n_labeled_now += 1
                break

            print("Invalid input. Use 1, 0, s, or q.")

    out_df = pd.DataFrame(
        sorted(label_map.items(), key=lambda x: x[0]),
        columns=["image_path", "label"],
    )
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(labels_csv, index=False)

    class_counts = out_df["label"].value_counts().to_dict() if len(out_df) else {}
    print(f"Saved labels to {labels_csv}.")
    print(f"Total labeled images: {len(out_df)}")
    print(f"Class counts: {class_counts}")


if __name__ == "__main__":
    main()
