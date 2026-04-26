from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

# Ensure local scripts/ imports work when executed as a top-level script.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from fetch_euclid_q1_data import fetch_cutouts


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Euclid Q1 cutouts for many targets and build a labeling queue CSV."
        )
    )
    parser.add_argument(
        "--targets-csv",
        default="data/euclid_q1/targets.csv",
        help=(
            "CSV with either target_name column OR ra_deg,dec_deg columns. "
            "Optional columns: target_id,label_prior"
        ),
    )
    parser.add_argument("--out-dir", default="data/euclid_q1")
    parser.add_argument("--search-radius-arcsec", type=float, default=10.0)
    parser.add_argument("--cutout-size-arcmin", type=float, default=1.0)
    parser.add_argument("--img-collection", default="euclid_DpdMerBksMosaic")
    parser.add_argument("--max-cutouts-per-target", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "target"


def _resolve_target_coord(row) -> tuple[str, SkyCoord]:
    target_name = str(row.get("target_name", "")).strip()
    has_name = len(target_name) > 0 and target_name.lower() != "nan"

    has_radec = "ra_deg" in row and "dec_deg" in row and pd.notna(row["ra_deg"]) and pd.notna(row["dec_deg"])

    if has_name:
        coord = SkyCoord.from_name(target_name)
        return target_name, coord

    if has_radec:
        ra_deg = float(row["ra_deg"])
        dec_deg = float(row["dec_deg"])
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        target_name = f"radec_{ra_deg:.6f}_{dec_deg:.6f}"
        return target_name, coord

    raise ValueError("Each row must provide target_name or both ra_deg and dec_deg.")


def _iter_targets(df: pd.DataFrame):
    for idx, row in df.iterrows():
        target_id = str(row.get("target_id", "")).strip()
        if not target_id:
            target_id = f"row_{idx:04d}"
        yield idx, target_id, row


def main():
    args = parse_args()

    targets_csv = Path(args.targets_csv)
    out_dir = Path(args.out_dir)
    out_cutouts = out_dir / "cutouts"
    out_fits = out_dir / "cutouts_fits"
    per_target_dir = out_dir / "targets"
    label_queue_path = out_dir / "label_queue.csv"

    out_cutouts.mkdir(parents=True, exist_ok=True)
    out_fits.mkdir(parents=True, exist_ok=True)
    per_target_dir.mkdir(parents=True, exist_ok=True)

    if not targets_csv.exists():
        raise FileNotFoundError(f"Targets CSV not found: {targets_csv}")

    df = pd.read_csv(targets_csv)
    if len(df) == 0:
        raise ValueError("Targets CSV is empty.")

    queue_rows: list[dict] = []

    for _, target_id, row in _iter_targets(df):
        try:
            target_name, coord = _resolve_target_coord(row)
        except Exception as exc:
            print(f"Skipping {target_id}: failed to resolve target ({exc})")
            continue

        slug = _slugify(f"{target_id}_{target_name}")
        this_dir = per_target_dir / slug

        if args.skip_existing and (this_dir / "cutouts_manifest.csv").exists():
            print(f"Skipping {target_id}: manifest exists ({this_dir / 'cutouts_manifest.csv'})")
            continue

        this_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFetching target {target_id}: {target_name}")
        print(f"RA={coord.ra.deg:.6f}, Dec={coord.dec.deg:.6f}")

        try:
            manifest_rows, tile_id = fetch_cutouts(
                coord=coord,
                search_radius=args.search_radius_arcsec * u.arcsec,
                cutout_size=args.cutout_size_arcmin * u.arcmin,
                out_dir=this_dir,
                img_collection=args.img_collection,
                max_cutouts=max(args.max_cutouts_per_target, 0),
            )
        except Exception as exc:
            print(f"  Failed target {target_id}: {exc}")
            continue

        if not manifest_rows:
            print(f"  No cutouts found for target {target_id}.")
            continue

        label_prior = row.get("label_prior", "")
        label_prior = "" if pd.isna(label_prior) else str(label_prior).strip()

        for m in manifest_rows:
            src_png = Path(m["preview_png"])
            src_fits = Path(m["cutout_fits"])

            dst_png = out_cutouts / f"{slug}_{src_png.name}"
            dst_fits = out_fits / f"{slug}_{src_fits.name}"

            shutil.copy2(src_png, dst_png)
            shutil.copy2(src_fits, dst_fits)

            queue_rows.append(
                {
                    "target_id": target_id,
                    "target_name": target_name,
                    "ra_deg": f"{coord.ra.deg:.8f}",
                    "dec_deg": f"{coord.dec.deg:.8f}",
                    "tile_id": tile_id if tile_id is not None else "",
                    "filter": m["filter"],
                    "instrument": m["instrument"],
                    "bandpass": m["bandpass"],
                    "image_path": dst_png.name,
                    "fits_path": dst_fits.name,
                    "label": label_prior,
                }
            )

    if not queue_rows:
        print("\nNo cutouts were fetched from target list.")
        return

    with open(label_queue_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_id",
                "target_name",
                "ra_deg",
                "dec_deg",
                "tile_id",
                "filter",
                "instrument",
                "bandpass",
                "image_path",
                "fits_path",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(queue_rows)

    print("\nBatch fetch complete.")
    print(f"Label queue: {label_queue_path}")
    print(f"Preview images: {out_cutouts}")
    print(f"FITS cutouts: {out_fits}")
    print(f"Rows ready for labeling: {len(queue_rows)}")


if __name__ == "__main__":
    main()
