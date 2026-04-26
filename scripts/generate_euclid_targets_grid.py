from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dense Euclid Q1 target grid CSV from a center coordinate and angular span."
        )
    )
    parser.add_argument("--ra-deg", type=float, required=True)
    parser.add_argument("--dec-deg", type=float, required=True)
    parser.add_argument("--side-arcmin", type=float, default=12.0)
    parser.add_argument("--step-arcmin", type=float, default=1.5)
    parser.add_argument("--out-csv", default="data/euclid_q1/targets.csv")
    parser.add_argument("--prefix", default="grid")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.step_arcmin <= 0:
        raise ValueError("--step-arcmin must be positive")
    if args.side_arcmin <= 0:
        raise ValueError("--side-arcmin must be positive")

    half = args.side_arcmin / 2.0
    n_steps = int(round(args.side_arcmin / args.step_arcmin))
    if n_steps < 1:
        n_steps = 1

    start_offset = -half
    end_offset = half
    offsets = []
    current = start_offset
    while current <= end_offset + 1e-9:
        offsets.append(current)
        current += args.step_arcmin

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0
    for dec_off in offsets:
        for ra_off in offsets:
            idx += 1
            ra_deg = args.ra_deg + (ra_off / 60.0)
            dec_deg = args.dec_deg + (dec_off / 60.0)
            rows.append(
                {
                    "target_id": f"{args.prefix}_{idx:04d}",
                    "ra_deg": f"{ra_deg:.8f}",
                    "dec_deg": f"{dec_deg:.8f}",
                    "label_prior": "",
                }
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["target_id", "ra_deg", "dec_deg", "label_prior"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} targets to {out_csv}")
    print("Use this CSV with scripts/fetch_euclid_targets_batch.py")


if __name__ == "__main__":
    main()
