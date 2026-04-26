from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import astropy.units as u
import s3fs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from matplotlib import pyplot as plt


BUCKET_NAME = "nasa-irsa-euclid-q1"


def get_s3_fpath(cloud_access: str) -> str:
    cloud_info = json.loads(cloud_access)
    bucket_name = cloud_info["aws"]["bucket_name"]
    key = cloud_info["aws"]["key"]
    return f"{bucket_name}/{key}"


def get_filter_name(instrument: str, bandpass: str) -> str:
    return f"{instrument}_{bandpass}" if instrument != bandpass else instrument


def extract_tile_id_from_key(key: str) -> int | None:
    parts = key.split("/")
    # Example: q1/MER/102160339/VIS/<filename>.fits
    if len(parts) >= 4 and parts[0] == "q1" and parts[1] == "MER":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def write_manifest(manifest_path: Path, rows: list[dict]):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filter",
                "instrument",
                "bandpass",
                "s3_path",
                "cutout_fits",
                "preview_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def select_euclid_science_rows(img_tbl):
    return img_tbl[
        [
            row["facility_name"] == "Euclid"
            and row["dataproduct_subtype"] == "science"
            for row in img_tbl
        ]
    ]


def save_cutout_preview(cutout_data, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    norm = ImageNormalize(cutout_data, interval=PercentileInterval(99), stretch=AsinhStretch())
    ax.imshow(cutout_data, cmap="gray", origin="lower", norm=norm)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fetch_cutouts(
    coord: SkyCoord,
    search_radius: u.Quantity,
    cutout_size: u.Quantity,
    out_dir: Path,
    img_collection: str,
    max_cutouts: int,
):
    print("Querying IRSA SIA for mosaic products...")
    img_tbl = Irsa.query_sia(pos=(coord, search_radius), collection=img_collection)
    print(f"Found {len(img_tbl)} rows from SIA query.")
    euclid_sci_img_tbl = select_euclid_science_rows(img_tbl)
    print(f"Filtered to {len(euclid_sci_img_tbl)} Euclid science rows.")

    if len(euclid_sci_img_tbl) == 0:
        return [], None

    cutouts_dir = out_dir / "cutouts"
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    tile_id = None

    for idx, row in enumerate(euclid_sci_img_tbl, start=1):
        if len(manifest_rows) >= max_cutouts:
            print(f"Reached --max-cutouts={max_cutouts}; stopping cutout retrieval.")
            break
        s3_fpath = get_s3_fpath(row["cloud_access"])
        filter_name = get_filter_name(row["instrument_name"], row["energy_bandpassname"])
        fits_name = f"{filter_name}_cutout.fits"
        png_name = f"{filter_name}_cutout.png"
        out_fits = cutouts_dir / fits_name
        out_png = cutouts_dir / png_name
        print(f"[{idx}/{len(euclid_sci_img_tbl)}] Retrieving {filter_name} from {s3_fpath}")

        try:
            with fits.open(f"s3://{s3_fpath}", fsspec_kwargs={"anon": True}) as hdul:
                cutout = Cutout2D(
                    hdul[0].section,
                    position=coord,
                    size=cutout_size,
                    wcs=WCS(hdul[0].header),
                )

                hdu = fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header())
                hdu.writeto(out_fits, overwrite=True)
                save_cutout_preview(cutout.data, out_png, f"{filter_name} cutout")
        except Exception as exc:
            print(f"  Skipping {filter_name} due to error: {exc}")
            continue

        tile_id = tile_id or extract_tile_id_from_key(s3_fpath.split(f"{BUCKET_NAME}/", 1)[-1])
        manifest_rows.append(
            {
                "filter": filter_name,
                "instrument": row["instrument_name"],
                "bandpass": row["energy_bandpassname"],
                "s3_path": s3_fpath,
                "cutout_fits": str(out_fits),
                "preview_png": str(out_png),
            }
        )

    write_manifest(out_dir / "cutouts_manifest.csv", manifest_rows)
    return manifest_rows, tile_id


def fetch_object_id(coord: SkyCoord, mer_catalog: str, radius: u.Quantity) -> int | None:
    print("Querying MER object catalog for object_id...")
    mer_catalog_tbl = Irsa.query_region(
        coordinates=coord,
        spatial="Cone",
        catalog=mer_catalog,
        radius=radius,
    )
    if len(mer_catalog_tbl) == 0:
        print("No MER object match found in cone search.")
        return None
    print(f"Found {len(mer_catalog_tbl)} MER object match(es); using first row.")
    return int(mer_catalog_tbl["object_id"][0])


def fetch_spectrum_for_object(
    object_id: int,
    out_dir: Path,
    spec_association_catalog: str,
):
    print(f"Querying spectrum association for object_id={object_id}...")
    adql_query = f"SELECT * FROM {spec_association_catalog} WHERE objectid = {object_id}"
    spec_association_tbl = Irsa.query_tap(adql_query).to_table()

    if len(spec_association_tbl) == 0:
        print("No spectrum association found for this object.")
        return None

    spec_row = spec_association_tbl[0]
    spec_fpath_key = spec_row["path"].replace("api/spectrumdm/convert/euclid/", "").split("?")[0]
    object_hdu_idx = int(spec_row["hdu"])

    print(f"Retrieving spectrum HDU {object_hdu_idx} from s3://{BUCKET_NAME}/{spec_fpath_key}")
    with fits.open(f"s3://{BUCKET_NAME}/{spec_fpath_key}", fsspec_kwargs={"anon": True}) as hdul:
        spec_hdu = hdul[object_hdu_idx]
        spec_tbl = Table.read(spec_hdu)
        spec_header = spec_hdu.header

    spectra_dir = out_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    out_tbl = spectra_dir / f"{object_id}_spectrum.ecsv"
    out_meta = spectra_dir / f"{object_id}_spectrum_meta.json"
    out_plot = spectra_dir / f"{object_id}_spectrum.png"

    spec_tbl.write(out_tbl, format="ascii.ecsv", overwrite=True)

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "object_id": object_id,
                "hdu_index": object_hdu_idx,
                "s3_key": spec_fpath_key,
                "fscale": float(spec_header.get("FSCALE", 1.0)),
                "n_rows": len(spec_tbl),
            },
            f,
            indent=2,
        )

    # Plot with FSCALE correction when available.
    fscale = float(spec_header.get("FSCALE", 1.0))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(spec_tbl["WAVELENGTH"], fscale * spec_tbl["SIGNAL"], lw=1.0)
    ax.set_xlabel(str(spec_tbl["WAVELENGTH"].unit))
    ax.set_ylabel(str(spec_tbl["SIGNAL"].unit))
    ax.set_title(f"Euclid spectrum object {object_id}")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    return {
        "object_id": object_id,
        "spectrum_table": str(out_tbl),
        "spectrum_meta": str(out_meta),
        "spectrum_plot": str(out_plot),
    }


def list_bucket_examples(s3, bucket_name: str):
    print("Top-level q1 dirs:")
    print(s3.ls(f"{bucket_name}/q1"))
    print("\nMER tile examples:")
    print(s3.ls(f"{bucket_name}/q1/MER")[:10])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch Euclid Q1 cutouts and spectra from IRSA cloud-hosted data."
    )
    parser.add_argument("--target-name", default="TYC 4429-1677-1")
    parser.add_argument("--search-radius-arcsec", type=float, default=10.0)
    parser.add_argument("--object-search-radius-arcsec", type=float, default=5.0)
    parser.add_argument("--cutout-size-arcmin", type=float, default=1.0)
    parser.add_argument("--img-collection", default="euclid_DpdMerBksMosaic")
    parser.add_argument("--mer-catalog", default="euclid_q1_mer_catalogue")
    parser.add_argument(
        "--spec-association-catalog",
        default="euclid.objectid_spectrafile_association_q1",
    )
    parser.add_argument("--out-dir", default="data/euclid_q1")
    parser.add_argument(
        "--max-cutouts",
        type=int,
        default=4,
        help="Maximum number of science cutouts to retrieve (default: 4).",
    )
    parser.add_argument(
        "--skip-bucket-listing",
        action="store_true",
        help="Skip initial s3 directory listing to reduce startup time.",
    )
    parser.add_argument(
        "--skip-spectrum",
        action="store_true",
        help="Skip spectrum retrieval stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_name = args.target_name
    coord = SkyCoord.from_name(target_name)
    search_radius = args.search_radius_arcsec * u.arcsec
    object_search_radius = args.object_search_radius_arcsec * u.arcsec
    cutout_size = args.cutout_size_arcmin * u.arcmin

    # Browsing with s3fs mirrors the notebook's cloud exploration section.
    if not args.skip_bucket_listing:
        s3 = s3fs.S3FileSystem(anon=True)
        list_bucket_examples(s3, BUCKET_NAME)

    print(f"\nTarget: {target_name}")
    print(f"Coordinates: RA={coord.ra.deg:.6f}, Dec={coord.dec.deg:.6f}")

    manifest_rows, tile_id = fetch_cutouts(
        coord=coord,
        search_radius=search_radius,
        cutout_size=cutout_size,
        out_dir=out_dir,
        img_collection=args.img_collection,
        max_cutouts=max(args.max_cutouts, 0),
    )
    if not manifest_rows:
        print("No Euclid science mosaics found for this target/radius.")
        return

    object_id = fetch_object_id(
        coord=coord,
        mer_catalog=args.mer_catalog,
        radius=object_search_radius,
    )

    spectrum_result = None
    if object_id is not None and not args.skip_spectrum:
        spectrum_result = fetch_spectrum_for_object(
            object_id=object_id,
            out_dir=out_dir,
            spec_association_catalog=args.spec_association_catalog,
        )

    summary = {
        "target_name": target_name,
        "ra_deg": coord.ra.deg,
        "dec_deg": coord.dec.deg,
        "tile_id": tile_id,
        "object_id": object_id,
        "n_cutouts": len(manifest_rows),
        "cutout_manifest": str(out_dir / "cutouts_manifest.csv"),
        "spectrum": spectrum_result,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(f"- {out_dir / 'cutouts_manifest.csv'}")
    if spectrum_result is not None:
        print(f"- {spectrum_result['spectrum_table']}")
        print(f"- {spectrum_result['spectrum_meta']}")
        print(f"- {spectrum_result['spectrum_plot']}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
