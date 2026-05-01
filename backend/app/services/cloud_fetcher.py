"""Cloud data fetcher for Euclid Q1 images (S3-first, IRSA-optional)."""
import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import s3fs
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

logger = logging.getLogger(__name__)

BUCKET_NAME = "nasa-irsa-euclid-q1"
S3_FS = s3fs.S3FileSystem(anon=True)


class UpstreamServiceUnavailableError(RuntimeError):
    pass


class EuclidCloudFetcher:
    def __init__(self, cache_dir: str = "./euclid_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_name = BUCKET_NAME
        self.cache_file = self.cache_dir / "search_cache.json"

    # -------------------------------
    # MAIN SEARCH (IRSA + FALLBACK)
    # -------------------------------
    def search_images(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: float = 10.0,
    ) -> List[Dict]:

        coord = self._resolve_coordinates(target_name, ra, dec)

        # 1️⃣ Try IRSA
        try:
            logger.info("Trying IRSA search...")
            results = self._search_irsa(coord, radius_arcsec)

            if results:
                self._save_cache(results)
                return results

        except UpstreamServiceUnavailableError:
            logger.warning("IRSA unavailable → falling back")

        # 2️⃣ Try cache
        cached = self._load_cache()
        if cached:
            logger.info("Using cached results")
            return cached

        # 3️⃣ Final fallback (direct S3 known tile)
        logger.warning("Using hardcoded S3 fallback")
        return self._fallback_results(coord)

    # -------------------------------
    # IRSA SEARCH
    # -------------------------------
    def _search_irsa(self, coord: SkyCoord, radius_arcsec: float) -> List[Dict]:
        search_radius = radius_arcsec * u.arcsec
        img_collection = "euclid_DpdMerBksMosaic"

        img_tbl = self._query_sia_with_retries(
            coord=coord,
            search_radius=search_radius,
            img_collection=img_collection,
        )

        results = []
        for row in img_tbl:
            if row["facility_name"] != "Euclid":
                continue
            if row["dataproduct_subtype"] != "science":
                continue

            cloud_info = json.loads(row["cloud_access"])
            s3_key = cloud_info["aws"]["key"]

            results.append({
                "euclid_id": row.get("obs_id", "unknown"),
                "filter": row["energy_bandpassname"],
                "instrument": row["instrument_name"],
                "s3_url": f"s3://{self.bucket_name}/{s3_key}",
                "ra": coord.ra.deg,
                "dec": coord.dec.deg,
            })

        return results

    # -------------------------------
    # RETRY LOGIC
    # -------------------------------
    def _query_sia_with_retries(self, coord, search_radius, img_collection):
        for attempt in range(3):
            try:
                return Irsa.query_sia(
                    pos=(coord, search_radius),
                    collection=img_collection
                )
            except Exception as exc:
                logger.warning(f"IRSA attempt {attempt+1} failed: {exc}")
                time.sleep(2 ** attempt)

        raise UpstreamServiceUnavailableError("IRSA unavailable")

    # -------------------------------
    # FALLBACK (S3 DIRECT)
    # -------------------------------
    def _fallback_results(self, coord: SkyCoord) -> List[Dict]:
        # Known working Euclid tile
        fallback_file = (
            "q1/MER/102018211/VIS/"
            "EUC_MER_MOSAIC-VIS-RMS_TILE102018211-B3070B_20241018T142525.109753Z_00.00.fits"
        )

        return [{
            "euclid_id": "fallback_tile",
            "filter": "VIS",
            "instrument": "Euclid",
            "s3_url": f"s3://{self.bucket_name}/{fallback_file}",
            "ra": coord.ra.deg,
            "dec": coord.dec.deg,
        }]

    # -------------------------------
    # CACHE
    # -------------------------------
    def _save_cache(self, results: List[Dict]):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(results, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _load_cache(self) -> Optional[List[Dict]]:
        try:
            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
        return None

    # -------------------------------
    # COORD RESOLUTION
    # -------------------------------
    def _resolve_coordinates(self, target_name, ra, dec) -> SkyCoord:
        if target_name:
            coord = SkyCoord.from_name(target_name)
            logger.info(f"Resolved {target_name} → {coord}")
            return coord

        if ra is not None and dec is not None:
            return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        raise ValueError("Provide target_name OR ra+dec")

    # -------------------------------
    # CUTOUT (UNCHANGED)
    # -------------------------------
    def download_image_cutout(
        self,
        s3_url: str,
        output_path: str,
        cutout_size_arcmin: float = 1.0,
        target_ra: Optional[float] = None,
        target_dec: Optional[float] = None,
    ) -> str:

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fsspec_kwargs = {
            "anon": True,
            # "block_size": 1_000_000,
            # "cache_type": "bytes",
        }

        # Persistent FS already defined globally

        with fits.open(
            s3_url,
            use_fsspec=True,
            fsspec_kwargs={"anon": True},
            lazy_load_hdus=True,
        ) as hdul:

            hdu = hdul[1] if len(hdul) > 1 else hdul[0]

            # WCS cache
            if not hasattr(self, "_wcs_cache"):
                self._wcs_cache = {}

            if s3_url in self._wcs_cache:
                wcs = self._wcs_cache[s3_url]
            else:
                wcs = WCS(hdu.header)
                self._wcs_cache[s3_url] = wcs

            if target_ra is None or target_dec is None:
                nx = int(hdu.header.get("NAXIS1", 0))
                ny = int(hdu.header.get("NAXIS2", 0))
                target_ra, target_dec = wcs.pixel_to_world_values(nx / 2, ny / 2)

            coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)

            cutout = Cutout2D(
                hdu.section,
                position=coord,
                size=cutout_size_arcmin * u.arcmin,
                wcs=wcs,
            )

            # from astropy.io import fits
            fits.writeto(
                output_path,
                cutout.data,
                cutout.wcs.to_header(),
                overwrite=True
            )

        return str(output_path)


# Singleton
_fetcher_instance = None


def get_cloud_fetcher(cache_dir: str = "./euclid_cache") -> EuclidCloudFetcher:
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = EuclidCloudFetcher(cache_dir)
    return _fetcher_instance