"""Cloud data fetcher for Euclid Q1 images (using S3 + astroquery)."""
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
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
import requests

logger = logging.getLogger(__name__)

# Euclid AWS S3 bucket
BUCKET_NAME = "nasa-irsa-euclid-q1"
S3_FS = s3fs.S3FileSystem(anon=True)  # Anonymous access


class UpstreamServiceUnavailableError(RuntimeError):
    """Raised when the upstream IRSA service is temporarily unavailable."""


class EuclidCloudFetcher:
    """Fetch Euclid Q1 images and spectra from AWS S3 via astroquery."""

    def __init__(self, cache_dir: str = "./euclid_cache"):
        """
        Initialize the cloud fetcher.
        
        Args:
            cache_dir: Local directory to cache downloaded images
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_name = BUCKET_NAME

    def search_images(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: float = 10.0,
    ) -> List[Dict]:
        """
        Search for Euclid science images for a target.
        
        Uses astroquery to find available mosaics via IRSA SIA service.
        
        Args:
            target_name: Object name (e.g., "TYC 4429-1677-1") - resolved via astropy
            ra: RA in degrees (alternative to target_name)
            dec: Dec in degrees (alternative to target_name)
            radius_arcsec: Search radius in arcseconds
            
        Returns:
            List of dicts with: {euclid_id, filter, s3_url, instrument, ...}
        """
        try:
            # Resolve coordinates
            if target_name:
                coord = SkyCoord.from_name(target_name)
                logger.info(f"Resolved {target_name} to RA={coord.ra.deg:.4f}, Dec={coord.dec.deg:.4f}")
            elif ra is not None and dec is not None:
                coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            else:
                raise ValueError("Must provide either target_name or ra+dec")

            # Query Euclid mosaics via IRSA SIA
            search_radius = radius_arcsec * u.arcsec
            img_collection = "euclid_DpdMerBksMosaic"
            
            logger.info(f"Querying IRSA for {img_collection} near {coord}")
            img_tbl = self._query_sia_with_retries(
                coord=coord,
                search_radius=search_radius,
                img_collection=img_collection,
            )
            logger.info(f"Found {len(img_tbl)} images")

            # Filter to Euclid science images only
            euclid_sci_images = []
            for row in img_tbl:
                if row["facility_name"] == "Euclid" and row["dataproduct_subtype"] == "science":
                    euclid_sci_images.append(row)

            logger.info(f"Filtered to {len(euclid_sci_images)} Euclid science images")

            # Extract S3 paths and metadata
            results = []
            for row in euclid_sci_images:
                cloud_info = json.loads(row["cloud_access"])
                s3_key = cloud_info["aws"]["key"]
                s3_url = f"s3://{self.bucket_name}/{s3_key}"
                
                results.append({
                    "euclid_id": row.get("obs_id", "unknown"),
                    "filter": row["energy_bandpassname"],
                    "instrument": row["instrument_name"],
                    "s3_url": s3_url,
                    "ra": coord.ra.deg,
                    "dec": coord.dec.deg,
                    "access_url": row.get("access_url", ""),
                    "file_size_mb": row.get("filesize", 0) / 1024 / 1024,
                })

            return results

        except Exception as e:
            logger.error(f"Error searching Euclid images: {e}")
            raise

    def _query_sia_with_retries(
        self,
        coord: SkyCoord,
        search_radius: u.Quantity,
        img_collection: str,
    ):
        """Run IRSA SIA query with retry/backoff for transient upstream failures."""
        max_attempts = int(os.getenv("EUCLID_IRSA_MAX_ATTEMPTS", "3"))
        base_delay_sec = float(os.getenv("EUCLID_IRSA_RETRY_BASE_DELAY_SEC", "2.0"))

        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return Irsa.query_sia(pos=(coord, search_radius), collection=img_collection)
            except Exception as exc:  # pragma: no cover - third-party network failure path
                last_error = exc
                msg = str(exc).lower()
                transient_markers = (
                    "unable to access the capabilities endpoint",
                    "connection failed",
                    "timed out",
                    "service unavailable",
                    "temporary",
                    "503",
                    "dns",
                )
                is_transient = any(marker in msg for marker in transient_markers)
                if attempt < max_attempts and is_transient:
                    delay = base_delay_sec * (2 ** (attempt - 1))
                    logger.warning(
                        "IRSA query attempt %d/%d failed (%s). Retrying in %.1fs",
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                break

        if last_error is None:
            raise RuntimeError("IRSA query failed with an unknown error")

        msg = str(last_error).lower()
        if any(
            marker in msg
            for marker in (
                "unable to access the capabilities endpoint",
                "connection failed",
                "timed out",
                "service unavailable",
                "temporary",
                "503",
                "dns",
            )
        ):
            raise UpstreamServiceUnavailableError(
                "IRSA service is temporarily unavailable. Please retry in a few moments."
            ) from last_error

        raise last_error

    def download_image_cutout(
        self,
        s3_url: str,
        output_path: str,
        cutout_size_arcmin: float = 1.0,
        target_ra: Optional[float] = None,
        target_dec: Optional[float] = None,
    ) -> str:
        """
        Download a cutout from a Euclid mosaic using lazy FITS loading.
        
        Avoids downloading the entire mosaic (~1GB) by using Cutout2D.
        
        Args:
            s3_url: S3 URL of the FITS file (e.g., s3://nasa-irsa-euclid-q1/...)
            output_path: Local path to save cutout
            cutout_size_arcmin: Cutout size in arcminutes
            target_ra: RA of cutout center (if None, use mosaic center)
            target_dec: Dec of cutout center (if None, use mosaic center)
            
        Returns:
            Local path to saved cutout FITS file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading cutout from {s3_url}")

            # Tune remote reads for cloud-hosted FITS subsetting.
            # These settings follow Astropy/fsspec guidance for reducing total transfer and latency.
            fsspec_kwargs = {
                "anon": True,
                "block_size": 1_000_000,
                "cache_type": "bytes",
            }
            
            # Open FITS file from S3 with lazy loading
            with fits.open(
                s3_url,
                use_fsspec=True,
                lazy_load_hdus=True,
                fsspec_kwargs=fsspec_kwargs,
            ) as hdul:
                # Prefer the first image-like HDU that has array data.
                hdu = next((h for h in hdul if getattr(h, "data", None) is not None), hdul[0])
                if getattr(hdu, "data", None) is None:
                    raise ValueError("No image data found in FITS file")
                
                # Get mosaic center if target not specified
                if target_ra is None or target_dec is None:
                    # Use WCS center
                    wcs = WCS(hdu.header)
                    if wcs.pixel_shape is not None:
                        nx, ny = wcs.pixel_shape
                    else:
                        nx = int(hdu.header.get("NAXIS1", 0))
                        ny = int(hdu.header.get("NAXIS2", 0))
                    if nx <= 0 or ny <= 0:
                        raise ValueError("Could not determine image dimensions from FITS header")
                    target_ra, target_dec = wcs.pixel_to_world_values(nx / 2, ny / 2)
                    target_ra = float(target_ra)
                    target_dec = float(target_dec)

                coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
                cutout_size = cutout_size_arcmin * u.arcmin
                
                # Extract cutout using WCS.
                # Crucial for performance: pass hdu.section (lazy remote slicing), not hdu.data.
                cutout = Cutout2D(
                    hdu.section,
                    position=coord,
                    size=cutout_size,
                    wcs=WCS(hdu.header),
                )
                
                # Save cutout to FITS
                hdu.data = cutout.data
                hdu.header.update(cutout.wcs.to_header())
                hdu.writeto(output_path, overwrite=True)
                
                logger.info(f"Saved cutout to {output_path}")
                return str(output_path)

        except Exception as e:
            logger.error(f"Error downloading cutout: {e}")
            raise

    def get_spectrum(
        self,
        object_id: int,
        output_dir: str = None,
    ) -> Tuple[str, int]:
        """
        Fetch spectrum for a specific object ID from Euclid Q1 catalog.
        
        Args:
            object_id: Object ID from euclid_q1_mer_catalogue
            output_dir: Directory to save spectrum FITS
            
        Returns:
            Tuple of (spectrum_fits_path, spectrum_hdu_index)
        """
        try:
            if output_dir is None:
                output_dir = self.cache_dir / "spectra"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Query for spectrum file associated with this object
            euclid_spec_association_catalog = "euclid.objectid_spectrafile_association_q1"
            adql_query = f"SELECT * FROM {euclid_spec_association_catalog} WHERE objectid = {object_id}"
            
            logger.info(f"Querying spectrum for object {object_id}")
            spec_association_tbl = Irsa.query_tap(adql_query).to_table()

            if len(spec_association_tbl) == 0:
                raise ValueError(f"No spectrum found for object {object_id}")

            # Extract spectrum file path and HDU
            spec_fpath_key = spec_association_tbl["path"][0].replace(
                "api/spectrumdm/convert/euclid/", ""
            ).split("?")[0]
            hdu_idx = int(spec_association_tbl["hdu"][0])
            s3_url = f"s3://{self.bucket_name}/{spec_fpath_key}"
            
            # Download spectrum
            output_path = output_dir / f"spectrum_object_{object_id}.fits"
            with fits.open(s3_url, fsspec_kwargs={"anon": True}) as hdul:
                hdul.writeto(output_path, overwrite=True)

            logger.info(f"Saved spectrum to {output_path}")
            return str(output_path), hdu_idx

        except Exception as e:
            logger.error(f"Error fetching spectrum: {e}")
            raise

    def list_available_regions(self) -> List[str]:
        """List all available Euclid Q1 observation regions in S3."""
        try:
            regions = S3_FS.ls(f"{self.bucket_name}/q1")
            return regions
        except Exception as e:
            logger.error(f"Error listing regions: {e}")
            raise

    @staticmethod
    def cache_lookup(cache_dir: str, euclid_id: str) -> Optional[str]:
        """Check if image is already cached locally."""
        cache_path = Path(cache_dir) / f"{euclid_id}.fits"
        if cache_path.exists():
            return str(cache_path)
        return None


# Singleton instance for use across the app
_fetcher_instance = None


def get_cloud_fetcher(cache_dir: str = "./euclid_cache") -> EuclidCloudFetcher:
    """Get or create the cloud fetcher singleton."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = EuclidCloudFetcher(cache_dir=cache_dir)
    return _fetcher_instance
