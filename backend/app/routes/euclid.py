from typing import Optional
import time
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.cloud_fetcher import get_cloud_fetcher, UpstreamServiceUnavailableError
from app.services.analysis_log import get_analysis_logger
from app.models import Image

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/search")
def search_euclid(
    target_name: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    radius_arcsec: float = 10.0,
):
    """Search for Euclid images near a target name or coordinates."""
    fetcher = get_cloud_fetcher()
    try:
        results = fetcher.search_images(
            target_name=target_name, ra=ra, dec=dec, radius_arcsec=radius_arcsec
        )
        return {"count": len(results), "results": results}
    except UpstreamServiceUnavailableError as exc:
        logger.warning("Euclid search upstream unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "15"},
        )
    except Exception as exc:
        logger.error("Euclid search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fetch")
def fetch_cutout(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    """Fetch a cutout from S3 and create an Image record.

    Expected payload keys: `s3_url` (required), optional `euclid_id`, `cutout_size_arcmin`, `target_ra`, `target_dec`.
    """
    s3_url = payload.get("s3_url")
    if not s3_url:
        raise HTTPException(status_code=400, detail="s3_url is required in payload")

    euclid_id = payload.get("euclid_id")
    cutout_size_arcmin = float(payload.get("cutout_size_arcmin", 1.0))
    target_ra = payload.get("target_ra")
    target_dec = payload.get("target_dec")

    fetcher = get_cloud_fetcher()

    # Determine local output path
    cache_dir = Path(fetcher.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    if euclid_id:
        out_name = f"{euclid_id}.fits"
    else:
        out_name = f"cutout_{timestamp}.fits"
    out_path = cache_dir / out_name

    try:
        saved_path = fetcher.download_image_cutout(
            s3_url=s3_url,
            output_path=str(out_path),
            cutout_size_arcmin=cutout_size_arcmin,
            target_ra=target_ra,
            target_dec=target_dec,
        )

        # Create DB record
        logger.info("Creating image record for %s", saved_path)
        analysis_logger = get_analysis_logger(db)
        metadata = {
            "s3_url": s3_url,
            "cutout_size_arcmin": cutout_size_arcmin,
            "target_ra": target_ra,
            "target_dec": target_dec,
        }
        image = analysis_logger.create_image_record(
            euclid_id=euclid_id,
            source="euclid",
            s3_url=s3_url,
            local_path=saved_path,
            metadata=metadata,
        )

        return {"image_id": image.id, "local_path": saved_path}

    except Exception as exc:
        logger.exception("Failed to fetch and save cutout: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/images")
def list_images(limit: int = 50, db: Session = Depends(get_db)):
    """List images cached in the system."""
    imgs = db.query(Image).order_by(Image.fetch_date.desc()).limit(limit).all()
    return {"count": len(imgs), "images": [
        {
            "id": i.id,
            "euclid_id": i.euclid_id,
            "source": i.source,
            "s3_url": i.s3_url,
            "local_path": i.local_path,
            "fetch_date": i.fetch_date.isoformat() if i.fetch_date else None,
        }
        for i in imgs
    ]}


@router.get("/images/{image_id}/metadata")
def image_metadata(image_id: int, db: Session = Depends(get_db)):
    """Get metadata for a specific image."""
    image = db.query(Image).filter_by(id=image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return {
        "id": image.id,
        "euclid_id": image.euclid_id,
        "s3_url": image.s3_url,
        "local_path": image.local_path,
            "metadata": image.metadata_json,
        "fetch_date": image.fetch_date.isoformat() if image.fetch_date else None,
    }
