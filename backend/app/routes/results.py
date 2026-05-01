"""Results gallery and export routes."""
import logging
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Image, AnalysisRun

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/gallery")
def get_results_gallery(
    limit: int = 50,
    skip: int = 0,
    min_confidence: float = 0.0,
    db: Session = Depends(get_db),
):
    """Get paginated gallery of analyzed images sorted by confidence."""
    # Query all analysis runs with completed status
    runs = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.status == "completed")
        .order_by(AnalysisRun.run_timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    results = []
    for run in runs:
        consensus = run.consensus_result or {}
        confidence = consensus.get("final_confidence", 0.0)
        
        if confidence >= min_confidence:
            results.append({
                "run_id": run.id,
                "image_id": run.image_id,
                "euclid_id": run.image.euclid_id,
                "confidence": confidence,
                "prediction": consensus.get("final_prediction", 0),
                "analysis_time_seconds": run.analysis_time_seconds,
                "timestamp": run.run_timestamp.isoformat(),
                "heatmap_url": f"/api/v1/analyze/runs/{run.id}/heatmap",
            })
    
    return {"count": len(results), "results": results}


@router.get("/gallery/by_confidence")
def get_gallery_sorted_by_confidence(
    threshold_low: float = 0.0,
    threshold_high: float = 1.0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Get analyzed images filtered by confidence range."""
    runs = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.status == "completed")
        .order_by(AnalysisRun.run_timestamp.desc())
        .limit(limit)
        .all()
    )
    
    filtered = []
    for run in runs:
        consensus = run.consensus_result or {}
        confidence = consensus.get("final_confidence", 0.0)
        
        if threshold_low <= confidence <= threshold_high:
            filtered.append({
                "run_id": run.id,
                "image_id": run.image_id,
                "confidence": confidence,
                "prediction": consensus.get("final_prediction", 0),
                "euclid_id": run.image.euclid_id,
            })
    
    return {"count": len(filtered), "results": filtered}


@router.post("/export")
def export_results(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    """Export selected analysis results to CSV/JSON.
    
    Expected payload:
    {
        "run_ids": [1, 2, 3],
        "format": "csv" or "json",
        "include_fields": ["run_id", "confidence", "prediction", "analysis_time_seconds"]
    }
    """
    run_ids = payload.get("run_ids", [])
    export_format = payload.get("format", "json").lower()
    include_fields = payload.get("include_fields", ["run_id", "confidence", "prediction"])
    
    if not run_ids:
        raise HTTPException(status_code=400, detail="run_ids required")
    
    if export_format not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="format must be 'csv' or 'json'")
    
    runs = db.query(AnalysisRun).filter(AnalysisRun.id.in_(run_ids)).all()
    
    # Build export data
    export_data = []
    for run in runs:
        consensus = run.consensus_result or {}
        row = {
            "run_id": run.id,
            "image_id": run.image_id,
            "euclid_id": run.image.euclid_id,
            "confidence": consensus.get("final_confidence", 0.0),
            "prediction": consensus.get("final_prediction", 0),
            "analysis_time_seconds": run.analysis_time_seconds,
            "status": run.status,
        }
        export_data.append(row)
    
    if export_format == "json":
        return {
            "format": "json",
            "count": len(export_data),
            "data": export_data,
        }
    else:  # CSV
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=include_fields)
        writer.writeheader()
        
        for row in export_data:
            filtered_row = {k: v for k, v in row.items() if k in include_fields}
            writer.writerow(filtered_row)
        
        # Return as downloadable file
        csv_content = output.getvalue()
        return {
            "format": "csv",
            "count": len(export_data),
            "preview": csv_content[:500],  # First 500 chars
        }


@router.get("/report/{analysis_run_id}")
def get_analysis_report(analysis_run_id: int, db: Session = Depends(get_db)):
    """Get full analysis report for a run."""
    run = db.query(AnalysisRun).filter_by(id=analysis_run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    
    return {
        "run_id": run.id,
        "image_id": run.image_id,
        "euclid_id": run.image.euclid_id,
        "status": run.status,
        "model_version": run.model_version,
        "stage_1_result": run.stage_1_result,
        "stage_2_results": run.stage_2_results,
        "stage_3_results": run.stage_3_results,
        "consensus_result": run.consensus_result,
        "analysis_time_seconds": run.analysis_time_seconds,
        "run_timestamp": run.run_timestamp.isoformat(),
        "heatmap_url": f"/api/v1/analyze/runs/{run.id}/heatmap" if run.heatmap_path else None,
    }


@router.get("/stats")
def get_results_stats(db: Session = Depends(get_db)):
    """Get statistics about all analyzed results."""
    total_runs = db.query(AnalysisRun).count()
    completed_runs = db.query(AnalysisRun).filter_by(status="completed").count()
    failed_runs = db.query(AnalysisRun).filter_by(status="failed").count()
    
    # Count predicted lenses
    lens_count = 0
    non_lens_count = 0
    for run in db.query(AnalysisRun).filter_by(status="completed").all():
        consensus = run.consensus_result or {}
        pred = consensus.get("final_prediction", 0)
        if pred == 1:
            lens_count += 1
        else:
            non_lens_count += 1
    
    return {
        "total_analyzed": completed_runs,
        "total_runs": total_runs,
        "completed": completed_runs,
        "failed": failed_runs,
        "lenses_found": lens_count,
        "non_lenses": non_lens_count,
    }
