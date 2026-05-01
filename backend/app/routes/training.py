"""Training history and monitoring routes."""
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import TrainingRun

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/history")
def get_training_history(
    limit: int = 20,
    skip: int = 0,
    db: Session = Depends(get_db),
):
    """Get list of training runs, ordered by most recent first."""
    runs = (
        db.query(TrainingRun)
        .order_by(TrainingRun.start_time.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {
        "count": len(runs),
        "runs": [
            {
                "id": r.id,
                "config_name": r.config_name,
                "model_name": r.model_name,
                "start_time": r.start_time.isoformat(),
                "end_time": r.end_time.isoformat() if r.end_time else None,
                "total_epochs": r.total_epochs,
                "best_epoch": r.best_epoch,
                "best_val_auc": r.best_val_auc,
                "best_val_tpr_1e2": r.best_val_tpr_1e2,
                "status": r.status,
            }
            for r in runs
        ],
    }


@router.get("/{run_id}")
def get_training_run(run_id: int, db: Session = Depends(get_db)):
    """Get details of a specific training run."""
    run = db.query(TrainingRun).filter_by(id=run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    return {
        "id": run.id,
        "config_name": run.config_name,
        "model_name": run.model_name,
        "start_time": run.start_time.isoformat(),
        "end_time": run.end_time.isoformat() if run.end_time else None,
        "total_epochs": run.total_epochs,
        "best_epoch": run.best_epoch,
        "best_val_auc": run.best_val_auc,
        "best_val_tpr_1e2": run.best_val_tpr_1e2,
        "best_val_tpr_1e3": run.best_val_tpr_1e3,
        "best_val_tpr_1e4": run.best_val_tpr_1e4,
        "status": run.status,
        "checkpoint_path": run.checkpoint_path,
        "dataset_stats": run.dataset_stats,
    }


@router.get("/{run_id}/metrics")
def get_training_metrics(run_id: int, db: Session = Depends(get_db)):
    """Get full training metrics (epoch history) for a run."""
    run = db.query(TrainingRun).filter_by(id=run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    return {
        "run_id": run.id,
        "config_name": run.config_name,
        "best_epoch": run.best_epoch,
        "best_val_auc": run.best_val_auc,
        "metrics_history": run.metrics_json or {},
    }


@router.get("/{run_id}/comparison")
def compare_training_runs(
    run_ids: str,  # comma-separated IDs like "1,2,3"
    db: Session = Depends(get_db),
):
    """Compare multiple training runs (for visualization)."""
    try:
        ids = [int(x.strip()) for x in run_ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="run_ids must be comma-separated integers")
    
    runs = db.query(TrainingRun).filter(TrainingRun.id.in_(ids)).all()
    if not runs:
        raise HTTPException(status_code=404, detail="No training runs found")
    
    return {
        "count": len(runs),
        "runs": [
            {
                "id": r.id,
                "config_name": r.config_name,
                "best_val_auc": r.best_val_auc,
                "total_epochs": r.total_epochs,
                "start_time": r.start_time.isoformat(),
                "metrics_history": r.metrics_json or {},
            }
            for r in runs
        ],
    }
