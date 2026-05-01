"""Evaluation metrics and reporting routes."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import TrainingRun, EvaluationResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{training_run_id}")
def get_evaluation(training_run_id: int, db: Session = Depends(get_db)):
    """Get all evaluation metrics for a training run."""
    training_run = db.query(TrainingRun).filter_by(id=training_run_id).first()
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    evaluations = db.query(EvaluationResult).filter_by(training_run_id=training_run_id).all()
    
    return {
        "training_run_id": training_run_id,
        "config_name": training_run.config_name,
        "evaluations": [
            {
                "id": e.id,
                "split": e.split,
                "roc_auc": e.roc_auc,
                "precision": e.precision,
                "recall": e.recall,
                "f1_score": e.f1_score,
                "confusion_matrix": e.confusion_matrix,
                "eval_timestamp": e.eval_timestamp.isoformat(),
            }
            for e in evaluations
        ],
    }


@router.get("/{training_run_id}/confusion_matrix")
def get_confusion_matrix(training_run_id: int, split: str = "test", db: Session = Depends(get_db)):
    """Get confusion matrix for a specific split."""
    eval_result = (
        db.query(EvaluationResult)
        .filter_by(training_run_id=training_run_id, split=split)
        .first()
    )
    if not eval_result:
        raise HTTPException(status_code=404, detail="Evaluation result not found for this split")
    
    cm = eval_result.confusion_matrix or [[0, 0], [0, 0]]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    return {
        "split": split,
        "confusion_matrix": cm,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
    }


@router.get("/{training_run_id}/roc_data")
def get_roc_data(training_run_id: int, split: str = "test", db: Session = Depends(get_db)):
    """Get ROC curve data for plotting."""
    eval_result = (
        db.query(EvaluationResult)
        .filter_by(training_run_id=training_run_id, split=split)
        .first()
    )
    if not eval_result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    
    # Extract ROC data from metrics_json if available
    metrics = eval_result.metrics_json or {}
    roc_curve = metrics.get("roc_curve", {})
    
    return {
        "split": split,
        "roc_auc": eval_result.roc_auc,
        "fpr": roc_curve.get("fpr", []),
        "tpr": roc_curve.get("tpr", []),
        "thresholds": roc_curve.get("thresholds", []),
    }


@router.get("/{training_run_id}/threshold_metrics")
def get_threshold_metrics(training_run_id: int, threshold: float = 0.5, db: Session = Depends(get_db)):
    """Get metrics at a specific confidence threshold."""
    eval_result = (
        db.query(EvaluationResult)
        .filter_by(training_run_id=training_run_id, split="test")
        .first()
    )
    if not eval_result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    
    cm = eval_result.confusion_matrix or [[0, 0], [0, 0]]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    return {
        "threshold": threshold,
        "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
    }
