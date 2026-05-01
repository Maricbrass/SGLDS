from typing import Optional, Dict, Any
import logging
import time
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models import Image
from app.services.analysis_log import get_analysis_logger

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_analysis_background(run_id: int, image_id: int, model_version: str) -> None:
    """Background task: execute inference pipeline and update run."""
    db = SessionLocal()
    try:
        analysis_logger = get_analysis_logger(db)
        run = analysis_logger.get_analysis_run(run_id)
        if not run:
            logger.error("Run %s not found", run_id)
            return

        # Check pipeline availability
        from app.main import app_state

        # If an in-process pipeline exists, use it. Otherwise, if an external Python
        # interpreter is configured (INFERENCE_PYTHON), spawn a subprocess worker
        # that runs the inference in that interpreter (must have torch installed).
        if not app_state.inference_pipeline:
            external_py = app_state.inference_python
            if external_py:
                logger.info("Using external inference python %s for run %s", external_py, run_id)
                try:
                    import subprocess, os

                    repo_root = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent))
                    worker_path = os.path.join(repo_root, "backend", "inference_worker.py")
                    model_checkpoint = os.getenv("MODEL_CHECKPOINT_PATH", os.path.join(repo_root, "runs", "training_data_swin", "best.pt"))
                    model_name = os.getenv("INFERENCE_MODEL_NAME", "swin_tiny_patch4_window7_224")

                    cmd = [external_py, worker_path, "--image", run.image.local_path, "--checkpoint", model_checkpoint, "--model", model_name, "--device", app_state.device or "cpu"]
                    env = os.environ.copy()
                    # Ensure worker can import project packages
                    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

                    logger.info("Spawning worker: %s", " ".join(cmd))
                    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
                    out = proc.stdout.strip()
                    err = proc.stderr.strip()

                    if proc.returncode != 0:
                        error_msg = f"External inference worker failed (rc={proc.returncode}): {err or out}"
                        logger.error(error_msg)
                        analysis_logger.update_analysis_run(run_id, status="failed", error_message=error_msg)
                        return

                    # Parse JSON output
                    payload = json.loads(out)
                    if not payload.get("ok"):
                        error_msg = payload.get("error") or payload.get("traceback") or "Unknown worker error"
                        analysis_logger.update_analysis_run(run_id, status="failed", error_message=error_msg)
                        logger.error("Worker returned error: %s", error_msg)
                        return

                    results = payload.get("results", {})
                    elapsed = 0.0
                    # Save heatmap if available
                    heatmap_path = None
                    if "heatmap_image_path" in results:
                        heatmap_path = results["heatmap_image_path"]

                    analysis_logger.update_analysis_run(
                        run_id,
                        status="completed",
                        stage_1_result=results.get("stages", {}).get("stage_1"),
                        stage_2_results=results.get("stages", {}).get("stage_2"),
                        stage_3_results=results.get("stages", {}).get("stage_3"),
                        consensus_result={
                            "final_confidence": results.get("final_confidence"),
                            "final_prediction": results.get("final_prediction"),
                        },
                        analysis_time_seconds=elapsed,
                        heatmap_path=heatmap_path,
                    )
                    logger.info("External worker completed run %s", run_id)
                    return
                except Exception as exc:
                    logger.exception("Failed running external inference worker: %s", exc)
                    analysis_logger.update_analysis_run(run_id, status="failed", error_message=str(exc))
                    return
            else:
                error_msg = "Inference pipeline not available (Torch may be missing)"
                analysis_logger.update_analysis_run(run_id, status="failed", error_message=error_msg)
                logger.error(error_msg)
                return

        # Run analysis
        logger.info("Starting background analysis for run %s, image %s", run_id, image_id)
        start = time.time()
        results = app_state.inference_pipeline.analyze(run.image.local_path)
        elapsed = time.time() - start

        # Save heatmap if available
        heatmap_path = None
        if "heatmap_image_path" in results:
            heatmap_path = results["heatmap_image_path"]

        # Update run with results
        analysis_logger.update_analysis_run(
            run_id,
            status="completed",
            stage_1_result=results.get("stages", {}).get("stage_1"),
            stage_2_results=results.get("stages", {}).get("stage_2"),
            stage_3_results=results.get("stages", {}).get("stage_3"),
            consensus_result={
                "final_confidence": results.get("final_confidence"),
                "final_prediction": results.get("final_prediction"),
            },
            analysis_time_seconds=elapsed,
            heatmap_path=heatmap_path,
        )
        logger.info("Completed background analysis for run %s in %.2f sec", run_id, elapsed)

    except Exception as exc:
        logger.exception("Background analysis failed for run %s: %s", run_id, exc)
        analysis_logger = get_analysis_logger(db)
        analysis_logger.update_analysis_run(run_id, status="failed", error_message=str(exc))
    finally:
        db.close()


@router.post("/image/{image_id}")
def analyze_image(
    image_id: int,
    model_version: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Start analysis for an image by id.

    If a completed cached result exists for the image+model_version, it is returned.
    Otherwise creates an AnalysisRun and queues the inference pipeline as a background task.
    Returns immediately with run_id and status=queued.
    """
    analysis_logger = get_analysis_logger(db)

    model_version = model_version or "swin_euclid_baseline"

    image = db.query(Image).filter_by(id=image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Check for cached completed analysis
    cached = analysis_logger.check_image_already_analyzed(image_id, model_version=model_version)
    if cached:
        return {
            "run_id": cached.id,
            "status": cached.status,
            "cached": True,
            "consensus_result": cached.consensus_result,
        }

    # Create run record with queued status
    run = analysis_logger.create_analysis_run(image_id=image_id, model_version=model_version)
    
    # Queue background task
    if background_tasks:
        background_tasks.add_task(_run_analysis_background, run.id, image_id, model_version)
        logger.info("Queued background analysis for run %s", run.id)
    
    return {
        "run_id": run.id,
        "status": run.status,
        "cached": False,
    }


@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Get analysis run details."""
    analysis_logger = get_analysis_logger(db)
    run = analysis_logger.get_analysis_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": run.id,
        "image_id": run.image_id,
        "status": run.status,
        "stage_1_result": run.stage_1_result,
        "stage_2_results": run.stage_2_results,
        "stage_3_results": run.stage_3_results,
        "consensus_result": run.consensus_result,
        "analysis_time_seconds": run.analysis_time_seconds,
        "error_message": run.error_message,
        "heatmap_path": run.heatmap_path,
    }


@router.get("/runs/{run_id}/status")
def get_run_status(run_id: int, db: Session = Depends(get_db)):
    """Get run status (quick check for polling)."""
    analysis_logger = get_analysis_logger(db)
    run = analysis_logger.get_analysis_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run.id,
        "status": run.status,
        "progress_percent": 100 if run.status == "completed" else 50 if run.status == "running" else 0,
        "completed_at": run.run_timestamp.isoformat() if run.run_timestamp else None,
    }


@router.get("/runs/{run_id}/heatmap")
def get_run_heatmap(run_id: int, db: Session = Depends(get_db)):
    """Get heatmap image for a run (returns PNG file)."""
    analysis_logger = get_analysis_logger(db)
    run = analysis_logger.get_analysis_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status == "failed":
        detail = run.error_message or "Analysis failed before a heatmap could be generated"
        raise HTTPException(status_code=409, detail=detail)
    if not run.heatmap_path or not Path(run.heatmap_path).exists():
        raise HTTPException(status_code=404, detail="Heatmap not yet available")
    return FileResponse(run.heatmap_path, media_type="image/png")


@router.get("/image/{image_id}/history")
def image_history(image_id: int, limit: int = 10, db: Session = Depends(get_db)):
    """Get recent analysis runs for an image."""
    analysis_logger = get_analysis_logger(db)
    runs = analysis_logger.get_image_analysis_history(image_id, limit=limit)
    return {"count": len(runs), "runs": [
        {
            "id": r.id,
            "status": r.status,
            "run_timestamp": r.run_timestamp.isoformat() if r.run_timestamp else None,
            "consensus_result": r.consensus_result,
        }
        for r in runs
    ]}
