"""Database operations for analysis logging."""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.models import Image, AnalysisRun
from app.schemas import AnalysisRunResponse

logger = logging.getLogger(__name__)


class AnalysisLogger:
    """Log analysis runs to database."""

    def __init__(self, db: Session):
        """
        Initialize analysis logger.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def create_image_record(
        self,
        euclid_id: Optional[str] = None,
        source: str = "upload",
        s3_url: Optional[str] = None,
        local_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Image:
        """
        Create or get image record.
        
        Args:
            euclid_id: Euclid Q1 identifier (unique)
            source: 'euclid', 'upload', 'local'
            s3_url: S3 URL if from cloud
            local_path: Local file path
            metadata: Additional metadata (RA, Dec, filter, etc.)
            
        Returns:
            Image model instance
        """
        # Check if image already exists
        if euclid_id:
            existing = self.db.query(Image).filter_by(euclid_id=euclid_id).first()
            if existing:
                logger.info(f"Image {euclid_id} already exists")
                return existing

        image = Image(
            euclid_id=euclid_id,
            source=source,
            s3_url=s3_url,
            local_path=local_path,
            metadata=metadata or {},
            fetch_date=datetime.utcnow(),
        )
        
        self.db.add(image)
        self.db.commit()
        self.db.refresh(image)
        
        logger.info(f"Created image record: {image.id}")
        return image

    def create_analysis_run(
        self,
        image_id: int,
        model_version: str = "swin_euclid_baseline",
    ) -> AnalysisRun:
        """
        Create analysis run record.
        
        Args:
            image_id: ID of image to analyze
            model_version: Model identifier
            
        Returns:
            AnalysisRun model instance
        """
        run = AnalysisRun(
            image_id=image_id,
            model_version=model_version,
            status="pending",
            run_timestamp=datetime.utcnow(),
        )
        
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        
        logger.info(f"Created analysis run: {run.id} for image {image_id}")
        return run

    def update_analysis_run(
        self,
        run_id: int,
        status: str = "running",
        stage_1_result: Optional[Dict] = None,
        stage_2_results: Optional[Dict] = None,
        stage_3_results: Optional[Dict] = None,
        consensus_result: Optional[Dict] = None,
        analysis_time_seconds: Optional[float] = None,
        heatmap_url: Optional[str] = None,
        error_message: Optional[str] = None,
        gpu_used: Optional[str] = None,
    ) -> AnalysisRun:
        """
        Update analysis run with results.
        
        Args:
            run_id: Analysis run ID
            status: 'pending', 'running', 'completed', 'failed'
            stage_*_result: Results from each stage
            consensus_result: Final consensus
            analysis_time_seconds: Total time taken
            heatmap_url: URL/path to heatmap image
            error_message: Error details if failed
            gpu_used: GPU device used
            
        Returns:
            Updated AnalysisRun
        """
        run = self.db.query(AnalysisRun).filter_by(id=run_id).first()
        if not run:
            raise ValueError(f"Analysis run {run_id} not found")

        run.status = status
        if stage_1_result:
            run.stage_1_result = stage_1_result
        if stage_2_results:
            run.stage_2_results = stage_2_results
        if stage_3_results:
            run.stage_3_results = stage_3_results
        if consensus_result:
            run.consensus_result = consensus_result
        if analysis_time_seconds:
            run.analysis_time_seconds = analysis_time_seconds
        if heatmap_url:
            run.heatmap_url = heatmap_url
        if error_message:
            run.error_message = error_message
        if gpu_used:
            run.gpu_used = gpu_used

        self.db.commit()
        self.db.refresh(run)
        
        logger.info(f"Updated analysis run {run_id}: status={status}")
        return run

    def get_analysis_run(self, run_id: int) -> Optional[AnalysisRun]:
        """Get analysis run by ID."""
        return self.db.query(AnalysisRun).filter_by(id=run_id).first()

    def get_image_analysis_history(
        self,
        image_id: int,
        limit: int = 10,
    ) -> list[AnalysisRun]:
        """
        Get analysis history for an image.
        
        Args:
            image_id: Image ID
            limit: Max results to return
            
        Returns:
            List of AnalysisRun records
        """
        return (
            self.db.query(AnalysisRun)
            .filter_by(image_id=image_id)
            .order_by(AnalysisRun.run_timestamp.desc())
            .limit(limit)
            .all()
        )

    def check_image_already_analyzed(
        self,
        image_id: int,
        model_version: str = "swin_euclid_baseline",
    ) -> Optional[AnalysisRun]:
        """
        Check if image has already been analyzed with this model.
        
        Returns cached result if available.
        
        Args:
            image_id: Image ID
            model_version: Model version to check
            
        Returns:
            Cached AnalysisRun if exists, None otherwise
        """
        run = (
            self.db.query(AnalysisRun)
            .filter_by(
                image_id=image_id,
                model_version=model_version,
                status="completed",
            )
            .order_by(AnalysisRun.run_timestamp.desc())
            .first()
        )
        
        if run:
            logger.info(f"Found cached analysis result for image {image_id}")
            return run
        
        return None

    def get_latest_analyses(
        self,
        limit: int = 20,
    ) -> list[AnalysisRun]:
        """
        Get most recent completed analyses.
        
        Args:
            limit: Number to return
            
        Returns:
            List of recent AnalysisRun records
        """
        return (
            self.db.query(AnalysisRun)
            .filter_by(status="completed")
            .order_by(AnalysisRun.run_timestamp.desc())
            .limit(limit)
            .all()
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics.
        
        Returns:
            Dict with: total_analyzed, lenses_found, avg_time, etc.
        """
        total_analyzed = self.db.query(AnalysisRun).filter_by(status="completed").count()
        
        # Get lenses found (final_prediction=1 in consensus_result)
        all_runs = self.db.query(AnalysisRun).filter_by(status="completed").all()
        lenses_found = sum(
            1 for run in all_runs
            if run.consensus_result and run.consensus_result.get("final_prediction") == 1
        )
        
        # Average analysis time
        avg_time = None
        if total_analyzed > 0:
            times = [run.analysis_time_seconds for run in all_runs if run.analysis_time_seconds]
            if times:
                avg_time = sum(times) / len(times)

        return {
            "total_analyzed": total_analyzed,
            "lenses_found": lenses_found,
            "non_lenses_found": total_analyzed - lenses_found,
            "avg_analysis_time_seconds": avg_time,
        }


def get_analysis_logger(db: Session) -> AnalysisLogger:
    """Create and return an analysis logger."""
    return AnalysisLogger(db)
