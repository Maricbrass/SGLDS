"""SQLAlchemy ORM models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base


class Image(Base):
    """Store metadata about images (from Euclid or uploaded)."""
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    euclid_id = Column(String(100), unique=True, index=True, nullable=True)
    source = Column(String(50), default="euclid")  # 'euclid', 'upload', 'local'
    fetch_date = Column(DateTime, default=datetime.utcnow)
    s3_url = Column(String(500), nullable=True)
    local_path = Column(String(500), nullable=True)
    image_size_pixels = Column(Integer, nullable=True)  # e.g., 1024 for 1024x1024
    metadata = Column(JSON, default={})  # RA, Dec, filter, etc.

    # Relationships
    analysis_runs = relationship("AnalysisRun", back_populates="image", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Image(euclid_id={self.euclid_id}, source={self.source})>"


class AnalysisRun(Base):
    """Store results of single-image multi-stage inference."""
    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), index=True)
    model_version = Column(String(50), default="swin_euclid_baseline")
    run_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Stage results (JSON for flexibility)
    stage_1_result = Column(JSON, nullable=True)  # {confidence, prediction}
    stage_2_results = Column(JSON, nullable=True)  # [{tile_idx, x, y, confidence}, ...]
    stage_3_results = Column(JSON, nullable=True)  # [{tile_idx, x, y, confidence}, ...]
    consensus_result = Column(JSON, nullable=True)  # {final_confidence, is_lens, heatmap_path}
    
    heatmap_url = Column(String(500), nullable=True)
    analysis_time_seconds = Column(Float, nullable=True)
    gpu_used = Column(String(50), nullable=True)
    status = Column(String(20), default="pending")  # 'pending', 'running', 'completed', 'failed'
    error_message = Column(Text, nullable=True)

    # Relationships
    image = relationship("Image", back_populates="analysis_runs")
    flagged = relationship("FlaggedImage", back_populates="analysis_run", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<AnalysisRun(image_id={self.image_id}, status={self.status})>"


class TrainingRun(Base):
    """Store metadata about training jobs."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), default="swin_tiny_patch4_window7_224")
    config_name = Column(String(50), index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_epochs = Column(Integer)
    best_epoch = Column(Integer)
    best_val_auc = Column(Float)
    best_val_tpr_1e2 = Column(Float, nullable=True)
    best_val_tpr_1e3 = Column(Float, nullable=True)
    best_val_tpr_1e4 = Column(Float, nullable=True)
    metrics_json = Column(JSON)  # Full epoch history
    checkpoint_path = Column(String(500))
    dataset_stats = Column(JSON)
    status = Column(String(20), default="completed")  # 'running', 'completed', 'failed'

    # Relationships
    evaluations = relationship("EvaluationResult", back_populates="training_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TrainingRun(config={self.config_name}, best_auc={self.best_val_auc})>"


class EvaluationResult(Base):
    """Store test set evaluation metrics."""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), index=True)
    split = Column(String(20), default="test")  # 'val', 'test'
    roc_auc = Column(Float)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    confusion_matrix = Column(JSON)  # [[tn, fp], [fn, tp]]
    metrics_json = Column(JSON)
    eval_timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    training_run = relationship("TrainingRun", back_populates="evaluations")

    def __repr__(self):
        return f"<EvaluationResult(training_run_id={self.training_run_id}, auc={self.roc_auc})>"


class FlaggedImage(Base):
    """User-flagged images for QA review."""
    __tablename__ = "flagged_images"

    id = Column(Integer, primary_key=True, index=True)
    analysis_run_id = Column(Integer, ForeignKey("analysis_runs.id"), unique=True)
    reason = Column(String(100))  # 'high_confidence_lens', 'uncertain', 'false_positive', 'user_marked'
    flag_timestamp = Column(DateTime, default=datetime.utcnow)
    user_notes = Column(Text, nullable=True)
    reviewed = Column(Boolean, default=False)
    final_label = Column(Integer, nullable=True)  # 0 or 1 after human review

    # Relationships
    analysis_run = relationship("AnalysisRun", back_populates="flagged")

    def __repr__(self):
        return f"<FlaggedImage(analysis_run_id={self.analysis_run_id}, reason={self.reason})>"


class User(Base):
    """Store user info for team-shared dashboard (Phase 2)."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    role = Column(String(20), default="scientist")  # 'admin', 'scientist', 'labeler'
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<User(username={self.username}, role={self.role})>"
