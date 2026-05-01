"""Pydantic request/response schemas."""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel


# Image Schemas
class ImageMetadata(BaseModel):
    ra: Optional[float] = None
    dec: Optional[float] = None
    filter: Optional[str] = None
    instrument: Optional[str] = None


class ImageCreate(BaseModel):
    euclid_id: Optional[str] = None
    source: str = "upload"
    s3_url: Optional[str] = None
    local_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ImageResponse(BaseModel):
    id: int
    euclid_id: Optional[str]
    source: str
    fetch_date: datetime
    s3_url: Optional[str]
    local_path: Optional[str]
    metadata: Dict[str, Any]

    class Config:
        from_attributes = True


# Analysis Schemas
class AnalysisResultDetail(BaseModel):
    confidence: float
    is_lens: bool
    chunks_detected: Optional[int] = None


class AnalysisRunResponse(BaseModel):
    id: int
    image_id: int
    model_version: str
    run_timestamp: datetime
    status: str
    stage_1_result: Optional[Dict[str, Any]] = None
    stage_2_results: Optional[List[Dict[str, Any]]] = None
    stage_3_results: Optional[List[Dict[str, Any]]] = None
    consensus_result: Optional[Dict[str, Any]] = None
    analysis_time_seconds: Optional[float] = None
    heatmap_url: Optional[str] = None

    class Config:
        from_attributes = True


# Training Schemas
class TrainingRunResponse(BaseModel):
    id: int
    model_name: str
    config_name: str
    start_time: datetime
    end_time: Optional[datetime]
    total_epochs: int
    best_epoch: int
    best_val_auc: float
    best_val_tpr_1e2: Optional[float]
    best_val_tpr_1e3: Optional[float]
    best_val_tpr_1e4: Optional[float]
    checkpoint_path: str
    status: str

    class Config:
        from_attributes = True


# Evaluation Schemas
class EvaluationResultResponse(BaseModel):
    id: int
    training_run_id: int
    split: str
    roc_auc: float
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    confusion_matrix: List[List[int]]
    eval_timestamp: datetime

    class Config:
        from_attributes = True


# Statistics
class SystemStatsResponse(BaseModel):
    total_images_analyzed: int
    total_lenses_found: int
    avg_analysis_time_seconds: float
    total_training_runs: int
    latest_model_auc: Optional[float] = None
    gpu_available: bool


class HealthCheckResponse(BaseModel):
    status: str
    database_connected: bool
    model_loaded: bool
    gpu_available: bool
    device: str
