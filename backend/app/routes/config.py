"""Configuration management routes."""
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
import json

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory config (can be extended to use database or file-based config)
_config = {
    "inference": {
        "stage_1_threshold": 0.5,
        "stage_2_threshold": 0.6,
        "stage_3_threshold": 0.65,
        "chunk_size_pixels": 512,
        "sub_chunk_size_pixels": 128,
        "overlap_percent": 10,
    },
    "model": {
        "current_model": "swin_euclid_baseline",
        "available_models": ["swin_euclid_baseline", "training_data_swin"],
        "device": "cuda",
    },
    "data": {
        "cache_dir": "./euclid_cache",
        "max_cache_size_gb": 100,
        "cutout_size_arcmin": 1.0,
    },
    "export": {
        "include_heatmaps": True,
        "pdf_report_template": "default",
        "csv_fields": [
            "run_id",
            "euclid_id",
            "confidence",
            "prediction",
            "analysis_time_seconds",
        ],
    },
}


@router.get("")
def get_config() -> Dict[str, Any]:
    """Get current system configuration."""
    return {
        "config": _config,
        "version": "1.0",
    }


@router.get("/inference")
def get_inference_config() -> Dict[str, Any]:
    """Get inference-specific configuration."""
    return {
        "inference": _config.get("inference", {}),
    }


@router.get("/model")
def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return {
        "model": _config.get("model", {}),
    }


@router.get("/data")
def get_data_config() -> Dict[str, Any]:
    """Get data handling configuration."""
    return {
        "data": _config.get("data", {}),
    }


@router.put("")
def update_config(
    payload: dict = Body(...),
) -> Dict[str, Any]:
    """Update system configuration.
    
    Expected payload structure:
    {
        "inference": {...},
        "model": {...},
        "data": {...},
        "export": {...}
    }
    """
    global _config
    
    try:
        # Merge updates (don't overwrite entire config)
        for section, values in payload.items():
            if section in _config and isinstance(values, dict):
                _config[section].update(values)
            elif section in _config:
                _config[section] = values
            else:
                logger.warning(f"Unknown config section: {section}")
        
        logger.info(f"Config updated with sections: {list(payload.keys())}")
        return {
            "status": "success",
            "config": _config,
        }
    except Exception as exc:
        logger.exception(f"Failed to update config: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/inference")
def update_inference_config(
    payload: dict = Body(...),
) -> Dict[str, Any]:
    """Update inference thresholds and parameters."""
    global _config
    _config["inference"].update(payload)
    logger.info(f"Inference config updated: {payload}")
    return {"status": "success", "inference": _config["inference"]}


@router.put("/model")
def update_model_config(
    current_model: str = Body(...),
) -> Dict[str, Any]:
    """Switch to a different model."""
    if current_model not in _config["model"].get("available_models", []):
        raise HTTPException(
            status_code=400,
            detail=f"Model {current_model} not in available models",
        )
    
    _config["model"]["current_model"] = current_model
    logger.info(f"Model switched to: {current_model}")
    return {"status": "success", "model": _config["model"]}


@router.get("/stages")
def get_stage_parameters() -> Dict[str, Any]:
    """Get multi-stage inference parameters."""
    inf_cfg = _config.get("inference", {})
    return {
        "stage_1": {
            "threshold": inf_cfg.get("stage_1_threshold", 0.5),
            "description": "Full-image classification",
        },
        "stage_2": {
            "threshold": inf_cfg.get("stage_2_threshold", 0.6),
            "chunk_size_pixels": inf_cfg.get("chunk_size_pixels", 512),
            "overlap_percent": inf_cfg.get("overlap_percent", 10),
            "description": "Grid-based chunk analysis",
        },
        "stage_3": {
            "threshold": inf_cfg.get("stage_3_threshold", 0.65),
            "sub_chunk_size_pixels": inf_cfg.get("sub_chunk_size_pixels", 128),
            "description": "Fine-grained sub-chunk analysis",
        },
    }


@router.post("/reset")
def reset_config_to_defaults() -> Dict[str, Any]:
    """Reset configuration to defaults."""
    global _config
    _config = {
        "inference": {
            "stage_1_threshold": 0.5,
            "stage_2_threshold": 0.6,
            "stage_3_threshold": 0.65,
            "chunk_size_pixels": 512,
            "sub_chunk_size_pixels": 128,
            "overlap_percent": 10,
        },
        "model": {
            "current_model": "swin_euclid_baseline",
            "available_models": ["swin_euclid_baseline", "training_data_swin"],
            "device": "cuda",
        },
        "data": {
            "cache_dir": "./euclid_cache",
            "max_cache_size_gb": 100,
            "cutout_size_arcmin": 1.0,
        },
        "export": {
            "include_heatmaps": True,
            "pdf_report_template": "default",
        },
    }
    logger.info("Configuration reset to defaults")
    return {"status": "success", "config": _config}
