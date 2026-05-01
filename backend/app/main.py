"""FastAPI main application."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app import routes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AppState:
    """Global application state."""

    def __init__(self) -> None:
        self.model = None
        self.device = None
        self.inference_pipeline = None
        self.torch_available = False
        self.torch_import_error = None


app_state = AppState()


def _try_import_torch():
    """Import torch lazily so the API can boot even if Torch DLLs fail."""
    try:
        import torch

        return torch, True, None
    except Exception as exc:  # pragma: no cover - runtime fallback
        return None, False, exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    logger.info("=== APPLICATION STARTUP ===")

    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")

    torch, torch_available, torch_error = _try_import_torch()
    app_state.torch_available = torch_available
    app_state.torch_import_error = torch_error

    if torch_available:
        app_state.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        app_state.device = "cpu"
        logger.warning("Torch is unavailable at startup: %s", torch_error)

    logger.info("Using device: %s", app_state.device)

    if torch_available and app_state.device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    elif not torch_available:
        logger.warning(
            "Running without Torch; model inference will be disabled until the DLL issue is fixed."
        )

    model_path = os.getenv("MODEL_CHECKPOINT_PATH", "../runs/training_data_swin/best.pt")
    if torch_available and Path(model_path).exists():
        try:
            logger.info("Loading model from %s", model_path)

            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from lens_detection.models import build_model

            app_state.model = build_model(
                "swin_tiny_patch4_window7_224",
                num_classes=2,
                pretrained=False,
            )
            checkpoint = torch.load(model_path, map_location=app_state.device)
            app_state.model.load_state_dict(checkpoint)
            app_state.model = app_state.model.to(app_state.device).eval()

            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.error("Error loading model: %s", exc)
            app_state.model = None
    elif not torch_available:
        logger.info("Skipping model load because Torch could not be imported.")
    else:
        logger.warning("Model checkpoint not found at %s", model_path)

    if app_state.model:
        from app.services.multistage_inference import get_inference_pipeline

        app_state.inference_pipeline = get_inference_pipeline(
            app_state.model,
            device=app_state.device,
        )
        logger.info("Inference pipeline ready")

    logger.info("=== APPLICATION READY ===")

    yield

    logger.info("=== APPLICATION SHUTDOWN ===")
    app_state.model = None
    app_state.inference_pipeline = None
    logger.info("=== GOODBYE ===")


app = FastAPI(
    title="SGLDS Backend API",
    description="Multi-stage gravitational lens detection system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routers from the routes package.
try:
    routes.register_routes(app)
except Exception:
    logger.exception("Could not register routes at startup")
    raise
else:
    logger.info("API routers registered successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SGLDS Backend API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "database": "connected",
        "model_loaded": app_state.model is not None,
        "device": app_state.device,
        "gpu_available": bool(app_state.torch_available and app_state.device == "cuda"),
        "torch_available": app_state.torch_available,
    }


@app.get("/api/v1/stats")
async def get_stats():
    """Get overall system statistics."""
    from app.database import SessionLocal
    from app.services.analysis_log import get_analysis_logger

    db = SessionLocal()
    try:
        analysis_logger = get_analysis_logger(db)
        return analysis_logger.get_statistics()
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "False").lower() == "true",
    )
