"""Routes package.

This module exposes a helper to register API routers on the FastAPI app.
Routers are imported lazily when `register_routes` is called so importing
the package at startup is cheap.
"""
from fastapi import FastAPI


def register_routes(app: FastAPI) -> None:
	"""Import and include all route modules on the given FastAPI `app`.

	This function is called from `app.main` after the FastAPI `app` instance
	has been created.
	"""
	# Import routers here to avoid circular imports at module import time
	from app.routes import euclid as _euclid
	from app.routes import analyze as _analyze
	from app.routes import training as _training
	from app.routes import evaluation as _evaluation
	from app.routes import results as _results
	from app.routes import config as _config

	# Register all routers with their prefixes
	app.include_router(_euclid.router, prefix="/api/v1/euclid", tags=["euclid"])
	app.include_router(_analyze.router, prefix="/api/v1/analyze", tags=["analyze"])
	app.include_router(_training.router, prefix="/api/v1/training", tags=["training"])
	app.include_router(_evaluation.router, prefix="/api/v1/evaluation", tags=["evaluation"])
	app.include_router(_results.router, prefix="/api/v1/results", tags=["results"])
	app.include_router(_config.router, prefix="/api/v1/config", tags=["config"])

__all__ = ["register_routes"]
# Re-export the route registration helper for package-level imports.

