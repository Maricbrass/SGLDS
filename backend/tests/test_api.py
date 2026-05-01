"""API tests aligned with the implemented backend routes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import AnalysisRun, EvaluationResult, Image, TrainingRun
import app.database as app_database
import app.routes.analyze as analyze_routes


SQLALCHEMY_DATABASE_URL = "sqlite://"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    app_database.SessionLocal = TestingSessionLocal
    analyze_routes.SessionLocal = TestingSessionLocal
    yield


@pytest.fixture
def mock_cloud_fetcher():
    fetcher = MagicMock()
    fetcher.cache_dir = Path("./test-cache")
    fetcher.search_images.return_value = [
        {
            "euclid_id": "euclid-1",
            "filter": "VIS",
            "instrument": "VIS",
            "s3_url": "s3://bucket/image.fits",
            "ra": 10.0,
            "dec": 20.0,
            "access_url": "https://example.invalid/image.fits",
            "file_size_mb": 12.5,
        }
    ]
    fetcher.download_image_cutout.return_value = str(Path("./test-cache/euclid-1.fits"))
    return fetcher


def create_image(db, euclid_id: str = "test_001", local_path: str = "/tmp/test.fits"):
    image = Image(
        euclid_id=euclid_id,
        source="euclid",
        local_path=local_path,
        s3_url="s3://test/test.fits",
        metadata_json={"filter": "VIS"},
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


class TestEuclidRoutes:
    def test_search_euclid_uses_cloud_fetcher(self, mock_cloud_fetcher):
        with patch("app.routes.euclid.get_cloud_fetcher", return_value=mock_cloud_fetcher):
            response = client.get("/api/v1/euclid/search?target_name=tyc 4429-1677-1")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["euclid_id"] == "euclid-1"

    def test_fetch_cutout_missing_s3_url(self):
        response = client.post("/api/v1/euclid/fetch", json={})
        assert response.status_code == 400

    def test_fetch_cutout_creates_image_record(self, mock_cloud_fetcher):
        db = TestingSessionLocal()
        try:
            with patch("app.routes.euclid.get_cloud_fetcher", return_value=mock_cloud_fetcher):
                response = client.post(
                    "/api/v1/euclid/fetch",
                    json={
                        "s3_url": "s3://bucket/image.fits",
                        "euclid_id": "euclid-1",
                        "cutout_size_arcmin": 1.5,
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data["image_id"] > 0

            image = db.query(Image).filter_by(id=data["image_id"]).first()
            assert image is not None
            assert image.euclid_id == "euclid-1"
            assert image.s3_url == "s3://bucket/image.fits"
        finally:
            db.close()

    def test_list_images_and_metadata(self, mock_cloud_fetcher):
        db = TestingSessionLocal()
        try:
            image = create_image(db)

            response = client.get("/api/v1/euclid/images")
            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 1
            assert data["images"][0]["id"] == image.id

            response = client.get(f"/api/v1/euclid/images/{image.id}/metadata")
            assert response.status_code == 200
            metadata = response.json()
            assert metadata["euclid_id"] == "test_001"
        finally:
            db.close()


class TestAnalyzeRoutes:
    def test_analyze_image_nonexistent(self):
        response = client.post("/api/v1/analyze/image/99999")
        assert response.status_code == 404

    def test_get_run_nonexistent(self):
        response = client.get("/api/v1/analyze/runs/99999")
        assert response.status_code == 404

    def test_get_run_status_and_history(self):
        db = TestingSessionLocal()
        try:
            image = create_image(db)
            run = AnalysisRun(
                image_id=image.id,
                status="completed",
                model_version="test_model",
                consensus_result={"final_confidence": 0.95, "final_prediction": 1},
            )
            db.add(run)
            db.commit()
            db.refresh(run)

            response = client.get(f"/api/v1/analyze/runs/{run.id}/status")
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == run.id
            assert data["status"] == "completed"

            response = client.get(f"/api/v1/analyze/image/{image.id}/history")
            assert response.status_code == 200
            history = response.json()
            assert history["count"] == 1
        finally:
            db.close()

    def test_analyze_image_runs_background_pipeline(self):
        db = TestingSessionLocal()
        try:
            image = create_image(db, local_path="/tmp/test-image.fits")

            pipeline = MagicMock()
            pipeline.analyze.return_value = {
                "stages": {
                    "stage_1": {"confidence": 0.9, "prediction": 1, "proceed_to_stage_2": True},
                    "stage_2": {"proceed_to_stage_3": True, "high_confidence_count": 1},
                    "stage_3": {"final_confidence": 0.95, "num_high_confidence_sub": 1},
                },
                "final_confidence": 0.95,
                "final_prediction": 1,
            }

            from app.main import app_state

            app_state.inference_pipeline = pipeline

            response = client.post(f"/api/v1/analyze/image/{image.id}")
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] > 0

            run = db.query(AnalysisRun).filter_by(id=data["run_id"]).first()
            assert run is not None
            assert run.status in {"pending", "completed"}
        finally:
            db.close()
            from app.main import app_state
            app_state.inference_pipeline = None


class TestTrainingRoutes:
    def test_training_history_and_lookup(self):
        db = TestingSessionLocal()
        try:
            run = TrainingRun(
                config_name="smoke",
                model_name="swin_tiny_patch4_window7_224",
                total_epochs=3,
                best_epoch=2,
                best_val_auc=0.91,
                checkpoint_path="/tmp/model.pt",
                metrics_json={"epochs": [1, 2, 3]},
                dataset_stats={},
            )
            db.add(run)
            db.commit()
            db.refresh(run)

            response = client.get("/api/v1/training/history")
            assert response.status_code == 200
            assert response.json()["count"] == 1

            response = client.get(f"/api/v1/training/{run.id}")
            assert response.status_code == 200
            assert response.json()["id"] == run.id

            response = client.get(f"/api/v1/training/{run.id}/metrics")
            assert response.status_code == 200
            assert response.json()["run_id"] == run.id
        finally:
            db.close()

    def test_training_comparison_validation(self):
        response = client.get("/api/v1/training/1/comparison?run_ids=invalid")
        assert response.status_code == 400


class TestEvaluationRoutes:
    def test_evaluation_endpoints(self):
        db = TestingSessionLocal()
        try:
            training_run = TrainingRun(
                config_name="smoke",
                model_name="swin_tiny_patch4_window7_224",
                total_epochs=3,
                best_epoch=2,
                best_val_auc=0.91,
                checkpoint_path="/tmp/model.pt",
                metrics_json={"roc_curve": {"fpr": [0, 0.1], "tpr": [0, 0.9], "thresholds": [1.0, 0.5]}},
                dataset_stats={},
            )
            db.add(training_run)
            db.commit()
            db.refresh(training_run)

            eval_result = EvaluationResult(
                training_run_id=training_run.id,
                split="test",
                roc_auc=0.95,
                precision=0.9,
                recall=0.92,
                f1_score=0.91,
                confusion_matrix=[[850, 50], [30, 70]],
                metrics_json={"roc_curve": {"fpr": [0, 0.1], "tpr": [0, 0.9], "thresholds": [1.0, 0.5]}},
            )
            db.add(eval_result)
            db.commit()

            response = client.get(f"/api/v1/evaluation/{training_run.id}")
            assert response.status_code == 200
            assert response.json()["training_run_id"] == training_run.id

            response = client.get(f"/api/v1/evaluation/{training_run.id}/confusion_matrix")
            assert response.status_code == 200
            assert response.json()["true_positives"] == 70

            response = client.get(f"/api/v1/evaluation/{training_run.id}/roc_data")
            assert response.status_code == 200
            assert response.json()["roc_auc"] == 0.95
        finally:
            db.close()


class TestResultsRoutes:
    def test_gallery_stats_export_and_report(self):
        db = TestingSessionLocal()
        try:
            image = create_image(db)
            run = AnalysisRun(
                image_id=image.id,
                status="completed",
                consensus_result={"final_confidence": 0.9, "final_prediction": 1},
                analysis_time_seconds=25.5,
                heatmap_path=None,
            )
            db.add(run)
            db.commit()
            db.refresh(run)

            response = client.get("/api/v1/results/gallery")
            assert response.status_code == 200
            assert response.json()["count"] == 1

            response = client.get("/api/v1/results/stats")
            assert response.status_code == 200
            assert response.json()["lenses_found"] == 1

            response = client.post("/api/v1/results/export", json={"run_ids": [run.id], "format": "json"})
            assert response.status_code == 200
            assert response.json()["count"] == 1

            response = client.get(f"/api/v1/results/report/{run.id}")
            assert response.status_code == 200
            assert response.json()["run_id"] == run.id
        finally:
            db.close()


class TestConfigRoutes:
    def test_config_read_update_reset(self):
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        assert "config" in response.json()

        response = client.put("/api/v1/config/inference", json={"stage_1_threshold": 0.7})
        assert response.status_code == 200
        assert response.json()["inference"]["stage_1_threshold"] == 0.7

        response = client.get("/api/v1/config/stages")
        assert response.status_code == 200
        assert "stage_1" in response.json()

        response = client.post("/api/v1/config/reset")
        assert response.status_code == 200
        assert response.json()["config"]["inference"]["stage_1_threshold"] == 0.5


class TestHealthCheck:
    def test_health_check_and_stats(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "ok"

        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "total_analyzed" in stats
        assert "lenses_found" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
