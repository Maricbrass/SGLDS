# SGLDS Backend API (Phase 1)

Production-grade FastAPI backend for gravitational lens detection in Euclid space telescope images.

## Features

✅ **Multi-Stage Inference Pipeline**
- Stage 1: Quick full-image analysis (224×224)
- Stage 2: Adaptive chunk grid analysis (512×512 tiles)
- Stage 3: Sub-chunk refinement (128×128 tiles) with heatmap generation

✅ **Euclid Q1 Cloud Integration**
- Query Euclid images via astroquery + IRSA
- Lazy-load FITS files from AWS S3 (no download of full mosaics)
- Automatic local caching

✅ **Database Logging**
- PostgreSQL for persistent analysis logs
- Full tracking of inference runs, training metadata, evaluation metrics
- Support for team-shared dashboard (multi-user ready)

✅ **RESTful API**
- FastAPI with async/await support
- OpenAPI documentation (Swagger UI at `/docs`)
- WebSocket-ready for real-time streaming (Phase 2)

## Architecture

```
FastAPI Server (main.py)
    ├── Models (SQLAlchemy ORM)
    │   ├── Image
    │   ├── AnalysisRun
    │   ├── TrainingRun
    │   ├── EvaluationResult
    │   └── FlaggedImage
    │
    ├── Services
    │   ├── cloud_fetcher.py (Euclid S3 + astroquery)
    │   ├── multistage_inference.py (3-stage lens detection)
    │   └── analysis_log.py (DB operations)
    │
    ├── Routes (Phase 2)
    │   ├── euclid.py
    │   ├── analyze.py
    │   ├── training.py
    │   ├── evaluation.py
    │   ├── results.py
    │   └── config.py
    │
    └── Database
        └── PostgreSQL (via Docker)
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker & Docker Compose (for PostgreSQL)
- NVIDIA GPU (RTX 4050+, for real-time inference)
- PyTorch 2.5.1+cu121 (already installed in main project)

### 2. Setup Backend Environment

```bash
# Navigate to backend folder
cd backend

# Create a fresh virtual environment for this backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If pip check shows unrelated conflicts, you are in a shared or preloaded environment.
# Use this clean venv instead of the system interpreter.
```

### 3. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your settings
# For local dev, defaults should work:
# - DATABASE_URL=postgresql://sglds_user:sglds_password@localhost:5432/sglds_db
# - DEBUG=True
# - DEVICE=cuda (or cpu)
# - MODEL_CHECKPOINT_PATH=../runs/training_data_swin/best.pt
```

### 4. Start PostgreSQL Database

```bash
# Start PostgreSQL container
docker-compose up -d postgres

# Verify it's running
docker-compose logs -f postgres

# You should see: "database system is ready to accept connections"
```

### 5. Run Backend Server

```bash
# From backend/ directory
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### 6. Test API

- **OpenAPI Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health
- **Root**: http://localhost:8000/

## API Endpoints (Phase 1 - Core)

### Health & Status
```
GET  /api/v1/health              Health check + GPU status
GET  /api/v1/stats               Overall statistics
```

### Euclid Cloud (Phase 1)
```
GET  /api/v1/euclid/search?target_name=...   Search for Euclid images
POST /api/v1/euclid/fetch                     Download and cache image
GET  /api/v1/images                          List cached images
```

### Analysis (Phase 1)
```
POST /api/v1/analyze/image/{image_id}        Run single-image inference
GET  /api/v1/analyze/status/{image_id}       Check analysis status
GET  /api/v1/analyze/results/{image_id}      Get analysis results
```

### Training (Phase 2+)
```
GET  /api/v1/training/history                List training runs
GET  /api/v1/training/{run_id}/metrics       Get training metrics
```

### Evaluation (Phase 2+)
```
GET  /api/v1/evaluation/{training_run_id}    Get eval metrics
```

## File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                  FastAPI app initialization
│   ├── database.py              SQLAlchemy configuration
│   ├── models.py                ORM models (Image, AnalysisRun, etc.)
│   ├── schemas.py               Pydantic request/response schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cloud_fetcher.py     Euclid S3 + astroquery integration
│   │   ├── multistage_inference.py   3-stage inference pipeline
│   │   └── analysis_log.py      Database operations
│   │
│   └── routes/
│       ├── __init__.py
│       ├── euclid.py            (To be created - Phase 1)
│       ├── analyze.py           (To be created - Phase 1)
│       ├── training.py          (To be created - Phase 2)
│       ├── evaluation.py        (To be created - Phase 2)
│       ├── results.py           (To be created - Phase 3)
│       └── config.py            (To be created - Phase 2)
│
├── tests/
│   ├── __init__.py
│   └── test_api.py              (To be created)
│
├── requirements.txt             Python dependencies
├── docker-compose.yml           PostgreSQL container definition
├── .env.example                 Environment template
└── README.md                    This file
```

## Database Schema

**Images Table**
```sql
id | euclid_id | source | fetch_date | s3_url | local_path | metadata
```

**AnalysisRuns Table**
```sql
id | image_id | model_version | run_timestamp | stage_1_result | stage_2_results | stage_3_results | consensus_result | status
```

**TrainingRuns Table**
```sql
id | model_name | config_name | best_epoch | best_val_auc | checkpoint_path | metrics_json
```

**EvaluationResults Table**
```sql
id | training_run_id | split | roc_auc | precision | recall | confusion_matrix
```

**FlaggedImages Table** (QA workflow)
```sql
id | analysis_run_id | reason | user_notes | reviewed | final_label
```

## Multi-Stage Inference Pipeline

### Stage 1: Full-Image Analysis (Fast)
```
Input: Full image (any size)
  ↓
Resize to 224×224
  ↓
Model inference
  ↓
Output: confidence score

If confidence < threshold → Return NON-LENS (stop)
Else → Continue to Stage 2
```

**Why**: Quickly reject obvious non-lenses without expensive computation

### Stage 2: Chunk Grid Analysis (Medium)
```
Input: Full image + Stage 1 confidence
  ↓
Divide into 512×512 tiles (configurable stride)
  ↓
Model inference on each tile
  ↓
Output: confidence for each tile

If no tiles > threshold → Return NON-LENS
Else → Continue to Stage 3
```

**Why**: Identify lens candidates in image regions

### Stage 3: Sub-Chunk Refinement (Fine)
```
Input: Full image + high-confidence tiles from Stage 2
  ↓
On each Stage 2 region: Divide into 128×128 sub-tiles
  ↓
Model inference on sub-tiles
  ↓
Generate heatmap
  ↓
Output: final consensus prediction + heatmap

Final prediction = consensus of Stage 2/3 results
```

**Why**: High-resolution detection + explainability via heatmap

## Usage Examples

### Example 1: Analyze Euclid Image

```python
import requests

# Search for Euclid images
response = requests.get(
    "http://localhost:8000/api/v1/euclid/search",
    params={"target_name": "TYC 4429-1677-1", "radius_arcsec": 10}
)
images = response.json()

# Fetch first image
response = requests.post(
    "http://localhost:8000/api/v1/euclid/fetch",
    json={"euclid_id": images[0]["euclid_id"]}
)
cached_image = response.json()

# Run inference
response = requests.post(
    "http://localhost:8000/api/v1/analyze/image/1"
)
analysis = response.json()
print(f"Prediction: {analysis['final_prediction']}")
print(f"Confidence: {analysis['final_confidence']:.3f}")
```

### Example 2: Check GPU Status

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "ok",
  "database": "connected",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true
}
```

## Development Commands

### Start Database Only
```bash
docker-compose up -d postgres
```

### Stop & Remove Containers
```bash
docker-compose down
```

### View Database Logs
```bash
docker-compose logs -f postgres
```

### Access PostgreSQL CLI
```bash
docker-compose exec postgres psql -U sglds_user -d sglds_db
```

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black app/
isort app/
```

### Type Check
```bash
mypy app/
```

## Troubleshooting

### Error: "Could not translate host name "postgres" to address"
- Ensure docker-compose is running: `docker-compose up -d postgres`
- Check network: `docker network ls`

### Error: "Model checkpoint not found"
- Verify path in .env: `MODEL_CHECKPOINT_PATH`
- Train model first: `cd .. && python -m lens_detection.train --config configs/training_data.yaml`

### GPU Not Detected
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify PyTorch installation: `pip install torch==2.5.1+cu121`

### Database Connection Failed
- Check PostgreSQL is running: `docker-compose ps`
- Verify credentials in .env match docker-compose.yml

## Next Steps (Phase 2)

- [ ] Implement route modules (euclid.py, analyze.py, training.py, etc.)
- [ ] Add WebSocket support for real-time streaming
- [ ] Add caching layer (Redis)
- [ ] Implement batch analysis endpoint
- [ ] Add authentication & role-based access control

## Performance Targets

| Component | Target Time |
|-----------|------------|
| Stage 1   | 1–2 sec    |
| Stage 2   | 5–10 sec   |
| Stage 3   | 10–20 sec  |
| **Total** | **<30 sec**|

Measured on RTX 4050, 512×512 image input.

## Team Deployment

For team-shared dashboard on local network:

1. **Start backend on your machine**:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. **Other team members connect** (replace with your IP):
   ```
   http://<your-machine-ip>:8000
   ```

3. **Database**: PostgreSQL runs in Docker container (accessible to all)

4. **Frontend** (Phase 1B): Will connect to this backend

## Contributing

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write tests for new features
- Commit messages: `[FEATURE/BUGFIX] Description`

## License

Academic - SGLDS College Project

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review IMPLEMENTATION_PLAN.md for architectural details
3. Check database health: `GET /api/v1/health`
