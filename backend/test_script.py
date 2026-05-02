import os
import sys

# Ensure we're in the backend dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, init_db
from app.services.analysis_log import get_analysis_logger
from app.routes.analyze import _run_analysis_background
from app.main import app_state

def run_test():
    # Force app_state to be initialized just like in main
    app_state.torch_available = False
    app_state.inference_pipeline = None
    app_state.inference_python = os.getenv("INFERENCE_PYTHON")
    
    db = SessionLocal()
    analysis_logger = get_analysis_logger(db)
    
    # Create a dummy run
    run = analysis_logger.create_analysis_run(image_id=2)
    print(f"Created run {run.id}")
    
    # Run the background task directly
    print("Executing background task...")
    _run_analysis_background(run.id, 2, "swin_euclid_baseline")
    
    # Fetch result
    updated_run = analysis_logger.get_analysis_run(run.id)
    print(f"Run {run.id} Status: {updated_run.status}")
    print(f"Stage 1 Result: {updated_run.stage_1_result}")
    print(f"Heatmap Path: {updated_run.heatmap_path}")
    print(f"Error Message: {updated_run.error_message}")
    
    db.close()

if __name__ == "__main__":
    run_test()
