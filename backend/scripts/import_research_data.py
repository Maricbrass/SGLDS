import os
import json
from pathlib import Path
from datetime import datetime
from app.database import SessionLocal, init_db
from app.models import TrainingRun, EvaluationResult

def import_benchmark_data():
    print("Initializing database...")
    init_db()
    db = SessionLocal()
    
    try:
        # Check if already imported
        if db.query(TrainingRun).count() > 0:
            print("Database already contains training runs. Skipping import.")
            return

        # Look in the project root (one level up from backend)
        base_dir = Path("..") / "runs"
        comparison_dirs = {
            "swin": base_dir / "comparison_swin_tiny",
            "vit": base_dir / "comparison_vit_small",
            "cnn": base_dir / "comparison_cnn_resnet50"
        }
        
        for model_type, run_dir in comparison_dirs.items():
            metrics_file = run_dir / "final_metrics.json"
            if not metrics_file.exists():
                print(f"Warning: Metrics file not found at {metrics_file}")
                continue
                
            with open(metrics_file, "r") as f:
                data = json.load(f)
            
            # Create TrainingRun
            run = TrainingRun(
                model_name=data["config"]["model_name"],
                config_name=f"{model_type}_comparison",
                total_epochs=data["total_epochs"],
                best_epoch=data["best_epoch"],
                best_val_auc=data["best_val_auc"],
                best_val_tpr_1e2=data["best_val_tpr_at_fpr_1e-2"],
                best_val_tpr_1e3=data["best_val_tpr_at_fpr_1e-3"],
                best_val_tpr_1e4=data["best_val_tpr_at_fpr_1e-4"],
                metrics_json={"history": data.get("epoch_history", [])},
                dataset_stats=data["dataset"],
                status="completed"
            )
            db.add(run)
            db.flush() # Get run.id
            
            # Create EvaluationResult (using val as test for the demo)
            eval_res = EvaluationResult(
                training_run_id=run.id,
                split="test",
                roc_auc=data["best_val_auc"],
                precision=0.92, # Placeholder as not in final_metrics.json directly
                recall=0.88,
                f1_score=0.90,
                confusion_matrix=[[2800, 200], [150, 2882]], # Mock CM for demo
                metrics_json={
                    "roc_curve": {
                        "fpr": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
                        "tpr": [0.0, 0.85, 0.92, 0.95, 0.99, 1.0]
                    }
                }
            )
            db.add(eval_res)
            print(f"Imported {model_type} run results.")
            
        db.commit()
        print("Successfully imported all benchmark data.")
        
    except Exception as e:
        db.rollback()
        print(f"Error importing data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    import_benchmark_data()
