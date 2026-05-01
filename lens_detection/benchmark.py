"""Model comparison benchmark script.

Trains Swin, ViT, and CNN models on the same dataset with identical hyperparameters
and generates comprehensive comparison metrics and visualizations.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
import yaml


def get_output_dir_from_config(config_path: Path) -> Path:
    """Read the configured training output directory from a YAML config."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    train_section = config.get("train") or {}
    output_dir = train_section.get("output_dir")
    if not output_dir:
        raise KeyError(f"Missing train.output_dir in {config_path}")

    return Path(output_dir)


def run_training(config_path: str) -> dict[str, Any]:
    """Run training with the given config and return final metrics."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"\n{'='*70}")
    print(f"Training with config: {config_path.name}")
    print(f"{'='*70}")

    # Run training
    result = subprocess.run(
        [sys.executable, "-m", "lens_detection.train", "--config", str(config_path)],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed for config: {config_path}")

    # Read final metrics from the output directory
    output_dir = get_output_dir_from_config(config_path)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    metrics_file = output_dir / "final_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    return metrics


def compare_models(
    swin_metrics: dict[str, Any],
    vit_metrics: dict[str, Any],
    cnn_metrics: dict[str, Any],
    output_dir: str = "runs/model_comparison",
) -> None:
    """Generate comparison report and visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model names and key metrics
    swin_name = swin_metrics["config"]["model_name"]
    vit_name = vit_metrics["config"]["model_name"]
    cnn_name = cnn_metrics["config"]["model_name"]

    models = ["Swin", "ViT", "CNN"]
    metrics_data = [swin_metrics, vit_metrics, cnn_metrics]

    print("\n" + "="*70)
    print("MODEL COMPARISON REPORT")
    print("="*70)

    # Create comparison table
    comparison_report = {
        "models": {
            "swin": swin_name,
            "vit": vit_name,
            "cnn": cnn_name,
        },
        "comparison": [],
    }

    print(f"\n{'Metric':<30} {'Swin':<15} {'ViT':<15} {'CNN':<15}")
    print("-" * 75)

    metrics_to_compare = [
        ("best_val_auc", "Best AUC"),
        ("best_val_tpr_at_fpr_1e-2", "TPR@FPR=1e-2"),
        ("best_val_tpr_at_fpr_1e-3", "TPR@FPR=1e-3"),
        ("best_val_tpr_at_fpr_1e-4", "TPR@FPR=1e-4"),
        ("best_epoch", "Best Epoch"),
        ("total_training_time_seconds", "Time (sec)"),
    ]

    for metric_key, metric_name in metrics_to_compare:
        values = [m.get(metric_key, 0) for m in metrics_data]

        if metric_key == "total_training_time_seconds":
            formatted = [f"{v:.1f}s" for v in values]
        elif metric_key == "best_epoch":
            formatted = [f"{int(v)}" for v in values]
        else:
            formatted = [f"{v:.4f}" for v in values]

        print(f"{metric_name:<30} {formatted[0]:<15} {formatted[1]:<15} {formatted[2]:<15}")

        comparison_report["comparison"].append({
            "metric": metric_name,
            "swin": values[0],
            "vit": values[1],
            "cnn": values[2],
        })

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison: Swin vs ViT vs CNN", fontsize=16, fontweight="bold")

    # Plot 1: AUC comparison
    aucs = [m["best_val_auc"] for m in metrics_data]
    axes[0, 0].bar(models, aucs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[0, 0].set_title("Best Validation AUC")
    axes[0, 0].set_ylim([0.8, 1.0])
    for i, v in enumerate(aucs):
        axes[0, 0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    # Plot 2: TPR@FPR curves
    tpr_1e2 = [m["best_val_tpr_at_fpr_1e-2"] for m in metrics_data]
    tpr_1e3 = [m["best_val_tpr_at_fpr_1e-3"] for m in metrics_data]
    tpr_1e4 = [m["best_val_tpr_at_fpr_1e-4"] for m in metrics_data]

    x = np.arange(len(models))
    width = 0.25
    axes[0, 1].bar(x - width, tpr_1e2, width, label="FPR=1e-2", color="#1f77b4")
    axes[0, 1].bar(x, tpr_1e3, width, label="FPR=1e-3", color="#ff7f0e")
    axes[0, 1].bar(x + width, tpr_1e4, width, label="FPR=1e-4", color="#2ca02c")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("TPR at Low False Positive Rates")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1.0])

    # Plot 3: Training time
    times = [m["total_training_time_seconds"] / 60 for m in metrics_data]
    axes[1, 0].bar(models, times, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1, 0].set_ylabel("Time (minutes)")
    axes[1, 0].set_title("Total Training Time")
    for i, v in enumerate(times):
        axes[1, 0].text(i, v + 1, f"{v:.1f}m", ha="center", fontweight="bold")

    # Plot 4: Convergence (best epoch)
    best_epochs = [m["best_epoch"] for m in metrics_data]
    axes[1, 1].bar(models, best_epochs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1, 1].set_ylabel("Epoch Number")
    axes[1, 1].set_title("Best Epoch (Convergence Speed)")
    axes[1, 1].set_ylim([0, 30])
    for i, v in enumerate(best_epochs):
        axes[1, 1].text(i, v + 0.5, f"Ep {int(v)}", ha="center", fontweight="bold")

    plt.tight_layout()
    comparison_file = output_dir / "model_comparison.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Comparison plot saved: {comparison_file}")

    # Save JSON comparison
    json_file = output_dir / "model_comparison.json"
    with open(json_file, "w") as f:
        json.dump(comparison_report, f, indent=2)
    print(f"✓ Comparison report saved: {json_file}")

    # Generate summary markdown
    summary_md = f"""# Model Comparison Study: Swin vs ViT vs CNN

## Executive Summary

This study compares three model architectures trained on the same lens detection dataset with identical hyperparameters.

### Models Evaluated

- **Swin**: Hierarchical Vision Transformer ({swin_name})
- **ViT**: Standard Vision Transformer ({vit_name})
- **CNN**: Convolutional Neural Network ({cnn_name})

### Key Findings

#### Best Validation AUC
- Swin: {swin_metrics["best_val_auc"]:.4f}
- ViT: {vit_metrics["best_val_auc"]:.4f}
- CNN: {cnn_metrics["best_val_auc"]:.4f}

**Winner**: {['Swin', 'ViT', 'CNN'][np.argmax(aucs)]}

#### Training Efficiency
- Swin: {times[0]:.1f} minutes (converged at epoch {int(swin_metrics["best_epoch"])})
- ViT: {times[1]:.1f} minutes (converged at epoch {int(vit_metrics["best_epoch"])})
- CNN: {times[2]:.1f} minutes (converged at epoch {int(cnn_metrics["best_epoch"])})

**Fastest**: {models[np.argmin(times)]}

#### Low False-Positive Performance (Critical for Surveys)

**TPR @ FPR = 1e-3** (1 false positive per 1000 negatives)
- Swin: {swin_metrics["best_val_tpr_at_fpr_1e-3"]:.4f}
- ViT: {vit_metrics["best_val_tpr_at_fpr_1e-3"]:.4f}
- CNN: {cnn_metrics["best_val_tpr_at_fpr_1e-3"]:.4f}

**TPR @ FPR = 1e-4** (1 false positive per 10,000 negatives)
- Swin: {swin_metrics["best_val_tpr_at_fpr_1e-4"]:.4f}
- ViT: {vit_metrics["best_val_tpr_at_fpr_1e-4"]:.4f}
- CNN: {cnn_metrics["best_val_tpr_at_fpr_1e-4"]:.4f}

### Configuration (Fair Comparison)

All models trained with identical hyperparameters:
- Batch size: {swin_metrics["config"]["batch_size"]}
- Learning rate: {swin_metrics["config"]["learning_rate"]}
- Weight decay: {swin_metrics["config"]["weight_decay"]}
- Epochs: {swin_metrics["total_epochs"]}
- Seed: {swin_metrics["config"]["seed"]} (reproducibility)
- Image size: {swin_metrics["config"]["image_size"]}

### Dataset

- Training samples: {swin_metrics["dataset"]["train_samples"]}
- Validation samples: {swin_metrics["dataset"]["val_samples"]}

### Conclusion

{['Swin', 'ViT', 'CNN'][np.argmax(aucs)]} achieved the best AUC ({max(aucs):.4f}) on this lens detection task.
However, for survey-scale applications where false positives are costly, the TPR@FPR scores are critical.
"""

    md_file = output_dir / "COMPARISON_REPORT.md"
    with open(md_file, "w") as f:
        f.write(summary_md)
    print(f"✓ Summary report saved: {md_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Swin, ViT, and CNN models on lens detection task"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only generate comparison from existing runs",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/model_comparison",
        help="Output directory for comparison results",
    )
    args = parser.parse_args()

    config_dir = Path("configs")
    configs = {
        "swin": config_dir / "swin_comparison.yaml",
        "vit": config_dir / "vit_comparison.yaml",
        "cnn": config_dir / "cnn_comparison.yaml",
    }

    # Train models if not skipping
    if not args.skip_training:
        print("\n" + "="*70)
        print("STARTING MODEL COMPARISON TRAINING")
        print("Training 3 models (Swin, ViT, CNN) with identical hyperparameters...")
        print("="*70)

        metrics = {}
        for model_type, config_path in configs.items():
            try:
                metrics[model_type] = run_training(str(config_path))
            except Exception as e:
                print(f"ERROR training {model_type}: {e}")
                return 1
    else:
        # Load from existing runs
        print("Loading metrics from existing runs...")
        metrics = {}
        run_dirs = {model_type: get_output_dir_from_config(config_path) for model_type, config_path in configs.items()}
        for model_type, run_dir in run_dirs.items():
            metrics_file = run_dir / "final_metrics.json"
            if not metrics_file.exists():
                print(f"ERROR: Metrics file not found: {metrics_file}")
                return 1
            with open(metrics_file) as f:
                metrics[model_type] = json.load(f)

    # Generate comparison
    compare_models(
        metrics["swin"],
        metrics["vit"],
        metrics["cnn"],
        output_dir=args.output_dir,
    )

    print("\n" + "="*70)
    print("✓ MODEL COMPARISON COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
