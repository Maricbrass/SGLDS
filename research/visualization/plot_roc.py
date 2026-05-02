import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def plot_comparison_metrics(json_path, output_dir):
    """
    Generate professional plots from the comparison JSON.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = data["comparison"]
    models = ["swin", "vit", "cnn"]
    model_labels = ["Swin-T", "ViT-S", "ResNet-50"]
    
    # 1. Plot AUC and TPR comparison
    plt.figure(figsize=(12, 6))
    
    # Extract data
    auc_values = []
    tpr_1e3_values = []
    for m in metrics:
        if m["metric"] == "Best AUC":
            auc_values = [m[model] for model in models]
        if m["metric"] == "TPR@FPR=1e-3":
            tpr_1e3_values = [m[model] for model in models]
            
    x = np.arange(len(model_labels))
    width = 0.35
    
    plt.bar(x - width/2, auc_values, width, label="ROC AUC", color="#00e5ff", alpha=0.8)
    plt.bar(x + width/2, tpr_1e3_values, width, label="TPR @ FPR=10⁻³", color="#bf5aff", alpha=0.8)
    
    plt.ylabel("Score")
    plt.title("Performance Comparison across Architectures", fontsize=14, fontweight="bold")
    plt.xticks(x, model_labels)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.ylim(0.7, 1.0)
    
    plt.savefig(output_dir / "performance_bars.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated performance bars: {output_dir / 'performance_bars.png'}")

if __name__ == "__main__":
    json_path = "research/artifacts/data/model_comparison.json"
    if Path(json_path).exists():
        plot_comparison_metrics(json_path, "research/artifacts/plots")
