import os
import sys
import json
from pathlib import Path
import shutil

# Ensure the root of the project is in the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.visualization.plot_roc import plot_comparison_metrics
from research.visualization.attention_maps import generate_swin_attention_map

def main():
    print("Starting Research Data Collection...")
    
    # 1. Setup Directories
    base_dir = Path("research/artifacts")
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    reports_dir = base_dir / "reports"
    
    for d in [data_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Collect existing benchmark results
    src_data = Path("runs/model_comparison")
    if src_data.exists():
        print(f"Collecting data from {src_data}...")
        for f in src_data.glob("*"):
            shutil.copy(f, data_dir)
    else:
        print("⚠️ Warning: runs/model_comparison not found. Please run the benchmark script first.")
        return

    # 3. Generate Visualizations
    print("Generating figures for paper...")
    json_path = data_dir / "model_comparison.json"
    if json_path.exists():
        plot_comparison_metrics(json_path, plots_dir)
    
    # Generate a demo attention map from one of the images
    # We use a fallback if no real image is found
    demo_image = Path("data/example_lens.png") # Check if exists
    if not demo_image.exists():
        # Look for any fits/png in euclid_cache
        matches = list(Path("euclid_cache").glob("*.png")) + list(Path("euclid_cache").glob("*.fits"))
        if matches:
            demo_image = matches[0]
            
    if demo_image.exists():
        generate_swin_attention_map(demo_image, plots_dir / "fig3_attention_analysis.png")

    # 4. Generate Summary Report
    print("Creating summary report...")
    with open(json_path, "r") as f:
        comp_data = json.load(f)
        
    report_content = f"""# Research Data Summary: SGLDS Architecture Study

## Data Overview
This folder contains collected metrics and visualizations comparing Vision Transformer (Swin, ViT) and CNN architectures for Strong Gravitational Lens detection in Euclid Q1 data.

## Key Performance Indicators
"""
    for m in comp_data["comparison"]:
        report_content += f"- **{m['metric']}**: Swin ({m['swin']:.4f}), ViT ({m['vit']:.4f}), CNN ({m['cnn']:.4f})\n"

    report_content += "\n## Figure List\n"
    report_content += "1. `performance_bars.png`: Comparison of AUC and TPR@FPR across architectures.\n"
    report_content += "2. `fig3_attention_analysis.png`: Simulated Swin Transformer attention maps highlighting lens features.\n"
    report_content += "3. `model_comparison.png`: Consolidated metrics dashboard.\n"

    with open(reports_dir / "RESEARCH_SUMMARY.md", "w") as f:
        f.write(report_content)
        
    print(f"Research data collection complete. Results stored in {base_dir}")

if __name__ == "__main__":
    main()
    