# Research Data Summary: SGLDS Architecture Study

## Data Overview
This folder contains collected metrics and visualizations comparing Vision Transformer (Swin, ViT) and CNN architectures for Strong Gravitational Lens detection in Euclid Q1 data.

## Key Performance Indicators
- **Best AUC**: Swin (0.9918), ViT (0.9900), CNN (0.9800)
- **TPR@FPR=1e-2**: Swin (1.0000), ViT (0.9993), CNN (1.0000)
- **TPR@FPR=1e-3**: Swin (0.9993), ViT (0.9993), CNN (1.0000)
- **TPR@FPR=1e-4**: Swin (0.9990), ViT (0.9993), CNN (1.0000)
- **Best Epoch**: Swin (10.0000), ViT (5.0000), CNN (9.0000)
- **Time (sec)**: Swin (2133.3953), ViT (1929.6706), CNN (3001.8385)

## Figure List
1. `performance_bars.png`: Comparison of AUC and TPR@FPR across architectures.
2. `fig3_attention_analysis.png`: Simulated Swin Transformer attention maps highlighting lens features.
3. `model_comparison.png`: Consolidated metrics dashboard.
