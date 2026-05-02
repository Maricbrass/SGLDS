# Research Data Collection Plan

The goal is to consolidate all experimental results and generate publication-quality figures for the research paper.

## Completed Tasks
- [x] Create directory structure in `research/artifacts/` (data, plots, reports).
- [x] Collect raw benchmark results from `runs/model_comparison/`.
- [x] Implement `research/visualization/plot_roc.py` for professional performance plotting.
- [x] Implement `research/visualization/attention_maps.py` for model interpretability visualization.
- [x] Implement `research/runner/generate_paper_data.py` to automate the collection and generation process.
- [x] Generate `research/artifacts/reports/RESEARCH_SUMMARY.md`.

## Data Artifacts Generated
- `research/artifacts/data/model_comparison.json`: Raw comparison metrics.
- `research/artifacts/plots/performance_bars.png`: Architecture comparison chart.
- `research/artifacts/plots/fig3_attention_analysis.png`: Swin Transformer attention visualization.
- `research/artifacts/reports/RESEARCH_SUMMARY.md`: Executive summary of findings.

## Findings Summary
- **Swin Transformer** leads with an AUC of **0.9918**.
- **CNN (ResNet-50)** maintains slightly higher performance at extremely low FPR (1e-3), but Swin is more consistent across scales.
- **Inference Time**: ViT-Small is the fastest ({1929.67/60:.1f}m), while CNN is significantly slower ({3001.8/60:.1f}m) on the same dataset.
