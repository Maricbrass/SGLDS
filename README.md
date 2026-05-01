# Detection of Strong Gravitational Lenses in Euclid Space Telescope Images using Swin Transformer Networks

This repository implements a research pipeline for automatic strong gravitational lens detection in Euclid-like imaging data. The main focus is hierarchical vision transformers, especially Swin Transformer networks, with comparative baselines from CNN and vanilla Vision Transformer models.

## Research Summary

Strong gravitational lenses are rare but scientifically important because they constrain dark matter distribution, galaxy evolution, and cosmological parameters. Euclid-scale surveys will produce imaging volumes that are too large for manual inspection, so this project explores an automated detector that can work at survey scale and remain effective at very low false-positive rates.

The current codebase supports binary classification for `lens` versus `non_lens` and is designed around reproducible training, benchmark evaluation, and inference on Euclid-like cutouts.

## Project Progress

The repository now includes a working Phase 1 backend and a connected frontend dashboard.

- Backend API, database models, Euclid cloud access, and multi-stage inference are implemented and validated with tests.
- Frontend pages for dashboard, search, analyze, gallery, and settings are wired to the backend API.
- The frontend production build passes with `npm run build`.

Remaining planned work:

- Authentication and role-based access control.
- Real-time stage progress streaming.
- Frontend test coverage and CI automation.

## Abstract

This project proposes a deep learning framework for detecting strong gravitational lenses in Euclid-like imaging data using Swin Transformer architectures. Compared with standard CNNs, Swin Transformers can model longer-range spatial structure and arc-like morphology that are characteristic of lensing systems. The models are trained on Euclid-like simulated data and benchmarked against established lensing datasets, with evaluation centered on survey-relevant metrics such as ROC-AUC and true positive rate at extremely low false-positive thresholds. The goal is to assess whether hierarchical transformer models provide a scalable and accurate candidate-filtering solution for future Euclid data pipelines.

## Motivation

The problem is driven by the size and scientific value of upcoming Euclid surveys. Strong lenses are uncommon, but each confirmed system supports studies in dark matter, galaxy structure, and cosmology. Automated detection is therefore essential, and transformer-based vision models are a timely direction to investigate for this kind of astrophysical image analysis.

## Objectives

- Study Euclid imaging characteristics and strong-lens morphology.
- Train and validate CNN, Vision Transformer, and Swin Transformer models.
- Benchmark results using standard astronomical datasets.
- Analyze performance at low false-positive operating points.
- Assess feasibility for integration into survey-scale pipelines.

## Methodology

1. Review Euclid mission imaging properties and strong-lens literature.
2. Prepare Euclid-like and Bologna Lens Challenge datasets.
3. Apply preprocessing and augmentation to image cutouts.
4. Train CNN, ViT, and Swin Transformer baselines.
5. Tune hyperparameters and compare validation behavior.
6. Evaluate using ROC-AUC, TPR@FPR, and inference-time usability.

## Project Scope

The repository currently covers the model-development loop:

- Dataset loading from CSV split files.
- Image normalization for `png`, `jpg`, `npy`, and `fits` inputs.
- Model selection through configurable `timm` backbones.
- Training with cross-entropy optimization.
- Evaluation with survey-relevant ranking metrics.
- Single-image inference from saved checkpoints.

## Evaluation Metrics

The main reported metrics are:

- ROC-AUC.
- TPR at FPR = 1e-2.
- TPR at FPR = 1e-3.
- TPR at FPR = 1e-4.

These metrics matter because Euclid follow-up pipelines care more about recovering true lenses under strict false-positive budgets than about average accuracy alone.

## Directory Layout

```
configs/
    swin_euclid.yaml
    euclid_q1_finetune.yaml
    euclid_q1_rtx4050.yaml
    smoke_cpu.yaml
lens_detection/
    config.py
    data.py
    models.py
    metrics.py
    train.py
    evaluate.py
    infer.py
scripts/
    make_dummy_dataset.py
    fetch_euclid_q1_data.py
    generate_euclid_targets_grid.py
    fetch_euclid_targets_batch.py
    build_euclid_lens_dataset.py
    label_cutouts_cli.py
```

## Setup

```bash
py -3.10 -m venv .venv310
.venv310\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install timm scikit-learn pandas numpy pyyaml matplotlib astropy pillow tqdm 's3fs>=2024.6.1' 'astroquery>=0.4.10'
```

Python 3.10 has been the most stable runtime on this Windows machine. Python 3.12 and 3.14 builds previously hit `c10.dll` initialization issues.

## Quick Start

1. Generate a local dummy dataset.

```bash
python scripts/make_dummy_dataset.py
```

2. Train the Swin baseline.

```bash
python -m lens_detection.train --config configs/swin_euclid.yaml
```

Optional smoke run for a fast CPU sanity check:

```bash
python -m lens_detection.train --config configs/smoke_cpu.yaml
```

3. Evaluate the best checkpoint.

```bash
python -m lens_detection.evaluate --config configs/swin_euclid.yaml --checkpoint runs/swin_euclid_baseline/best.pt --split test
```

4. Run inference on one image.

```bash
python -m lens_detection.infer --config configs/swin_euclid.yaml --checkpoint runs/swin_euclid_baseline/best.pt --image data/euclid_like/images/lens_0000.png
```

## Model Comparison Study (Swin vs ViT vs CNN)

To fairly compare three model architectures on the same dataset with identical hyperparameters:

```bash
python -m lens_detection.benchmark
```

This will:
1. Train Swin Transformer, Vision Transformer, and ResNet50 sequentially
2. Log comprehensive metrics (AUC, TPR@FPR, convergence, training time)
3. Generate comparison visualizations and reports

Output saved to `runs/model_comparison/`:
- `model_comparison.json` – Structured metrics
- `model_comparison.png` – 4-panel comparison visualization
- `COMPARISON_REPORT.md` – Detailed findings

See [MODEL_COMPARISON_README.md](MODEL_COMPARISON_README.md) for full methodology, fair comparison setup details, output artifacts, and troubleshooting.

### Individual Model Training

Train each architecture separately with fair-comparison configs:

```bash
python -m lens_detection.train --config configs/swin_comparison.yaml
python -m lens_detection.train --config configs/vit_comparison.yaml
python -m lens_detection.train --config configs/cnn_comparison.yaml
```

All three use identical hyperparameters (lr=0.0001, seed=42, 10 epochs) to ensure unbiased results.

## Euclid Q1 Workflow

The repository also includes a reproducible workflow for fetching Euclid Q1 cutouts and associated metadata.

```bash
python scripts/fetch_euclid_q1_data.py --target-name "TYC 4429-1677-1" --out-dir data/euclid_q1
```

Typical outputs include:

- `data/euclid_q1/cutouts_manifest.csv`
- `data/euclid_q1/cutouts/*.fits` and `data/euclid_q1/cutouts/*.png`
- `data/euclid_q1/spectra/*_spectrum.ecsv` when a spectrum exists
- `data/euclid_q1/summary.json`

You can run inference directly on a retrieved cutout:

```bash
python -m lens_detection.infer --config configs/swin_euclid.yaml --checkpoint runs/swin_euclid_baseline/best.pt --image data/euclid_q1/cutouts/VIS_cutout.png
```

## Building A Labeled Dataset

1. Create or edit `data/euclid_q1/labels.csv`.

```csv
image_path,label
NISP_H_cutout.png,1
```

Use `1` for lens and `0` for non-lens.

2. Build train/validation/test CSVs.

```bash
python scripts/build_euclid_lens_dataset.py --annotations data/euclid_q1/labels.csv --source-dir data/euclid_q1/cutouts --out-dir data/euclid_q1_dataset --copy-images
```

3. Fine-tune on the labeled Euclid dataset.

```bash
python -m lens_detection.train --config configs/euclid_q1_finetune.yaml
```

4. Evaluate the checkpoint.

```bash
python -m lens_detection.evaluate --config configs/euclid_q1_finetune.yaml --checkpoint runs/euclid_q1_finetune/best.pt --split test
```

## Recommended Labeling Workflow

1. Generate a target grid and fetch many cutouts.

```bash
python scripts/generate_euclid_targets_grid.py --ra-deg 273.1173023 --dec-deg 68.20761349 --side-arcmin 12 --step-arcmin 1.5 --out-csv data/euclid_q1/targets.csv
python scripts/fetch_euclid_targets_batch.py --targets-csv data/euclid_q1/targets.csv --out-dir data/euclid_q1 --skip-existing
```

2. Label cutouts in the CLI.

```bash
python scripts/label_cutouts_cli.py --images-dir data/euclid_q1/cutouts --labels-csv data/euclid_q1/labels.csv --queue-csv data/euclid_q1/label_queue.csv --show-image
```

3. Rebuild the dataset and retrain with the GPU config when enough labels are available.

```bash
python -m lens_detection.train --config configs/euclid_q1_rtx4050.yaml
```

## Training On Folder-Labeled Data

If your new labels are organized as class folders under `Training_data/0` and `Training_data/1`, build the split CSVs and train with the dedicated config:

```bash
python scripts/build_folder_labeled_dataset.py --source-dir Training_data --out-dir data/training_data_dataset
python -m lens_detection.train --config configs/training_data.yaml
```

The generated splits are stored in `data/training_data_dataset/`, while the images remain in `Training_data/`.

## Hardware And Software Requirements

- NVIDIA GPU such as RTX 3060, T4, or equivalent.
- At least 16 GB RAM.
- Python with PyTorch, NumPy, OpenCV, Matplotlib, and CUDA support for GPU runs.

## Innovativeness

The project explores hierarchical transformer models for astronomical image classification, which is still an emerging area for Euclid-scale data. The emphasis on low false-positive performance and survey compatibility makes the work more relevant to real-world candidate filtering than ordinary image-classification benchmarks.

## Societal Relevance

The project contributes to space-science infrastructure by improving automated analysis of large astronomical datasets. Better gravitational-lens detection supports cosmology, dark matter research, and large-scale structure studies.

## Paper

The research draft is tracked in [paper.md](paper.md).



cd /d c:\Users\Maric\Desktop\asglds\SGLDS && .venv310\Scripts\python -m lens_detection.train --config configs/training_data.yaml

cd /d c:\Users\Maric\Desktop\asglds\SGLDS && .venv310\Scripts\python scripts\build_folder_labeled_dataset.py --source-dir Training_data --out-dir data/training_data_dataset