# Detection of Strong Gravitational Lenses in Euclid Images

Research pipeline for strong lens detection using Swin Transformer networks, with baseline comparisons to CNN and ViT models.

## Project Focus

- Binary classification: `lens` vs `non_lens`
- Survey-relevant evaluation: ROC-AUC and TPR at very low FPR
- Architectures:
    - `swin_tiny_patch4_window7_224`
    - `vit_base_patch16_224`
    - `resnet50`

## Current Data Flow

1. Input image + label lists (`train.csv`, `val.csv`, `test.csv`) are read by `lens_detection/data.py`.
2. Images (`png/jpg/npy/fits`) are normalized and transformed.
3. Model is built via `lens_detection/models.py` using `timm`.
4. Training loop in `lens_detection/train.py` optimizes cross-entropy.
5. Evaluation in `lens_detection/evaluate.py` reports:
     - ROC-AUC
     - TPR@FPR=1e-2
     - TPR@FPR=1e-3
     - TPR@FPR=1e-4
6. Inference on a single image is available in `lens_detection/infer.py`.

## Directory Layout

```
configs/
    swin_euclid.yaml
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
python -m pip install timm scikit-learn pandas numpy pyyaml matplotlib astropy pillow tqdm s3fs 'astroquery>=0.4.10'
```

For this Windows machine, Python 3.10 with the CPU wheels above was the stable runtime. Python 3.12/3.14 builds hit `c10.dll` initialization issues.

## Quick Start (Runnable End-to-End)

1. Generate a local dummy dataset:

```bash
python scripts/make_dummy_dataset.py
```

2. Train Swin baseline:

```bash
python -m lens_detection.train --config configs/swin_euclid.yaml
```

Optional smoke run (fast CPU sanity check):

```bash
python -m lens_detection.train --config configs/smoke_cpu.yaml
```

3. Evaluate best checkpoint:

```bash
python -m lens_detection.evaluate --config configs/swin_euclid.yaml --checkpoint runs/swin_euclid_baseline/best.pt --split test
```

4. Inference on one image:

```bash
python -m lens_detection.infer --config configs/swin_euclid.yaml --checkpoint runs/swin_euclid_baseline/best.pt --image data/euclid_like/images/lens_0000.png
```

## Adapting to Euclid / Bologna Data

- Keep CSV schema as:
    - `image_path,label`
- Place images under `data/euclid_like/images`
- Update paths and training settings in `configs/swin_euclid.yaml`

## Fetch Euclid Q1 Data From Cloud

You can reproduce the Euclid Q1 cloud-access workflow (S3 browse, MER cutouts, MER object ID, associated spectrum) with:

```bash
python scripts/fetch_euclid_q1_data.py --target-name "TYC 4429-1677-1" --out-dir data/euclid_q1
```

Key outputs:

- `data/euclid_q1/cutouts_manifest.csv`
- `data/euclid_q1/cutouts/*.fits` and `data/euclid_q1/cutouts/*.png`
- `data/euclid_q1/spectra/*_spectrum.ecsv` (if spectrum exists)
- `data/euclid_q1/summary.json`

If a target has no associated spectrum in Q1, the script still saves cutouts and summary metadata.

You can run inference on a retrieved cutout image directly:

```bash
python -m lens_detection.infer --config configs/swin_euclid.yaml --checkpoint runs/swin_euclid_baseline/best.pt --image data/euclid_q1/cutouts/VIS_cutout.png
```

## Build Labeled Euclid Dataset and Train

1. Create or edit labels file at `data/euclid_q1/labels.csv`:

```csv
image_path,label
NISP_H_cutout.png,1
```

Use `label=1` for lens and `label=0` for non-lens. Add more rows as you fetch more targets.

2. Build stratified train/val/test CSVs for the training pipeline:

```bash
python scripts/build_euclid_lens_dataset.py --annotations data/euclid_q1/labels.csv --source-dir data/euclid_q1/cutouts --out-dir data/euclid_q1_dataset --copy-images
```

If you only have one class while testing the pipeline, add `--allow-single-class` (dry run only). For actual training, keep both classes (0 and 1) in train/val/test.

3. Train on your labeled Euclid dataset:

```bash
python -m lens_detection.train --config configs/euclid_q1_finetune.yaml
```

4. Evaluate checkpoint:

```bash
python -m lens_detection.evaluate --config configs/euclid_q1_finetune.yaml --checkpoint runs/euclid_q1_finetune/best.pt --split test
```

Note: meaningful lens detection requires both classes in labels (0 and 1) and enough examples per class.

## Fix Label Bottleneck (Recommended Workflow)

1. Fetch cutouts for many targets into a labeling queue:

```bash
python scripts/generate_euclid_targets_grid.py --ra-deg 273.1173023 --dec-deg 68.20761349 --side-arcmin 12 --step-arcmin 1.5 --out-csv data/euclid_q1/targets.csv
python scripts/fetch_euclid_targets_batch.py --targets-csv data/euclid_q1/targets.csv --out-dir data/euclid_q1 --skip-existing
```

This generates:

- `data/euclid_q1/label_queue.csv`
- `data/euclid_q1/cutouts/*.png` (for fast manual labeling)
- `data/euclid_q1/cutouts_fits/*.fits`

2. Label images quickly in CLI (`1` lens, `0` non_lens, `s` skip, `q` quit):

```bash
python scripts/label_cutouts_cli.py --images-dir data/euclid_q1/cutouts --labels-csv data/euclid_q1/labels.csv --queue-csv data/euclid_q1/label_queue.csv --show-image
```

3. Build strict train/val/test splits (builder enforces both classes):

```bash
python scripts/build_euclid_lens_dataset.py --annotations data/euclid_q1/labels.csv --source-dir data/euclid_q1/cutouts --out-dir data/euclid_q1_dataset --copy-images
```

4. Train lens detector:

```bash
python -m lens_detection.train --config configs/euclid_q1_rtx4050.yaml
```

5. Evaluate:

```bash
python -m lens_detection.evaluate --config configs/euclid_q1_rtx4050.yaml --checkpoint runs/euclid_q1_rtx4050/best.pt --split test
```

For your RTX 4050, keep `amp: true` and use the GPU config above. The training loop enables cuDNN benchmark mode, TF32 matmul paths, and AMP automatically when CUDA is available.

## Repository Scope

The repository now targets the lens detection pipeline only:

- `lens_detection/` for model code and training/evaluation/inference entrypoints
- `configs/` for experiment configs
- `scripts/` for utility data generation

Legacy simulation code and related legacy docs/tests were removed from active use.



