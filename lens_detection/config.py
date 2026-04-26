from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    root_dir: str
    train_csv: str
    val_csv: str
    test_csv: str
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2


@dataclass
class ModelConfig:
    name: str = "swin_tiny_patch4_window7_224"
    num_classes: int = 2
    pretrained: bool = True


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 42
    output_dir: str = "runs/swin_baseline"
    positive_class: int = 1


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_cfg = DataConfig(**raw["data"])
    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))

    return ExperimentConfig(data=data_cfg, model=model_cfg, train=train_cfg)
