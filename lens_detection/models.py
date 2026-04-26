from __future__ import annotations

import timm
import torch.nn as nn


SUPPORTED_MODELS = {
    "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",
    "resnet50",
}


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported: {sorted(SUPPORTED_MODELS)}"
        )

    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
