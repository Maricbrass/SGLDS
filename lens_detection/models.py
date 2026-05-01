"""Model definitions for lens detection.

Supports Swin Transformers, Vision Transformers, and CNN architectures.
"""
from __future__ import annotations

import timm
import torch.nn as nn


# Model registry grouped by architecture family
MODEL_REGISTRY = {
    # Swin Transformer models (hierarchical attention)
    "swin_tiny_patch4_window7_224": {
        "family": "swin",
        "size": "tiny",
        "description": "Swin Tiny (Hierarchical Vision Transformer)",
    },
    "swin_small_patch4_window7_224": {"family": "swin", "size": "small", "description": "Swin Small"},
    "swin_base_patch4_window7_224": {"family": "swin", "size": "base", "description": "Swin Base"},
    # Vision Transformer models (standard ViT)
    "vit_tiny_patch16_224": {"family": "vit", "size": "tiny", "description": "ViT Tiny"},
    "vit_small_patch16_224": {"family": "vit", "size": "small", "description": "ViT Small"},
    "vit_base_patch16_224": {"family": "vit", "size": "base", "description": "ViT Base"},
    # CNN models (ResNet)
    "resnet18": {"family": "cnn_resnet", "size": "18", "description": "ResNet-18 (CNN)"},
    "resnet34": {"family": "cnn_resnet", "size": "34", "description": "ResNet-34 (CNN)"},
    "resnet50": {"family": "cnn_resnet", "size": "50", "description": "ResNet-50 (CNN)"},
    "resnet101": {"family": "cnn_resnet", "size": "101", "description": "ResNet-101 (CNN)"},
    # CNN models (EfficientNet)
    "efficientnet_b0": {"family": "cnn_efficientnet", "size": "b0", "description": "EfficientNet-B0 (CNN)"},
    "efficientnet_b1": {"family": "cnn_efficientnet", "size": "b1", "description": "EfficientNet-B1 (CNN)"},
    "efficientnet_b2": {"family": "cnn_efficientnet", "size": "b2", "description": "EfficientNet-B2 (CNN)"},
}

SUPPORTED_MODELS = set(MODEL_REGISTRY.keys())


def get_model_family(model_name: str) -> str:
    """Get the family (swin, vit, cnn_resnet, cnn_efficientnet) of a model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]["family"]


def get_model_description(model_name: str) -> str:
    """Get human-readable description of a model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]["description"]


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a model from the registry.

    Args:
        model_name: Name of the model (must be in SUPPORTED_MODELS)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights

    Returns:
        Instantiated model
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported: {sorted(SUPPORTED_MODELS)}"
        )

    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
