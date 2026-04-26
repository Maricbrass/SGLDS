from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lens_detection.config import load_config


def load_image(path: str, image_size: int) -> torch.Tensor:
    p = Path(path)
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        img = Image.open(p).convert("RGB")
    elif p.suffix.lower() == ".npy":
        arr = np.load(p)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[0] in {1, 3}:
            arr = np.moveaxis(arr, 0, -1)
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        raise ValueError(f"Unsupported extension: {p.suffix}")

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Inference for lens detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from lens_detection.models import build_model

    model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = load_image(args.image, cfg.data.image_size).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    print(f"p(non_lens)={probs[0].item():.6f}")
    print(f"p(lens)={probs[1].item():.6f}")


if __name__ == "__main__":
    main()
