from __future__ import annotations

import argparse
import json

import torch

from lens_detection.config import load_config
from lens_detection.data import build_loaders
from lens_detection.metrics import classification_metrics


def run_eval(model, loader, device, pos_label):
    model.eval()
    probs_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, pos_label]
            probs_all.extend(probs.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    return classification_metrics(labels_all, probs_all, pos_label=pos_label)


def main():
    parser = argparse.ArgumentParser(description="Evaluate lens detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from lens_detection.models import build_model

    _, val_loader, test_loader = build_loaders(cfg.data)
    loader = val_loader if args.split == "val" else test_loader

    model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    metrics = run_eval(model, loader, device, cfg.train.positive_class)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
