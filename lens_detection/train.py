from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm

from lens_detection.config import load_config
from lens_detection.data import build_loaders
from lens_detection.metrics import classification_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device, amp_enabled: bool):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if amp_enabled and device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def evaluate(model, loader, device, pos_label=1):
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
    parser = argparse.ArgumentParser(description="Train lens detector")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    from lens_detection.models import build_model

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.train.amp and device.type == "cuda")
    configure_runtime(device, amp_enabled)

    train_loader, val_loader, _ = build_loaders(cfg.data)
    model = build_model(cfg.model.name, cfg.model.num_classes, cfg.model.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scaler = amp.GradScaler(enabled=amp_enabled)

    best_auc = -1.0
    best_epoch = -1
    best_path = out_dir / "best.pt"
    best_val_metrics = {}
    
    # Track epoch history for comprehensive logging
    epoch_history = []
    start_time = time.time()

    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_metrics = evaluate(model, val_loader, device, pos_label=cfg.train.positive_class)

        # Record epoch metrics
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_auc": float(val_metrics["roc_auc"]),
            "val_tpr_at_fpr_1e-2": float(val_metrics["tpr_at_fpr_1e-2"]),
            "val_tpr_at_fpr_1e-3": float(val_metrics["tpr_at_fpr_1e-3"]),
            "val_tpr_at_fpr_1e-4": float(val_metrics["tpr_at_fpr_1e-4"]),
        }
        epoch_history.append(epoch_record)

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_epoch = epoch + 1
            best_val_metrics = val_metrics.copy()
            torch.save({"model_state": model.state_dict(), "config": args.config}, best_path)

        print(
            f"epoch={epoch + 1} train_loss={train_loss:.5f} "
            f"val_auc={val_metrics['roc_auc']:.5f} "
            f"tpr@1e-3={val_metrics['tpr_at_fpr_1e-3']:.5f}"
        )

    total_time = time.time() - start_time

    # Save comprehensive metrics
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc),
        "best_val_tpr_at_fpr_1e-2": float(best_val_metrics.get("tpr_at_fpr_1e-2", 0.0)),
        "best_val_tpr_at_fpr_1e-3": float(best_val_metrics.get("tpr_at_fpr_1e-3", 0.0)),
        "best_val_tpr_at_fpr_1e-4": float(best_val_metrics.get("tpr_at_fpr_1e-4", 0.0)),
        "total_epochs": cfg.train.epochs,
        "total_training_time_seconds": float(total_time),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "config": {
            "model_name": cfg.model.name,
            "pretrained": cfg.model.pretrained,
            "learning_rate": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            "batch_size": cfg.data.batch_size,
            "image_size": cfg.data.image_size,
            "seed": cfg.train.seed,
        },
        "dataset": {
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        },
        "epoch_history": epoch_history,
    }

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Saved best checkpoint: {best_path}")
    print(f"Training completed in {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
