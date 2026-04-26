from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LensImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, root_dir: str | Path, image_size: int = 224):
        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: Path) -> Image.Image:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            img = Image.open(path).convert("RGB")
            return img

        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim == 3 and arr.shape[0] in {1, 3}:
                arr = np.moveaxis(arr, 0, -1)
            arr = arr.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        if path.suffix.lower() == ".fits":
            from astropy.io import fits

            with fits.open(path) as hdu:
                arr = hdu[0].data.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        raise ValueError(f"Unsupported image extension: {path.suffix}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root_dir / str(row["image_path"])
        image = self._load_image(img_path)
        image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label


def build_loaders(data_cfg):
    train_ds = LensImageDataset(data_cfg.train_csv, data_cfg.root_dir, data_cfg.image_size)
    val_ds = LensImageDataset(data_cfg.val_csv, data_cfg.root_dir, data_cfg.image_size)
    test_ds = LensImageDataset(data_cfg.test_csv, data_cfg.root_dir, data_cfg.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
