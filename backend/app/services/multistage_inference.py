"""Multi-stage inference pipeline for lens detection."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import time

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

logger = logging.getLogger(__name__)


class MultiStageInferencePipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        stage_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device).eval()
        self.device = device

        self.stage_config = {
            "stage_1": {"enabled": True, "input_size": 224, "threshold": 0.5},
            "stage_2": {"enabled": True, "tile_size": 512, "stride": 512, "threshold": 0.5},
            "stage_3": {"enabled": True, "tile_size": 128, "threshold": 0.5},
        }

        if stage_config:
            self._update_config(stage_config)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _update_config(self, updates: Dict):
        for stage, config in updates.items():
            if stage in self.stage_config:
                self.stage_config[stage].update(config)

    def load_image(self, image_path: str) -> np.ndarray:
        img = PILImage.open(image_path).convert("RGB")
        return np.array(img)

    def _predict_batch(self, image_batch: torch.Tensor):
        with torch.no_grad():
            logits = self.model(image_batch)
            probs = torch.softmax(logits, dim=1)
            lens_probs = probs[:, 1].cpu().numpy()
            predictions = (lens_probs > 0.5).astype(int)
            return lens_probs, predictions

    def _prepare_tiles(self, image, tile_size, stride):
        h, w, _ = image.shape
        tiles = []
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append((tile, x, y))
        return tiles

    def _process_tiles(self, tiles, batch_size=8):
        results = {"high_confidence_tiles": [], "all_results": [], "num_tiles": len(tiles)}

        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_images = []
            coords = []

            for tile, x, y in batch_tiles:
                img = PILImage.fromarray(tile.astype("uint8")).resize((224, 224))
                batch_images.append(self.transform(img))
                coords.append((x, y))

            batch = torch.stack(batch_images).to(self.device)
            confs, preds = self._predict_batch(batch)

            for j, (c, p) in enumerate(zip(confs, preds)):
                x, y = coords[j]
                result = {"x": x, "y": y, "confidence": float(c), "prediction": int(p)}
                results["all_results"].append(result)

                if c > 0.7:
                    results["high_confidence_tiles"].append(result)

        return results

    def run_stage_1(self, image):
        img = PILImage.fromarray(image.astype("uint8")).resize((224, 224))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        conf, pred = self._predict_batch(tensor)

        return {
            "confidence": float(conf[0]),
            "prediction": int(pred[0]),
            "proceed_to_stage_2": float(conf[0]) > self.stage_config["stage_1"]["threshold"],
        }

    def run_stage_2(self, image):
        cfg = self.stage_config["stage_2"]
        tiles = self._prepare_tiles(image, cfg["tile_size"], cfg["stride"])
        results = self._process_tiles(tiles)

        return {
            "num_tiles": results["num_tiles"],
            "high_confidence_tiles": results["high_confidence_tiles"],
            "proceed_to_stage_3": len(results["high_confidence_tiles"]) > 0,
        }

    def run_stage_3(self, image, high_tiles):
        tile2 = self.stage_config["stage_2"]["tile_size"]
        tile3 = self.stage_config["stage_3"]["tile_size"]

        h, w, _ = image.shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        all_sub = []

        for t in high_tiles:
            region = image[t["y"]:t["y"]+tile2, t["x"]:t["x"]+tile2]
            sub_tiles = self._prepare_tiles(region, tile3, tile3)
            sub_results = self._process_tiles(sub_tiles)

            for r in sub_results["all_results"]:
                abs_x = t["x"] + r["x"]
                abs_y = t["y"] + r["y"]
                heatmap[abs_y:abs_y+tile3, abs_x:abs_x+tile3] += r["confidence"]
                r["abs_x"], r["abs_y"] = abs_x, abs_y
                all_sub.append(r)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        high_conf = [r["confidence"] for r in all_sub if r["confidence"] > 0.7]
        final_conf = float(np.mean(high_conf)) if high_conf else 0.0

        return {
            "heatmap": heatmap,
            "final_confidence": final_conf,
            "num_sub_chunks": len(all_sub),
        }

    # 🔥 FIXED: heatmap saving
    def save_heatmap(self, heatmap: np.ndarray, output_path: str):
        from matplotlib.cm import get_cmap
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=0, vmax=1)
        cmap = get_cmap("hot")
        img = cmap(norm(heatmap))[:, :, :3]
        img = (img * 255).astype("uint8")

        PILImage.fromarray(img).save(output_path)
        logger.info(f"Saved heatmap: {output_path}")

    def analyze(self, image_path: str, run_id: Optional[int] = None):
        start = time.time()

        image = self.load_image(image_path)

        s1 = self.run_stage_1(image)
        if not s1["proceed_to_stage_2"]:
            return {
                "final_prediction": 0,
                "final_confidence": s1["confidence"],
                "stages": {"stage_1": s1},
                "elapsed": time.time() - start,
            }

        s2 = self.run_stage_2(image)
        if not s2["proceed_to_stage_3"]:
            return {
                "final_prediction": 0,
                "final_confidence": 0.0,
                "stages": {"stage_1": s1, "stage_2": s2},
                "elapsed": time.time() - start,
            }

        s3 = self.run_stage_3(image, s2["high_confidence_tiles"])

        # SAVE HEATMAP HERE
        output_dir = Path("euclid_cache")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"heatmap_{run_id if run_id else int(time.time())}.png"
        heatmap_path = output_dir / filename

        self.save_heatmap(s3["heatmap"], str(heatmap_path))

        # Return results including stages for logging
        return {
            "final_prediction": 1 if s3["final_confidence"] > 0.5 else 0,
            "final_confidence": s3["final_confidence"],
            "heatmap_path": str(heatmap_path),
            "stages": {
                "stage_1": s1,
                "stage_2": s2,
                "stage_3": {
                    "final_confidence": s3["final_confidence"],
                    "num_sub_chunks": s3["num_sub_chunks"],
                }
            },
            "elapsed": time.time() - start,
        }


def get_inference_pipeline(model, device="cuda", stage_config=None):
    return MultiStageInferencePipeline(model, device, stage_config)