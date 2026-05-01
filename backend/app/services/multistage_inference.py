"""Multi-stage inference pipeline for lens detection."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms

logger = logging.getLogger(__name__)


class MultiStageInferencePipeline:
    """
    Three-stage inference for gravitational lens detection:
    
    Stage 1: Full-image analysis (224x224)
      - Quick pass on full image
      - If confidence < threshold, return non-lens (skip stages 2-3)
      - Otherwise proceed to stage 2
      
    Stage 2: Adaptive chunk grid (e.g., 512x512 tiles)
      - Divide image into overlapping chunks
      - Run model on each chunk
      - Collect high-confidence detections
      - If no chunks above threshold, return non-lens
      - Otherwise proceed to stage 3
      
    Stage 3: Sub-chunk refinement (e.g., 128x128 tiles)
      - On high-confidence chunks from stage 2
      - Further divide into sub-chunks
      - Generate heatmap
      - Final consensus prediction
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        stage_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained PyTorch model (expects 224x224 RGB input)
            device: 'cuda' or 'cpu'
            stage_config: Configuration for stages (thresholds, chunk sizes, etc.)
        """
        self.model = model.to(device).eval()
        self.device = device
        
        # Default configuration
        self.stage_config = {
            "stage_1": {
                "enabled": True,
                "input_size": 224,
                "threshold": 0.5,  # If stage 1 < 0.5, return non-lens
            },
            "stage_2": {
                "enabled": True,
                "tile_size": 512,
                "stride": 512,  # No overlap (can increase stride < tile_size for overlap)
                "threshold": 0.5,  # Min confidence to mark chunk as lens-like
            },
            "stage_3": {
                "enabled": True,
                "tile_size": 128,
                "threshold": 0.5,
            },
        }
        
        # Override with provided config
        if stage_config:
            self._update_config(stage_config)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet defaults
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _update_config(self, updates: Dict):
        """Update stage configuration."""
        for stage, config in updates.items():
            if stage in self.stage_config:
                self.stage_config[stage].update(config)

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image and convert to numpy array.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as RGB numpy array (H, W, 3)
        """
        try:
            img = PILImage.open(image_path).convert("RGB")
            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def _predict_batch(self, image_batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model inference on batch of images.
        
        Args:
            image_batch: Tensor of shape (B, 3, 224, 224)
            
        Returns:
            Tuple of (confidences, predictions)
              - confidences: Array of shape (B,) with probability of lens (0-1)
              - predictions: Array of shape (B,) with 0/1 predictions
        """
        with torch.no_grad():
            logits = self.model(image_batch)
            # Assume model outputs logits for binary classification
            probs = torch.softmax(logits, dim=1)  # (B, 2)
            lens_probs = probs[:, 1].cpu().numpy()  # Probability of lens (class 1)
            predictions = (lens_probs > 0.5).astype(int)
            return lens_probs, predictions

    def _prepare_tiles(
        self,
        image: np.ndarray,
        tile_size: int,
        stride: int,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Extract tiles from image.
        
        Args:
            image: Image as numpy array (H, W, 3)
            tile_size: Size of tiles to extract
            stride: Stride between tiles
            
        Returns:
            List of (tile_array, x, y) tuples
        """
        h, w, _ = image.shape
        tiles = []
        
        y = 0
        while y + tile_size <= h:
            x = 0
            while x + tile_size <= w:
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append((tile, x, y))
                x += stride
            y += stride
        
        return tiles

    def _process_tiles(
        self,
        tiles: List[Tuple[np.ndarray, int, int]],
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Process tiles through model.
        
        Args:
            tiles: List of (tile_array, x, y)
            batch_size: Batch size for inference
            
        Returns:
            Dict with results: {high_confidence_tiles: [...], all_results: [...]}
        """
        results = {
            "high_confidence_tiles": [],
            "all_results": [],
            "num_tiles": len(tiles),
        }
        
        if not tiles:
            return results

        # Process in batches
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_images = []
            batch_coords = []
            
            for tile, x, y in batch_tiles:
                # Resize to 224x224 for model
                tile_pil = PILImage.fromarray(tile.astype("uint8"))
                tile_resized = tile_pil.resize((224, 224))
                tile_tensor = self.transform(tile_resized)
                batch_images.append(tile_tensor)
                batch_coords.append((x, y))
            
            batch = torch.stack(batch_images).to(self.device)
            confidences, predictions = self._predict_batch(batch)
            
            for j, (conf, pred) in enumerate(zip(confidences, predictions)):
                x, y = batch_coords[j]
                result = {
                    "x": int(x),
                    "y": int(y),
                    "confidence": float(conf),
                    "prediction": int(pred),
                }
                results["all_results"].append(result)
                
                # Mark high-confidence detections
                if conf > 0.7:  # Configurable threshold
                    results["high_confidence_tiles"].append(result)
        
        return results

    def run_stage_1(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Stage 1: Quick full-image analysis.
        
        Args:
            image: Image as numpy array (H, W, 3)
            
        Returns:
            Dict with {confidence, prediction, proceed_to_stage_2}
        """
        logger.info("Running Stage 1: Full-image analysis")
        
        # Resize to 224x224
        img_pil = PILImage.fromarray(image.astype("uint8"))
        img_resized = img_pil.resize((224, 224))
        img_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        confidence, prediction = self._predict_batch(img_tensor)
        
        result = {
            "confidence": float(confidence[0]),
            "prediction": int(prediction[0]),
            "proceed_to_stage_2": float(confidence[0]) > self.stage_config["stage_1"]["threshold"],
        }
        
        logger.info(f"Stage 1 result: confidence={result['confidence']:.3f}, proceed={result['proceed_to_stage_2']}")
        
        return result

    def run_stage_2(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Stage 2: Chunk grid analysis.
        
        Args:
            image: Image as numpy array (H, W, 3)
            
        Returns:
            Dict with {num_tiles, high_confidence_count, proceed_to_stage_3, tile_results}
        """
        logger.info("Running Stage 2: Chunk grid analysis")
        
        tile_size = self.stage_config["stage_2"]["tile_size"]
        stride = self.stage_config["stage_2"]["stride"]
        
        tiles = self._prepare_tiles(image, tile_size, stride)
        logger.info(f"Extracted {len(tiles)} tiles of size {tile_size}x{tile_size}")
        
        tile_results = self._process_tiles(tiles)
        
        proceed = len(tile_results["high_confidence_tiles"]) > 0
        
        result = {
            "num_tiles": tile_results["num_tiles"],
            "high_confidence_count": len(tile_results["high_confidence_tiles"]),
            "proceed_to_stage_3": proceed,
            "tile_results": tile_results["all_results"],
            "high_confidence_tiles": tile_results["high_confidence_tiles"],
        }
        
        logger.info(f"Stage 2 result: {result['high_confidence_count']} high-confidence tiles, proceed={proceed}")
        
        return result

    def run_stage_3(
        self,
        image: np.ndarray,
        high_confidence_tiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Stage 3: Sub-chunk refinement on high-confidence regions.
        
        Args:
            image: Image as numpy array (H, W, 3)
            high_confidence_tiles: Results from stage 2
            
        Returns:
            Dict with {sub_chunk_results, heatmap_path, final_confidence}
        """
        logger.info(f"Running Stage 3: Sub-chunk refinement on {len(high_confidence_tiles)} regions")
        
        tile_size_2 = self.stage_config["stage_2"]["tile_size"]
        tile_size_3 = self.stage_config["stage_3"]["tile_size"]
        stride_3 = tile_size_3  # No overlap for sub-chunks
        
        all_sub_results = []
        
        for parent_tile in high_confidence_tiles:
            x2, y2 = parent_tile["x"], parent_tile["y"]
            # Extract parent region
            region = image[y2:y2+tile_size_2, x2:x2+tile_size_2]
            
            # Prepare sub-tiles
            sub_tiles = self._prepare_tiles(region, tile_size_3, stride_3)
            sub_results = self._process_tiles(sub_tiles)
            
            # Store results with absolute coordinates
            for sub_result in sub_results["all_results"]:
                sub_result["parent_x"] = x2
                sub_result["parent_y"] = y2
                sub_result["abs_x"] = x2 + sub_result["x"]
                sub_result["abs_y"] = y2 + sub_result["y"]
                all_sub_results.append(sub_result)
        
        # Compute final heatmap
        h, w, _ = image.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for sub_result in all_sub_results:
            x = sub_result["abs_x"]
            y = sub_result["abs_y"]
            size = tile_size_3
            heatmap[y:y+size, x:x+size] += sub_result["confidence"]
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Final confidence: average of high-confidence sub-chunks
        high_conf_sub = [r for r in all_sub_results if r["confidence"] > 0.7]
        final_confidence = np.mean([r["confidence"] for r in high_conf_sub]) if high_conf_sub else 0.0
        
        result = {
            "num_sub_chunks_analyzed": len(all_sub_results),
            "num_high_confidence_sub": len(high_conf_sub),
            "final_confidence": float(final_confidence),
            "sub_chunk_results": all_sub_results,
            "heatmap": heatmap,
        }
        
        logger.info(f"Stage 3 result: final_confidence={final_confidence:.3f}")
        
        return result

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Run complete multi-stage analysis.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dict with complete analysis results
        """
        start_time = time.time()
        logger.info(f"Starting multi-stage analysis of {image_path}")
        
        # Load image
        image = self.load_image(image_path)
        h, w, _ = image.shape
        logger.info(f"Loaded image: {w}x{h}")
        
        results = {
            "image_path": str(image_path),
            "image_size": {"width": w, "height": h},
            "timestamp": datetime.utcnow().isoformat(),
            "stages": {},
        }
        
        # Stage 1
        stage_1_result = self.run_stage_1(image)
        results["stages"]["stage_1"] = stage_1_result
        
        if not stage_1_result["proceed_to_stage_2"]:
            results["final_prediction"] = stage_1_result["prediction"]
            results["final_confidence"] = stage_1_result["confidence"]
            results["reason"] = "Rejected at stage 1"
            results["elapsed_seconds"] = time.time() - start_time
            logger.info(f"Analysis complete: {results['final_prediction']} (rejected at stage 1)")
            return results
        
        # Stage 2
        stage_2_result = self.run_stage_2(image)
        results["stages"]["stage_2"] = stage_2_result
        
        if not stage_2_result["proceed_to_stage_3"]:
            results["final_prediction"] = 0  # Non-lens
            results["final_confidence"] = 0.0
            results["reason"] = "No high-confidence chunks at stage 2"
            results["elapsed_seconds"] = time.time() - start_time
            logger.info(f"Analysis complete: {results['final_prediction']} (rejected at stage 2)")
            return results
        
        # Stage 3
        stage_3_result = self.run_stage_3(image, stage_2_result["high_confidence_tiles"])
        results["stages"]["stage_3"] = {
            "num_sub_chunks_analyzed": stage_3_result["num_sub_chunks_analyzed"],
            "num_high_confidence_sub": stage_3_result["num_high_confidence_sub"],
            "final_confidence": stage_3_result["final_confidence"],
        }
        
        # Final prediction
        results["final_prediction"] = 1 if stage_3_result["final_confidence"] > 0.5 else 0
        results["final_confidence"] = stage_3_result["final_confidence"]
        results["reason"] = "Completed all stages"
        results["elapsed_seconds"] = time.time() - start_time
        
        # Store heatmap (can be saved separately)
        results["heatmap_data"] = stage_3_result["heatmap"].tolist()
        
        logger.info(f"Analysis complete: prediction={results['final_prediction']}, confidence={results['final_confidence']:.3f}")
        
        return results

    def save_heatmap(self, heatmap: np.ndarray, output_path: str):
        """Save heatmap as image."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import get_cmap
        
        norm = Normalize(vmin=0, vmax=1)
        cmap = get_cmap("hot")
        heatmap_rgb = cmap(norm(heatmap))
        heatmap_img = PILImage.fromarray((heatmap_rgb[:, :, :3] * 255).astype("uint8"))
        heatmap_img.save(output_path)
        logger.info(f"Saved heatmap to {output_path}")


def get_inference_pipeline(
    model: torch.nn.Module,
    device: str = "cuda",
    stage_config: Optional[Dict] = None,
) -> MultiStageInferencePipeline:
    """Create and return an inference pipeline."""
    return MultiStageInferencePipeline(model, device, stage_config)
