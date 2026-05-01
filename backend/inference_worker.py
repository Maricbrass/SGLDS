#!/usr/bin/env python
"""Inference worker script.

Run this with a Python interpreter that has PyTorch installed. It loads the model
checkpoint, runs the backend inference pipeline on a single image, and emits a
JSON result to stdout.

Usage:
  python inference_worker.py --image /path/to/image.png --checkpoint /path/to/best.pt --model swin_tiny_patch4_window7_224 --device cuda
"""
import argparse
import json
import os
import sys
import traceback


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
    return checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Ensure repo root is on sys.path so imports work when launched from backend
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import torch
        from lens_detection.models import build_model
        from app.services.multistage_inference import get_inference_pipeline

        # Build model and load checkpoint
        model = build_model(args.model, num_classes=2, pretrained=False)
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(_extract_state_dict(ckpt))

        model = model.to(args.device).eval()

        pipeline = get_inference_pipeline(model, device=args.device)
        results = pipeline.analyze(args.image)

        # Optionally save heatmap to a temp file and include path; the pipeline may return heatmap_data
        # We'll emit the results JSON to stdout
        print(json.dumps({"ok": True, "results": results}))
    except Exception as exc:
        tb = traceback.format_exc()
        print(json.dumps({"ok": False, "error": str(exc), "traceback": tb}))
        sys.exit(2)

if __name__ == "__main__":
    main()
