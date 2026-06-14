"""Run Cellpose inference on a single image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from train import MODEL_NAME

MODEL_PATH = Path.home() / ".cellpose" / "models" / (MODEL_NAME + ".pt")


def predict(
    image_path: str | Path,
    model=None,
) -> tuple[np.ndarray, dict]:
    """
    Run Cellpose inference on a single image.

    Returns:
        masks:   (H, W) int32 instance label map.
        details: Dict with 'n_instances'.
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    if model is None:
        from cellpose import models

        model = models.CellposeModel(gpu=True, pretrained_model=str(MODEL_PATH))

    masks, flows, styles = model.eval(img)
    n_instances = int(masks.max())

    details = {"n_instances": n_instances}
    return masks.astype(np.int32), details


def summarize(masks: np.ndarray, details: dict) -> None:
    """Print a short summary of prediction results."""
    n = details.get("n_instances", int(masks.max()))
    print(f"Detected {n} nucleus instance(s).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cellpose inference on a single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", "-o", help="Save instance label map as TIFF")
    parser.add_argument("--model-path", default=None, help="Path to trained cellpose .pt model")
    args = parser.parse_args()

    if args.model_path:
        from cellpose import models

        m = models.CellposeModel(gpu=True, pretrained_model=args.model_path)
    else:
        m = None

    masks, details = predict(args.image, model=m)
    summarize(masks, details)

    if args.output:
        from tifffile import imwrite

        imwrite(args.output, masks.astype(np.uint16))
        print(f"Saved → {args.output}")
