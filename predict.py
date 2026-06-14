"""Run Classpose inference on a single image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from train import MODEL_NAME, N_CLASSES
from data import CELL_TYPES

MODEL_PATH = Path("models") / MODEL_NAME / (MODEL_NAME + ".pt")


def predict(
    image_path: str | Path,
    model=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run Classpose inference on a single image.

    Returns:
        masks:       (H, W) int32 instance label map.
        class_masks: (H, W) int32 class map (0=bg, 1..5=cell types).
        details:     Dict with 'n_instances' and per-class counts.
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    if model is None:
        from classpose.models import ClassposeModel

        model = ClassposeModel(
            gpu=True,
            pretrained_model=str(MODEL_PATH),
            nclasses=N_CLASSES,
        )

    masks, flows, class_masks, styles = model.eval(img)
    n_instances = int(masks.max())

    class_counts = {}
    for inst_id in range(1, n_instances + 1):
        cls = int(class_masks[masks == inst_id][0])
        if 1 <= cls <= len(CELL_TYPES):
            name = CELL_TYPES[cls - 1]
        else:
            name = f"class_{cls}"
        class_counts[name] = class_counts.get(name, 0) + 1

    details = {"n_instances": n_instances, "class_counts": class_counts}
    return masks.astype(np.int32), class_masks.astype(np.int32), details


def summarize(masks: np.ndarray, class_masks: np.ndarray, details: dict) -> None:
    """Print a short summary of prediction results."""
    n = details.get("n_instances", int(masks.max()))
    print(f"Detected {n} nucleus instance(s).")
    cc = details.get("class_counts", {})
    if cc:
        print("Per-class counts:")
        for name, count in sorted(cc.items()):
            print(f"  {name}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classpose inference on a single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", "-o", help="Save instance label map as TIFF")
    parser.add_argument("--output-classes", help="Save class map as TIFF")
    parser.add_argument("--model-path", default=None, help="Path to trained classpose .pt model")
    args = parser.parse_args()

    if args.model_path:
        from classpose.models import ClassposeModel

        m = ClassposeModel(gpu=True, pretrained_model=args.model_path, nclasses=N_CLASSES)
    else:
        m = None

    masks, class_masks, details = predict(args.image, model=m)
    summarize(masks, class_masks, details)

    if args.output:
        from tifffile import imwrite

        imwrite(args.output, masks.astype(np.uint16))
        print(f"Instance masks → {args.output}")

    if args.output_classes:
        from tifffile import imwrite

        imwrite(args.output_classes, class_masks.astype(np.uint8))
        print(f"Class map → {args.output_classes}")
