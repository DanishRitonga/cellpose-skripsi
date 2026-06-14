"""CLI entry point for cellpose-skripsi."""

from __future__ import annotations

import argparse
from pathlib import Path

from train import MODEL_NAME

MODEL_PATH = Path.home() / ".cellpose" / "models" / (MODEL_NAME + ".pt")


def cmd_train(args: argparse.Namespace) -> None:
    from train import train_model

    train_model(max_samples=args.max_samples)


def cmd_predict(args: argparse.Namespace) -> None:
    from predict import predict, summarize

    if args.model_path:
        from cellpose import models

        model = models.CellposeModel(gpu=True, pretrained_model=args.model_path)
    else:
        model = None

    masks, details = predict(args.image, model=model)
    summarize(masks, details)

    if args.output:
        import numpy as np
        from tifffile import imwrite

        imwrite(args.output, masks.astype(np.uint16))
        print(f"Saved → {args.output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate import evaluate

    evaluate(max_samples=args.max_samples, model_path=args.model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cellpose-skripsi",
        description="Cellpose nucleus segmentation fine-tuned on PanNuke",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _positive_int(v: str) -> int:
        n = int(v)
        if n < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
        return n

    # ── train ──
    t = sub.add_parser("train", help="Fine-tune Cellpose-SAM on PanNuke (fold1+fold2)")
    t.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Cap samples per fold for quick tests (e.g. --max-samples 5)",
    )

    # ── predict ──
    p = sub.add_parser("predict", help="Run inference on a single image")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--output", "-o", metavar="FILE", help="Save instance map as TIFF")
    p.add_argument("--model-path", default=None, help="Path to trained cellpose .pt model")

    # ── evaluate ──
    e = sub.add_parser("evaluate", help="Evaluate on PanNuke fold3 (LSP-DETR protocol)")
    e.add_argument("--max-samples", type=_positive_int, default=None, metavar="N", help="Cap test samples")
    e.add_argument("--model-path", default=None, help="Path to trained cellpose .pt model")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
