"""CLI entry point for classpose-skripsi."""

from __future__ import annotations

import argparse

from train import MODEL_NAME, N_CLASSES
from data import CELL_TYPES


def cmd_train(args: argparse.Namespace) -> None:
    from train import train_model

    train_model(
        max_samples=args.max_samples,
        freeze=args.freeze,
        use_class_weights=not args.no_class_weights,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    from predict import predict, summarize

    if args.model_path:
        from classpose.models import ClassposeModel

        model = ClassposeModel(
            gpu=True,
            pretrained_model=args.model_path,
            nclasses=N_CLASSES,
        )
    else:
        model = None

    masks, class_masks, details = predict(args.image, model=model)
    summarize(masks, class_masks, details)

    if args.output:
        import numpy as np
        from tifffile import imwrite

        imwrite(args.output, masks.astype(np.uint16))
        print(f"Instance masks → {args.output}")

    if args.output_classes:
        from tifffile import imwrite

        imwrite(args.output_classes, class_masks.astype(np.uint8))
        print(f"Class map → {args.output_classes}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate import evaluate

    evaluate(max_samples=args.max_samples, model_path=args.model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cellpose-skripsi",
        description="Classpose nucleus segmentation & classification fine-tuned on PanNuke",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _positive_int(v: str) -> int:
        n = int(v)
        if n < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
        return n

    # ── train ──
    t = sub.add_parser("train", help="Fine-tune Classpose on PanNuke (fold1+fold2)")
    t.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Cap samples per fold for quick tests (e.g. --max-samples 5)",
    )
    t.add_argument(
        "--freeze",
        nargs="+",
        choices=["backbone", "segmentation_head", "neck"],
        default=[],
        help="Parts to freeze during training",
    )
    t.add_argument(
        "--no-class-weights",
        action="store_true",
        default=False,
        help="Disable class weighting for classification loss",
    )

    # ── predict ──
    p = sub.add_parser("predict", help="Run inference on a single image")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--output", "-o", metavar="FILE", help="Save instance map as TIFF")
    p.add_argument("--output-classes", metavar="FILE", help="Save class map as TIFF")
    p.add_argument("--model-path", default=None, help="Path to trained classpose .pt model")

    # ── evaluate ──
    e = sub.add_parser("evaluate", help="Evaluate on PanNuke fold3 (mPQ / bPQ)")
    e.add_argument("--max-samples", type=_positive_int, default=None, metavar="N", help="Cap test samples")
    e.add_argument("--model-path", default=None, help="Path to trained classpose .pt model")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
