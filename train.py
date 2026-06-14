"""Fine-tune Classpose CPSAM on PanNuke."""

from __future__ import annotations

import argparse
from pathlib import Path

from classpose import train
from classpose.models import ClassposeModel
from classpose.train_utils import process_and_build_dataset, split_dataset

MODEL_NAME = "classpose_pannuke"
N_EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
BATCH_SIZE = 1
N_CLASSES = 6


def train_model(
    max_samples: int | None = None,
    freeze: list[str] | None = None,
    feature_transformation_structure: list[int] | None = None,
    use_class_weights: bool = True,
) -> None:
    from data import prepare_dataset

    freeze = freeze or []

    print("Preparing PanNuke dataset...")
    train_imgs, train_lbls, val_imgs, val_lbls = prepare_dataset(
        max_samples=max_samples,
    )

    print("Building Classpose training datasets (computing flows)...")
    full_images = train_imgs + val_imgs
    full_labels = train_lbls + val_lbls

    full_dataset = process_and_build_dataset(
        images=full_images,
        labels=full_labels,
        compute_flows=True,
    )

    n_classes = full_dataset.n_classes
    print(f"  n_classes = {n_classes} (from dataset)")

    train_fraction = len(train_imgs) / len(full_images)
    train_dataset, test_dataset = split_dataset(
        full_dataset, train_fraction, seed=42
    )

    print(f"\nInitializing Classpose model (pretrained CPSAM)...")
    model = ClassposeModel(
        gpu=True,
        pretrained_model="cpsam",
        nclasses=n_classes,
        feature_transformation_structure=feature_transformation_structure,
    )

    model.net.freeze(
        backbone="backbone" in freeze,
        instance_classification="segmentation_head" in freeze,
        neck="neck" in freeze,
    )

    class_weights = train_dataset.class_weights if use_class_weights else None

    save_path = str(Path("models") / MODEL_NAME)
    save_path_parent = str(Path("models"))

    print(
        f"\nTraining:  epochs={N_EPOCHS}  "
        f"lr={LEARNING_RATE}  weight_decay={WEIGHT_DECAY}  "
        f"batch_size={BATCH_SIZE}  "
        f"train_samples={len(train_dataset)}  val_samples={len(test_dataset)}\n"
    )

    model_path, train_losses, test_losses = train.train_class_seg(
        net=model.net,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        save_path=save_path_parent,
        model_name=MODEL_NAME,
        class_weights=class_weights,
    )

    print(f"\nDone. Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Classpose on PanNuke")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit samples per fold for quick smoke tests (e.g. --max-samples 5)",
    )
    parser.add_argument(
        "--freeze",
        nargs="+",
        choices=["backbone", "segmentation_head", "neck"],
        default=[],
        help="Parts to freeze during training",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        default=False,
        help="Disable class weighting for classification loss",
    )
    args = parser.parse_args()
    train_model(
        max_samples=args.max_samples,
        freeze=args.freeze,
        use_class_weights=not args.no_class_weights,
    )
