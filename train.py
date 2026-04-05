"""Fine-tune Cellpose CPSAM on PanNuke."""

from __future__ import annotations

import argparse
from pathlib import Path

from cellpose import models, train, io

io.logger_setup()

MODEL_NAME = "cellpose_pannuke"
MODEL_DIR = Path("models") / MODEL_NAME
N_EPOCHS = 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
BATCH_SIZE = 1


def train_model(max_samples: int | None = None) -> None:
    from data import prepare_dataset

    print("Preparing PanNuke dataset...")
    train_imgs, train_lbls, val_imgs, val_lbls = prepare_dataset(
        max_samples=max_samples,
    )

    print(f"\nInitializing Cellpose-SAM model (pretrained CPSAM)...")
    model = models.CellposeModel(gpu=True)

    print(
        f"\nTraining:  epochs={N_EPOCHS}  "
        f"lr={LEARNING_RATE}  weight_decay={WEIGHT_DECAY}  "
        f"batch_size={BATCH_SIZE}  "
        f"train_samples={len(train_imgs)}  val_samples={len(val_imgs)}\n"
    )

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_imgs,
        train_labels=train_lbls,
        test_data=val_imgs,
        test_labels=val_lbls,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        nimg_per_epoch=max(2, len(train_imgs)),
        model_name=MODEL_NAME,
    )

    print(f"\nDone. Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Cellpose on PanNuke")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit samples per fold for quick smoke tests (e.g. --max-samples 5)",
    )
    args = parser.parse_args()
    train_model(max_samples=args.max_samples)
