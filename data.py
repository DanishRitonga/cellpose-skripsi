"""PanNuke dataset loading and label-map conversion for Cellpose training."""

from __future__ import annotations

import gc
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_ID = "RationAI/PanNuke"
DATA_DIR = Path("data") / "pannuke"

# PanNuke cell-type classes (0-indexed, matching `categories` field values)
CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

FOLD_TO_SPLIT = {"fold1": "train", "fold2": "val", "fold3": "test"}


def _instances_to_labelmap(instances: list, img_h: int, img_w: int) -> np.ndarray:
    """Convert a list of binary instance masks into a single integer label map.

    Returns:
        (img_h, img_w) int32 array where 0=background, 1,2,...=instance IDs.
    """
    labelmap = np.zeros((img_h, img_w), dtype=np.int32)
    for idx, mask in enumerate(instances, start=1):
        m = np.array(mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(bool)
        labelmap[m] = idx
    return labelmap


def _split_has_data(split_dir: Path) -> bool:
    """Check whether a split directory already contains cached .npy files."""
    if not split_dir.exists():
        return False
    return any(split_dir.glob("*.npy"))


def _process_fold(
    fold_name: str,
    split: str,
    max_samples: int | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Stream one PanNuke fold, cache to disk, and return (images, labels).

    Images are saved as uint8 .npy (H, W, 3).
    Labels are saved as int32 .npy (H, W) label maps.
    """
    split_dir = DATA_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID, split=fold_name, streaming=True)
    if max_samples is not None:
        ds = ds.take(max_samples)

    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    i = -1
    for i, sample in enumerate(tqdm(ds, desc=fold_name, unit="img")):
        img: Image.Image = sample["image"]
        img_w, img_h = img.size
        img_np = np.array(img, dtype=np.uint8)  # (H, W, 3)

        labelmap = _instances_to_labelmap(sample["instances"], img_h, img_w)

        # Cache to disk
        np.save(split_dir / f"{i:06d}.npy", img_np)
        np.save(split_dir / f"{i:06d}_label.npy", labelmap)

        images.append(img_np)
        labels.append(labelmap)

        if i % 100 == 0:
            gc.collect()

    n_images = i + 1 if i >= 0 else 0
    n_instances = sum(int(l.max()) for l in labels) if labels else 0
    print(f"  {fold_name} → {split}: {n_images} images, {n_instances} instances")

    return images, labels


def _load_cached_split(split_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load cached images and label maps from a split directory."""
    img_files = sorted(split_dir.glob("[0-9]*.npy"))
    images = []
    labels = []
    for img_path in tqdm(img_files, desc=f"Loading {split_dir.name}", unit="img"):
        label_path = split_dir / (img_path.stem + "_label.npy")
        if not label_path.exists():
            continue
        images.append(np.load(img_path))
        labels.append(np.load(label_path))
    return images, labels


def prepare_dataset(
    max_samples: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Prepare the PanNuke dataset for Cellpose training.

    fold1 → train, fold2 → val, fold3 → test

    Returns:
        (train_images, train_labels, val_images, val_labels)
        Each is a list of numpy arrays.
    """
    all_data: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}

    for fold, split in FOLD_TO_SPLIT.items():
        split_dir = DATA_DIR / split

        if max_samples is None and _split_has_data(split_dir):
            print(f"Split '{split}' already cached, loading from disk.")
            all_data[split] = _load_cached_split(split_dir)
        else:
            if split_dir.exists():
                shutil.rmtree(split_dir)
            all_data[split] = _process_fold(fold, split, max_samples)

    train_imgs, train_lbls = all_data["train"]
    val_imgs, val_lbls = all_data["val"]

    print(f"\nDataset ready: {len(train_imgs)} train, {len(val_imgs)} val")
    return train_imgs, train_lbls, val_imgs, val_lbls
