"""Evaluate trained Cellpose model on PanNuke fold3 using mPQ and bPQ."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from cellpose import models
from tqdm import tqdm

from data import DATA_DIR, FOLD_TO_SPLIT, CELL_TYPES, _instances_to_labelmap
from train import MODEL_NAME

MODEL_PATH = Path.home() / ".cellpose" / "models" / (MODEL_NAME + ".pt")

IOU_THRESHOLD = 0.5


def _load_test_split(max_samples: int | None = None):
    """Load fold3 (test) images, GT label maps, and per-instance cell-type arrays."""
    from data import _split_has_data, _process_fold, _load_cached_split

    split_dir = DATA_DIR / "test"

    if max_samples is None and _split_has_data(split_dir):
        print("Loading cached test split from disk.")
        images, labels, categories = _load_cached_split(split_dir)
    else:
        images, labels, categories = _process_fold("fold3", "test", max_samples)

    return images, labels, categories


def _build_gt_info(labelmap: np.ndarray, cell_type_arr: np.ndarray):
    """Extract per-instance GT masks and cell types.

    Args:
        cell_type_arr: (N_instances,) int32 array of cell-type indices, may be empty.

    Returns:
        instance_masks: dict[int, np.ndarray]  instance_id -> binary mask
        instance_types: dict[int, int]          instance_id -> cell type index (0-4)
    """
    instance_ids = np.unique(labelmap)
    instance_ids = instance_ids[instance_ids != 0]

    instance_masks = {}
    instance_types = {}

    for idx in instance_ids:
        instance_masks[int(idx)] = (labelmap == idx)
        if len(cell_type_arr) > 0 and int(idx) - 1 < len(cell_type_arr):
            instance_types[int(idx)] = int(cell_type_arr[int(idx) - 1])

    return instance_masks, instance_types


def _build_pred_info(labelmap: np.ndarray):
    """Extract per-instance prediction masks."""
    instance_ids = np.unique(labelmap)
    instance_ids = instance_ids[instance_ids != 0]

    instance_masks = {}
    for idx in instance_ids:
        instance_masks[int(idx)] = (labelmap == idx)
    return instance_masks


def _compute_iou_matrix(gt_masks: dict, pred_masks: dict) -> np.ndarray:
    """Compute IoU matrix of shape (n_gt, n_pred)."""
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float64)

    gt_ids = sorted(gt_masks.keys())
    pred_ids = sorted(pred_masks.keys())

    for i, gid in enumerate(gt_ids):
        gt_mask = gt_masks[gid]
        for j, pid in enumerate(pred_masks):
            pred_mask = pred_masks[pid]
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou_mat[i, j] = intersection / union

    return iou_mat, gt_ids, pred_ids


def _match_instances(iou_mat: np.ndarray, iou_thresh: float = IOU_THRESHOLD):
    """Greedy-match GT and pred instances by IoU (one-to-one).

    Returns:
        matched_pairs: list of (gt_idx, pred_idx, iou)
        unmatched_gt: set of gt indices
        unmatched_pred: set of pred indices
    """
    n_gt, n_pred = iou_mat.shape
    matched_pairs = []
    used_gt = set()
    used_pred = set()

    while True:
        best_iou = 0.0
        best_pair = None
        for i in range(n_gt):
            if i in used_gt:
                continue
            for j in range(n_pred):
                if j in used_pred:
                    continue
                if iou_mat[i, j] > best_iou:
                    best_iou = iou_mat[i, j]
                    best_pair = (i, j)

        if best_pair is None or best_iou < iou_thresh:
            break

        i, j = best_pair
        matched_pairs.append((i, j, best_iou))
        used_gt.add(i)
        used_pred.add(j)

    unmatched_gt = set(range(n_gt)) - used_gt
    unmatched_pred = set(range(n_pred)) - used_pred

    return matched_pairs, unmatched_gt, unmatched_pred


def compute_pq(
    gt_masks: dict,
    pred_masks: dict,
    gt_types: dict[int, int] | None = None,
    class_idx: int | None = None,
) -> float | None:
    """Compute Panoptic Quality, optionally filtered to a single class.

    If class_idx is given, only GT instances of that class and all pred
    instances are considered (pred has no class info, so all preds are
    candidate matches but FP are counted against this class).

    Returns None if there are no GT instances for this class.
    """
    if class_idx is not None and gt_types is not None:
        filtered_gt = {k: v for k, v in gt_masks.items() if gt_types.get(k) == class_idx}
    else:
        filtered_gt = gt_masks

    if len(filtered_gt) == 0:
        return None

    iou_mat, gt_ids, pred_ids = _compute_iou_matrix(filtered_gt, pred_masks)
    matched, unmatched_gt, unmatched_pred = _match_instances(iou_mat)

    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    iou_sum = sum(iou for _, _, iou in matched)

    denom = tp + 0.5 * fp + 0.5 * fn
    if denom == 0:
        return 0.0

    return iou_sum / denom


def compute_mpq(
    gt_masks: dict,
    pred_masks: dict,
    gt_types: dict[int, int],
) -> float:
    """Compute multi-class PQ (mPQ): PQ per cell type, then averaged."""
    pq_values = []
    for class_idx in range(len(CELL_TYPES)):
        pq = compute_pq(gt_masks, pred_masks, gt_types, class_idx)
        if pq is not None:
            pq_values.append(pq)

    if not pq_values:
        return 0.0
    return float(np.mean(pq_values))


def compute_bpq(gt_masks: dict, pred_masks: dict) -> float:
    """Compute binary PQ (bPQ): all instances treated as one class."""
    return compute_pq(gt_masks, pred_masks) or 0.0


def evaluate(
    max_samples: int | None = None,
    model_path: str | Path | None = None,
) -> dict:
    """Run evaluation on PanNuke fold3.

    Returns dict with 'mPQ', 'bPQ', and per-image details.
    """
    print("Loading PanNuke fold3 (test) ...")
    images, gt_labels, categories = _load_test_split(max_samples)
    print(f"  {len(images)} test images loaded.")

    print("Loading trained model ...")
    mpath = str(model_path) if model_path else str(MODEL_PATH)
    model = models.CellposeModel(gpu=True, pretrained_model=mpath)

    mpq_values = []
    bpq_values = []
    details = []

    for idx in tqdm(range(len(images)), desc="Evaluating", unit="img"):
        img = images[idx]
        gt_lmap = gt_labels[idx]
        cat_arr = categories[idx]

        pred_masks_arr, _, _ = model.eval(img)
        pred_lmap = pred_masks_arr.astype(np.int32)

        gt_masks, gt_types = _build_gt_info(gt_lmap, cat_arr)
        pred_masks = _build_pred_info(pred_lmap)

        mpq = compute_mpq(gt_masks, pred_masks, gt_types)
        bpq = compute_bpq(gt_masks, pred_masks)

        mpq_values.append(mpq)
        bpq_values.append(bpq)
        details.append({"image_idx": idx, "mPQ": mpq, "bPQ": bpq})

    mean_mpq = float(np.mean(mpq_values)) if mpq_values else 0.0
    mean_bpq = float(np.mean(bpq_values)) if bpq_values else 0.0

    results = {
        "mPQ": mean_mpq,
        "bPQ": mean_bpq,
        "per_image": details,
    }

    print(f"\n{'='*40}")
    print(f"  Evaluation Results (fold3, {len(images)} images)")
    print(f"{'='*40}")
    print(f"  mPQ  = {mean_mpq:.4f}")
    print(f"  bPQ  = {mean_bpq:.4f}")
    print(f"{'='*40}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Cellpose on PanNuke fold3")
    parser.add_argument("--max-samples", type=int, default=None, metavar="N")
    parser.add_argument("--model-path", default=None, help="Path to .pt model")
    args = parser.parse_args()
    evaluate(max_samples=args.max_samples, model_path=args.model_path)
