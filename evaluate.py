"""Evaluate Cellpose model on PanNuke fold3 — LSP-DETR aligned metrics.

Metrics:
    AJI, AP@0.5, AP@0.7, AP@0.9, AP@0.5:0.05:0.95,
    bPQ, bMPQ,
    F1 (centroid, r=12), Precision, Recall.

All instance-matching metrics use Hungarian assignment.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --max-samples 10
    uv run python main.py evaluate --max-samples 10
"""

from __future__ import annotations

import argparse
import gc
import resource
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from data import DATA_DIR, CELL_TYPES
from train import MODEL_NAME

MODEL_PATH = Path.home() / ".cellpose" / "models" / (MODEL_NAME + ".pt")

IOU_THRESHOLDS = sorted(set(round(x, 2) for x in np.arange(0.5, 1.0, 0.05)))
CENTROID_RADIUS = 12


def _load_test_split(max_samples: int | None = None):
    from data import _split_has_data, _process_fold, _load_cached_split

    split_dir = DATA_DIR / "test"
    if max_samples is None and _split_has_data(split_dir):
        print("Loading cached test split from disk.")
        return _load_cached_split(split_dir)
    return _process_fold("fold3", "test", max_samples)


def _extract_instances(
    mask_array: np.ndarray,
    flows: np.ndarray | None = None,
):
    """Extract per-instance binary masks and centroids from a label map.

    Returns:
        masks:     list of (H,W) bool arrays
        centroids: (N, 2) float64 array — (cy, cx)
        confs:     (N,) float64 array — mean flow magnitude per instance
    """
    instance_ids = np.unique(mask_array)
    instance_ids = instance_ids[instance_ids != 0]

    masks = []
    centroids_cy = []
    centroids_cx = []
    confs = []

    if flows is not None:
        flow_mag = np.sqrt(
            flows[..., 0].astype(np.float64) ** 2
            + flows[..., 1].astype(np.float64) ** 2
        )

    for inst_id in instance_ids:
        mask = mask_array == inst_id
        masks.append(mask)

        ys, xs = np.where(mask)
        centroids_cy.append(float(ys.mean()) if len(ys) else 0.0)
        centroids_cx.append(float(xs.mean()) if len(xs) else 0.0)

        if flows is not None:
            confs.append(float(flow_mag[mask].mean()))

    centroids = (
        np.column_stack([centroids_cy, centroids_cx])
        if masks
        else np.zeros((0, 2))
    )
    confs_arr = (
        np.array(confs, dtype=np.float64)
        if confs
        else np.zeros(len(masks), dtype=np.float64)
    )
    return masks, centroids, confs_arr


def _mask_iou_matrix(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]
) -> np.ndarray:
    """Compute pairwise mask IoU matrix (n_pred × n_gt)."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)

    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)

    intersection = pred_stack @ gt_stack.T
    pred_area = pred_stack.sum(axis=1, keepdims=True)
    gt_area = gt_stack.sum(axis=1, keepdims=True)
    union = pred_area + gt_area.T - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def compute_aji(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> float:
    """Aggregated Jaccard Index with Hungarian matching."""
    n_gt = len(gt_masks)
    if n_gt == 0:
        return 0.0
    n_pred = len(pred_masks)
    if n_pred == 0:
        return 0.0

    iou_mat = _mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    intersection = 0.0
    union = 0.0
    matched_gt = set()

    for pi, gi in zip(row_ind, col_ind):
        if iou_mat[pi, gi] > 0:
            intersection += np.logical_and(pred_masks[pi], gt_masks[gi]).sum()
            union += np.logical_or(pred_masks[pi], gt_masks[gi]).sum()
            matched_gt.add(gi)

    for gi in range(n_gt):
        if gi not in matched_gt:
            union += gt_masks[gi].sum()

    return float(intersection / union) if union > 0 else 0.0


def _compute_pq_masked(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
    fg_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute PQ with Hungarian matching.  Returns (pq, sq, dq)."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    if fg_mask is not None:
        pred_masks = [m & fg_mask for m in pred_masks]
        gt_masks = [m & fg_mask for m in gt_masks]

    iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold

    tp = int(valid.sum())
    fp = n_pred - tp
    fn = n_gt - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0
    return float(sq * dq), float(sq), float(dq)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute average precision from recall/precision curves."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def compute_metrics_streaming(image_results: list[dict]) -> dict:
    """Compute all metrics in a single pass, one image at a time."""
    aji_scores: list[float] = []
    bpq_scores: list[float] = []
    bmpq_scores: list[float] = []

    centroid_tp = 0
    centroid_fp = 0
    centroid_fn = 0

    ap_stats: dict[float, dict] = {
        t: {"tp": [], "fp": [], "conf": [], "n_gt": 0} for t in IOU_THRESHOLDS
    }

    n_total = len(image_results)
    for i, r in enumerate(image_results):
        gt_masks = r["gt_masks"]
        pred_masks = r["pred_masks"]
        pred_confs = r["pred_confs"]
        gt_centroids = r["gt_centroids"]
        pred_centroids = r["pred_centroids"]
        im_h, im_w = r["imgsz"]

        if i == 0:
            print(
                f"  First image: {len(pred_masks)} preds, {len(gt_masks)} GT, "
                f"size={im_h}x{im_w}",
                flush=True,
            )

        aji_scores.append(compute_aji(pred_masks, gt_masks))

        # bPQ / bMPQ
        pred_binary = (
            np.stack(pred_masks).max(axis=0).astype(np.uint8)
            if pred_masks
            else np.zeros((im_h, im_w), dtype=np.uint8)
        )
        gt_binary = (
            np.stack(gt_masks).max(axis=0).astype(np.uint8)
            if gt_masks
            else np.zeros((im_h, im_w), dtype=np.uint8)
        )

        bpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary])
        bpq_scores.append(bpq)

        if gt_binary.sum() > 0:
            fg = gt_binary > 0
            bmpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary], fg_mask=fg)
        else:
            bmpq = 0.0
        bmpq_scores.append(bmpq)

        # AP — single-class, Hungarian matching
        n_pred = len(pred_masks)
        n_gt = len(gt_masks)

        for t in IOU_THRESHOLDS:
            ap_stats[t]["n_gt"] += n_gt

        if n_pred > 0 and n_gt > 0:
            iou_mat = _mask_iou_matrix(pred_masks, gt_masks)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            matched_iou = iou_mat[row_ind, col_ind]
        else:
            row_ind = np.array([], dtype=int)
            matched_iou = np.array([], dtype=float)

        for t in IOU_THRESHOLDS:
            valid_mask = matched_iou >= t if n_gt > 0 else np.zeros(0, dtype=bool)
            matched_pred = set(row_ind[valid_mask].tolist())
            for pi in range(n_pred):
                is_tp = pi in matched_pred
                ap_stats[t]["conf"].append(
                    float(pred_confs[pi] if pi < len(pred_confs) else 0.0)
                )
                ap_stats[t]["tp"].append(is_tp)
                ap_stats[t]["fp"].append(not is_tp)

        # Centroid F1
        if n_pred > 0 and n_gt > 0:
            dist_matrix = np.linalg.norm(
                pred_centroids[:, None, :] - gt_centroids[None, :, :], axis=2
            )
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            tp = int((dist_matrix[row_ind, col_ind] <= CENTROID_RADIUS).sum())
        else:
            tp = 0
        centroid_tp += tp
        centroid_fp += n_pred - tp
        centroid_fn += n_gt - tp

        del gt_masks, pred_masks
        gc.collect()

        if (i + 1) % 500 == 0:
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(
                f"  Processed {i + 1}/{n_total} images (RSS={mem_mb:.0f}MB)",
                flush=True,
            )

    # Aggregate
    mean_aji = float(np.mean(aji_scores)) if aji_scores else 0.0
    mean_bpq = float(np.mean(bpq_scores)) if bpq_scores else 0.0
    mean_bmpq = float(np.mean(bmpq_scores)) if bmpq_scores else 0.0

    ap_results: dict[float, dict] = {}
    for t in IOU_THRESHOLDS:
        stats = ap_stats[t]
        n_gt = stats["n_gt"]
        if n_gt == 0:
            ap_results[t] = {"AP": 0.0}
            continue
        confs = np.array(stats["conf"])
        tps = np.array(stats["tp"])
        fps = np.array(stats["fp"])
        if len(confs) == 0:
            ap_results[t] = {"AP": 0.0}
            continue
        order = np.argsort(-confs)
        tps = tps[order]
        fps = fps[order]
        precision = tps.cumsum() / (tps.cumsum() + fps.cumsum())
        recall = tps.cumsum() / n_gt
        ap_results[t] = {"AP": _compute_ap(recall, precision)}

    prec = (
        centroid_tp / (centroid_tp + centroid_fp)
        if (centroid_tp + centroid_fp) > 0
        else 0.0
    )
    rec = (
        centroid_tp / (centroid_tp + centroid_fn)
        if (centroid_tp + centroid_fn) > 0
        else 0.0
    )
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "aji": mean_aji,
        "bpq": mean_bpq,
        "bmpq": mean_bmpq,
        "ap": ap_results,
        "centroid": {"precision": prec, "recall": rec, "f1": f1},
    }


def _diagnose_recall(image_results: list[dict]) -> None:
    """Break down recall by GT size bin and nearest-prediction distance."""
    gt_areas_matched: list[float] = []
    gt_areas_unmatched: list[float] = []
    unmatched_dists: list[float] = []
    unmatched_nearest_conf: list[float] = []

    for r in image_results:
        gt_masks = r["gt_masks"]
        pred_masks = r["pred_masks"]
        pred_confs = r["pred_confs"]
        gt_centroids = r["gt_centroids"]
        pred_centroids = r["pred_centroids"]

        n_gt = len(gt_masks)
        n_pred = len(pred_masks)
        if n_gt == 0:
            continue

        gt_areas = np.array([m.sum() for m in gt_masks], dtype=np.float64)

        if n_pred > 0:
            dist_matrix = np.linalg.norm(
                pred_centroids[:, None, :] - gt_centroids[None, :, :], axis=2
            )
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            matched = np.zeros(n_gt, dtype=bool)
            for ri, ci, d in zip(row_ind, col_ind, dist_matrix[row_ind, col_ind]):
                if d <= CENTROID_RADIUS:
                    matched[ci] = True
        else:
            matched = np.zeros(n_gt, dtype=bool)

        for j in range(n_gt):
            if matched[j]:
                gt_areas_matched.append(float(gt_areas[j]))
            else:
                gt_areas_unmatched.append(float(gt_areas[j]))
                if n_pred > 0:
                    dists_j = np.linalg.norm(pred_centroids - gt_centroids[j], axis=1)
                    nearest_idx = int(np.argmin(dists_j))
                    unmatched_dists.append(float(dists_j[nearest_idx]))
                    unmatched_nearest_conf.append(float(pred_confs[nearest_idx]))
                else:
                    unmatched_dists.append(float("inf"))
                    unmatched_nearest_conf.append(0.0)

    total_gt = len(gt_areas_matched) + len(gt_areas_unmatched)
    total_matched = len(gt_areas_matched)
    recall = total_matched / total_gt if total_gt > 0 else 0.0

    print(f"\n{'=' * 65}")
    print("  Recall Diagnosis (centroid-based)")
    print(f"{'=' * 65}")
    print(f"  Overall: {total_matched}/{total_gt} = {recall:.4f}")

    # Size bins
    all_areas = np.array(gt_areas_matched + gt_areas_unmatched)
    if len(all_areas) >= 3:
        p33 = np.percentile(all_areas, 33)
        p67 = np.percentile(all_areas, 67)
        bins = [
            ("Small (<P33)", lambda a: a < p33),
            ("Medium (P33-P67)", lambda a: (a >= p33) & (a < p67)),
            ("Large (>P67)", lambda a: a >= p67),
        ]
        print(
            f"\n  {'Size Bin':<20} {'Area Range':<18} {'Total':>8} {'Matched':>8} {'Recall':>8}"
        )
        print(f"  {'-' * 20} {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 8}")
        for label, cond in bins:
            t = sum(1 for a in all_areas if cond(a))
            m = sum(1 for a in gt_areas_matched if cond(a))
            if t > 0:
                lo = min(a for a in all_areas if cond(a))
                hi = max(a for a in all_areas if cond(a))
                rng = f"{lo:.0f}-{hi:.0f}px²"
            else:
                rng = "-"
            rec_b = m / t if t > 0 else 0.0
            print(f"  {label:<20} {rng:<18} {t:>8} {m:>8} {rec_b:>8.4f}")

    # Unmatched GT distance breakdown
    if unmatched_dists:
        ud = np.array(unmatched_dists)
        finite = ud[np.isfinite(ud)]
        print(f"\n  Unmatched GT — Distance to nearest prediction:")
        n_un = len(unmatched_dists)
        print(
            f"    <5px (near miss): {int(np.sum(finite < 5))}/{n_un} "
            f"({100 * sum(finite < 5) / n_un:.1f}%)"
        )
        print(
            f"    5-12px (drifted):  {int(np.sum((finite >= 5) & (finite <= 12)))}/{n_un} "
            f"({100 * sum((finite >= 5) & (finite <= 12)) / n_un:.1f}%)"
        )
        print(
            f"    >12px (truly miss):{int(np.sum(finite > 12))}/{n_un} "
            f"({100 * sum(finite > 12) / n_un:.1f}%)"
        )
        no_det = int(np.isinf(ud).sum())
        if no_det > 0:
            print(
                f"    No predictions:    {no_det}/{n_un} "
                f"({100 * no_det / n_un:.1f}%)"
            )

    # Unmatched GT: confidence of nearest prediction within 12px
    if unmatched_nearest_conf:
        umc = np.array(unmatched_nearest_conf)
        nearest_12 = np.array([d <= 12.0 for d in unmatched_dists])
        conf_within_12 = umc[nearest_12]
        if len(conf_within_12) > 0:
            print(f"\n  Unmatched GT — Confidence of nearest pred within 12px:")
            print(f"    Mean conf:          {conf_within_12.mean():.4f}")
            print(f"    Median conf:        {float(np.median(conf_within_12)):.4f}")
            above = int((conf_within_12 >= 0.2).sum())
            print(
                f"    Conf ≥0.2:          {above}/{len(conf_within_12)} "
                f"({100 * above / max(len(conf_within_12), 1):.1f}%)"
            )

    print(f"{'=' * 65}")


def evaluate(
    max_samples: int | None = None,
    model_path: str | Path | None = None,
) -> dict:
    """Run evaluation on PanNuke fold3 — LSP-DETR protocol (binary)."""
    print("Loading PanNuke fold3 (test) ...")
    images, gt_labels = _load_test_split(max_samples)
    print(f"  {len(images)} test images loaded.")

    print("Loading Cellpose model ...")
    from cellpose import models

    mpath = str(model_path) if model_path else str(MODEL_PATH)
    cpmodel = models.CellposeModel(gpu=True, pretrained_model=mpath)

    image_results: list[dict] = []
    t_start = time.perf_counter()

    for idx in tqdm(range(len(images)), desc="Inference", unit="img"):
        img = images[idx]
        gt_label = gt_labels[idx]

        pred_masks_arr, flows, _ = cpmodel.eval(img)
        im_h, im_w = img.shape[:2]

        pred_masks, pred_centroids, pred_confs = _extract_instances(
            pred_masks_arr.astype(np.int32), flows
        )
        gt_masks, gt_centroids, _ = _extract_instances(gt_label.astype(np.int32))

        image_results.append(
            {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "pred_confs": pred_confs,
                "pred_centroids": pred_centroids,
                "gt_centroids": gt_centroids,
                "imgsz": (im_h, im_w),
            }
        )

    t_inf = time.perf_counter() - t_start
    n_gt_total = sum(len(r["gt_masks"]) for r in image_results)
    n_pred_total = sum(len(r["pred_masks"]) for r in image_results)
    print(
        f"\n  Inference: {len(images)} images in {t_inf:.1f}s "
        f"({t_inf / len(images) * 1000:.1f} ms/img)"
    )
    print(f"  GT instances: {n_gt_total},  Pred instances: {n_pred_total}")

    print("\nComputing metrics (streaming)...")
    mtx = compute_metrics_streaming(image_results)

    ap_results = mtx["ap"]
    ap50 = ap_results.get(0.5, {}).get("AP", 0.0)
    ap70 = ap_results.get(0.7, {}).get("AP", 0.0)
    ap90 = ap_results.get(0.9, {}).get("AP", 0.0)
    ap50_95 = float(np.mean([ap_results[t]["AP"] for t in sorted(ap_results.keys())]))
    f12 = mtx["centroid"]

    print(f"\n{'=' * 60}")
    print(f"  PanNuke Fold3 Evaluation Results (LSP-DETR Protocol)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'-' * 42}")
    print(f"  {'AJI':<30} {mtx['aji']:>12.4f}")
    print(f"  {'AP@0.5':<30} {ap50:>12.4f}")
    print(f"  {'AP@0.7':<30} {ap70:>12.4f}")
    print(f"  {'AP@0.9':<30} {ap90:>12.4f}")
    print(f"  {'AP@0.5:0.05:0.95':<30} {ap50_95:>12.4f}")
    print(f"  {'bPQ':<30} {mtx['bpq']:>12.4f}")
    print(f"  {'bMPQ':<30} {mtx['bmpq']:>12.4f}")
    print(f"  {'F1 (centroid, r=12)':<30} {f12['f1']:>12.4f}")
    print(f"  {'Precision (centroid)':<30} {f12['precision']:>12.4f}")
    print(f"  {'Recall (centroid)':<30} {f12['recall']:>12.4f}")
    print(f"  {'Inference (ms/img)':<30} {t_inf / len(images) * 1000:>12.1f}")
    print(f"{'=' * 60}")

    _diagnose_recall(image_results)

    return {
        "aji": mtx["aji"],
        "bpq": mtx["bpq"],
        "bmpq": mtx["bmpq"],
        "ap": ap_results,
        "centroid_f1": f12,
        "inference_ms_per_img": t_inf / len(images) * 1000,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Cellpose on PanNuke fold3 (LSP-DETR protocol)"
    )
    parser.add_argument("--max-samples", type=int, default=None, metavar="N")
    parser.add_argument("--model-path", default=None, help="Path to trained cellpose .pt model")
    args = parser.parse_args()
    evaluate(max_samples=args.max_samples, model_path=args.model_path)
