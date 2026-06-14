"""Evaluate Classpose model on PanNuke fold3 — LSP-DETR aligned metrics.

Metrics:
    AJI, AP@0.5, AP@0.7, AP@0.9, AP@0.5:0.05:0.95,
    bPQ, bMPQ, mPQ, mMPQ,
    F1 (centroid, r=12), Precision, Recall.

All instance-matching metrics use Hungarian assignment (not greedy)
to match LSP-DETR exactly.

Usage:
    python evaluate.py
    python evaluate.py --max-samples 10
    python evaluate.py --model-path path/to/model.pt
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
from train import MODEL_NAME, N_CLASSES

MODEL_PATH = Path("models") / MODEL_NAME / (MODEL_NAME + ".pt")

IOU_THRESHOLDS = sorted(set(round(x, 2) for x in np.arange(0.5, 1.0, 0.05)))
CENTROID_RADIUS = 12


def _load_test_split(max_samples: int | None = None):
    from data import _split_has_data, _process_fold, _load_cached_split

    split_dir = DATA_DIR / "test"
    if max_samples is None and _split_has_data(split_dir):
        print("Loading cached test split from disk.")
        images, labels, categories = _load_cached_split(split_dir)
    else:
        images, labels, categories = _process_fold("fold3", "test", max_samples)
    return images, labels, categories


def _extract_instances(
    mask_array: np.ndarray,
    class_array: np.ndarray,
    flows: np.ndarray | None = None,
):
    """Extract per-instance masks, classes, centroids, conf from cellpose output.

    Returns:
        masks:     list of (H,W) bool arrays
        classes:   list of 1-indexed int class labels
        centroids: (N, 2) float64 array — (cy, cx)
        confs:     (N,) float64 array — mean flow magnitude per instance
    """
    instance_ids = np.unique(mask_array)
    instance_ids = instance_ids[instance_ids != 0]

    masks = []
    classes = []
    centroids_cy = []
    centroids_cx = []
    confs = []

    if flows is not None:
        flow_mag = np.sqrt(flows[..., 0].astype(np.float64) ** 2
                          + flows[..., 1].astype(np.float64) ** 2)

    for inst_id in instance_ids:
        mask = mask_array == inst_id
        masks.append(mask)
        classes.append(int(class_array[mask][0]))

        ys, xs = np.where(mask)
        if len(ys) == 0:
            centroids_cy.append(0.0)
            centroids_cx.append(0.0)
        else:
            centroids_cy.append(float(ys.mean()))
            centroids_cx.append(float(xs.mean()))

        if flows is not None:
            confs.append(float(flow_mag[mask].mean()))

    centroids = np.column_stack([centroids_cy, centroids_cx]) if masks else np.zeros((0, 2))
    confs_arr = np.array(confs, dtype=np.float64) if confs else np.zeros(len(masks), dtype=np.float64)
    return masks, classes, centroids, confs_arr


def _extract_gt_instances(label_2ch: np.ndarray, cell_type_arr: np.ndarray):
    """Extract per-instance GT masks, classes, and centroids.

    Args:
        label_2ch: (H, W, 2) — channel 0=instance ID, channel 1=class.
        cell_type_arr: (N_instances,) 0-indexed cell-type indices.

    Returns:
        masks:     list of (H,W) bool arrays
        classes:   list of 1-indexed int class labels
        centroids: (N, 2) float64 array — (cy, cx)
    """
    inst_map = label_2ch[..., 0] if label_2ch.ndim == 3 else label_2ch
    instance_ids = np.unique(inst_map)
    instance_ids = instance_ids[instance_ids != 0]

    masks = []
    classes = []
    centroids_cy = []
    centroids_cx = []

    for inst_id in instance_ids:
        iid = int(inst_id)
        mask = inst_map == inst_id
        masks.append(mask)
        if len(cell_type_arr) > 0 and iid - 1 < len(cell_type_arr):
            classes.append(int(cell_type_arr[iid - 1]) + 1)
        else:
            classes.append(0)

        ys, xs = np.where(mask)
        centroids_cy.append(float(ys.mean()) if len(ys) else 0.0)
        centroids_cx.append(float(xs.mean()) if len(xs) else 0.0)

    centroids = np.column_stack([centroids_cy, centroids_cx]) if masks else np.zeros((0, 2))
    return masks, classes, centroids


def _mask_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
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
        intersection, union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def compute_aji(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> float:
    """Aggregated Jaccard Index with Hungarian matching."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0:
        return 0.0

    if n_pred == 0:
        return 0.0

    iou_mat = _mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    intersection = 0.0
    union = 0.0
    matched_gt = set()

    for pi, gi in zip(row_ind, col_ind):
        if iou_mat[pi, gi] > 0:
            inter = np.logical_and(pred_masks[pi], gt_masks[gi]).sum()
            uni = np.logical_or(pred_masks[pi], gt_masks[gi]).sum()
            intersection += inter
            union += uni
            matched_gt.add(gi)

    for gi in range(n_gt):
        if gi not in matched_gt:
            union += gt_masks[gi].sum()

    return float(intersection / union) if union > 0 else 0.0


def _compute_pq_masked(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
    mask: np.ndarray | None = None,
):
    """Compute PQ with Hungarian matching.  Returns (pq, sq, dq)."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    if mask is not None:
        pred_masks = [m & mask for m in pred_masks]
        gt_masks = [m & mask for m in gt_masks]

    iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold

    tp = int(valid.sum())
    fp = n_pred - tp
    fn = n_gt - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0
    pq = sq * dq
    return float(pq), float(sq), float(dq)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def compute_metrics_streaming(
    image_results: list[dict],
    num_classes: int,
) -> dict:
    """Compute all metrics in a single pass, one image at a time.

    Peak memory = O(max_instances_per_image * H * W).
    """
    # Accumulators
    aji_scores: list[float] = []
    bpq_scores: list[float] = []
    bmpq_scores: list[float] = []
    class_pq: dict[int, list[float]] = {c: [] for c in range(1, num_classes)}   # 1-indexed
    class_mpq: dict[int, list[float]] = {c: [] for c in range(1, num_classes)}

    centroid_tp = 0
    centroid_fp = 0
    centroid_fn = 0

    # AP accumulators — per class, per threshold
    seen_classes: set[int] = set()
    for r in image_results:
        seen_classes.update(r["gt_classes"])
        seen_classes.update(r["pred_classes"])

    ap_stats: dict[int, dict[float, dict]] = {}
    for cls_id in sorted(seen_classes):
        ap_stats[cls_id] = {t: {"tp": [], "fp": [], "conf": [], "n_gt": 0}
                           for t in IOU_THRESHOLDS}

    n_total = len(image_results)
    for i, r in enumerate(image_results):
        gt_masks = r["gt_masks"]
        pred_masks = r["pred_masks"]
        gt_classes = r["gt_classes"]
        pred_classes = r["pred_classes"]
        pred_confs = r["pred_confs"]
        gt_centroids = r["gt_centroids"]
        pred_centroids = r["pred_centroids"]
        im_h, im_w = r["imgsz"]

        if i == 0:
            print(f"  First image: {len(pred_masks)} preds, {len(gt_masks)} GT, "
                  f"size={im_h}x{im_w}", flush=True)

        # --- AJI ---
        aji_val = compute_aji(pred_masks, gt_masks)
        aji_scores.append(aji_val)

        # --- bPQ / bMPQ ---
        if len(pred_masks) > 0:
            pred_binary = np.stack(pred_masks).max(axis=0).astype(np.uint8)
        else:
            pred_binary = np.zeros((im_h, im_w), dtype=np.uint8)

        if len(gt_masks) > 0:
            gt_binary = np.stack(gt_masks).max(axis=0).astype(np.uint8)
        else:
            gt_binary = np.zeros((im_h, im_w), dtype=np.uint8)

        bpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary])
        bpq_scores.append(bpq)

        if gt_binary.sum() > 0:
            fg = gt_binary > 0
            bmpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary], mask=fg)
        else:
            bmpq = 0.0
        bmpq_scores.append(bmpq)

        # --- mPQ / mMPQ ---
        gt_any = np.stack(gt_masks).max(axis=0).astype(np.uint8) if gt_masks else np.zeros((im_h, im_w), dtype=np.uint8)
        fg_mask = gt_any > 0 if gt_any.sum() > 0 else None

        class_pq_img: dict[int, float] = {}
        for cls_id in range(1, num_classes):
            pred_idx = [j for j, c in enumerate(pred_classes) if c == cls_id]
            gt_idx = [j for j, c in enumerate(gt_classes) if c == cls_id]

            pred_cls_masks = [pred_masks[j] for j in pred_idx]
            gt_cls_masks = [gt_masks[j] for j in gt_idx]

            pq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks)
            class_pq[cls_id].append(pq)
            class_pq_img[cls_id] = pq

            if fg_mask is not None:
                mpq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks, mask=fg_mask)
            else:
                mpq = 0.0
            class_mpq[cls_id].append(mpq)

        # --- AP (per-class Hungarian matching, sorted by confidence) ---
        for cls_id in sorted(seen_classes):
            cls_pred_mask = np.array(pred_classes) == cls_id
            cls_pred_idx = np.where(cls_pred_mask)[0]
            cls_gt_idx = np.array([j for j, c in enumerate(gt_classes) if c == cls_id])
            cls_confs = pred_confs[cls_pred_idx]

            n_pred_cls = len(cls_pred_idx)
            n_gt_cls = len(cls_gt_idx)

            for t in IOU_THRESHOLDS:
                ap_stats[cls_id][t]["n_gt"] += n_gt_cls

            if n_pred_cls > 0 and n_gt_cls > 0:
                cls_pred_masks = [pred_masks[j] for j in cls_pred_idx]
                cls_gt_masks = [gt_masks[j] for j in cls_gt_idx]
                iou_mat = _mask_iou_matrix(cls_pred_masks, cls_gt_masks)
                row_ind, col_ind = linear_sum_assignment(-iou_mat)
                matched_iou = iou_mat[row_ind, col_ind]
            else:
                row_ind = np.array([], dtype=int)
                matched_iou = np.array([], dtype=float)

            for t in IOU_THRESHOLDS:
                if n_gt_cls > 0:
                    valid_mask = matched_iou >= t
                    matched_pred = set(row_ind[valid_mask].tolist())
                else:
                    matched_pred = set()

                for pi in range(n_pred_cls):
                    is_tp = pi in matched_pred
                    ap_stats[cls_id][t]["conf"].append(float(cls_confs[pi]) if pi < len(cls_confs) else 0.0)
                    ap_stats[cls_id][t]["tp"].append(is_tp)
                    ap_stats[cls_id][t]["fp"].append(not is_tp)

        # --- Centroid F1 ---
        n_pred = len(pred_masks)
        n_gt = len(gt_masks)
        if n_pred > 0 and n_gt > 0:
            dist_matrix = np.linalg.norm(
                pred_centroids[:, None, :] - gt_centroids[None, :, :], axis=2
            )
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            tp = int((dist_matrix[row_ind, col_ind] <= CENTROID_RADIUS).sum())
        elif n_pred > 0:
            tp = 0
        else:
            tp = 0
        centroid_tp += tp
        centroid_fp += n_pred - tp
        centroid_fn += n_gt - tp

        del gt_masks, pred_masks
        gc.collect()

        if (i + 1) % 500 == 0:
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"  Processed {i + 1}/{n_total} images (RSS={mem_mb:.0f}MB)", flush=True)

    # --- Aggregate ---

    mean_aji = float(np.mean(aji_scores)) if aji_scores else 0.0

    mean_bpq = float(np.mean(bpq_scores)) if bpq_scores else 0.0
    mean_bmpq = float(np.mean(bmpq_scores)) if bmpq_scores else 0.0

    mpq_values = []
    mmpq_values = []
    for c in range(1, num_classes):
        valid_pq = [v for v in class_pq[c] if v > 0]
        valid_mpq = [v for v in class_mpq[c] if v > 0]
        if valid_pq:
            mpq_values.append(np.mean(valid_pq))
        if valid_mpq:
            mmpq_values.append(np.mean(valid_mpq))
    mean_mpq = float(np.mean(mpq_values)) if mpq_values else 0.0
    mean_mmpq = float(np.mean(mmpq_values)) if mmpq_values else 0.0

    # AP
    ap_results: dict[float, dict] = {}
    for t in IOU_THRESHOLDS:
        aps = []
        for cls_id in sorted(seen_classes):
            stats = ap_stats[cls_id][t]
            n_gt = stats["n_gt"]
            if n_gt == 0:
                continue
            confs = np.array(stats["conf"])
            tps = np.array(stats["tp"])
            fps = np.array(stats["fp"])
            if len(confs) == 0:
                aps.append(0.0)
                continue
            order = np.argsort(-confs)
            tps = tps[order]
            fps = fps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / n_gt
            aps.append(_compute_ap(recall, precision))
        ap_results[t] = {"AP": float(np.mean(aps)) if aps else 0.0}

    # Centroid F1
    prec = centroid_tp / (centroid_tp + centroid_fp) if (centroid_tp + centroid_fp) > 0 else 0.0
    rec = centroid_tp / (centroid_tp + centroid_fn) if (centroid_tp + centroid_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "aji": mean_aji,
        "bpq": mean_bpq,
        "bmpq": mean_bmpq,
        "mpq": mean_mpq,
        "mmpq": mean_mmpq,
        "ap": ap_results,
        "centroid": {"precision": prec, "recall": rec, "f1": f1},
        "class_pq": class_pq,
        "class_mpq": class_mpq,
    }


def _per_class_breakdown(
    image_results: list[dict],
    num_classes: int,
    names: list[str],
    class_pq: dict[int, list[float]],
    class_mpq: dict[int, list[float]],
) -> None:
    """Per-class centroid F1 (class-matched) + PQ / mPQ."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    class_gt = [0] * num_classes
    class_pred = [0] * num_classes

    for r in image_results:
        pred_classes = r["pred_classes"]
        gt_classes = r["gt_classes"]
        pred_centroids = r["pred_centroids"]
        gt_centroids = r["gt_centroids"]

        for c in gt_classes:
            if 0 <= c < num_classes:
                class_gt[c] += 1
        for c in pred_classes:
            if 0 <= c < num_classes:
                class_pred[c] += 1

        n_gt = len(gt_classes)
        n_pred = len(pred_classes)
        if n_gt == 0 or n_pred == 0:
            continue

        dist = np.linalg.norm(
            pred_centroids[:, None, :] - gt_centroids[None, :, :], axis=2
        )
        row_ind, col_ind = linear_sum_assignment(dist)

        for ri, ci in zip(row_ind, col_ind):
            if dist[ri, ci] <= CENTROID_RADIUS:
                pc = pred_classes[ri]
                gc = gt_classes[ci]
                if pc == gc and 0 <= gc < num_classes:
                    tp[gc] += 1

        for ri in range(n_pred):
            if ri not in row_ind or dist[ri, col_ind[np.where(row_ind == ri)[0][0]] if ri in row_ind else 0] > CENTROID_RADIUS:
                pass  # handled by total - tp below

    print(f"\n{'=' * 80}")
    print(f"  Nuclei Class Breakdown (Centroid F1, class-matched)")
    print(f"{'=' * 80}")
    header = f"  {'Class':<16} {'GT':>6} {'Pred':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'PQ':>8} {'mPQ':>8}"
    print(header)
    print(f"  {'-' * 16} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for c in range(1, num_classes):
        name = names[c - 1] if c - 1 < len(names) else f"class_{c}"
        prec_c = tp[c] / class_pred[c] if class_pred[c] else 0.0
        rec_c = tp[c] / class_gt[c] if class_gt[c] else 0.0
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) else 0.0
        pq_vals = [v for v in class_pq.get(c, []) if v > 0]
        mpq_vals = [v for v in class_mpq.get(c, []) if v > 0]
        pq_c = float(np.mean(pq_vals)) if pq_vals else 0.0
        mpq_c = float(np.mean(mpq_vals)) if mpq_vals else 0.0
        print(f"  {name:<16} {class_gt[c]:>6} {class_pred[c]:>6} "
              f"{prec_c:>8.4f} {rec_c:>8.4f} {f1_c:>8.4f} {pq_c:>8.4f} {mpq_c:>8.4f}")

    tot_gt = sum(class_gt)
    tot_pred = sum(class_pred)
    tot_tp = sum(tp)
    prec_t = tot_tp / tot_pred if tot_pred else 0.0
    rec_t = tot_tp / tot_gt if tot_gt else 0.0
    f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) else 0.0
    print(f"  {'-' * 16} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    print(f"  {'TOTAL':<16} {tot_gt:>6} {tot_pred:>6} "
          f"{prec_t:>8.4f} {rec_t:>8.4f} {f1_t:>8.4f}")
    print(f"{'=' * 80}")


def _diagnose_recall(image_results: list[dict], num_classes: int) -> None:
    """Break down recall by GT size bin, class, and nearest-prediction distance."""
    names = CELL_TYPES

    gt_areas_matched: list[float] = []
    gt_areas_unmatched: list[float] = []
    class_total = [0] * num_classes
    class_matched_list = [0] * num_classes
    unmatched_dists: list[float] = []
    unmatched_nearest_conf: list[float] = []
    unmatched_nearest_cls: list[int] = []

    for r in image_results:
        gt_masks = r["gt_masks"]
        pred_masks = r["pred_masks"]
        pred_confs = r["pred_confs"]
        pred_classes = r["pred_classes"]
        gt_classes = r["gt_classes"]
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
            pred_centroids = np.zeros((0, 2))

        for j in range(n_gt):
            cls_id = gt_classes[j]
            if 0 <= cls_id < num_classes:
                class_total[cls_id] += 1
                if matched[j]:
                    class_matched_list[cls_id] += 1

            if matched[j]:
                gt_areas_matched.append(float(gt_areas[j]))
            else:
                gt_areas_unmatched.append(float(gt_areas[j]))
                if n_pred > 0:
                    dists_j = np.linalg.norm(pred_centroids - gt_centroids[j], axis=1)
                    nearest_idx = int(np.argmin(dists_j))
                    unmatched_dists.append(float(dists_j[nearest_idx]))
                    unmatched_nearest_conf.append(float(pred_confs[nearest_idx]))
                    unmatched_nearest_cls.append(int(pred_classes[nearest_idx]))
                else:
                    unmatched_dists.append(float("inf"))
                    unmatched_nearest_conf.append(0.0)
                    unmatched_nearest_cls.append(-1)

    total_gt = sum(class_total)
    total_matched = sum(class_matched_list)
    recall = total_matched / total_gt if total_gt > 0 else 0.0

    print(f"\n{'=' * 65}")
    print("  Recall Diagnosis (centroid-based)")
    print(f"{'=' * 65}")
    print(f"  Overall: {total_matched}/{total_gt} = {recall:.4f}")

    # By class
    print(f"\n  {'Class':<18} {'Total':>8} {'Matched':>8} {'Recall':>8}")
    print(f"  {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 8}")
    for c in range(1, num_classes):
        if class_total[c] > 0:
            r_c = class_matched_list[c] / class_total[c]
            name = names[c - 1] if c - 1 < len(names) else f"class_{c}"
            print(f"  {name:<18} {class_total[c]:>8} {class_matched_list[c]:>8} {r_c:>8.4f}")

    # By size bin
    all_areas = np.array(gt_areas_matched + gt_areas_unmatched)
    if len(all_areas) >= 3:
        p33 = np.percentile(all_areas, 33)
        p67 = np.percentile(all_areas, 67)
        bins = [
            ("Small (<P33)", lambda a: a < p33),
            ("Medium (P33-P67)", lambda a: (a >= p33) & (a < p67)),
            ("Large (>P67)", lambda a: a >= p67),
        ]
        print(f"\n  {'Size Bin':<20} {'Area Range':<18} {'Total':>8} {'Matched':>8} {'Recall':>8}")
        print(f"  {'-' * 20} {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 8}")
        for label, cond in bins:
            t = sum(1 for a in all_areas if cond(a))
            m = sum(1 for a in gt_areas_matched if cond(a))
            if t > 0:
                min_a = min(a for a in all_areas if cond(a))
                max_a = max(a for a in all_areas if cond(a))
                rng = f"{min_a:.0f}-{max_a:.0f}px²"
            else:
                rng = "-"
            rec_b = m / t if t > 0 else 0.0
            print(f"  {label:<20} {rng:<18} {t:>8} {m:>8} {rec_b:>8.4f}")

    # Unmatched GT: distance to nearest prediction
    if unmatched_dists:
        ud = np.array(unmatched_dists)
        finite = ud[np.isfinite(ud)]
        print(f"\n  Unmatched GT — Distance to nearest prediction:")
        print(f"    <5px (near miss): {int(np.sum(finite < 5))}/{len(unmatched_dists)} "
              f"({100 * sum(finite < 5) / len(unmatched_dists):.1f}%)")
        print(f"    5-12px (drifted):  {int(np.sum((finite >= 5) & (finite <= 12)))}/{len(unmatched_dists)} "
              f"({100 * sum((finite >= 5) & (finite <= 12)) / len(unmatched_dists):.1f}%)")
        print(f"    >12px (truly miss):{int(np.sum(finite > 12))}/{len(unmatched_dists)} "
              f"({100 * sum(finite > 12) / len(unmatched_dists):.1f}%)")

    # Unmatched GT: confidence of nearest prediction within 12px
    if unmatched_nearest_conf:
        umc = np.array(unmatched_nearest_conf)
        nearest_12 = np.array([d <= 12.0 for d in unmatched_dists])
        conf_within_12 = umc[nearest_12]
        print(f"\n  Unmatched GT — Confidence of nearest pred within 12px:")
        if len(conf_within_12) > 0:
            print(f"    Mean conf:          {conf_within_12.mean():.4f}")
            print(f"    Median conf:        {float(np.median(conf_within_12)):.4f}")
            above_thresh = int((conf_within_12 >= 0.2).sum())
            print(f"    Conf ≥0.2:          {above_thresh}/{len(conf_within_12)} "
                  f"({100 * above_thresh / max(len(conf_within_12), 1):.1f}%)")

    print(f"{'=' * 65}")


def evaluate(
    max_samples: int | None = None,
    model_path: str | Path | None = None,
) -> dict:
    """Run evaluation on PanNuke fold3 — LSP-DETR protocol."""
    print("Loading PanNuke fold3 (test) ...")
    images, gt_labels_2ch, categories = _load_test_split(max_samples)
    print(f"  {len(images)} test images loaded.")

    print("Loading Classpose model ...")
    from classpose.models import ClassposeModel

    mpath = str(model_path) if model_path else str(MODEL_PATH)
    model = ClassposeModel(gpu=True, pretrained_model=mpath, nclasses=N_CLASSES)

    image_results: list[dict] = []
    t_start = time.perf_counter()

    for idx in tqdm(range(len(images)), desc="Inference", unit="img"):
        img = images[idx]
        gt_l2ch = gt_labels_2ch[idx]
        cat_arr = categories[idx]

        pred_masks_arr, flows, pred_class_masks, _ = model.eval(img)
        im_h, im_w = img.shape[:2]

        pred_masks, pred_classes, pred_centroids, pred_confs = _extract_instances(
            pred_masks_arr.astype(np.int32),
            pred_class_masks.astype(np.int32),
            flows,
        )
        gt_masks, gt_classes, gt_centroids = _extract_gt_instances(gt_l2ch, cat_arr)

        image_results.append({
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "pred_classes": pred_classes,
            "gt_classes": gt_classes,
            "pred_confs": pred_confs,
            "pred_centroids": pred_centroids,
            "gt_centroids": gt_centroids,
            "imgsz": (im_h, im_w),
        })

    t_inf = time.perf_counter() - t_start
    n_gt_total = sum(len(r["gt_masks"]) for r in image_results)
    n_pred_total = sum(len(r["pred_masks"]) for r in image_results)
    print(f"\n  Inference: {len(images)} images in {t_inf:.1f}s "
          f"({t_inf / len(images) * 1000:.1f} ms/img)")
    print(f"  GT instances: {n_gt_total},  Pred instances: {n_pred_total}")

    print("\nComputing metrics (streaming)...")
    metrics = compute_metrics_streaming(image_results, num_classes=N_CLASSES)

    ap_results = metrics["ap"]
    ap50 = ap_results.get(0.5, {}).get("AP", 0.0)
    ap70 = ap_results.get(0.7, {}).get("AP", 0.0)
    ap90 = ap_results.get(0.9, {}).get("AP", 0.0)
    ap50_95 = float(np.mean([ap_results[t]["AP"] for t in sorted(ap_results.keys())]))
    f12 = metrics["centroid"]

    # --- Print results ---
    print(f"\n{'=' * 60}")
    print(f"  PanNuke Fold3 Evaluation Results (LSP-DETR Protocol)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'-' * 42}")
    print(f"  {'AJI':<30} {metrics['aji']:>12.4f}")
    print(f"  {'AP@0.5':<30} {ap50:>12.4f}")
    print(f"  {'AP@0.7':<30} {ap70:>12.4f}")
    print(f"  {'AP@0.9':<30} {ap90:>12.4f}")
    print(f"  {'AP@0.5:0.05:0.95':<30} {ap50_95:>12.4f}")
    print(f"  {'bPQ':<30} {metrics['bpq']:>12.4f}")
    print(f"  {'bMPQ':<30} {metrics['bmpq']:>12.4f}")
    print(f"  {'mPQ':<30} {metrics['mpq']:>12.4f}")
    print(f"  {'mMPQ':<30} {metrics['mmpq']:>12.4f}")
    print(f"  {'F1 (centroid, r=12)':<30} {f12['f1']:>12.4f}")
    print(f"  {'Precision (centroid)':<30} {f12['precision']:>12.4f}")
    print(f"  {'Recall (centroid)':<30} {f12['recall']:>12.4f}")
    print(f"  {'Inference (ms/img)':<30} {t_inf / len(images) * 1000:>12.1f}")
    print(f"{'=' * 60}")

    # Per-class breakdown
    _per_class_breakdown(
        image_results, N_CLASSES, CELL_TYPES,
        metrics["class_pq"], metrics["class_mpq"],
    )

    # Recall diagnosis
    _diagnose_recall(image_results, N_CLASSES)

    results = {
        "aji": metrics["aji"],
        "bpq": metrics["bpq"],
        "bmpq": metrics["bmpq"],
        "mpq": metrics["mpq"],
        "mmpq": metrics["mmpq"],
        "ap": ap_results,
        "centroid_f1": f12,
        "inference_ms_per_img": t_inf / len(images) * 1000,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Classpose on PanNuke fold3 (LSP-DETR protocol)"
    )
    parser.add_argument("--max-samples", type=int, default=None, metavar="N")
    parser.add_argument("--model-path", default=None, help="Path to .pt model")
    args = parser.parse_args()
    evaluate(max_samples=args.max_samples, model_path=args.model_path)
