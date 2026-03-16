"""
Pure metric computation functions.

All functions are stateless and operate on plain Python/NumPy values so they
can be tested in isolation without any label-loading machinery.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two YOLO-format boxes (cx, cy, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax0, ax1 = ax - aw / 2, ax + aw / 2
    ay0, ay1 = ay - ah / 2, ay + ah / 2
    bx0, bx1 = bx - bw / 2, bx + bw / 2
    by0, by1 = by - bh / 2, by + bh / 2

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter

    return inter / union if union > 0.0 else 0.0


def iou_matrix(
    boxes_a: list[tuple[float, float, float, float]],
    boxes_b: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Return an (N, M) matrix of IoU values between two lists of boxes."""
    n, m = len(boxes_a), len(boxes_b)
    mat = np.zeros((n, m), dtype=np.float64)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            mat[i, j] = compute_iou(a, b)
    return mat


# ---------------------------------------------------------------------------
# Object matching
# ---------------------------------------------------------------------------


def greedy_match(
    iou_mat: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching: highest-IoU pairs first.

    Returns a list of (gt_idx, pred_idx) pairs where IoU >= iou_threshold.
    Each GT and each prediction appears at most once.
    """
    if iou_mat.size == 0:
        return []

    matched: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    # Flatten indices sorted by IoU descending
    flat_order = np.argsort(iou_mat, axis=None)[::-1]
    rows, cols = np.unravel_index(flat_order, iou_mat.shape)

    for gt_idx, pred_idx in zip(rows.tolist(), cols.tolist()):
        if iou_mat[gt_idx, pred_idx] < iou_threshold:
            break
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        matched.append((gt_idx, pred_idx))
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)

    return matched


# ---------------------------------------------------------------------------
# Ranging metrics
# ---------------------------------------------------------------------------


def mae(true: list[float], pred: list[float]) -> float:
    """Mean Absolute Error.  Returns NaN when the list is empty."""
    if not true:
        return float("nan")
    arr_t = np.asarray(true, dtype=np.float64)
    arr_p = np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.abs(arr_t - arr_p)))


def rmse(true: list[float], pred: list[float]) -> float:
    """Root Mean Squared Error.  Returns NaN when the list is empty."""
    if not true:
        return float("nan")
    arr_t = np.asarray(true, dtype=np.float64)
    arr_p = np.asarray(pred, dtype=np.float64)
    return float(np.sqrt(np.mean((arr_t - arr_p) ** 2)))


def mean_relative_error(true: list[float], pred: list[float]) -> float:
    """Mean relative error = mean(|pred - true| / true).

    Returns NaN when the list is empty or all true values are zero.
    """
    if not true:
        return float("nan")
    arr_t = np.asarray(true, dtype=np.float64)
    arr_p = np.asarray(pred, dtype=np.float64)
    nonzero = arr_t != 0.0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs(arr_t[nonzero] - arr_p[nonzero]) / arr_t[nonzero]))


# ---------------------------------------------------------------------------
# Detection AP
# ---------------------------------------------------------------------------


def compute_ap(
    gt_boxes_per_image: list[list[tuple[float, float, float, float]]],
    pred_boxes_per_image: list[list[tuple[float, float, float, float]]],
    pred_scores_per_image: list[list[float]],
    iou_threshold: float = 0.5,
) -> float:
    """Average Precision at `iou_threshold` (VOC-style area under PR curve).

    Parameters
    ----------
    gt_boxes_per_image:
        One list of GT boxes per frame (may be empty).
    pred_boxes_per_image:
        One list of predicted boxes per frame (may be empty).
    pred_scores_per_image:
        Confidence scores corresponding to `pred_boxes_per_image`.
        If a frame has no confidence scores (GT-only run) pass empty lists;
        all predictions will be treated as having score 1.0.
    iou_threshold:
        IoU threshold for a prediction to be a true positive.

    Returns
    -------
    float
        AP ∈ [0, 1], or NaN if there are zero GT objects.
    """
    total_gt = sum(len(g) for g in gt_boxes_per_image)
    if total_gt == 0:
        return float("nan")

    # Collect all predictions with their frame index and score
    all_preds: list[tuple[float, int, int]] = []  # (score, frame_idx, box_idx)
    for img_idx, (boxes, scores) in enumerate(
        zip(pred_boxes_per_image, pred_scores_per_image)
    ):
        for box_idx, box in enumerate(boxes):
            score = scores[box_idx] if box_idx < len(scores) else 1.0
            all_preds.append((score, img_idx, box_idx))

    # Sort by descending confidence
    all_preds.sort(key=lambda x: x[0], reverse=True)

    matched_gt: list[set[int]] = [set() for _ in gt_boxes_per_image]
    tp_list: list[int] = []
    fp_list: list[int] = []

    for _score, img_idx, box_idx in all_preds:
        gt_boxes = gt_boxes_per_image[img_idx]
        pred_box = pred_boxes_per_image[img_idx][box_idx]

        best_iou = iou_threshold - 1e-9
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt[img_idx]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched_gt[img_idx].add(best_gt_idx)
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list, dtype=np.float64)
    fp_cum = np.cumsum(fp_list, dtype=np.float64)

    recalls = tp_cum / total_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    # Prepend sentinel (0, 1) and append (1, 0) for stable AUC integration
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Make precision monotonically non-increasing from right to left
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Sum areas of trapezoids where recall changes
    idx = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = float(np.sum((recalls[idx] - recalls[idx - 1]) * precisions[idx]))
    return ap
