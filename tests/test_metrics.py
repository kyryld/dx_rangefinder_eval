"""Tests for pure metric functions."""

import math

import pytest

from eval_rangefinder.metrics import (
    compute_ap,
    compute_iou,
    greedy_match,
    iou_matrix,
    mae,
    mean_relative_error,
    rmse,
)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

class TestComputeIoU:
    def test_identical_boxes(self):
        box = (0.5, 0.5, 0.2, 0.2)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.1, 0.1, 0.1, 0.1)
        b = (0.9, 0.9, 0.1, 0.1)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        a = (0.25, 0.5, 0.5, 0.5)
        b = (0.75, 0.5, 0.5, 0.5)
        # x overlap = 0.0, actually just touching — use shifted boxes
        a = (0.3, 0.5, 0.4, 0.4)
        b = (0.5, 0.5, 0.4, 0.4)
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area_box_returns_zero(self):
        a = (0.5, 0.5, 0.0, 0.2)
        b = (0.5, 0.5, 0.2, 0.2)
        assert compute_iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

class TestGreedyMatch:
    def test_simple_match(self):
        box = (0.5, 0.5, 0.2, 0.2)
        mat = iou_matrix([box], [box])
        matches = greedy_match(mat, iou_threshold=0.5)
        assert matches == [(0, 0)]

    def test_no_match_below_threshold(self):
        a = (0.1, 0.1, 0.1, 0.1)
        b = (0.9, 0.9, 0.1, 0.1)
        mat = iou_matrix([a], [b])
        assert greedy_match(mat, iou_threshold=0.5) == []

    def test_one_to_one(self):
        boxes = [(0.2, 0.5, 0.2, 0.2), (0.8, 0.5, 0.2, 0.2)]
        mat = iou_matrix(boxes, boxes)
        matches = greedy_match(mat, iou_threshold=0.5)
        assert len(matches) == 2
        gt_idxs = {m[0] for m in matches}
        pred_idxs = {m[1] for m in matches}
        assert gt_idxs == {0, 1}
        assert pred_idxs == {0, 1}

    def test_empty_inputs(self):
        mat = iou_matrix([], [])
        assert greedy_match(mat, 0.5) == []


# ---------------------------------------------------------------------------
# Ranging metrics
# ---------------------------------------------------------------------------

class TestRangingMetrics:
    def test_mae_perfect(self):
        assert mae([10.0, 20.0], [10.0, 20.0]) == pytest.approx(0.0)

    def test_mae_known(self):
        assert mae([10.0, 20.0], [12.0, 18.0]) == pytest.approx(2.0)

    def test_mae_empty(self):
        assert math.isnan(mae([], []))

    def test_rmse_known(self):
        # errors are 2 and -2, rmse = 2
        assert rmse([10.0, 20.0], [12.0, 18.0]) == pytest.approx(2.0)

    def test_rel_err_known(self):
        # |12-10|/10 = 0.2, |18-20|/20 = 0.1 → mean = 0.15
        assert mean_relative_error([10.0, 20.0], [12.0, 18.0]) == pytest.approx(0.15)

    def test_rel_err_empty(self):
        assert math.isnan(mean_relative_error([], []))


# ---------------------------------------------------------------------------
# AP
# ---------------------------------------------------------------------------

class TestComputeAP:
    def test_perfect_detector(self):
        box = (0.5, 0.5, 0.2, 0.2)
        gt = [[box]]
        pred = [[box]]
        scores = [[1.0]]
        ap = compute_ap(gt, pred, scores, iou_threshold=0.5)
        assert ap == pytest.approx(1.0)

    def test_no_predictions(self):
        box = (0.5, 0.5, 0.2, 0.2)
        gt = [[box]]
        pred = [[]]
        scores = [[]]
        ap = compute_ap(gt, pred, scores, iou_threshold=0.5)
        assert ap == pytest.approx(0.0)

    def test_no_gt_returns_nan(self):
        box = (0.5, 0.5, 0.2, 0.2)
        gt = [[]]
        pred = [[box]]
        scores = [[0.9]]
        ap = compute_ap(gt, pred, scores, iou_threshold=0.5)
        assert math.isnan(ap)

    def test_ap_between_0_and_1(self):
        box_gt = (0.5, 0.5, 0.2, 0.2)
        box_fp = (0.1, 0.1, 0.1, 0.1)
        gt = [[box_gt], [box_gt]]
        pred = [[box_gt, box_fp], [box_fp]]  # one TP, two FP, one FN
        scores = [[0.9, 0.8], [0.7]]
        ap = compute_ap(gt, pred, scores, iou_threshold=0.5)
        assert 0.0 <= ap <= 1.0
