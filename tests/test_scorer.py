"""Integration tests for the scorer using the example_data fixtures."""

import math
from pathlib import Path

import pytest

from eval_rangefinder.scorer import CONFIGS, Scorer

EXAMPLE_GT = Path(__file__).parent.parent / "example_data" / "gt"
EXAMPLE_PRED = Path(__file__).parent.parent / "example_data" / "predictions"


class TestScorerWithExampleData:
    def setup_method(self):
        self.scorer = Scorer(config=CONFIGS["default_v1"])
        self.result = self.scorer.score(EXAMPLE_GT, EXAMPLE_PRED, dataset_name="example")

    def test_matched_files(self):
        assert self.result.matched_files == 2

    def test_no_unmatched_files(self):
        assert self.result.gt_only_files == 0
        assert self.result.pred_only_files == 0

    def test_all_classes_present(self):
        assert set(self.result.per_class.keys()) == {"person", "vehicle", "building"}

    def test_ap_in_range(self):
        for cls, m in self.result.per_class.items():
            if not math.isnan(m.ap):
                assert 0.0 <= m.ap <= 1.0, f"AP out of range for {cls}"

    def test_person_recall_is_one(self):
        """All 3 GT persons have matching predictions in example data."""
        m = self.result.per_class["person"]
        assert m.total_gt == 3
        assert m.matched_gt == 3

    def test_vehicle_recall_is_one(self):
        m = self.result.per_class["vehicle"]
        assert m.total_gt == 2
        assert m.matched_gt == 2

    def test_building_recall_is_one(self):
        m = self.result.per_class["building"]
        assert m.total_gt == 1
        assert m.matched_gt == 1

    def test_mae_positive(self):
        for cls, m in self.result.per_class.items():
            if not math.isnan(m.mae):
                assert m.mae >= 0.0

    def test_to_dict_has_expected_keys(self):
        d = self.result.to_dict()
        for cls in ("person", "vehicle", "building"):
            assert f"AP_{cls}" in d
            assert f"MAE_{cls}" in d
            assert f"RMSE_{cls}" in d
            assert f"RelErr_{cls}" in d
        assert "total_gt_all" in d
        assert "detection_recall_all" in d

    def test_scorer_version_in_result(self):
        from eval_rangefinder.scorer import SCORER_VERSION
        assert self.result.scorer_version == SCORER_VERSION

    def test_config_id_in_result(self):
        assert self.result.config_id == "default_v1"


class TestScorerWithStrictConfig:
    def test_strict_config_runs(self):
        scorer = Scorer(config=CONFIGS["strict_v1"])
        result = scorer.score(EXAMPLE_GT, EXAMPLE_PRED)
        assert result.config_id == "strict_v1"
