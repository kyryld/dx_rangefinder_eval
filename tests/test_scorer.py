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

    def test_no_skipped_files(self):
        assert self.result.skipped_files == 0

    def test_all_classes_present(self):
        assert set(self.result.per_class.keys()) == {"person", "vehicle", "building"}

    def test_ap_valid_because_confidence_present(self):
        for cls, m in self.result.per_class.items():
            assert m.ap_valid, f"AP should be valid for {cls} (confidence scores present)"
            assert 0.0 <= m.ap <= 1.0

    def test_person_recall_is_one(self):
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

    def test_gt_hash_is_12_hex_chars(self):
        assert len(self.result.gt_hash) == 12
        assert all(c in "0123456789abcdef" for c in self.result.gt_hash)

    def test_to_dict_has_expected_keys(self):
        d = self.result.to_dict()
        for cls in ("person", "vehicle", "building"):
            assert f"AP_{cls}" in d
            assert f"MAE_{cls}" in d
            assert f"RMSE_{cls}" in d
            assert f"RelErr_{cls}" in d
        assert "total_gt_all" in d
        assert "detection_recall_all" in d
        assert "gt_hash" in d

    def test_scorer_version_in_result(self):
        from eval_rangefinder.scorer import SCORER_VERSION
        assert self.result.scorer_version == SCORER_VERSION

    def test_config_id_in_result(self):
        assert self.result.config_id == "default_v1"


class TestScorerAPNullWithoutConfidence:
    """AP must be NaN when no predictions carry confidence_detection."""

    def test_ap_nan_when_no_confidence(self, tmp_path):
        import json

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        gt_frame = {
            "format_version": 2,
            "frame_id": "f1",
            "source_image": "f1.png",
            "objects": [
                {
                    "object_id": "g0",
                    "class_name": "person",
                    "bbox": {"x_center_rel": 0.5, "y_center_rel": 0.5, "width_rel": 0.1, "height_rel": 0.2},
                    "distance_m": 20.0,
                }
            ],
        }
        pred_frame = {
            "format_version": 2,
            "frame_id": "f1",
            "source_image": "f1.png",
            "objects": [
                {
                    "object_id": "p0",
                    "class_name": "person",
                    "bbox": {"x_center_rel": 0.5, "y_center_rel": 0.5, "width_rel": 0.1, "height_rel": 0.2},
                    "distance_m": 21.0,
                    # no confidence_detection
                }
            ],
        }
        (gt_dir / "f1.json").write_text(json.dumps(gt_frame))
        (pred_dir / "f1.json").write_text(json.dumps(pred_frame))

        scorer = Scorer()
        result = scorer.score(gt_dir, pred_dir)
        m = result.per_class["person"]
        assert not m.ap_valid
        assert math.isnan(m.ap)


class TestScorerSkipsBadFile:
    """A single malformed file should be skipped, not crash the run."""

    def test_bad_file_skipped(self, tmp_path):
        import json
        import shutil

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        shutil.copytree(EXAMPLE_GT, gt_dir)
        shutil.copytree(EXAMPLE_PRED, pred_dir)

        # Corrupt one prediction file
        bad = pred_dir / "frame_001.json"
        bad.write_text("{ not valid json !!!")

        scorer = Scorer()
        result = scorer.score(gt_dir, pred_dir)
        # matched_files = files in the intersection; skipped is a subset of those
        assert result.matched_files == 2
        assert result.skipped_files == 1
        # Only frame_002 contributed: it has 2 persons, frame_001 (skipped) had 1
        assert result.per_class["person"].total_gt == 2


class TestScorerWithStrictConfig:
    def test_strict_config_runs(self):
        scorer = Scorer(config=CONFIGS["strict_v1"])
        result = scorer.score(EXAMPLE_GT, EXAMPLE_PRED)
        assert result.config_id == "strict_v1"
