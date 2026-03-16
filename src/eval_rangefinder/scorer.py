"""
Scorer: loads GT and prediction folders, matches files, computes all metrics.

The scorer is self-describing: it embeds a SCORER_VERSION string and the
active config identifier in every result dict so that spreadsheet rows are
always traceable to the exact evaluation code and parameters that produced them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import warnings
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .metrics import (
    compute_ap,
    greedy_match,
    iou_matrix,
    mae,
    mean_relative_error,
    rmse,
)
from .schema import VALID_CLASSES, FrameLabel

# Bump when the scoring algorithm changes in a way that would alter results.
SCORER_VERSION: str = "1.1.0"

ALL_CLASSES: list[str] = ["person", "vehicle", "building"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ScorerConfig:
    """All tunable parameters for the scorer.

    Add a new named config here (and register it in CONFIGS) if you want to
    run evaluations under different matching strictness settings.
    """

    config_id: str = "default_v1"
    iou_thresholds: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"person": 0.5, "vehicle": 0.5, "building": 0.5}
    )
    ap_iou_threshold: float = 0.5

    def iou_for(self, class_name: str) -> float:
        return self.iou_thresholds.get(class_name, 0.5)


CONFIGS: dict[str, ScorerConfig] = {
    "default_v1": ScorerConfig(),
    "strict_v1": ScorerConfig(
        config_id="strict_v1",
        iou_thresholds={"person": 0.6, "vehicle": 0.6, "building": 0.6},
        ap_iou_threshold=0.5,
    ),
}


# ---------------------------------------------------------------------------
# Per-class accumulators
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ClassAccumulator:
    """Collects matched pairs for one class across all frames."""

    gt_distances: list[float] = dataclasses.field(default_factory=list)
    pred_distances: list[float] = dataclasses.field(default_factory=list)

    gt_boxes_per_image: list[list[tuple[float, float, float, float]]] = dataclasses.field(
        default_factory=list
    )
    pred_boxes_per_image: list[list[tuple[float, float, float, float]]] = dataclasses.field(
        default_factory=list
    )
    pred_scores_per_image: list[list[float]] = dataclasses.field(default_factory=list)

    total_gt: int = 0
    matched_gt: int = 0
    total_pred: int = 0
    # Whether at least one prediction for this class carried a real confidence score.
    # AP is set to NaN when this is False — ranking all predictions equally is not
    # a meaningful AP measurement.
    has_confidence: bool = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ClassMetrics:
    class_name: str
    ap: float  # NaN when no confidence scores were present
    ap_valid: bool  # False when AP was suppressed due to missing confidence
    mae: float
    rmse: float
    rel_err: float
    total_gt: int
    matched_gt: int
    total_pred: int
    detection_recall: float


@dataclasses.dataclass
class ScorerResult:
    scorer_version: str
    config_id: str
    dataset_name: str
    gt_dir: str
    gt_hash: str
    pred_dir: str
    matched_files: int
    gt_only_files: int
    pred_only_files: int
    skipped_files: int
    per_class: dict[str, ClassMetrics]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scorer_version": self.scorer_version,
            "config_id": self.config_id,
            "dataset_name": self.dataset_name,
            "gt_dir": self.gt_dir,
            "gt_hash": self.gt_hash,
            "pred_dir": self.pred_dir,
            "matched_files": self.matched_files,
            "gt_only_files": self.gt_only_files,
            "pred_only_files": self.pred_only_files,
            "skipped_files": self.skipped_files,
            # Derived file counts for spreadsheet columns
            "gt_files": self.matched_files + self.gt_only_files,
            "pred_files": self.matched_files + self.pred_only_files,
        }
        for cls, m in self.per_class.items():
            d[f"AP_{cls}"] = m.ap
            d[f"AP_{cls}_valid"] = m.ap_valid
            d[f"MAE_{cls}"] = m.mae
            d[f"RMSE_{cls}"] = m.rmse
            d[f"RelErr_{cls}"] = m.rel_err
            d[f"total_gt_{cls}"] = m.total_gt
            d[f"matched_gt_{cls}"] = m.matched_gt
            d[f"total_pred_{cls}"] = m.total_pred
            d[f"recall_{cls}"] = m.detection_recall

        d["total_gt_all"] = sum(m.total_gt for m in self.per_class.values())
        d["matched_gt_all"] = sum(m.matched_gt for m in self.per_class.values())
        d["total_pred_all"] = sum(m.total_pred for m in self.per_class.values())
        total_gt_all = d["total_gt_all"]
        d["detection_recall_all"] = (
            d["matched_gt_all"] / total_gt_all if total_gt_all > 0 else float("nan")
        )
        return d


# ---------------------------------------------------------------------------
# GT dataset hash
# ---------------------------------------------------------------------------


def compute_gt_hash(gt_dir: Path) -> str:
    """Stable 12-char hex fingerprint of all GT JSON files (sorted by name)."""
    h = hashlib.sha256()
    for p in sorted(gt_dir.glob("*.json")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class Scorer:
    def __init__(self, config: ScorerConfig | None = None) -> None:
        self.config = config or CONFIGS["default_v1"]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score(
        self,
        gt_dir: Path,
        pred_dir: Path,
        dataset_name: str = "",
        console: Console | None = None,
    ) -> ScorerResult:
        c = console or Console()

        gt_files = {p.name: p for p in gt_dir.glob("*.json")}
        pred_files = {p.name: p for p in pred_dir.glob("*.json")}

        shared = sorted(gt_files.keys() & pred_files.keys())
        gt_only = sorted(gt_files.keys() - pred_files.keys())
        pred_only = sorted(pred_files.keys() - gt_files.keys())

        accum: dict[str, _ClassAccumulator] = {cls: _ClassAccumulator() for cls in ALL_CLASSES}
        skipped = 0

        for fname in shared:
            try:
                gt_frame = FrameLabel.from_file(gt_files[fname])
                pred_frame = FrameLabel.from_file(pred_files[fname])
            except Exception as exc:
                c.print(f"[yellow]  Skipping {fname}: {exc}[/yellow]")
                skipped += 1
                continue
            self._process_frame(gt_frame, pred_frame, accum)

        per_class: dict[str, ClassMetrics] = {}
        for cls in ALL_CLASSES:
            per_class[cls] = self._finalise_class(cls, accum[cls], console=c)

        gt_hash = compute_gt_hash(gt_dir)

        return ScorerResult(
            scorer_version=SCORER_VERSION,
            config_id=self.config.config_id,
            dataset_name=dataset_name or gt_dir.name,
            gt_dir=str(gt_dir),
            gt_hash=gt_hash,
            pred_dir=str(pred_dir),
            matched_files=len(shared),
            gt_only_files=len(gt_only),
            pred_only_files=len(pred_only),
            skipped_files=skipped,
            per_class=per_class,
        )

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        gt_frame: FrameLabel,
        pred_frame: FrameLabel,
        accum: dict[str, _ClassAccumulator],
    ) -> None:
        for cls in ALL_CLASSES:
            iou_thresh = self.config.iou_for(cls)

            gt_objs = [o for o in gt_frame.objects if o.class_name == cls]
            pred_objs = [o for o in pred_frame.objects if o.class_name == cls]

            gt_boxes = [o.bbox.as_tuple() for o in gt_objs]
            pred_boxes = [o.bbox.as_tuple() for o in pred_objs]
            pred_scores: list[float] = []
            for o in pred_objs:
                if o.confidence_detection is not None:
                    pred_scores.append(o.confidence_detection)
                    accum[cls].has_confidence = True
                else:
                    pred_scores.append(1.0)

            accum[cls].total_gt += len(gt_objs)
            accum[cls].total_pred += len(pred_objs)

            accum[cls].gt_boxes_per_image.append(gt_boxes)
            accum[cls].pred_boxes_per_image.append(pred_boxes)
            accum[cls].pred_scores_per_image.append(pred_scores)

            if not gt_objs or not pred_objs:
                continue

            mat = iou_matrix(gt_boxes, pred_boxes)
            matches = greedy_match(mat, iou_thresh)

            accum[cls].matched_gt += len(matches)

            for gt_idx, pred_idx in matches:
                accum[cls].gt_distances.append(gt_objs[gt_idx].distance_m)
                accum[cls].pred_distances.append(pred_objs[pred_idx].distance_m)

    # ------------------------------------------------------------------
    # Per-class finalisation
    # ------------------------------------------------------------------

    def _finalise_class(
        self, cls: str, acc: _ClassAccumulator, console: Console | None = None
    ) -> ClassMetrics:
        if acc.has_confidence:
            ap = compute_ap(
                acc.gt_boxes_per_image,
                acc.pred_boxes_per_image,
                acc.pred_scores_per_image,
                iou_threshold=self.config.ap_iou_threshold,
            )
            ap_valid = True
        else:
            ap = float("nan")
            ap_valid = False
            if acc.total_pred > 0 and console:
                console.print(
                    f"[yellow]  AP for '{cls}' set to N/A: no confidence_detection "
                    f"scores found in predictions.[/yellow]"
                )

        recall = acc.matched_gt / acc.total_gt if acc.total_gt > 0 else float("nan")
        return ClassMetrics(
            class_name=cls,
            ap=ap,
            ap_valid=ap_valid,
            mae=mae(acc.gt_distances, acc.pred_distances),
            rmse=rmse(acc.gt_distances, acc.pred_distances),
            rel_err=mean_relative_error(acc.gt_distances, acc.pred_distances),
            total_gt=acc.total_gt,
            matched_gt=acc.matched_gt,
            total_pred=acc.total_pred,
            detection_recall=recall,
        )


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

_NAN_STR = "—"


def _fmt(val: float, decimals: int = 3) -> str:
    import math

    if math.isnan(val):
        return _NAN_STR
    return f"{val:.{decimals}f}"


def print_results_table(result: ScorerResult, console: Console | None = None) -> None:
    """Render a Rich table of scoring results to the terminal."""
    c = console or Console()

    c.print(
        f"\n[bold]Scorer v{result.scorer_version}[/bold]  "
        f"config: [cyan]{result.config_id}[/cyan]  "
        f"dataset: [green]{result.dataset_name}[/green]  "
        f"GT hash: [dim]{result.gt_hash}[/dim]"
    )
    c.print(
        f"Files: {result.matched_files} matched, "
        f"{result.gt_only_files} GT-only, "
        f"{result.pred_only_files} pred-only"
        + (f", [yellow]{result.skipped_files} skipped[/yellow]" if result.skipped_files else "")
        + "\n"
    )

    table = Table(title="Per-class metrics", show_lines=True)
    table.add_column("Class", style="bold")
    table.add_column("AP@0.5", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("MAE (m)", justify="right")
    table.add_column("RMSE (m)", justify="right")
    table.add_column("RelErr", justify="right")
    table.add_column("GT objs", justify="right")
    table.add_column("Matched GT", justify="right")
    table.add_column("Pred objs", justify="right")

    for cls in ALL_CLASSES:
        m = result.per_class[cls]
        ap_str = _fmt(m.ap) if m.ap_valid else "[dim]N/A[/dim]"
        table.add_row(
            cls,
            ap_str,
            _fmt(m.detection_recall),
            _fmt(m.mae),
            _fmt(m.rmse),
            _fmt(m.rel_err),
            str(m.total_gt),
            str(m.matched_gt),
            str(m.total_pred),
        )

    c.print(table)

    d = result.to_dict()
    c.print(
        f"[dim]Overall: {d['total_gt_all']} GT objects, "
        f"{d['matched_gt_all']} matched, "
        f"{d['total_pred_all']} predictions, "
        f"recall {_fmt(d['detection_recall_all'])}[/dim]\n"
    )
