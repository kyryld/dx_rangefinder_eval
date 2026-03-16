"""
Label format specification for the rangefinder evaluation system.

Version history
---------------
1  Initial version — bbox, distance, optional confidence fields.

Design notes
------------
- Ground truth and predictions share the same schema; predictions are a
  superset that may include `confidence_detection` and `confidence_distance`.
- Object IDs are free-form strings; they are not used for matching during
  evaluation (matching is done by IoU).
- All bbox coordinates are in YOLO-style normalized [0, 1] form with explicit
  named subfields instead of a positional list.
- The format is designed for forward compatibility: any unknown extra fields at
  the `FrameLabel` or `ObjectLabel` level are silently preserved so that future
  format versions remain readable by older tooling that ignores new fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

LABEL_FORMAT_VERSION: int = 1

VALID_CLASSES: frozenset[str] = frozenset({"person", "vehicle", "building"})


class BBox(BaseModel):
    """YOLO-style normalized bounding box with named subfields."""

    x_center_rel: float = Field(..., ge=0.0, le=1.0, description="Horizontal center, [0, 1]")
    y_center_rel: float = Field(..., ge=0.0, le=1.0, description="Vertical center, [0, 1]")
    width_rel: float = Field(..., gt=0.0, le=1.0, description="Width, (0, 1]")
    height_rel: float = Field(..., gt=0.0, le=1.0, description="Height, (0, 1]")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return (x_center, y_center, width, height)."""
        return (self.x_center_rel, self.y_center_rel, self.width_rel, self.height_rel)


class ObjectLabel(BaseModel):
    """A single labeled object within a frame.

    Extra fields beyond those declared here are allowed (for forward
    compatibility) and will be preserved on round-trip serialization.
    """

    model_config = {"populate_by_name": True, "extra": "allow"}

    object_id: str = Field(..., description="Unique identifier within this frame")
    class_name: str = Field(
        ...,
        alias="class",
        description="Object class: person | vehicle | building",
    )
    bbox: BBox
    distance_m: float = Field(..., gt=0.0, description="Distance in meters")

    # Prediction-only fields — omit from ground truth files
    confidence_detection: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Detection confidence [0, 1]. Present in predictions only.",
    )
    confidence_distance: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Distance estimate confidence [0, 1]. Present in predictions only.",
    )

    @model_validator(mode="after")
    def _validate_class(self) -> "ObjectLabel":
        if self.class_name not in VALID_CLASSES:
            raise ValueError(
                f"Unknown class '{self.class_name}'. Must be one of {sorted(VALID_CLASSES)}."
            )
        return self

    def model_dump_json_compat(self) -> dict[str, Any]:
        """Serialize to a dict using 'class' as the key (not 'class_name')."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        return data


class FrameLabel(BaseModel):
    """All labeled objects for a single camera frame.

    Extra fields are allowed and preserved for forward compatibility.
    """

    model_config = {"extra": "allow"}

    format_version: int = Field(LABEL_FORMAT_VERSION, description="Label format version")
    frame_id: str = Field(..., description="Unique frame identifier")
    source_image: str = Field(..., description="Filename of the source image")
    objects: list[ObjectLabel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_version(self) -> "FrameLabel":
        if self.format_version != LABEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format_version {self.format_version}. "
                f"This tool supports version {LABEL_FORMAT_VERSION}."
            )
        return self

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Path) -> "FrameLabel":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def to_file(self, path: Path) -> None:
        path.write_text(
            json.dumps(self._to_serializable(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _to_serializable(self) -> dict[str, Any]:
        data = self.model_dump(by_alias=True, exclude_none=True)
        # Replace 'class_name' key with 'class' in each object
        for obj in data.get("objects", []):
            if "class_name" in obj:
                obj["class"] = obj.pop("class_name")
        return data
