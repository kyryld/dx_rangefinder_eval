# Label Format Specification

**Version:** 1  
**Filename convention:** `<frame_id>.json`  
**Encoding:** UTF-8

---

## Overview

One JSON file per camera frame. The same schema is used for **ground truth (GT)**
and **predictions**. Predictions are a strict superset of GT: they may include
optional confidence fields that GT files omit.

---

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `format_version` | integer | yes | Schema version. Currently `1`. |
| `frame_id` | string | yes | Globally unique frame identifier (e.g. `2026_03_04_000042`). |
| `source_image` | string | yes | Filename (not path) of the source image file. |
| `objects` | array | yes | List of labeled objects. May be empty `[]`. |

Unknown top-level fields are preserved on round-trip (forward compatibility).

---

## Object fields

Each element of `objects` has the following fields:

| Field | Type | Required in GT | Required in pred | Description |
|---|---|---|---|---|
| `object_id` | string | yes | yes | Unique within this frame. No global uniqueness guarantee. |
| `class` | string | yes | yes | One of `person`, `vehicle`, `building`. |
| `bbox` | object | yes | yes | See *BBox fields* below. |
| `distance_m` | number | yes | yes | Estimated or ground-truth distance in metres. Must be > 0. |
| `confidence_detection` | number 0–1 | no | recommended | Detection confidence score. Used for AP calculation. If absent in predictions, all detections are treated as having score 1.0 (affects AP ranking). |
| `confidence_distance` | number 0–1 | no | optional | Confidence in the distance estimate. Not used by the scorer currently; reserved for future metrics. |

Unknown object fields are preserved on round-trip.

### BBox fields

YOLO-style normalized coordinates. All values are in **[0, 1]** relative to the
image dimensions.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `x_center_rel` | number | [0, 1] | Horizontal centre of the box. |
| `y_center_rel` | number | [0, 1] | Vertical centre of the box. |
| `width_rel` | number | (0, 1] | Box width. |
| `height_rel` | number | (0, 1] | Box height. |

---

## Example — ground truth file

```json
{
  "format_version": 1,
  "frame_id": "2026_03_04_000042",
  "source_image": "2026_03_04_000042.png",
  "objects": [
    {
      "object_id": "gt_0",
      "class": "person",
      "bbox": {
        "x_center_rel": 0.512,
        "y_center_rel": 0.438,
        "width_rel": 0.045,
        "height_rel": 0.120
      },
      "distance_m": 34.2
    },
    {
      "object_id": "gt_1",
      "class": "vehicle",
      "bbox": {
        "x_center_rel": 0.310,
        "y_center_rel": 0.550,
        "width_rel": 0.180,
        "height_rel": 0.130
      },
      "distance_m": 87.5
    }
  ]
}
```

## Example — predictions file (same filename as GT)

```json
{
  "format_version": 1,
  "frame_id": "2026_03_04_000042",
  "source_image": "2026_03_04_000042.png",
  "objects": [
    {
      "object_id": "pred_0",
      "class": "person",
      "bbox": {
        "x_center_rel": 0.514,
        "y_center_rel": 0.440,
        "width_rel": 0.047,
        "height_rel": 0.118
      },
      "distance_m": 36.1,
      "confidence_detection": 0.92,
      "confidence_distance": 0.75
    },
    {
      "object_id": "pred_1",
      "class": "vehicle",
      "bbox": {
        "x_center_rel": 0.312,
        "y_center_rel": 0.548,
        "width_rel": 0.178,
        "height_rel": 0.132
      },
      "distance_m": 91.0,
      "confidence_detection": 0.88
    }
  ]
}
```

---

## File naming

Files in the GT folder and the predictions folder are **matched by filename**.
The frame_id inside the file is informational and is not used for matching.
Recommended convention: `<frame_id>.json`.

---

## Versioning and extension

- **Adding new optional fields** to `ObjectLabel` or `FrameLabel` does not
  require a version bump, provided old tooling can ignore them.
- **Removing or renaming existing fields**, or changing their semantics,
  **requires incrementing `format_version`** and updating the scorer to handle
  both old and new versions.
- The current scorer validates that `format_version == 1` and raises an error
  for any other value, so consumers of future versions can opt in explicitly.
