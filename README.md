# eval-rangefinder

Quality evaluation system for the DeepX rangefinder CV pipeline.

Measures the impact of individual component changes (demosaicing, color correction,
super resolution, detection model, segmentation, pose estimation, ranging method)
on final detection and ranging quality against a shared dataset. Results are
published to a shared Google Spreadsheet so the whole team can compare runs.

---

## Components

| Component | Description |
|---|---|
| **Label format** | Versioned JSON schema for GT and predictions — see [`docs/label_format.md`](docs/label_format.md) |
| **Scorer** (`eval_rangefinder.scorer`) | Loads GT + predictions, matches objects by IoU, computes per-class AP / MAE / RMSE / RelErr |
| **TUI** (`eval-rangefinder`) | Interactive terminal app — runs the scorer, optionally publishes to Google Sheets |

---

## Quick start

### 1. Install

```bash
# from the project root
uv sync
```

### 2. Set up credentials

```bash
mkdir -p ~/.eval_rangefinder
cp creds.json.example ~/.eval_rangefinder/creds.json
# Edit creds.json — fill in author name and get the service_account block from
# your project admin.  See "Google Sheets setup" below.
```

### 3. Run

```bash
uv run eval-rangefinder
```

You will be prompted for the predictions folder and the GT folder, then the
scorer runs and prints results. Optionally publish to the shared spreadsheet.

---

## Scorer only (no TUI)

```python
from pathlib import Path
from eval_rangefinder.scorer import Scorer, CONFIGS, print_results_table

scorer = Scorer(config=CONFIGS["default_v1"])
result = scorer.score(Path("gt/"), Path("predictions/"))
print_results_table(result)
print(result.to_dict())  # structured dict for programmatic use
```

---

## Label format

See [`docs/label_format.md`](docs/label_format.md) for the full spec.

Short version: one JSON file per frame, same filename in GT and predictions folders.

```
gt/
  frame_001.json
  frame_002.json
predictions/
  frame_001.json   ← matched by filename
  frame_002.json
```

---

## Scorer configs

| Config ID | IoU threshold | Notes |
|---|---|---|
| `default_v1` | 0.5 for all classes | Default |
| `strict_v1` | 0.6 for all classes | Stricter matching |

Add new configs in `scorer.py` → `CONFIGS` dict.

---

## Metrics

| Metric | Description |
|---|---|
| **AP@0.5** | Average Precision at IoU 0.5 (per class) |
| **Recall** | Matched GT objects / total GT objects |
| **MAE** | Mean Absolute Error on distance (matched objects only) |
| **RMSE** | Root Mean Squared Error on distance (matched objects only) |
| **RelErr** | Mean relative error = mean(\|pred−gt\| / gt) (matched objects only) |

---

## Spreadsheet columns

```
Timestamp | Run name | Author | Dataset | Predictions path | Scorer version |
Scorer config | Intended task |
AP_person | AP_vehicle | AP_building |
MAE_person | MAE_vehicle | MAE_building |
RMSE_person | RMSE_vehicle | RMSE_building |
RelErr_person | RelErr_vehicle | RelErr_building |
Total GT objects | Matched GT objects | Total predictions | Matched predictions |
Detection recall | Tags | Notes
```

---

## Google Sheets setup (admin, one-time)

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → create a project.
2. Enable **Google Sheets API** and **Google Drive API**.
3. Create a **Service Account** → download the JSON key file.
4. Share the target spreadsheet with the service account's e-mail (give it **Editor** access).
5. Get the `spreadsheet_id` from the spreadsheet URL:
   `https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>/edit`
6. Fill in `creds.json.example` with the service account JSON and the spreadsheet ID.
7. Distribute `creds.json` to the team. Each member replaces `"author"` with their name.

The service account key is a **shared secret** — treat it like a password.
Keep `creds.json` out of git (it is in `.gitignore`).

The TUI will create a `Results` worksheet with the correct header row automatically
on the first publish.

---

## Project structure

```
eval_rangefinder/
├── pyproject.toml
├── creds.json.example
├── docs/
│   └── label_format.md
├── example_data/
│   ├── gt/             ← example GT JSON files
│   └── predictions/    ← example prediction JSON files
└── src/
    └── eval_rangefinder/
        ├── schema.py   ← Pydantic label models
        ├── metrics.py  ← IoU, AP, MAE, RMSE, RelErr
        ├── scorer.py   ← ScorerConfig, Scorer, print_results_table
        ├── sheets.py   ← Google Sheets append_row
        └── tui.py      ← interactive TUI entry point
```

---

## Adding a new object class

1. Add the class name to `VALID_CLASSES` in `schema.py`.
2. Add the class to `ALL_CLASSES` in `scorer.py`.
3. Bump `format_version` in `schema.py` if the change is breaking.
4. Add the new `AP_<class>`, `MAE_<class>`, … columns to `SPREADSHEET_COLUMNS` in `sheets.py`.
