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
Timestamp | Run name | Author | Dataset name | GT hash | Scorer version |
Scorer config | Intended task |
AP_person | AP_vehicle | AP_building |
MAE_person | MAE_vehicle | MAE_building |
RMSE_person | RMSE_vehicle | RMSE_building |
RelErr_person | RelErr_vehicle | RelErr_building |
Total GT objects | Matched GT objects | Total predictions |
Detection recall | Predictions (Drive) | Notes
```

---

## Google Sheets + Drive setup (admin, one-time)

### What is a service account?

A service account is a special Google account that represents the application
rather than a person. Instead of logging in with a username and password, the
app authenticates using a JSON key file. That file is what you download and
distribute to the team.

### Step 1 — Create a Google Cloud project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project selector at the top → **New Project**
3. Give it a name (e.g. `rangefinder-eval`) → **Create**
4. Make sure your new project is selected in the top bar

### Step 2 — Enable the APIs

In the left sidebar: **APIs & Services → Library**

Search for and enable both:
- **Google Sheets API** → Enable
- **Google Drive API** → Enable

### Step 3 — Create the service account

1. Left sidebar: **APIs & Services → Credentials**
2. Click **+ Create Credentials → Service account**
3. Name it (e.g. `eval-rangefinder`) → **Create and Continue**
4. Skip the optional role/permissions steps → **Done**

### Step 4 — Download the JSON key file

1. On the Credentials page, click the service account you just created (its
   email looks like `eval-rangefinder@your-project.iam.gserviceaccount.com`)
2. Go to the **Keys** tab
3. Click **Add Key → Create new key**
4. Choose **JSON** → **Create**
5. A `.json` file downloads automatically — this is your key file

### Step 5 — Share the spreadsheet with the service account

The service account has its own email address (`client_email` in the JSON).
Share your Google Spreadsheet with it exactly as you would share with a
colleague:

1. Open your Google Spreadsheet
2. Click **Share** (top right)
3. Paste the `client_email` value
4. Give it **Editor** access → **Send**

Get the spreadsheet ID from the URL:
`https://docs.google.com/spreadsheets/d/`**`THIS_PART`**`/edit`

### Step 6 — (Optional) Create a shared Drive folder for prediction zips

If you want the TUI to upload prediction archives to Drive:

1. Go to [drive.google.com](https://drive.google.com)
2. Create a new folder (e.g. `rangefinder-eval-predictions`)
3. Share it with the service account email — give it **Editor** access
4. Also share it with your teammates' personal Google accounts (view access)
   so they can browse uploaded zips
5. Get the folder ID from the URL when the folder is open:
   `https://drive.google.com/drive/folders/`**`THIS_PART_IS_THE_ID`**

Each uploaded zip is named `{run_name}_predictions.zip` and lands in this
shared folder. The "Predictions (Drive)" spreadsheet column stores the direct
link. The upload prompt only appears when `drive_folder_id` is set in `creds.json`.

### Step 7 — Build and distribute `creds.json`

There are two ways to supply the service account credentials.

**Option A — file path (recommended, avoids copy-paste issues):**

```json
{
  "author": "Your Name",
  "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
  "drive_folder_id": "1AbCdEfGhIjKlMnOpQrStUvWxYz",
  "service_account_file": "/path/to/your-downloaded-key.json"
}
```

Just point `service_account_file` at the JSON file Google gave you.
No copy-pasting of the private key required.

**Option B — embedded inline:**

```json
{
  "author": "Your Name",
  "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
  "drive_folder_id": "1AbCdEfGhIjKlMnOpQrStUvWxYz",
  "service_account": {
    "type": "service_account",
    "project_id": "rangefinder-eval",
    "private_key_id": "abc123",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
    "client_email": "eval-rangefinder@rangefinder-eval.iam.gserviceaccount.com",
    ...
  }
}
```

Paste the **entire contents** of the downloaded JSON file as the value of
`"service_account"`. The private key must keep its `\n` escape sequences
intact — do not replace them with actual line breaks.

Each team member copies `creds.json` to `~/.eval_rangefinder/creds.json` and
replaces `"author"` with their own name. Everything else is identical for everyone.

The service account key is a **shared secret** — treat it like a password.
`creds.json` is in `.gitignore` and must never be committed.

The TUI creates the `Results` worksheet with the correct header row
automatically on the first publish.

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
├── prompts/
│   └── vibe-coding-session.md   ← full AI-assisted design session transcript
└── src/
    └── eval_rangefinder/
        ├── schema.py   ← Pydantic label models
        ├── metrics.py  ← IoU, AP, MAE, RMSE, RelErr
        ├── scorer.py   ← ScorerConfig, Scorer, print_results_table
        ├── sheets.py   ← Google Sheets + Drive integration
        └── tui.py      ← interactive TUI entry point
```

---

## Adding a new object class

1. Add the class name to `VALID_CLASSES` in `schema.py`.
2. Add the class to `ALL_CLASSES` in `scorer.py`.
3. Bump `format_version` in `schema.py` if the change is breaking.
4. Add the new `AP_<class>`, `MAE_<class>`, … columns to `SPREADSHEET_COLUMNS` in `sheets.py`.
