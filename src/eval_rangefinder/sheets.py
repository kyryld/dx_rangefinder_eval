"""
Google Sheets (and Google Drive) integration.

Authentication strategy
-----------------------
We use a **service account** — a single set of machine credentials shared
across the team.  This avoids per-user OAuth browser flows and works in
headless / SSH environments.

Setup (one-time, done by the project admin)
-------------------------------------------
1. In Google Cloud Console, create a project and enable the Sheets API and
   the Drive API.
2. Create a service account; download its JSON key file.
3. Share the target spreadsheet with the service account's e-mail address
   (give it Editor access).
4. (Optional) Create a shared Drive folder, share it with the service account
   (Editor), and add its ID as ``drive_folder_id`` in creds.json.  This
   enables uploading prediction zips from the TUI.
5. Distribute ``creds.json`` (see ``creds.json.example``) to the team.  Each
   member fills in their own ``author`` name; the service account block is
   identical for everyone.

``creds.json`` schema
---------------------
{
  "author": "Jane Doe",
  "spreadsheet_id": "<sheet id from the URL>",
  "drive_folder_id": "<Drive folder id — optional, enables zip upload>",
  "service_account": { ...standard GCP service account JSON key fields... }
}

The file must NOT be committed to git (it is listed in .gitignore).
"""

from __future__ import annotations

import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPREADSHEET_COLUMNS: list[str] = [
    "Timestamp",
    "Run name",
    "Author",
    "Dataset name",
    "GT hash",
    "Scorer version",
    "Scorer config",
    "Intended task",
    # --- Detection ---
    "AP_person",
    "AP_vehicle",
    "AP_building",
    # --- Ranging ---
    "MAE_person",
    "MAE_vehicle",
    "MAE_building",
    "RMSE_person",
    "RMSE_vehicle",
    "RMSE_building",
    "RelErr_person",
    "RelErr_vehicle",
    "RelErr_building",
    # --- Dataset coverage ---
    "Total GT objects",
    "Matched GT objects",
    "Total predictions",
    "Detection recall",
    # --- Predictions archive ---
    "Predictions (Drive)",
    # --- Free text ---
    "Notes",
]

DEFAULT_CREDS_PATH = Path.home() / ".eval_rangefinder" / "creds.json"
FALLBACK_CREDS_PATH = Path("creds.json")


def load_creds(path: Path | None = None) -> dict[str, Any]:
    """Load and return the credentials dict.

    Search order (first found wins):
      1. Explicitly supplied ``path``
      2. ``~/.eval_rangefinder/creds.json``
      3. ``./creds.json`` in the current working directory
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(DEFAULT_CREDS_PATH)
    candidates.append(FALLBACK_CREDS_PATH)

    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No credentials file found. Searched: {searched}\n"
        "Copy creds.json.example to one of the above paths and fill in your details."
    )


# ---------------------------------------------------------------------------
# Google Drive upload
# ---------------------------------------------------------------------------


def upload_predictions_zip(
    pred_dir: Path,
    run_name: str,
    drive_folder_id: str,
    service_account_info: dict[str, Any],
) -> str:
    """Zip the predictions folder and upload it to a shared Drive folder.

    Returns the ``webViewLink`` of the uploaded file so it can be stored in
    the spreadsheet.  The file is readable by anyone who has access to the
    parent folder (folder-level sharing is expected to be set up by the admin).

    Parameters
    ----------
    pred_dir:
        Local folder containing prediction JSON files.
    run_name:
        Used to name the uploaded zip: ``{run_name}_predictions.zip``.
    drive_folder_id:
        ID of the target Drive folder (from the folder's URL).
    service_account_info:
        Service account JSON key dict (same one used for Sheets auth).
    """
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise ImportError(
            "google-api-python-client is required for Drive upload. "
            "Run: uv add google-api-python-client"
        ) from exc

    scopes = [
        "https://www.googleapis.com/auth/drive.file",
    ]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    # Create zip in a temp file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for json_file in sorted(pred_dir.glob("*.json")):
                zf.write(json_file, json_file.name)

        file_metadata: dict[str, Any] = {
            "name": f"{run_name}_predictions.zip",
            "parents": [drive_folder_id],
        }
        media = MediaFileUpload(str(tmp_path), mimetype="application/zip", resumable=False)
        uploaded = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id,webViewLink")
            .execute()
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return uploaded.get("webViewLink", "")


# ---------------------------------------------------------------------------
# Spreadsheet row
# ---------------------------------------------------------------------------


def _make_row(
    run_name: str,
    author: str,
    dataset_name: str,
    gt_hash: str,
    scorer_version: str,
    scorer_config: str,
    intended_task: str,
    result_dict: dict[str, Any],
    predictions_link: str,
    notes: str,
) -> list[Any]:
    """Build a flat row matching SPREADSHEET_COLUMNS order."""

    def _v(val: Any) -> Any:
        if isinstance(val, float) and math.isnan(val):
            return ""
        if isinstance(val, float):
            return round(val, 5)
        return val

    d = result_dict
    return [
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        run_name,
        author,
        dataset_name,
        gt_hash,
        scorer_version,
        scorer_config,
        intended_task,
        # Detection
        _v(d.get("AP_person")),
        _v(d.get("AP_vehicle")),
        _v(d.get("AP_building")),
        # Ranging
        _v(d.get("MAE_person")),
        _v(d.get("MAE_vehicle")),
        _v(d.get("MAE_building")),
        _v(d.get("RMSE_person")),
        _v(d.get("RMSE_vehicle")),
        _v(d.get("RMSE_building")),
        _v(d.get("RelErr_person")),
        _v(d.get("RelErr_vehicle")),
        _v(d.get("RelErr_building")),
        # Coverage
        _v(d.get("total_gt_all")),
        _v(d.get("matched_gt_all")),
        _v(d.get("total_pred_all")),
        _v(d.get("detection_recall_all")),
        # Predictions archive
        predictions_link,
        # Free text
        notes,
    ]


def append_row(
    run_name: str,
    author: str,
    dataset_name: str,
    gt_hash: str,
    scorer_version: str,
    scorer_config: str,
    intended_task: str,
    result_dict: dict[str, Any],
    predictions_link: str,
    notes: str,
    spreadsheet_id: str,
    service_account_info: dict[str, Any],
    worksheet_name: str = "Results",
) -> str:
    """Append one result row to the shared Google Spreadsheet.

    Returns the URL of the spreadsheet for display.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise ImportError(
            "gspread and google-auth are required for spreadsheet publishing. "
            "Run: uv add gspread google-auth"
        ) from exc

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    client = gspread.authorize(creds)

    spreadsheet = client.open_by_key(spreadsheet_id)

    try:
        ws = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(
            title=worksheet_name, rows=1000, cols=len(SPREADSHEET_COLUMNS)
        )

    # Ensure header row exists (idempotent)
    first_row = ws.row_values(1)
    if not first_row or first_row[0] != "Timestamp":
        ws.insert_row(SPREADSHEET_COLUMNS, index=1, value_input_option="USER_ENTERED")

    row = _make_row(
        run_name=run_name,
        author=author,
        dataset_name=dataset_name,
        gt_hash=gt_hash,
        scorer_version=scorer_version,
        scorer_config=scorer_config,
        intended_task=intended_task,
        result_dict=result_dict,
        predictions_link=predictions_link,
        notes=notes,
    )
    ws.append_row(row, value_input_option="USER_ENTERED")

    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
