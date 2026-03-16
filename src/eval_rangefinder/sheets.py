"""
Google Sheets integration.

Authentication strategy
-----------------------
We use a **service account** — a single set of machine credentials shared
across the team.  This avoids per-user OAuth browser flows and works in
headless / SSH environments.

Setup (one-time, done by the project admin)
-------------------------------------------
1. In Google Cloud Console, create a project and enable the Sheets API.
2. Create a service account; download its JSON key file.
3. Share the target spreadsheet with the service account's e-mail address
   (give it Editor access).
4. Distribute `creds.json` (see `creds.json.example`) to the team.  Each
   member fills in their own `author` name; the service account block is
   identical for everyone.

`creds.json` schema
-------------------
{
  "author": "Jane Doe",
  "spreadsheet_id": "<sheet id from the URL>",
  "service_account": { ...standard GCP service account JSON key fields... }
}

The file must NOT be committed to git (it is listed in .gitignore).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPREADSHEET_COLUMNS: list[str] = [
    "Timestamp",
    "Run name",
    "Author",
    "Dataset",
    "Predictions path",
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
    "Matched predictions",
    "Detection recall",
    # --- Free text ---
    "Tags",
    "Notes",
]

DEFAULT_CREDS_PATH = Path.home() / ".eval_rangefinder" / "creds.json"
FALLBACK_CREDS_PATH = Path("creds.json")


def load_creds(path: Path | None = None) -> dict[str, Any]:
    """Load and return the credentials dict.

    Search order (first found wins):
      1. Explicitly supplied `path`
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


def _make_row(
    run_name: str,
    author: str,
    dataset: str,
    pred_dir: str,
    scorer_version: str,
    scorer_config: str,
    intended_task: str,
    result_dict: dict[str, Any],
    tags: str,
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
        dataset,
        pred_dir,
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
        _v(d.get("matched_pred_all")),
        _v(d.get("detection_recall_all")),
        # Free text
        tags,
        notes,
    ]


def append_row(
    run_name: str,
    author: str,
    dataset: str,
    pred_dir: str,
    scorer_version: str,
    scorer_config: str,
    intended_task: str,
    result_dict: dict[str, Any],
    tags: str,
    notes: str,
    spreadsheet_id: str,
    service_account_info: dict[str, Any],
    worksheet_name: str = "Results",
) -> str:
    """Append one result row to the shared Google Spreadsheet.

    Returns the URL of the spreadsheet for display.

    Raises
    ------
    ImportError
        If `gspread` or `google-auth` are not installed.
    gspread.exceptions.SpreadsheetNotFound
        If the spreadsheet ID is wrong or the service account has no access.
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

    # Get or create the worksheet
    try:
        ws = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=len(SPREADSHEET_COLUMNS))
        ws.append_row(SPREADSHEET_COLUMNS, value_input_option="USER_ENTERED")

    row = _make_row(
        run_name=run_name,
        author=author,
        dataset=dataset,
        pred_dir=pred_dir,
        scorer_version=scorer_version,
        scorer_config=scorer_config,
        intended_task=intended_task,
        result_dict=result_dict,
        tags=tags,
        notes=notes,
    )
    ws.append_row(row, value_input_option="USER_ENTERED")

    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"


def ensure_header_row(
    spreadsheet_id: str,
    service_account_info: dict[str, Any],
    worksheet_name: str = "Results",
) -> None:
    """Ensure the header row exists in the worksheet (idempotent).

    Safe to call multiple times — only writes the header if row 1 is empty.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        return

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
        ws = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=len(SPREADSHEET_COLUMNS))

    first_row = ws.row_values(1)
    if not first_row or first_row[0] != "Timestamp":
        ws.insert_row(SPREADSHEET_COLUMNS, index=1, value_input_option="USER_ENTERED")
