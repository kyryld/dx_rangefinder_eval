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
    "GT files",
    "Prediction files",
    "Matched files",
    "Total GT objects",
    "Matched GT objects",
    "Total predictions",
    "Detection recall",
    # --- Free text ---
    "Notes",
]

DEFAULT_CREDS_PATH = Path.home() / ".eval_rangefinder" / "creds.json"


def _find_project_root() -> Path | None:
    """Walk up from cwd looking for pyproject.toml as the project root marker."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def load_creds(path: Path | None = None) -> dict[str, Any]:
    """Load and return the credentials dict.

    Search order (first found wins):
      1. Explicitly supplied ``path``
      2. ``~/.eval_rangefinder/creds.json``
      3. ``<project_root>/creds.json``  (project root found via pyproject.toml)
      4. ``./creds.json``               (current working directory)

    Service account resolution
    --------------------------
    Two formats are supported for the service account credentials:

    **Embedded** (inline JSON object):

        { "service_account": { "type": "service_account", ... } }

    **By file path** (avoids copy-paste issues with the private key):

        { "service_account_file": "/path/to/downloaded-key.json" }

    If both keys are present, ``service_account_file`` takes precedence.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(DEFAULT_CREDS_PATH)
    project_root = _find_project_root()
    if project_root:
        candidates.append(project_root / "creds.json")
    candidates.append(Path("creds.json"))

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        resolved = c.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(c)

    for candidate in unique:
        if candidate.exists():
            creds = json.loads(candidate.read_text(encoding="utf-8"))
            # Resolve service_account_file → service_account
            sa_file = creds.pop("service_account_file", None)
            if sa_file:
                sa_path = Path(sa_file).expanduser().resolve()
                if not sa_path.exists():
                    raise FileNotFoundError(
                        f"service_account_file not found: {sa_path}\n"
                        "Check the path in your creds.json."
                    )
                creds["service_account"] = json.loads(sa_path.read_text(encoding="utf-8"))
            return creds

    searched = "\n  ".join(str(c) for c in unique)
    raise FileNotFoundError(
        f"No credentials file found. Searched:\n  {searched}\n"
        "Copy creds.json.example to one of the above paths and fill in your details."
    )


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
        _v(d.get("gt_files")),
        _v(d.get("pred_files")),
        _v(d.get("matched_files")),
        _v(d.get("total_gt_all")),
        _v(d.get("matched_gt_all")),
        _v(d.get("total_pred_all")),
        _v(d.get("detection_recall_all")),
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

    # Only the spreadsheets scope is needed; drive.file is not required and
    # will fail if the Drive API is not enabled in the GCP project.
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
    except Exception as exc:
        raise PermissionError(
            f"Could not open spreadsheet '{spreadsheet_id}'.\n"
            f"Make sure the spreadsheet is shared with the service account: "
            f"{service_account_info.get('client_email', '(unknown)')}\n"
            f"Original error: {exc}"
        ) from exc

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
        notes=notes,
    )
    ws.append_row(row, value_input_option="USER_ENTERED")

    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
