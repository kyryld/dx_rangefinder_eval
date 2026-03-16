"""
TUI entry point for the rangefinder evaluation workflow.

Usage
-----
    eval-rangefinder          # installed entry point
    uv run eval-rangefinder   # from the project root

Flow
----
1.  Ask for the predictions folder
2.  Ask for the ground-truth folder — dataset name inferred from folder name
3.  Sanity-check file counts (warn, don't fail)
4.  Run scorer → print results table
5.  Ask whether to save results to a JSON file
6.  Ask whether to publish to the shared Google Spreadsheet
7.  If publishing:
      a. Load credentials (fail fast before asking more questions)
      b. Suggest an auto-generated run name; let user edit it
      c. Ask for intended task (detection / ranging / both)
      d. Ask for a notes field (common component tags shown as a hint)
      e. Append row to spreadsheet; print URL
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel

from .scorer import CONFIGS, SCORER_VERSION, Scorer, print_results_table
from .sheets import DEFAULT_CREDS_PATH, append_row, load_creds

console = Console()

TAGS_HINT = (
    "demosaicing  color_correction  super_resolution  "
    "detection_model  segmentation_model  pose_model  ranging_method"
)

QUESTIONARY_STYLE = questionary.Style(
    [
        ("qmark", "fg:#5f87ff bold"),
        ("question", "bold"),
        ("answer", "fg:#00af87 bold"),
        ("pointer", "fg:#5f87ff bold"),
        ("highlighted", "fg:#5f87ff bold"),
        ("selected", "fg:#00af87"),
        ("instruction", "fg:#888888"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ask(prompt: str, **kwargs: object) -> str:
    try:
        result = questionary.text(prompt, style=QUESTIONARY_STYLE, **kwargs).ask()
    except KeyboardInterrupt:
        result = None
    if result is None:
        console.print("\n[yellow]Aborted.[/yellow]")
        sys.exit(0)
    return result


def _confirm(prompt: str, default: bool = True) -> bool:
    try:
        result = questionary.confirm(prompt, default=default, style=QUESTIONARY_STYLE).ask()
    except KeyboardInterrupt:
        result = None
    if result is None:
        console.print("\n[yellow]Aborted.[/yellow]")
        sys.exit(0)
    return result


def _select(prompt: str, choices: list[str]) -> str:
    try:
        result = questionary.select(prompt, choices=choices, style=QUESTIONARY_STYLE).ask()
    except KeyboardInterrupt:
        result = None
    if result is None:
        console.print("\n[yellow]Aborted.[/yellow]")
        sys.exit(0)
    return result


def _validate_dir(text: str) -> bool | str:
    p = Path(text.strip())
    if not p.exists():
        return f"Path does not exist: {p}"
    if not p.is_dir():
        return f"Not a directory: {p}"
    return True


def _count_json(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.json"))


def _auto_run_name(author: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_author = author.strip().lower().replace(" ", "_") or "unknown"
    return f"{safe_author}_{ts}"


def _save_results_json(result_dict: dict, dataset_name: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{dataset_name}_{ts}_results.json"
    raw = _ask("Save path:", default=default_name).strip() or default_name
    out_path = Path(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved → {out_path.resolve()}[/green]")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]eval-rangefinder[/bold cyan]  "
            f"[dim]scorer v{SCORER_VERSION}[/dim]",
            subtitle="[dim]Ctrl-C to abort at any time[/dim]",
        )
    )

    # ------------------------------------------------------------------ #
    # Step 1 & 2 — folder selection                                        #
    # ------------------------------------------------------------------ #

    pred_dir = Path(
        _ask("Path to predictions folder:", validate=_validate_dir).strip()
    )
    gt_dir = Path(
        _ask("Path to ground-truth folder:", validate=_validate_dir).strip()
    )
    dataset_name = gt_dir.name

    # ------------------------------------------------------------------ #
    # Step 3 — sanity check                                               #
    # ------------------------------------------------------------------ #

    gt_count = _count_json(gt_dir)
    pred_count = _count_json(pred_dir)

    if gt_count == 0:
        console.print("[red]No JSON files found in the GT folder. Aborting.[/red]")
        sys.exit(1)
    if pred_count == 0:
        console.print("[red]No JSON files found in the predictions folder. Aborting.[/red]")
        sys.exit(1)
    if gt_count != pred_count:
        console.print(
            f"[yellow]  File count mismatch: {gt_count} GT files vs "
            f"{pred_count} prediction files.[/yellow]"
        )

    # ------------------------------------------------------------------ #
    # Scorer config selection                                             #
    # ------------------------------------------------------------------ #

    config_names = list(CONFIGS.keys())
    config_id = _select("Scorer config:", config_names) if len(config_names) > 1 else config_names[0]
    scorer = Scorer(config=CONFIGS[config_id])

    # ------------------------------------------------------------------ #
    # Step 4 — run scorer                                                 #
    # ------------------------------------------------------------------ #

    console.print("\n[dim]Running scorer…[/dim]")
    try:
        result = scorer.score(gt_dir, pred_dir, dataset_name=dataset_name, console=console)
    except Exception as exc:
        console.print(f"[red]Scorer error: {exc}[/red]")
        sys.exit(1)

    print_results_table(result, console=console)
    result_dict = result.to_dict()

    # ------------------------------------------------------------------ #
    # Step 5 — save JSON?                                                 #
    # ------------------------------------------------------------------ #

    if _confirm("Save results to a JSON file?", default=False):
        _save_results_json(result_dict, dataset_name)

    # ------------------------------------------------------------------ #
    # Step 6 — publish?                                                   #
    # ------------------------------------------------------------------ #

    if not _confirm("Publish results to the shared Google Spreadsheet?"):
        console.print("[dim]Results not published.  Bye![/dim]")
        return

    # ------------------------------------------------------------------ #
    # Load credentials early — fail before asking more questions          #
    # ------------------------------------------------------------------ #

    try:
        creds = load_creds()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            f"[dim]Copy [bold]creds.json.example[/bold] to "
            f"[bold]{DEFAULT_CREDS_PATH}[/bold] (or the project root) "
            "and fill in your details.[/dim]"
        )
        sys.exit(1)

    author: str = creds.get("author", "").strip() or "unknown"
    spreadsheet_id: str = creds.get("spreadsheet_id", "").strip()
    service_account_info: dict = creds.get("service_account", {})

    if not spreadsheet_id:
        console.print("[red]'spreadsheet_id' is missing from creds.json.[/red]")
        sys.exit(1)
    if not service_account_info:
        console.print("[red]'service_account' block is missing from creds.json.[/red]")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 7a — run name                                                  #
    # ------------------------------------------------------------------ #

    suggested = _auto_run_name(author)
    run_name = (
        _ask("Run name (edit or press Enter to accept):", default=suggested).strip() or suggested
    )

    # ------------------------------------------------------------------ #
    # Step 7b — intended task                                             #
    # ------------------------------------------------------------------ #

    intended_task = _select(
        "What does this run evaluate?",
        ["detection", "ranging", "both"],
    )

    # ------------------------------------------------------------------ #
    # Step 7c — notes (with component-tag hint)                          #
    # ------------------------------------------------------------------ #

    console.print(f"[dim]Common component tags: {TAGS_HINT}[/dim]")
    notes = _ask(
        "Notes (tags, description, anything useful — optional):",
        default="",
    ).strip()

    # ------------------------------------------------------------------ #
    # Step 7d — append row                                                #
    # ------------------------------------------------------------------ #

    console.print("\n[dim]Publishing to Google Sheets…[/dim]")
    try:
        url = append_row(
            run_name=run_name,
            author=author,
            dataset_name=dataset_name,
            gt_hash=result.gt_hash,
            scorer_version=SCORER_VERSION,
            scorer_config=config_id,
            intended_task=intended_task,
            result_dict=result_dict,
            notes=notes,
            spreadsheet_id=spreadsheet_id,
            service_account_info=service_account_info,
        )
    except Exception as exc:
        console.print(f"[red]Failed to publish: {exc}[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Published.[/green]  Spreadsheet: [link={url}]{url}[/link]")


if __name__ == "__main__":
    main()
