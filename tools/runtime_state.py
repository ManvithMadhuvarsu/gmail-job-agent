"""Runtime state file used by the /health endpoint and dashboards.

Each daemon cycle calls record_cycle() so /health can answer
"is MailAI alive, when did it last run, what was the last error?" without
having to parse the audit log on every request.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/runtime_state.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        import json
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write(state: dict) -> None:
    try:
        atomic_write_json(STATE_PATH, state)
    except Exception as exc:
        logger.warning(f"runtime_state write failed: {exc}")


def record_cycle(
    *,
    processed: int,
    drafts: int,
    calendar_events: int,
    errors: int,
    dry_run: bool = False,
    error: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    state = _read()
    state["last_run_at"] = _now()
    state["last_processed"] = int(processed)
    state["last_drafts"] = int(drafts)
    state["last_calendar_events"] = int(calendar_events)
    state["last_errors"] = int(errors)
    state["last_duration_seconds"] = round(duration_seconds, 3) if duration_seconds is not None else None
    state["dry_run"] = bool(dry_run)
    state["pid"] = os.getpid()
    state["total_runs"] = int(state.get("total_runs", 0)) + 1
    state["total_processed"] = int(state.get("total_processed", 0)) + int(processed)
    state["total_drafts"] = int(state.get("total_drafts", 0)) + int(drafts)
    state["total_calendar_events"] = int(state.get("total_calendar_events", 0)) + int(calendar_events)

    if errors or error:
        state["consecutive_errors"] = int(state.get("consecutive_errors", 0)) + 1
        state["last_error"] = error or "see logs"
        state["last_error_at"] = _now()
    else:
        state["consecutive_errors"] = 0
        state.setdefault("last_error", None)

    _write(state)


def record_heartbeat() -> None:
    """Mark the daemon as alive without changing run stats."""
    state = _read()
    state["last_heartbeat_at"] = _now()
    state["pid"] = os.getpid()
    _write(state)


def snapshot() -> dict[str, Any]:
    return _read()
