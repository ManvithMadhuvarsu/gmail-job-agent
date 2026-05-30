"""Append-only audit log of every action MailAI took on a Gmail inbox.

Buyer trust depends on being able to answer "what did the AI do?" — this log is
the source of truth for the dashboard, undo, and health endpoints.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

from tools.atomic_io import append_jsonl, atomic_write_text


AUDIT_PATH = Path("data/audit.jsonl")
MAX_AUDIT_BYTES = 25 * 1024 * 1024  # 25 MB rolling cap

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_action(
    *,
    email: dict,
    category: str,
    action: str,
    label_id: str | None = None,
    label_name: str | None = None,
    draft_id: str | None = None,
    calendar_event_id: str | None = None,
    dry_run: bool = False,
    error: str | None = None,
    rule_id: str | None = None,
    notes: str | None = None,
) -> None:
    """Append one row describing what MailAI did (or would have done) for an email."""
    entry = {
        "ts": _now_iso(),
        "email_id": email.get("id", ""),
        "thread_id": email.get("thread_id", ""),
        "subject": (email.get("subject") or "")[:200],
        "sender": (email.get("sender") or "")[:200],
        "sender_email": email.get("sender_email", ""),
        "category": category,
        "action": action,
        "label_id": label_id,
        "label_name": label_name,
        "draft_id": draft_id,
        "calendar_event_id": calendar_event_id,
        "dry_run": bool(dry_run),
        "rule_id": rule_id,
        "error": error,
        "notes": notes,
    }
    try:
        _roll_if_needed()
        append_jsonl(AUDIT_PATH, entry)
    except Exception as exc:
        logger.warning(f"Audit log append failed: {exc}")


def _roll_if_needed() -> None:
    try:
        if not AUDIT_PATH.exists():
            return
        if AUDIT_PATH.stat().st_size < MAX_AUDIT_BYTES:
            return
        archive = AUDIT_PATH.with_suffix(
            f".{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
        )
        AUDIT_PATH.rename(archive)
        logger.info(f"Rolled audit log to {archive.name}")
    except Exception as exc:
        logger.warning(f"Audit log roll failed: {exc}")


def iter_entries(path: Path | None = None) -> Iterator[dict]:
    target = path or AUDIT_PATH
    if not target.exists():
        return
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def entries_since(seconds: int, path: Path | None = None) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(0, seconds))
    out: list[dict] = []
    for entry in iter_entries(path):
        ts = entry.get("ts")
        if not ts:
            continue
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when >= cutoff:
            out.append(entry)
    return out


def recent_entries(limit: int = 50, path: Path | None = None) -> list[dict]:
    target = path or AUDIT_PATH
    if not target.exists():
        return []
    lines: list[str] = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            lines.append(line)
            if len(lines) > limit * 4:
                lines = lines[-limit:]
    out: list[dict] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    out.reverse()
    return out


def summary_counts(entries: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        cat = entry.get("category") or "UNKNOWN"
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def mark_undone(
    *,
    email_ids: Iterable[str],
    note: str = "undone",
    path: Path | None = None,
) -> int:
    """Rewrite the audit log marking the given email_ids as undone.

    Returns the number of entries marked. Used by the undo command so that
    repeated undos do not double-process the same actions.
    """
    target = path or AUDIT_PATH
    if not target.exists():
        return 0
    ids = {eid for eid in email_ids if eid}
    if not ids:
        return 0

    marked = 0
    rewritten: list[str] = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                rewritten.append(line)
                continue
            if entry.get("email_id") in ids and not entry.get("undone"):
                entry["undone"] = True
                entry["undone_at"] = _now_iso()
                entry["undone_note"] = note
                marked += 1
            rewritten.append(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
    if marked:
        atomic_write_text(target, "\n".join(rewritten) + "\n")
    return marked
