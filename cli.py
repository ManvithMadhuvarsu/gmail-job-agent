"""MailAI command-line interface.

Run via:
    python cli.py <command> [args]

Commands:
    run         One-shot Gmail processing
    daemon      Continuous polling loop
    undo        Reverse labels, drafts, and calendar events from the recent window
    doctor      Self-diagnostic checks for install health
    rules       Show or scaffold pre-LLM allow/deny rules
    audit       Print the most recent audit log entries
    health      Print the runtime state and audit summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("mailai.cli")


# ── Helpers ───────────────────────────────────────────────────────────────────
DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)


def _parse_duration_to_seconds(value: str) -> int:
    """Parse 30m / 2h / 1d / 1w (and bare seconds) into total seconds."""
    if not value:
        raise ValueError("duration is empty")
    if value.isdigit():
        return int(value)
    match = DURATION_RE.match(value)
    if not match:
        raise ValueError(f"invalid duration: {value!r} (try 30m, 2h, 1d, 1w)")
    n = int(match.group(1))
    unit = match.group(2).lower()
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


def _color(text: str, code: str) -> str:
    if os.getenv("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _ok(text: str) -> str: return _color(text, "32")
def _warn(text: str) -> str: return _color(text, "33")
def _err(text: str) -> str: return _color(text, "31")
def _dim(text: str) -> str: return _color(text, "2")


# ── undo ──────────────────────────────────────────────────────────────────────
def cmd_undo(args: argparse.Namespace) -> int:
    from tools import audit_log
    from tools.gmail_tool import get_gmail_service, remove_label, delete_draft
    from tools.calendar_tool import delete_calendar_event, get_calendar_service

    seconds = _parse_duration_to_seconds(args.since)
    entries = audit_log.entries_since(seconds)
    candidates = [e for e in entries if not e.get("undone") and not e.get("dry_run")]
    if not candidates:
        print(_warn(f"No actions recorded in the last {args.since} (or all already undone)."))
        return 0

    print(f"Found {_ok(str(len(candidates)))} actions in the last {args.since}.")
    if args.dry_run:
        for e in candidates[:50]:
            print(
                f"  {_dim(e['ts'])}  [{e['category']:<10}] "
                f"label={e.get('label_name') or '-':<14}  "
                f"draft={'Y' if e.get('draft_id') else '-':<2}  "
                f"cal={'Y' if e.get('calendar_event_id') else '-':<2}  "
                f"{e.get('subject','')[:60]}"
            )
        if len(candidates) > 50:
            print(_dim(f"  ... and {len(candidates) - 50} more"))
        print(_dim("Dry run — nothing reverted. Re-run without --dry-run to apply."))
        return 0

    if not args.yes:
        confirm = input(f"Revert these {len(candidates)} actions in Gmail and Calendar? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print(_warn("Aborted."))
            return 1

    gmail = None
    calendar = None
    label_removed = draft_removed = event_removed = errors = 0
    undone_ids: list[str] = []

    for entry in candidates:
        email_id = entry.get("email_id")
        try:
            if entry.get("label_id") and email_id:
                if gmail is None:
                    gmail = get_gmail_service()
                if remove_label(gmail, email_id, entry["label_id"]):
                    label_removed += 1
            if entry.get("draft_id"):
                if gmail is None:
                    gmail = get_gmail_service()
                if delete_draft(gmail, entry["draft_id"]):
                    draft_removed += 1
            if entry.get("calendar_event_id"):
                if calendar is None:
                    calendar = get_calendar_service()
                if delete_calendar_event(calendar, entry["calendar_event_id"]):
                    event_removed += 1
            if email_id:
                undone_ids.append(email_id)
        except Exception as exc:
            errors += 1
            logger.warning(f"Undo failed for {email_id}: {exc}")

    audit_log.mark_undone(email_ids=undone_ids, note=f"undo --since {args.since}")

    print()
    print(_ok(f"Labels removed:        {label_removed}"))
    print(_ok(f"Drafts deleted:        {draft_removed}"))
    print(_ok(f"Calendar events removed: {event_removed}"))
    if errors:
        print(_err(f"Errors:                {errors}"))
    return 0 if errors == 0 else 2


# ── doctor ────────────────────────────────────────────────────────────────────
def cmd_doctor(args: argparse.Namespace) -> int:
    from tools.gmail_tool import (
        CREDENTIALS_PATH, TOKEN_PATH, SCOPES,
        _credentials_have_scopes, CALENDAR_SCOPES,
    )

    checks: list[tuple[str, bool, str]] = []

    py_ok = sys.version_info >= (3, 11)
    checks.append((
        "Python >= 3.11",
        py_ok,
        f"Detected {sys.version.split()[0]}" + ("" if py_ok else " (upgrade recommended)"),
    ))

    creds_ok = CREDENTIALS_PATH.exists() or bool(os.getenv("GMAIL_CREDENTIALS_JSON", "").strip())
    checks.append((
        "Gmail OAuth client present",
        creds_ok,
        "config/credentials.json or GMAIL_CREDENTIALS_JSON" if creds_ok
        else "Missing — see README Google OAuth setup",
    ))

    token_ok = TOKEN_PATH.exists()
    msg = "data/token.pickle present"
    scopes_msg = ""
    if token_ok:
        try:
            import pickle
            with open(TOKEN_PATH, "rb") as f:
                creds = pickle.load(f)
            if not _credentials_have_scopes(creds, SCOPES):
                token_ok = False
                scopes_msg = " — missing some required scopes (visit /login again)"
            else:
                cal_ok = _credentials_have_scopes(creds, CALENDAR_SCOPES)
                scopes_msg = " (calendar scope OK)" if cal_ok else " (calendar scope missing)"
        except Exception as exc:
            token_ok = False
            scopes_msg = f" — failed to read ({exc})"
    else:
        msg = "Run /login or `python main.py` once to authorize"
    checks.append(("Gmail token usable", token_ok, msg + scopes_msg))

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    writable = os.access(data_dir, os.W_OK)
    checks.append(("data/ writable", writable, str(data_dir.resolve())))

    use_ollama = os.getenv("USE_OLLAMA", "false").strip().lower() == "true"
    if use_ollama:
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        ok, info = _probe_ollama(url)
        checks.append(("Ollama reachable", ok, info))
    else:
        key = (os.getenv("GROQ_API_KEY") or "").strip()
        ok = bool(key) and key.lower() != "your_groq_api_key_here"
        checks.append(("Groq API key set", ok, "GROQ_API_KEY configured" if ok else "GROQ_API_KEY missing"))

    from tools.license_tool import verify_license, license_required
    lic = verify_license()
    lic_ok = (not license_required()) or lic.valid
    checks.append((
        "License",
        lic_ok,
        f"tier={lic.tier}, valid={lic.valid}, required={license_required()}",
    ))

    print()
    print("MailAI doctor")
    print("=============")
    failed = 0
    for label, ok, hint in checks:
        marker = _ok("PASS") if ok else _err("FAIL")
        print(f"  [{marker}] {label:<32}  {_dim(hint)}")
        if not ok:
            failed += 1
    print()
    if failed:
        print(_err(f"{failed} check(s) need attention."))
        return 1
    print(_ok("All checks passed."))
    return 0


def _probe_ollama(url: str) -> tuple[bool, str]:
    try:
        import urllib.request
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200, f"{url} responded {resp.status}"
    except Exception as exc:
        return False, f"{url} unreachable: {exc}"


# ── run / daemon ──────────────────────────────────────────────────────────────
def cmd_run(args: argparse.Namespace) -> int:
    if args.dry_run:
        os.environ["MAILAI_DRY_RUN"] = "true"
    from main import run as run_once
    run_once()
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    if args.dry_run:
        os.environ["MAILAI_DRY_RUN"] = "true"
    from daemon import start_daemon
    start_daemon()
    return 0


# ── rules / audit / health ────────────────────────────────────────────────────
def cmd_rules(args: argparse.Namespace) -> int:
    from tools import rules as user_rules
    if args.init:
        path = user_rules.write_example_rules()
        print(f"Wrote example rules to {path}")
        return 0
    summary = user_rules.ruleset_summary()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from tools.audit_log import recent_entries
    for entry in recent_entries(limit=args.limit):
        print(json.dumps(entry, ensure_ascii=False))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from tools.runtime_state import snapshot
    from tools.audit_log import entries_since, summary_counts
    state = snapshot()
    state["last_24h"] = {
        "actions": len(entries_since(86400)),
        "by_category": summary_counts(entries_since(86400)),
    }
    print(json.dumps(state, indent=2, default=str))
    return 0


# ── parser ────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mailai", description="MailAI command-line interface")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("run", help="Process emails once")
    s.add_argument("--dry-run", action="store_true", help="Classify but apply nothing")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("daemon", help="Run the continuous polling daemon")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_daemon)

    s = sub.add_parser("undo", help="Reverse labels, drafts, and calendar events")
    s.add_argument("--since", required=True, help="Window like 30m, 2h, 1d, 1w")
    s.add_argument("--dry-run", action="store_true", help="Preview without changing Gmail/Calendar")
    s.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    s.set_defaults(func=cmd_undo)

    s = sub.add_parser("doctor", help="Diagnose install health")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("rules", help="Show or scaffold pre-LLM rules")
    s.add_argument("--init", action="store_true", help="Write an example data/rules.json")
    s.set_defaults(func=cmd_rules)

    s = sub.add_parser("audit", help="Print recent audit log entries")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("health", help="Print runtime state snapshot")
    s.set_defaults(func=cmd_health)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
