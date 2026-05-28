"""
Calendar-only backfill for important job-search dates.

This scans a date window, classifies only likely calendar-relevant mail,
and creates Google Calendar events without applying labels or saving drafts.
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.errors import HttpError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except AttributeError:
    pass

# Safety: this command is for calendar backfill only.
os.environ["DISABLE_DRAFTS"] = "true"
os.environ.setdefault("ENABLE_CALENDAR_EVENTS", "true")

from agents.calendar_agent import IMPORTANT_KEYWORDS, extract_calendar_event
from agents.classifier_agent import process_email
from tools.calendar_tool import create_calendar_event_once, get_calendar_service
from tools.gmail_tool import _parse_email, fetch_emails_by_query, get_gmail_service


EXCLUDED_GMAIL_LABELS = {"SENT", "DRAFT", "TRASH", "SPAM"}
DATE_HINT_RE = re.compile(
    r"\b("
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?|"
    r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?|"
    r"today|tomorrow|tonight|deadline|due|by\s+\d{1,2}|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm|ist|utc|est|pst)"
    r")\b",
    re.IGNORECASE,
)
DEFAULT_SEARCH_TERMS = [
    "interview",
    '"screening call"',
    '"recruiter call"',
    "assessment",
    '"coding challenge"',
    '"online test"',
    '"technical test"',
    '"coding test"',
    '"technical round"',
    "meeting",
    "deadline",
    '"due date"',
    "submission",
    "assignment",
    "submit",
    "schedule",
]


def _gmail_date(value: datetime) -> str:
    return value.strftime("%Y/%m/%d")


def _parse_date_env(name: str) -> datetime | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return datetime.strptime(raw, "%Y-%m-%d")


def _date_window() -> tuple[datetime, datetime]:
    end = _parse_date_env("CALENDAR_BACKFILL_END_DATE") or datetime.now()
    start = _parse_date_env("CALENDAR_BACKFILL_START_DATE")
    if not start:
        days = int(os.getenv("CALENDAR_BACKFILL_DAYS", "").strip() or 30)
        start = end - timedelta(days=days)
    return start, end


def _build_query(start: datetime, end: datetime) -> str:
    # Gmail after/before are date-boundary searches. Pad one day on each side,
    # then rely on Calendar duplicate checks and local filtering for safety.
    after = start - timedelta(days=1)
    before = end + timedelta(days=1)
    return (
        f"after:{_gmail_date(after)} before:{_gmail_date(before)} "
        "in:anywhere -in:sent -in:drafts -in:spam -in:trash"
    )


def _search_terms() -> list[str]:
    raw = os.getenv("CALENDAR_BACKFILL_TERMS", "").strip()
    if not raw:
        return DEFAULT_SEARCH_TERMS
    return [term.strip() for term in raw.split(",") if term.strip()]


def _list_message_ids(service, query: str, max_total: int) -> list[str]:
    ids = []
    next_page_token = None
    while True:
        resp = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=min(100, max_total),
            pageToken=next_page_token,
        ).execute()
        ids.extend(msg["id"] for msg in resp.get("messages", []))
        next_page_token = resp.get("nextPageToken")
        if not next_page_token or len(ids) >= max_total:
            break
    return ids[:max_total]


def _fetch_email_details(service, message_ids: list[str]) -> list[dict]:
    emails = []
    total = len(message_ids)
    for idx, message_id in enumerate(message_ids, start=1):
        try:
            detail = service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()
            emails.append(_parse_email(detail))
        except HttpError as exc:
            print(f"warning: failed to fetch message {message_id}: {exc}")
        if idx % 25 == 0 or idx == total:
            print(f"Fetched message details: {idx}/{total}")
    return emails


def _fetch_candidate_emails(service, base_query: str, max_total: int) -> list[dict]:
    custom_query = os.getenv("CALENDAR_BACKFILL_QUERY", "").strip()
    if custom_query:
        print("Using custom CALENDAR_BACKFILL_QUERY.")
        return fetch_emails_by_query(service, query=custom_query, max_total=max_total)

    seen = set()
    message_ids = []
    terms = _search_terms()
    for term in terms:
        if len(message_ids) >= max_total:
            break
        query = f"{base_query} {term}"
        ids = _list_message_ids(service, query=query, max_total=max_total - len(message_ids))
        new_ids = [message_id for message_id in ids if message_id not in seen]
        seen.update(new_ids)
        message_ids.extend(new_ids)
        print(f"Search term {term}: {len(ids)} matches, {len(new_ids)} new")

    print(f"Unique likely event messages: {len(message_ids)}")
    return _fetch_email_details(service, message_ids[:max_total])


def _merged_text(email: dict) -> str:
    fields = [
        email.get("subject", ""),
        email.get("sender", ""),
        email.get("sender_email", ""),
        email.get("snippet", ""),
        email.get("body", ""),
    ]
    return " ".join(str(value).lower() for value in fields if value)


def _is_incoming_user_mail(email: dict) -> bool:
    labels = set(email.get("label_ids") or [])
    return labels.isdisjoint(EXCLUDED_GMAIL_LABELS)


def _looks_calendar_relevant(email: dict) -> bool:
    text = _merged_text(email)
    has_event_keyword = any(keyword in text for keyword in IMPORTANT_KEYWORDS)
    has_date_hint = bool(DATE_HINT_RE.search(text))
    return has_event_keyword and has_date_hint


def run() -> int:
    start, end = _date_window()
    max_total = int(os.getenv("CALENDAR_BACKFILL_MAX_TOTAL", "").strip() or 5000)
    sleep_seconds = float(os.getenv("CALENDAR_BACKFILL_SLEEP_SECONDS", "").strip() or 0.75)
    query = _build_query(start, end)

    print("MailAI calendar-only backfill")
    print(f"Window: {start.date()} -> {end.date()}")
    print(f"Query: {query}")
    print("Safety: drafts disabled; labels are not modified; Calendar events are de-duplicated.")

    gmail = get_gmail_service()
    calendar = get_calendar_service()
    emails = _fetch_candidate_emails(gmail, base_query=query, max_total=max_total)

    scanned = len(emails)
    candidates = 0
    extracted = 0
    created = 0
    duplicates = 0
    skipped = 0
    errors = 0

    print(f"Fetched {scanned} messages. Checking likely event emails...")

    for email in emails:
        subject = (email.get("subject") or "(no subject)")[:90]
        try:
            if not _is_incoming_user_mail(email) or not _looks_calendar_relevant(email):
                skipped += 1
                continue

            candidates += 1
            result = process_email(email)
            event = extract_calendar_event(email, result)
            if not event:
                skipped += 1
                print(f"skip: {subject}")
                time.sleep(sleep_seconds)
                continue

            extracted += 1
            was_created, event_id = create_calendar_event_once(calendar, email, event)
            if was_created:
                created += 1
                print(f"created: {event.get('event_type')} | {event.get('start')} | {subject} | {event_id}")
            else:
                duplicates += 1
                print(f"exists: {event.get('event_type')} | {event.get('start')} | {subject} | {event_id}")
            time.sleep(sleep_seconds)
        except Exception as exc:
            errors += 1
            print(f"error: {subject} | {exc}")

    print("\nCalendar backfill complete")
    print(f"Scanned: {scanned}")
    print(f"Likely candidates: {candidates}")
    print(f"Calendar events extracted: {extracted}")
    print(f"Calendar events created: {created}")
    print(f"Duplicates already present: {duplicates}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
