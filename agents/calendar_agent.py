"""Extract important job dates from emails for Google Calendar."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from email.utils import parsedate_to_datetime

from langchain_core.prompts import ChatPromptTemplate

from agents.classifier_agent import safe_invoke


logger = logging.getLogger(__name__)


IMPORTANT_CATEGORIES = {"INTERVIEW", "FOLLOW_UP"}
IMPORTANT_KEYWORDS = {
    "interview",
    "meeting",
    "assessment",
    "test",
    "coding challenge",
    "online test",
    "technical round",
    "screening",
    "assignment",
    "deadline",
    "due date",
    "submit",
    "submission",
    "schedule",
    "calendar invite",
}


CALENDAR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You extract only important job-search calendar events from email.

Return JSON only. Do not wrap it in markdown.

Create an event only when the email contains a concrete date or deadline for an important item:
- interview, recruiter meeting, screening call
- assessment, test, coding challenge, assignment
- submission deadline or requested document deadline

Do NOT create events for:
- application received confirmations
- rejections
- vague status updates
- requests for availability with no fixed date/time
- newsletters, OTPs, promos, or no-reply transactional mail

Reference date for resolving relative dates and dates without a year: {reference_date}
Default timezone: {timezone}

JSON shape:
{{
  "should_create": true,
  "event_type": "INTERVIEW | MEETING | ASSESSMENT | TEST | SUBMISSION | DEADLINE",
  "title": "short useful title",
  "start": "ISO-8601 datetime with offset, or YYYY-MM-DD for all-day deadline",
  "end": "ISO-8601 datetime with offset, or empty string",
  "all_day": false,
  "timezone": "{timezone}",
  "location": "location or meeting link if present",
  "description": "brief details, role/company/recruiter/instructions",
  "confidence": 0.0
}}

If no safe concrete calendar event exists, return:
{{"should_create": false, "reason": "why not", "confidence": 0.0}}

If the email gives a date without a year or a relative date, infer it from the reference date.
If the email gives a date but no explicit time, use all_day=true and start as YYYY-MM-DD.
Never invent a midnight time for date-only events."""),
    ("human", """Mail category: {category}
Mail action: {action}
Received date header: {email_date}
Subject: {subject}
From: {sender}

Email body:
{body}"""),
])


def _calendar_enabled() -> bool:
    return os.getenv("ENABLE_CALENDAR_EVENTS", "true").strip().lower() not in {"0", "false", "no", "off"}


def _timezone_name() -> str:
    return os.getenv("CALENDAR_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"


def _max_chars() -> int:
    raw = os.getenv("CALENDAR_CONTEXT_MAX_CHARS", "").strip()
    try:
        return int(raw or 1800)
    except ValueError:
        return 1800


def _min_confidence() -> float:
    raw = os.getenv("CALENDAR_MIN_CONFIDENCE", "").strip()
    try:
        return float(raw or 0.70)
    except ValueError:
        return 0.70


def _reference_date(email: dict) -> str:
    """Use the email's received date as the anchor for backfilled relative dates."""
    raw = (email.get("date") or "").strip()
    if raw:
        try:
            return parsedate_to_datetime(raw).date().isoformat()
        except (TypeError, ValueError, IndexError, AttributeError):
            logger.debug(f"Could not parse email date for calendar extraction: {raw!r}")
    return datetime.now().strftime("%Y-%m-%d")


def _merged_text(email: dict) -> str:
    fields = [
        email.get("subject", ""),
        email.get("sender", ""),
        email.get("sender_email", ""),
        email.get("snippet", ""),
        email.get("body", ""),
    ]
    return " ".join(str(value).lower() for value in fields if value)


def should_check_calendar_event(email: dict, result: dict) -> bool:
    """Cheap gate to avoid LLM/calendar work for unimportant mail."""
    if not _calendar_enabled():
        return False
    category = result.get("category", "IRRELEVANT")
    text = _merged_text(email)
    if category in IMPORTANT_CATEGORIES and any(keyword in text for keyword in IMPORTANT_KEYWORDS):
        return True
    if any(keyword in text for keyword in {"assessment", "coding challenge", "deadline", "due date", "submission", "interview"}):
        return category not in {"REJECTION", "IRRELEVANT"}
    return False


def _extract_json(raw: str) -> dict:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    candidates = [text]
    start = text.find("{")
    if start > 0:
        candidates.append(text[start:])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed, _ = decoder.raw_decode(candidate)
            return parsed
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _valid_event(event: dict) -> bool:
    if not event.get("should_create"):
        return False
    if float(event.get("confidence") or 0) < _min_confidence():
        return False
    if not event.get("start"):
        return False
    event_type = str(event.get("event_type", "")).upper()
    return event_type in {"INTERVIEW", "MEETING", "ASSESSMENT", "TEST", "SUBMISSION", "DEADLINE"}


def _normalize_event(event: dict) -> dict:
    """Normalize safe LLM output quirks before creating Calendar events."""
    start = str(event.get("start", ""))
    midnight_without_end = (
        not event.get("all_day")
        and re.search(r"T00:00(?::00)?", start)
        and not event.get("end")
    )
    if midnight_without_end and re.match(r"\d{4}-\d{2}-\d{2}", start):
        event["all_day"] = True
        event["start"] = start[:10]
    return event


def extract_calendar_event(email: dict, result: dict) -> dict | None:
    """Return a normalized calendar event payload, or None if nothing important."""
    if not should_check_calendar_event(email, result):
        return None

    body = (email.get("body") or email.get("snippet") or "").strip()
    body = body[:_max_chars()]
    response = safe_invoke(CALENDAR_PROMPT, {
        "reference_date": _reference_date(email),
        "timezone": _timezone_name(),
        "category": result.get("category", "IRRELEVANT"),
        "action": result.get("action", "SKIP"),
        "email_date": email.get("date", ""),
        "subject": email.get("subject", ""),
        "sender": email.get("sender", ""),
        "body": body,
    })

    try:
        event = _extract_json(response)
    except Exception as e:
        logger.warning(f"Calendar extraction returned invalid JSON for '{email.get('subject', '')[:60]}': {e}")
        return None

    if not _valid_event(event):
        logger.info(f"No important calendar event found for '{email.get('subject', '')[:60]}'")
        return None

    event["event_type"] = str(event.get("event_type", "DEADLINE")).upper()
    event["timezone"] = event.get("timezone") or _timezone_name()
    event["title"] = event.get("title") or f"{event['event_type'].title()}: {email.get('subject', '')}"
    event["description"] = event.get("description") or email.get("snippet", "")
    return _normalize_event(event)
