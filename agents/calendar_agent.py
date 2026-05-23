"""Extract important job dates from emails for Google Calendar."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime

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

Current date: {current_date}
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

If the email gives a date without a year, infer the next future occurrence from the current date.
If the email gives a deadline without a time, use all_day=true and start as YYYY-MM-DD."""),
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _valid_event(event: dict) -> bool:
    if not event.get("should_create"):
        return False
    if float(event.get("confidence") or 0) < _min_confidence():
        return False
    if not event.get("start"):
        return False
    event_type = str(event.get("event_type", "")).upper()
    return event_type in {"INTERVIEW", "MEETING", "ASSESSMENT", "TEST", "SUBMISSION", "DEADLINE"}


def extract_calendar_event(email: dict, result: dict) -> dict | None:
    """Return a normalized calendar event payload, or None if nothing important."""
    if not should_check_calendar_event(email, result):
        return None

    body = (email.get("body") or email.get("snippet") or "").strip()
    body = body[:_max_chars()]
    response = safe_invoke(CALENDAR_PROMPT, {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
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
    return event

