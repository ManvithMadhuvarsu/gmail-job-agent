"""Google Calendar helpers for MailAI important job dates."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from tools.gmail_tool import CALENDAR_SCOPES, get_google_credentials


logger = logging.getLogger(__name__)


def _calendar_api_setup_hint(error: HttpError) -> str:
    """Return a clear setup hint for the most common Calendar API project issue."""
    text = str(error)
    if "accessNotConfigured" in text or "calendar-json.googleapis.com" in text:
        return (
            " Google Calendar API is disabled for this OAuth project. "
            "Enable it in Google Cloud Console, then rerun MailAI."
        )
    return ""


def get_calendar_service():
    """Authenticate and return a Google Calendar API service."""
    creds = get_google_credentials(required_scopes=CALENDAR_SCOPES)
    return build("calendar", "v3", credentials=creds)


def _calendar_id() -> str:
    return os.getenv("GOOGLE_CALENDAR_ID", "primary").strip() or "primary"


def _timezone_name() -> str:
    return os.getenv("CALENDAR_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"


def _default_duration_minutes() -> int:
    raw = os.getenv("CALENDAR_DEFAULT_DURATION_MINUTES", "").strip()
    try:
        return max(15, int(raw or 60))
    except ValueError:
        return 60


def _reminder_minutes() -> list[int]:
    raw = os.getenv("CALENDAR_REMINDER_MINUTES", "1440,60").strip()
    reminders = []
    for item in raw.split(","):
        try:
            reminders.append(max(0, int(item.strip())))
        except ValueError:
            continue
    return reminders or [1440, 60]


def _parse_datetime(value: str, tz_name: str) -> datetime:
    text = (value or "").strip()
    if not text:
        raise ValueError("Missing event start time")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(tz_name))
    return parsed


def _parse_date(value: str) -> date:
    return date.fromisoformat((value or "").strip()[:10])


def _event_body(email: dict, event: dict) -> dict:
    tz_name = event.get("timezone") or _timezone_name()
    title = event.get("title") or f"MailAI: {event.get('event_type', 'Important date')}"
    event_type = (event.get("event_type") or "IMPORTANT").upper()
    description = "\n".join(
        part
        for part in [
            event.get("description", "").strip(),
            "",
            "Added by MailAI from Gmail.",
            f"Type: {event_type}",
            f"Subject: {email.get('subject', '')}",
            f"From: {email.get('sender', '')}",
            f"Gmail message id: {email.get('id', '')}",
            f"Confidence: {event.get('confidence', '')}",
        ]
        if part is not None
    ).strip()

    body = {
        "summary": f"MailAI: {title}",
        "description": description,
        "location": event.get("location", ""),
        "extendedProperties": {
            "private": {
                "mailaiEmailId": email.get("id", ""),
                "mailaiThreadId": email.get("thread_id", ""),
                "mailaiEventType": event_type,
            }
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": minutes}
                for minutes in _reminder_minutes()
            ],
        },
    }

    if event.get("all_day"):
        start_date = _parse_date(event.get("start", ""))
        end_date = _parse_date(event.get("end", "")) if event.get("end") else start_date + timedelta(days=1)
        if end_date <= start_date:
            end_date = start_date + timedelta(days=1)
        body["start"] = {"date": start_date.isoformat()}
        body["end"] = {"date": end_date.isoformat()}
        return body

    start_dt = _parse_datetime(event.get("start", ""), tz_name)
    end_dt = _parse_datetime(event.get("end", ""), tz_name) if event.get("end") else start_dt + timedelta(minutes=_default_duration_minutes())
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(minutes=_default_duration_minutes())

    body["start"] = {"dateTime": start_dt.isoformat(), "timeZone": tz_name}
    body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": tz_name}
    return body


def _existing_event_id(service, email_id: str) -> str | None:
    if not email_id:
        return None
    try:
        resp = service.events().list(
            calendarId=_calendar_id(),
            privateExtendedProperty=f"mailaiEmailId={email_id}",
            maxResults=1,
            singleEvents=True,
        ).execute()
        items = resp.get("items", [])
        return items[0].get("id") if items else None
    except HttpError as e:
        logger.warning(
            f"Calendar duplicate check failed for email {email_id}: {e}"
            f"{_calendar_api_setup_hint(e)}"
        )
        return None


def create_calendar_event_once(service, email: dict, event: dict) -> tuple[bool, str | None]:
    """Create one Calendar event for this email if one does not already exist."""
    email_id = email.get("id", "")
    existing_id = _existing_event_id(service, email_id)
    if existing_id:
        logger.info(f"Calendar event already exists for email {email_id}: {existing_id}")
        return False, existing_id

    try:
        body = _event_body(email, event)
        created = service.events().insert(
            calendarId=_calendar_id(),
            body=body,
            sendUpdates="none",
        ).execute()
        event_id = created.get("id")
        logger.info(f"Calendar event created for email {email_id}: {event_id}")
        return True, event_id
    except HttpError as e:
        logger.warning(
            f"Failed to create Calendar event for email {email_id}: {e}"
            f"{_calendar_api_setup_hint(e)}"
        )
        return False, None
    except ValueError as e:
        logger.warning(f"Failed to create Calendar event for email {email_id}: {e}")
        return False, None
