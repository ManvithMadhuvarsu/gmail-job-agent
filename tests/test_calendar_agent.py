from agents.calendar_agent import (
    _extract_json,
    _normalize_event,
    _reference_date,
    should_check_calendar_event,
)


def test_calendar_gate_allows_interview_with_schedule_keyword(monkeypatch):
    monkeypatch.setenv("ENABLE_CALENDAR_EVENTS", "true")
    email = {
        "subject": "Interview schedule for Software Engineer",
        "sender": "Recruiter <recruiter@example.com>",
        "body": "Your interview is scheduled for Monday at 10 AM.",
    }
    result = {"category": "INTERVIEW", "action": "DRAFT_CONFIRM"}

    assert should_check_calendar_event(email, result) is True


def test_calendar_gate_skips_application_confirmation(monkeypatch):
    monkeypatch.setenv("ENABLE_CALENDAR_EVENTS", "true")
    email = {
        "subject": "Application received",
        "sender": "no-reply@example.com",
        "body": "Thank you for applying. We received your application.",
    }
    result = {"category": "APPLIED", "action": "LABEL_ONLY"}

    assert should_check_calendar_event(email, result) is False


def test_calendar_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_CALENDAR_EVENTS", "false")
    email = {
        "subject": "Assessment deadline",
        "body": "Submit the coding assessment by Friday.",
    }
    result = {"category": "FOLLOW_UP", "action": "DRAFT_RESPONSE"}

    assert should_check_calendar_event(email, result) is False


def test_reference_date_uses_email_received_date():
    email = {"date": "Mon, 04 May 2026 09:30:00 +0530"}

    assert _reference_date(email) == "2026-05-04"


def test_extract_json_accepts_extra_text_after_object():
    raw = '{"should_create": false, "reason": "no fixed date", "confidence": 0.2}\nextra'

    assert _extract_json(raw)["should_create"] is False


def test_normalize_event_converts_invented_midnight_to_all_day():
    event = {
        "should_create": True,
        "event_type": "ASSESSMENT",
        "start": "2026-05-18T00:00:00+05:30",
        "end": "",
        "all_day": False,
        "confidence": 0.9,
    }

    normalized = _normalize_event(event)

    assert normalized["all_day"] is True
    assert normalized["start"] == "2026-05-18"
