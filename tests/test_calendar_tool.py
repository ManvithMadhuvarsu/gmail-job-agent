from tools.calendar_tool import _event_body


def test_event_body_uses_private_mailai_marker(monkeypatch):
    monkeypatch.setenv("CALENDAR_REMINDER_MINUTES", "1440,60")
    email = {
        "id": "msg-123",
        "thread_id": "thread-123",
        "subject": "Interview with ExampleCo",
        "sender": "Recruiter <recruiter@example.com>",
    }
    event = {
        "event_type": "INTERVIEW",
        "title": "Interview with ExampleCo",
        "start": "2026-06-01T10:00:00+05:30",
        "end": "2026-06-01T11:00:00+05:30",
        "timezone": "Asia/Kolkata",
        "description": "Technical interview",
        "confidence": 0.91,
    }

    body = _event_body(email, event)

    assert body["summary"] == "MailAI: Interview with ExampleCo"
    assert body["extendedProperties"]["private"]["mailaiEmailId"] == "msg-123"
    assert body["start"]["dateTime"].startswith("2026-06-01T10:00:00")
    assert body["reminders"]["overrides"] == [
        {"method": "popup", "minutes": 1440},
        {"method": "popup", "minutes": 60},
    ]


def test_event_body_supports_all_day_deadlines():
    email = {"id": "msg-456", "thread_id": "thread-456", "subject": "Assessment deadline", "sender": "HR"}
    event = {
        "event_type": "DEADLINE",
        "title": "Submit coding assessment",
        "start": "2026-06-03",
        "all_day": True,
        "description": "Submission deadline",
        "confidence": 0.85,
    }

    body = _event_body(email, event)

    assert body["start"] == {"date": "2026-06-03"}
    assert body["end"] == {"date": "2026-06-04"}

