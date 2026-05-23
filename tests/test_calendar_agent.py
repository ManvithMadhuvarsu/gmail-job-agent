from agents.calendar_agent import should_check_calendar_event


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

