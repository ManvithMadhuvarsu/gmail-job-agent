import json
from pathlib import Path

import pytest

from tools import audit_log


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    target = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit_log, "AUDIT_PATH", target)
    return target


def _email(eid: str = "abc1", subject: str = "Hello") -> dict:
    return {
        "id": eid,
        "thread_id": f"t-{eid}",
        "subject": subject,
        "sender": "Recruiter <r@example.com>",
        "sender_email": "r@example.com",
    }


def test_record_action_writes_jsonl(audit_path):
    audit_log.record_action(
        email=_email(),
        category="INTERVIEW",
        action="DRAFT_CONFIRM",
        label_id="Label_1",
        label_name="Job/Interview",
        draft_id="d-1",
        calendar_event_id="cal-1",
    )
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["email_id"] == "abc1"
    assert entry["category"] == "INTERVIEW"
    assert entry["draft_id"] == "d-1"
    assert entry["dry_run"] is False
    assert entry["ts"]


def test_dry_run_flag_round_trips(audit_path):
    audit_log.record_action(email=_email(), category="REJECTION", action="LABEL_ONLY", dry_run=True)
    entry = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["dry_run"] is True


def test_recent_entries_returns_most_recent_first(audit_path):
    for i in range(5):
        audit_log.record_action(email=_email(eid=f"id{i}"), category="APPLIED", action="LABEL_ONLY")
    recent = audit_log.recent_entries(limit=3)
    assert len(recent) == 3
    assert recent[0]["email_id"] == "id4"
    assert recent[-1]["email_id"] == "id2"


def test_mark_undone_only_marks_specified_ids(audit_path):
    audit_log.record_action(email=_email(eid="keep"), category="HOLD", action="LABEL_ONLY")
    audit_log.record_action(email=_email(eid="undo"), category="HOLD", action="LABEL_ONLY")
    marked = audit_log.mark_undone(email_ids=["undo"])
    assert marked == 1
    entries = list(audit_log.iter_entries())
    by_id = {e["email_id"]: e for e in entries}
    assert by_id["undo"]["undone"] is True
    assert by_id["undo"]["undone_at"]
    assert "undone" not in by_id["keep"]
