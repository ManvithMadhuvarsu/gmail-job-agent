import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import cli
from tools import audit_log
from tools.atomic_io import append_jsonl


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit_log, "AUDIT_PATH", audit_file)
    monkeypatch.chdir(tmp_path)
    yield audit_file


def _entry(ts: datetime, **overrides) -> dict:
    base = {
        "ts": ts.isoformat(timespec="seconds"),
        "email_id": "id-1",
        "thread_id": "t-1",
        "subject": "Interview",
        "sender": "Recruiter <r@bigco.com>",
        "sender_email": "r@bigco.com",
        "category": "INTERVIEW",
        "action": "DRAFT_CONFIRM",
        "label_id": "Label_1",
        "label_name": "Job/Interview",
        "draft_id": "draft-1",
        "calendar_event_id": "cal-1",
        "dry_run": False,
        "rule_id": None,
        "error": None,
        "notes": None,
    }
    base.update(overrides)
    return base


def test_parse_duration_to_seconds():
    assert cli._parse_duration_to_seconds("30m") == 1800
    assert cli._parse_duration_to_seconds("2h") == 7200
    assert cli._parse_duration_to_seconds("1d") == 86400
    assert cli._parse_duration_to_seconds("1w") == 604800
    assert cli._parse_duration_to_seconds("45") == 45
    with pytest.raises(ValueError):
        cli._parse_duration_to_seconds("nope")


def test_undo_dry_run_lists_actions(capsys, isolated_data):
    now = datetime.now(timezone.utc)
    append_jsonl(audit_log.AUDIT_PATH, _entry(now - timedelta(minutes=5)))
    append_jsonl(audit_log.AUDIT_PATH, _entry(now - timedelta(minutes=10), email_id="id-2", draft_id=None))
    rc = cli.main(["undo", "--since", "1h", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2" in out
    assert "INTERVIEW" in out


def test_undo_skips_already_undone(capsys, isolated_data):
    now = datetime.now(timezone.utc)
    append_jsonl(audit_log.AUDIT_PATH, _entry(now - timedelta(minutes=5), undone=True))
    rc = cli.main(["undo", "--since", "1h", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No actions" in out


def test_undo_skips_dry_run_entries(capsys, isolated_data):
    now = datetime.now(timezone.utc)
    append_jsonl(audit_log.AUDIT_PATH, _entry(now - timedelta(minutes=5), dry_run=True))
    rc = cli.main(["undo", "--since", "1h", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No actions" in out


def test_doctor_runs_without_crashing(capsys, isolated_data, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("USE_OLLAMA", raising=False)
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "MailAI doctor" in out
    assert rc in (0, 1)


def test_doctor_loads_dotenv_for_llm_config(capsys, isolated_data, monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("USE_OLLAMA=false\nGROQ_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("USE_OLLAMA", raising=False)

    rc = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert rc in (0, 1)
    assert "Groq API key set" in out
    assert "GROQ_API_KEY configured" in out


def test_rules_init_writes_example(capsys, tmp_path, monkeypatch):
    from tools import rules
    target = tmp_path / "rules.json"
    monkeypatch.setattr(rules, "RULES_PATH_JSON", target)
    monkeypatch.setattr(rules, "RULES_PATH_YAML", tmp_path / "rules.yaml")
    rules.reload_rules()
    rc = cli.main(["rules", "--init"])
    assert rc == 0
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "rules" in payload
