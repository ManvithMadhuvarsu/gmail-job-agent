import json
from pathlib import Path

import pytest

from tools import rules as user_rules


@pytest.fixture
def rules_file(tmp_path, monkeypatch):
    json_path = tmp_path / "rules.json"
    yaml_path = tmp_path / "rules.yaml"
    monkeypatch.setattr(user_rules, "RULES_PATH_JSON", json_path)
    monkeypatch.setattr(user_rules, "RULES_PATH_YAML", yaml_path)
    user_rules.reload_rules()
    yield json_path
    user_rules.reload_rules()


def _email(*, subject="Hello", sender_email="x@example.com", sender=None) -> dict:
    return {
        "subject": subject,
        "sender": sender or f"Someone <{sender_email}>",
        "sender_email": sender_email,
    }


def test_no_rules_returns_none(rules_file):
    assert user_rules.evaluate(_email()) is None


def test_skip_rule_matches_domain(rules_file):
    rules_file.write_text(json.dumps({"rules": [
        {"id": "skip-news", "when": {"sender_domain": "news.example.com"}, "then": {"action": "skip"}},
    ]}))
    user_rules.reload_rules()
    match = user_rules.evaluate(_email(sender_email="weekly@news.example.com"))
    assert match is not None
    assert match.action == "skip"
    assert match.rule_id == "skip-news"


def test_force_category(rules_file):
    rules_file.write_text(json.dumps({"rules": [
        {
            "id": "force-applied",
            "when": {"sender_email": "hr@bigco.com"},
            "then": {"action": "force", "category": "APPLIED"},
        },
    ]}))
    user_rules.reload_rules()
    match = user_rules.evaluate(_email(sender_email="hr@bigco.com"))
    assert match is not None
    assert match.action == "force"
    assert match.category == "APPLIED"


def test_invalid_category_for_force_is_ignored(rules_file):
    rules_file.write_text(json.dumps({"rules": [
        {"id": "bad", "when": {"sender_email": "x"}, "then": {"action": "force", "category": "BOGUS"}},
    ]}))
    user_rules.reload_rules()
    assert user_rules.evaluate(_email(sender_email="x@y.com")) is None


def test_allow_suppresses_later_skip(rules_file):
    rules_file.write_text(json.dumps({"rules": [
        {
            "id": "allow-vip",
            "when": {"sender_email": "vip@news.example.com"},
            "then": {"action": "allow"},
        },
        {
            "id": "skip-domain",
            "when": {"sender_domain": "news.example.com"},
            "then": {"action": "skip"},
        },
    ]}))
    user_rules.reload_rules()
    match = user_rules.evaluate(_email(sender_email="vip@news.example.com"))
    assert match is None


def test_subject_regex(rules_file):
    rules_file.write_text(json.dumps({"rules": [
        {
            "id": "skip-otp",
            "when": {"subject_regex": "(?i)one[- ]?time password"},
            "then": {"action": "skip"},
        },
    ]}))
    user_rules.reload_rules()
    assert user_rules.evaluate(_email(subject="Your One-time password")).action == "skip"
    assert user_rules.evaluate(_email(subject="Interview invite")) is None


def test_empty_when_does_not_wildcard(rules_file):
    rules_file.write_text(json.dumps({"rules": [
        {"id": "ghost", "when": {}, "then": {"action": "skip"}},
    ]}))
    user_rules.reload_rules()
    assert user_rules.evaluate(_email()) is None


def test_disabled_via_env(rules_file, monkeypatch):
    rules_file.write_text(json.dumps({"rules": [
        {"id": "skip", "when": {"sender_email": "x"}, "then": {"action": "skip"}},
    ]}))
    user_rules.reload_rules()
    monkeypatch.setenv("MAILAI_RULES_DISABLED", "true")
    assert user_rules.evaluate(_email(sender_email="x@y.com")) is None
