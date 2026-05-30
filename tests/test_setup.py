import json
from pathlib import Path

import pytest

from tools import setup_config
from tools.product_pages import render_setup


@pytest.fixture
def setup_path(tmp_path, monkeypatch):
    target = tmp_path / "config.json"
    monkeypatch.setattr(setup_config, "SETUP_PATH", target)
    yield target


def test_update_values_filters_unknown_keys(setup_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = setup_config.update_values(
        {"GROQ_API_KEY": "abc", "EVIL": "bad"},
        step="llm",
    )
    assert cfg.values["GROQ_API_KEY"] == "abc"
    assert "EVIL" not in cfg.values
    assert "llm" in cfg.completed_steps


def test_save_round_trip(setup_path):
    cfg = setup_config.update_values({"YOUR_NAME": "Ada"}, step="identity")
    on_disk = json.loads(setup_path.read_text(encoding="utf-8"))
    assert on_disk["values"]["YOUR_NAME"] == "Ada"
    assert on_disk["completed_steps"] == ["identity"]


def test_env_does_not_get_overridden_if_already_set(setup_path, monkeypatch):
    monkeypatch.setenv("YOUR_NAME", "From Env")
    setup_config.update_values({"YOUR_NAME": "From Setup"}, step="identity")
    setup_config.apply_to_env()
    import os
    assert os.environ["YOUR_NAME"] == "From Env"


def test_render_setup_each_step():
    license_status = {"valid": False, "tier": "community", "expiry_warning_level": None, "days_until_expiry": None}
    for step in ("identity", "gmail", "llm", "calendar", "labels", "safety", "done"):
        html = render_setup(
            step=step,
            values={},
            completed=[],
            auth=False,
            license_status=license_status,
        )
        assert "<form" in html or step in ("gmail", "done"), f"step {step} should have a form or special content"
        assert "MailAI setup wizard" in html
