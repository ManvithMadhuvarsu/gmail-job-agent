"""Persisted setup wizard configuration.

The /setup wizard writes a small JSON file at data/config.json. The various
modules (classifier, gmail tool, calendar tool, main.run) read from os.environ,
so this module exposes a helper that materializes the saved config into
os.environ at process start, taking effect for any later reads.

A real .env edit isn't safe — sellers ship `.env.example` and many users have
none. data/config.json is sidecar state we own.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


SETUP_PATH = Path("data/config.json")


_SETTABLE_KEYS = {
    # ── LLM ───────────────────────────────────────────────
    "USE_OLLAMA",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "GROQ_API_KEY",
    # ── Behavior ──────────────────────────────────────────
    "SCAN_DAYS",
    "POLL_INTERVAL_MINUTES",
    "MAILAI_DRY_RUN",
    "DISABLE_DRAFTS",
    "ENABLE_CALENDAR_EVENTS",
    "GOOGLE_CALENDAR_ID",
    "CALENDAR_TIMEZONE",
    # ── Identity ──────────────────────────────────────────
    "YOUR_NAME",
    "YOUR_PHONE",
    "YOUR_EMAIL",
    "YOUR_LINKEDIN",
    # ── Label prefix ──────────────────────────────────────
    "LABEL_REJECTION",
    "LABEL_INTERVIEW",
    "LABEL_HOLD",
    "LABEL_FOLLOWUP",
    "LABEL_APPLIED",
}


@dataclass
class SetupConfig:
    values: dict[str, str] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"values": dict(self.values), "completed_steps": list(self.completed_steps)}


def load_config() -> SetupConfig:
    if not SETUP_PATH.exists():
        return SetupConfig()
    try:
        with open(SETUP_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning(f"setup config read failed: {exc}")
        return SetupConfig()
    values = {k: str(v) for k, v in (raw.get("values") or {}).items() if k in _SETTABLE_KEYS}
    completed = [str(s) for s in (raw.get("completed_steps") or [])]
    return SetupConfig(values=values, completed_steps=completed)


def save_config(config: SetupConfig) -> None:
    atomic_write_json(SETUP_PATH, config.to_dict())


def update_values(updates: dict[str, str], *, step: str | None = None) -> SetupConfig:
    config = load_config()
    for key, value in updates.items():
        if key not in _SETTABLE_KEYS:
            continue
        if value is None or value == "":
            config.values.pop(key, None)
            continue
        config.values[key] = str(value)
    if step and step not in config.completed_steps:
        config.completed_steps.append(step)
    save_config(config)
    apply_to_env(config)
    return config


def mark_complete(step: str) -> SetupConfig:
    config = load_config()
    if step and step not in config.completed_steps:
        config.completed_steps.append(step)
        save_config(config)
    return config


def apply_to_env(config: SetupConfig | None = None) -> None:
    """Copy saved values into os.environ. Existing env wins (so .env stays authoritative)."""
    config = config or load_config()
    for key, value in config.values.items():
        if key not in os.environ or os.environ[key] == "":
            os.environ[key] = value


def probe_llm() -> dict[str, Any]:
    """Run a no-LLM connectivity probe; return user-friendly status."""
    use_ollama = (os.getenv("USE_OLLAMA", "false").lower() == "true")
    result: dict[str, Any] = {"use_ollama": use_ollama}
    if use_ollama:
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            import urllib.request
            req = urllib.request.Request(f"{url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                result["ok"] = resp.status == 200
                result["detail"] = f"Ollama at {url} responded {resp.status}"
        except Exception as exc:
            result["ok"] = False
            result["detail"] = f"Ollama at {url} unreachable: {exc}"
    else:
        key = (os.getenv("GROQ_API_KEY") or "").strip()
        ok = bool(key) and key.lower() != "your_groq_api_key_here"
        result["ok"] = ok
        result["detail"] = "GROQ_API_KEY is set" if ok else "GROQ_API_KEY missing or placeholder"
    return result
