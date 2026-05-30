"""Pre-LLM allow/deny rules.

Power users want escape hatches when the AI mislabels their mail and a way to
short-circuit obvious classifications without paying for an LLM call. This
module loads simple YAML/JSON rules from data/rules.yaml (preferred) or
data/rules.json and decides:

  - skip:    drop the email entirely (no label, no draft, no calendar)
  - force:   pin a specific category and skip the LLM
  - allow:   bypass any skip rules that would otherwise match

A rule has shape:

    - id: optional-name
      when:
        sender_domain: example.com         # exact or suffix match
        sender_email: hr@example.com       # substring match
        subject_regex: "(?i)assessment"    # python re
        subject_contains: "Interview"      # substring (case-insensitive)
      then:
        action: skip | force | allow
        category: INTERVIEW                # required when action=force

Rules are evaluated in order. The first matching `skip` or `force` wins
unless an earlier `allow` matched the same email.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RULES_PATH_YAML = Path("data/rules.yaml")
RULES_PATH_JSON = Path("data/rules.json")

VALID_CATEGORIES = {"REJECTION", "INTERVIEW", "HOLD", "FOLLOW_UP", "APPLIED", "IRRELEVANT"}
VALID_ACTIONS = {"skip", "force", "allow"}


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    action: str
    category: str | None = None


def _read_yaml(path: Path) -> list[dict] | None:
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("PyYAML not installed; rules.yaml will be ignored. Install pyyaml or use rules.json.")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning(f"Failed to parse {path}: {exc}")
        return None
    if isinstance(data, dict):
        data = data.get("rules", [])
    if not isinstance(data, list):
        return []
    return data


def _read_json(path: Path) -> list[dict] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to parse {path}: {exc}")
        return None
    if isinstance(data, dict):
        data = data.get("rules", [])
    if not isinstance(data, list):
        return []
    return data


def _load_rules_raw() -> list[dict]:
    if RULES_PATH_YAML.exists():
        data = _read_yaml(RULES_PATH_YAML)
        if data is not None:
            return data
    if RULES_PATH_JSON.exists():
        data = _read_json(RULES_PATH_JSON)
        if data is not None:
            return data
    return []


def _normalize(rule: dict, index: int) -> dict | None:
    if not isinstance(rule, dict):
        return None
    when = rule.get("when") or {}
    then = rule.get("then") or {}
    action = str(then.get("action", "")).lower()
    if action not in VALID_ACTIONS:
        logger.warning(f"Skipping rule[{index}] with invalid action: {action!r}")
        return None
    category = (then.get("category") or "").strip().upper() or None
    if action == "force" and category not in VALID_CATEGORIES:
        logger.warning(f"Skipping rule[{index}]: action=force requires a valid category, got {category!r}")
        return None
    subject_regex = when.get("subject_regex") or ""
    compiled = None
    if subject_regex:
        try:
            compiled = re.compile(subject_regex)
        except re.error as exc:
            logger.warning(f"Skipping rule[{index}]: invalid regex {subject_regex!r}: {exc}")
            return None
    return {
        "id": str(rule.get("id") or f"rule_{index}"),
        "action": action,
        "category": category,
        "sender_domain": (when.get("sender_domain") or "").strip().lower(),
        "sender_email": (when.get("sender_email") or "").strip().lower(),
        "subject_contains": (when.get("subject_contains") or "").strip().lower(),
        "subject_regex": compiled,
    }


@lru_cache(maxsize=1)
def _cached_rules() -> tuple[dict, ...]:
    raw = _load_rules_raw()
    normalized: list[dict] = []
    for index, rule in enumerate(raw):
        norm = _normalize(rule, index)
        if norm:
            normalized.append(norm)
    if normalized:
        logger.info(f"Loaded {len(normalized)} MailAI classification rule(s).")
    return tuple(normalized)


def reload_rules() -> int:
    """Drop the cached ruleset so the next call rereads disk. Returns the new count."""
    _cached_rules.cache_clear()
    return len(_cached_rules())


def _matches(rule: dict, email: dict) -> bool:
    sender_email = (email.get("sender_email") or "").lower()
    sender = (email.get("sender") or "").lower()
    subject = (email.get("subject") or "").lower()

    if rule["sender_domain"]:
        domain = rule["sender_domain"]
        if not (sender_email.endswith("@" + domain) or sender_email.endswith("." + domain) or domain in sender):
            return False
    if rule["sender_email"]:
        if rule["sender_email"] not in sender_email and rule["sender_email"] not in sender:
            return False
    if rule["subject_contains"]:
        if rule["subject_contains"] not in subject:
            return False
    if rule["subject_regex"]:
        if not rule["subject_regex"].search(email.get("subject") or ""):
            return False
    # If no condition was set, never match — empty rule is a config error, not a wildcard.
    if not any([
        rule["sender_domain"],
        rule["sender_email"],
        rule["subject_contains"],
        rule["subject_regex"],
    ]):
        return False
    return True


def evaluate(email: dict) -> RuleMatch | None:
    """Return the first decisive rule match for this email, or None."""
    if os.getenv("MAILAI_RULES_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    rules = _cached_rules()
    if not rules:
        return None
    allow_id: str | None = None
    for rule in rules:
        if not _matches(rule, email):
            continue
        if rule["action"] == "allow":
            allow_id = rule["id"]
            continue
        if allow_id:
            logger.debug(f"Rule {rule['id']} suppressed by earlier allow {allow_id}")
            return None
        return RuleMatch(rule_id=rule["id"], action=rule["action"], category=rule["category"])
    return None


def write_example_rules(path: Path | None = None) -> Path:
    """Write a friendly example rules file. Only called by setup/doctor on demand."""
    target = path or RULES_PATH_JSON
    if target.exists():
        return target
    example = {
        "rules": [
            {
                "id": "skip-newsletters",
                "when": {"sender_domain": "newsletter.example.com"},
                "then": {"action": "skip"},
            },
            {
                "id": "force-internal-hr-applied",
                "when": {"sender_email": "hr@yourcompany.com"},
                "then": {"action": "force", "category": "APPLIED"},
            },
            {
                "id": "always-process-recruiter",
                "when": {"sender_email": "recruiter@bigco.com"},
                "then": {"action": "allow"},
            },
        ]
    }
    from tools.atomic_io import atomic_write_json

    atomic_write_json(target, example)
    return target


def ruleset_summary() -> dict[str, Any]:
    rules = _cached_rules()
    return {
        "count": len(rules),
        "ids": [r["id"] for r in rules],
        "source": str(RULES_PATH_YAML) if RULES_PATH_YAML.exists() else (str(RULES_PATH_JSON) if RULES_PATH_JSON.exists() else None),
    }
