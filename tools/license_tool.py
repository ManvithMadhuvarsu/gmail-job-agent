"""Offline license verification for MailAI product builds."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


TOKEN_PREFIX = "mailai_v1"
LICENSE_PATH = Path("data/license.key")


@dataclass(frozen=True)
class LicenseStatus:
    valid: bool
    required: bool
    tier: str
    customer: str
    email: str
    license_id: str
    expires_at: str
    features: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "required": self.required,
            "tier": self.tier,
            "customer": self.customer,
            "email": self.email,
            "license_id": self.license_id,
            "expires_at": self.expires_at,
            "features": self.features,
            "reason": self.reason,
        }


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def license_required() -> bool:
    """Return True when this build should block processing without a valid license."""
    return _enabled(os.getenv("MAILAI_LICENSE_REQUIRED"))


def _b64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _b64_decode(value: str) -> bytes:
    return base64.b64decode(value.strip().encode("ascii"))


def load_license_key() -> str:
    """Load the configured license token from env or data/license.key."""
    env_value = os.getenv("MAILAI_LICENSE_KEY", "").strip()
    if env_value:
        return env_value
    if LICENSE_PATH.exists():
        return LICENSE_PATH.read_text(encoding="utf-8").strip()
    return ""


def save_license_key(token: str) -> None:
    """Persist a license token for local/self-hosted installs."""
    LICENSE_PATH.parent.mkdir(exist_ok=True)
    LICENSE_PATH.write_text(token.strip() + "\n", encoding="utf-8")


def _community_status(reason: str = "License not required for this build.") -> LicenseStatus:
    return LicenseStatus(
        valid=True,
        required=False,
        tier="community",
        customer="Local user",
        email="",
        license_id="",
        expires_at="",
        features=["gmail_labels", "gmail_drafts", "calendar_events", "local_llm"],
        reason=reason,
    )


def _invalid(reason: str) -> LicenseStatus:
    return LicenseStatus(
        valid=False,
        required=license_required(),
        tier="unlicensed",
        customer="",
        email="",
        license_id="",
        expires_at="",
        features=[],
        reason=reason,
    )


def _public_key() -> Ed25519PublicKey | None:
    raw = os.getenv("MAILAI_LICENSE_PUBLIC_KEY", "").strip()
    if not raw:
        return None
    return Ed25519PublicKey.from_public_bytes(_b64_decode(raw))


def verify_license(token: str | None = None) -> LicenseStatus:
    """Verify a MailAI license token."""
    required = license_required()
    token = (token or load_license_key()).strip()
    if not token:
        if required:
            return _invalid("No license key installed.")
        return _community_status()

    public_key = _public_key()
    if public_key is None:
        if required:
            return _invalid("MAILAI_LICENSE_PUBLIC_KEY is not configured.")
        return _community_status("License key present but verification is not configured.")

    try:
        prefix, payload_b64, signature_b64 = token.split(".", 2)
        if prefix != TOKEN_PREFIX:
            return _invalid("Unsupported license format.")
        signed_part = f"{prefix}.{payload_b64}".encode("ascii")
        public_key.verify(_b64url_decode(signature_b64), signed_part)
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (ValueError, json.JSONDecodeError, InvalidSignature):
        return _invalid("License signature is invalid.")
    except Exception as exc:
        return _invalid(f"License verification failed: {exc}")

    expires_at = str(payload.get("expires_at", "")).strip()
    if expires_at:
        try:
            expiry_text = expires_at
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expiry_text):
                expiry_text = f"{expiry_text}T23:59:59+00:00"
            expiry = datetime.fromisoformat(expiry_text.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expiry:
                return _invalid("License has expired.")
        except ValueError:
            return _invalid("License expiry date is invalid.")

    features = payload.get("features") or []
    if not isinstance(features, list):
        features = []

    return LicenseStatus(
        valid=True,
        required=required,
        tier=str(payload.get("tier", "pro")),
        customer=str(payload.get("customer", "")),
        email=str(payload.get("email", "")),
        license_id=str(payload.get("license_id", "")),
        expires_at=expires_at,
        features=[str(item) for item in features],
        reason="License verified.",
    )


def require_valid_license() -> LicenseStatus:
    """Raise when this build requires a license and the current token is invalid."""
    status = verify_license()
    if status.required and not status.valid:
        raise RuntimeError(status.reason)
    return status
