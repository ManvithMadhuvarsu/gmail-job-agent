import base64
import json
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.license_tool import TOKEN_PREFIX, verify_license


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _license_token(private_key, payload: dict) -> str:
    payload_b64 = _b64url(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    signed_part = f"{TOKEN_PREFIX}.{payload_b64}".encode("ascii")
    return f"{TOKEN_PREFIX}.{payload_b64}.{_b64url(private_key.sign(signed_part))}"


def _public_key_b64(private_key) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def test_license_not_required_defaults_to_community(monkeypatch):
    monkeypatch.delenv("MAILAI_LICENSE_REQUIRED", raising=False)
    monkeypatch.delenv("MAILAI_LICENSE_KEY", raising=False)
    monkeypatch.delenv("MAILAI_LICENSE_PUBLIC_KEY", raising=False)

    status = verify_license("")

    assert status.valid is True
    assert status.tier == "community"


def test_signed_license_verifies(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    token = _license_token(private_key, {
        "license_id": "lic_123",
        "customer": "Example Customer",
        "email": "buyer@example.com",
        "tier": "pro",
        "expires_at": expires,
        "features": ["calendar_events"],
    })
    monkeypatch.setenv("MAILAI_LICENSE_PUBLIC_KEY", _public_key_b64(private_key))
    monkeypatch.setenv("MAILAI_LICENSE_REQUIRED", "true")

    status = verify_license(token)

    assert status.valid is True
    assert status.tier == "pro"
    assert status.email == "buyer@example.com"


def test_expired_license_fails(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    expires = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    token = _license_token(private_key, {
        "license_id": "lic_expired",
        "customer": "Example Customer",
        "email": "buyer@example.com",
        "tier": "pro",
        "expires_at": expires,
        "features": ["calendar_events"],
    })
    monkeypatch.setenv("MAILAI_LICENSE_PUBLIC_KEY", _public_key_b64(private_key))
    monkeypatch.setenv("MAILAI_LICENSE_REQUIRED", "true")

    status = verify_license(token)

    assert status.valid is False
    assert "expired" in status.reason.lower()


def test_date_only_expiry_is_valid_through_that_day(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    expires = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    token = _license_token(private_key, {
        "license_id": "lic_date",
        "customer": "Example Customer",
        "email": "buyer@example.com",
        "tier": "pro",
        "expires_at": expires,
        "features": ["calendar_events"],
    })
    monkeypatch.setenv("MAILAI_LICENSE_PUBLIC_KEY", _public_key_b64(private_key))
    monkeypatch.setenv("MAILAI_LICENSE_REQUIRED", "true")

    status = verify_license(token)

    assert status.valid is True
