import base64
import json
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.license_tool import TOKEN_PREFIX, verify_license


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _token(private_key, payload: dict) -> str:
    payload_b64 = _b64url(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    signed = f"{TOKEN_PREFIX}.{payload_b64}".encode("ascii")
    return f"{TOKEN_PREFIX}.{payload_b64}.{_b64url(private_key.sign(signed))}"


def _pub(pk) -> str:
    return base64.b64encode(
        pk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii")


def test_days_until_expiry_14_day_warning(monkeypatch):
    pk = Ed25519PrivateKey.generate()
    expires = (datetime.now(timezone.utc) + timedelta(days=12)).isoformat()
    token = _token(pk, {
        "license_id": "lic", "customer": "C", "email": "c@x.com",
        "tier": "pro", "expires_at": expires, "features": [],
    })
    monkeypatch.setenv("MAILAI_LICENSE_PUBLIC_KEY", _pub(pk))
    monkeypatch.setenv("MAILAI_LICENSE_REQUIRED", "true")
    status = verify_license(token)
    assert status.valid
    assert status.expiry_warning_level() == "notice"
    days = status.days_until_expiry()
    assert days is not None and 11 <= days <= 12


def test_critical_warning_within_one_day(monkeypatch):
    pk = Ed25519PrivateKey.generate()
    expires = (datetime.now(timezone.utc) + timedelta(hours=20)).isoformat()
    token = _token(pk, {
        "license_id": "lic", "customer": "C", "email": "c@x.com",
        "tier": "pro", "expires_at": expires, "features": [],
    })
    monkeypatch.setenv("MAILAI_LICENSE_PUBLIC_KEY", _pub(pk))
    monkeypatch.setenv("MAILAI_LICENSE_REQUIRED", "true")
    status = verify_license(token)
    assert status.expiry_warning_level() == "critical"


def test_no_warning_when_far_future(monkeypatch):
    pk = Ed25519PrivateKey.generate()
    expires = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
    token = _token(pk, {
        "license_id": "lic", "customer": "C", "email": "c@x.com",
        "tier": "pro", "expires_at": expires, "features": [],
    })
    monkeypatch.setenv("MAILAI_LICENSE_PUBLIC_KEY", _pub(pk))
    monkeypatch.setenv("MAILAI_LICENSE_REQUIRED", "true")
    status = verify_license(token)
    assert status.expiry_warning_level() is None
