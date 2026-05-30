"""Generate MailAI license signing keys and signed customer licenses."""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.license_tool import TOKEN_PREFIX


DEFAULT_FEATURES = [
    "gmail_labels",
    "gmail_drafts",
    "calendar_events",
    "calendar_backfill",
    "local_llm",
]


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _raw_private_key(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _raw_public_key(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def init_keys(args: argparse.Namespace) -> int:
    private_key = Ed25519PrivateKey.generate()
    private_key_file = Path(args.private_key_file)
    if private_key_file.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing private key: {private_key_file}")

    private_key_file.parent.mkdir(parents=True, exist_ok=True)
    private_key_file.write_text(_b64(_raw_private_key(private_key)) + "\n", encoding="utf-8")

    print(f"Private key saved to: {private_key_file}")
    print("Keep it secret. Do not commit it.")
    print("\nSet this in paid builds:")
    print(f"MAILAI_LICENSE_PUBLIC_KEY={_b64(_raw_public_key(private_key))}")
    return 0


def _load_private_key(path: str) -> Ed25519PrivateKey:
    raw = base64.b64decode(Path(path).read_text(encoding="utf-8").strip())
    return Ed25519PrivateKey.from_private_bytes(raw)


def issue(args: argparse.Namespace) -> int:
    private_key = _load_private_key(args.private_key_file)
    features = [item.strip() for item in args.features.split(",") if item.strip()]
    payload = {
        "license_id": args.license_id or f"mailai_{secrets.token_hex(8)}",
        "customer": args.customer,
        "email": args.email,
        "tier": args.tier,
        "expires_at": args.expires_at,
        "features": features,
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url(payload_json)
    signed_part = f"{TOKEN_PREFIX}.{payload_b64}".encode("ascii")
    signature_b64 = _b64url(private_key.sign(signed_part))
    print(f"{TOKEN_PREFIX}.{payload_b64}.{signature_b64}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MailAI license utility")
    sub = parser.add_subparsers(dest="command", required=True)

    keys = sub.add_parser("init-keys", help="Create a MailAI license signing keypair")
    keys.add_argument("--private-key-file", default="data/license_private.key")
    keys.add_argument("--force", action="store_true")
    keys.set_defaults(func=init_keys)

    lic = sub.add_parser("issue", help="Issue a signed customer license")
    lic.add_argument("--private-key-file", default="data/license_private.key")
    lic.add_argument("--customer", required=True)
    lic.add_argument("--email", required=True)
    lic.add_argument("--tier", default="pro")
    lic.add_argument("--expires-at", required=True, help="YYYY-MM-DD or ISO datetime, e.g. 2027-05-29")
    lic.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    lic.add_argument("--license-id", default="")
    lic.set_defaults(func=issue)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
