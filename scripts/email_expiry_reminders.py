"""Send (or log) renewal reminder emails based on installed licenses.

Run from cron or a scheduled task once per day:

    python scripts/email_expiry_reminders.py --dry-run

Two delivery modes:
  - Resend: set RESEND_API_KEY (and FROM_EMAIL).
  - SMTP:   set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, FROM_EMAIL.

If neither is configured, this script prints what it WOULD have sent. That way
sellers can wire a real provider when ready without changing the calling cron.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.license_tool import verify_license  # noqa: E402

logger = logging.getLogger("mailai.expiry")


REMIND_AT_DAYS = (14, 7, 1)


def _subject(days: int) -> str:
    if days <= 0:
        return "Your MailAI license has expired"
    if days == 1:
        return "Your MailAI license expires in 1 day"
    return f"Your MailAI license expires in {days} days"


def _body(name: str, days: int, expires_at: str) -> str:
    if days <= 0:
        opener = "Your MailAI license has expired"
    else:
        opener = f"Your MailAI license will expire in {days} day(s)"
    return (
        f"Hi {name or 'there'},\n\n"
        f"{opener} (on {expires_at}).\n\n"
        "Renew here: https://mailai.app/license\n\n"
        "If you no longer use MailAI, you can ignore this message — your inbox is untouched.\n\n"
        "— MailAI"
    )


def _send_resend(to: str, subject: str, body: str) -> bool:
    key = os.getenv("RESEND_API_KEY", "").strip()
    sender = os.getenv("FROM_EMAIL", "").strip()
    if not (key and sender):
        return False
    try:
        import urllib.request
        payload = json.dumps({
            "from": sender,
            "to": [to],
            "subject": subject,
            "text": body,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        logger.warning(f"Resend send failed: {exc}")
        return False


def _send_smtp(to: str, subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        return False
    port = int(os.getenv("SMTP_PORT", "587") or 587)
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = os.getenv("FROM_EMAIL", user or "").strip()
    if not sender:
        return False
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
        return True
    except Exception as exc:
        logger.warning(f"SMTP send failed: {exc}")
        return False


def send_reminder(to: str, name: str, days: int, expires_at: str, *, dry_run: bool) -> str:
    subject = _subject(days)
    body = _body(name, days, expires_at)
    if dry_run:
        print(f"[DRY-RUN] to={to}\n  subject={subject}\n")
        return "dry-run"
    if _send_resend(to, subject, body):
        return "resend"
    if _send_smtp(to, subject, body):
        return "smtp"
    print(f"[LOG-ONLY] no provider configured. to={to} subject={subject!r}")
    return "log-only"


def main() -> int:
    parser = argparse.ArgumentParser(description="Email license-expiry reminders")
    parser.add_argument("--dry-run", action="store_true", help="Print emails without sending")
    parser.add_argument("--all", action="store_true", help="Send even outside the 14/7/1 windows")
    args = parser.parse_args()

    status = verify_license()
    if not status.valid and status.required:
        print("Installed license is not valid; renewal reminder still applies.")

    days = status.days_until_expiry()
    if days is None:
        print("No expiry on this license; nothing to remind.")
        return 0
    if not args.all and days not in REMIND_AT_DAYS and not (days <= 0):
        print(f"No reminder needed today (days_until_expiry={days}).")
        return 0
    if not status.email:
        print("License has no contact email; cannot send reminder.")
        return 1

    result = send_reminder(
        to=status.email,
        name=status.customer or "",
        days=days,
        expires_at=status.expires_at,
        dry_run=args.dry_run,
    )
    print(f"Reminder delivery: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
