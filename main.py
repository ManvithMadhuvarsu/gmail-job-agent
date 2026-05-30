"""
main.py
Job Email Agent — Main Orchestrator

Run once:
  python main.py

Run continuously (24/7 daemon mode):
  python daemon.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from colorama import Fore, Style, init

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
init(autoreset=True)
sys.path.insert(0, str(Path(__file__).parent))

# ── File logging setup ────────────────────────────────────────────────────────
LOG_DIR = Path("data")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Suppress noisy third-party loggers
logging.getLogger("googleapiclient.discovery").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from tools.gmail_tool import (
    get_gmail_service,
    fetch_recent_emails,
    get_or_create_label,
    apply_label,
    save_draft,
)
from agents.classifier_agent import process_email
from agents.calendar_agent import extract_calendar_event
from tools.calendar_tool import create_calendar_event_once, get_calendar_service
from tools.license_tool import require_valid_license
from tools.audit_log import record_action
from tools.atomic_io import atomic_write_json
from tools.runtime_state import record_cycle

STATS_FILE = Path("data/stats.json")


def _dry_run_enabled() -> bool:
    return os.getenv("MAILAI_DRY_RUN", "").strip().lower() in {"1", "true", "yes", "on"}

# ── Label Map ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "REJECTION": os.getenv("LABEL_REJECTION", "Job/Rejection"),
    "INTERVIEW": os.getenv("LABEL_INTERVIEW", "Job/Interview"),
    "HOLD":      os.getenv("LABEL_HOLD",      "Job/On-Hold"),
    "FOLLOW_UP": os.getenv("LABEL_FOLLOWUP",  "Job/Follow-Up"),
    "APPLIED":   os.getenv("LABEL_APPLIED",   "Job/Applied"),
}

# Terminal colour map
CATEGORY_COLOR = {
    "REJECTION":  Fore.RED,
    "INTERVIEW":  Fore.GREEN,
    "HOLD":       Fore.YELLOW,
    "FOLLOW_UP":  Fore.CYAN,
    "APPLIED":    Fore.BLUE,
    "IRRELEVANT": Fore.WHITE,
}

PROCESSED_LOG = Path("data/processed.json")
# Keep at most this many processed IDs to prevent unbounded file growth
MAX_PROCESSED_IDS = 5000


def load_processed() -> set:
    """Load already-processed email IDs from disk."""
    if PROCESSED_LOG.exists():
        try:
            with open(PROCESSED_LOG, encoding="utf-8") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read processed log, starting fresh: {e}")
    return set()


def save_processed(ids: set):
    """Persist processed email IDs atomically, trimming oldest if too large."""
    id_list = list(ids)
    if len(id_list) > MAX_PROCESSED_IDS:
        id_list = id_list[-MAX_PROCESSED_IDS:]
    atomic_write_json(PROCESSED_LOG, id_list)


def _save_daily_stats(stats: dict, drafts: int, errors: int, calendar_events: int = 0):
    """Append today's run statistics to a cumulative stats file (atomic)."""
    today = datetime.now().strftime("%Y-%m-%d")
    all_stats = {}
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, encoding="utf-8") as f:
                all_stats = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    day_entry = all_stats.get(today, {"runs": 0, "emails": {}, "drafts": 0, "calendar_events": 0, "errors": 0})
    day_entry["runs"] += 1
    day_entry["drafts"] += drafts
    day_entry["calendar_events"] = day_entry.get("calendar_events", 0) + calendar_events
    day_entry["errors"] += errors
    for cat, count in stats.items():
        day_entry["emails"][cat] = day_entry["emails"].get(cat, 0) + count
    all_stats[today] = day_entry

    if len(all_stats) > 90:
        sorted_keys = sorted(all_stats.keys())
        for key in sorted_keys[:-90]:
            del all_stats[key]

    atomic_write_json(STATS_FILE, all_stats)


def _thread_has_draft(service, thread_id: str) -> bool:
    """Check if a draft already exists for this thread to avoid duplicates.

    Uses the drafts.list response which includes message.threadId
    in each entry, avoiding costly per-draft GET calls.
    """
    if not thread_id:
        return False
    try:
        next_page_token = None
        while True:
            resp = service.users().drafts().list(
                userId="me", pageToken=next_page_token
            ).execute()
            for draft in resp.get("drafts", []):
                if draft.get("message", {}).get("threadId") == thread_id:
                    return True
            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break
    except Exception:
        pass  # If we can't check, proceed with creating the draft
    return False


def print_banner():
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    print(f"\n{Fore.CYAN}{'═' * 60}")
    print(f"  🤖  Job Email Agent")
    print(f"  📅  {now}")
    print(f"{'═' * 60}{Style.RESET_ALL}\n")


def print_result(email: dict, result: dict, draft_saved: bool, calendar_saved: bool = False):
    cat    = result.get("category", "IRRELEVANT")
    action = result.get("action", "SKIP")
    color  = CATEGORY_COLOR.get(cat, Fore.WHITE)

    subject = email["subject"][:55]
    sender  = email["sender"][:50]

    print(f"  {color}[{cat:<12}]{Style.RESET_ALL}  {subject}")
    print(f"               From:   {sender}")
    print(f"               Action: {action}", end="")
    if draft_saved:
        print(f"  {Fore.GREEN}→ Draft saved ✓{Style.RESET_ALL}", end="")
    if calendar_saved:
        print(f"  {Fore.GREEN}-> Calendar event added{Style.RESET_ALL}", end="")
    print("\n")


def _is_noreply(sender: str) -> bool:
    """Return True if the sender address looks like a no-reply address."""
    patterns = [
        "noreply", "no-reply", "donotreply", "do-not-reply",
        "notifications@", "mailer@", "mailer-daemon", "automated@",
    ]
    return any(p in sender.lower() for p in patterns)


def _email_has_noreply_details(email: dict) -> bool:
    """Return True when any important mail detail suggests replies should be skipped."""
    patterns = ["noreply", "no-reply", "donotreply", "do-not-reply"]
    fields = [
        email.get("sender", ""),
        email.get("sender_email", ""),
        email.get("sender_name", ""),
        email.get("reply_to", ""),
        email.get("subject", ""),
        email.get("body", ""),
        email.get("snippet", ""),
    ]
    merged = " ".join(str(value).lower() for value in fields if value)
    return any(pattern in merged for pattern in patterns)


def run():
    logger.info("=" * 50)
    logger.info("MailAI run starting")
    print_banner()

    dry_run = _dry_run_enabled()
    if dry_run:
        print(f"{Fore.YELLOW}DRY RUN: classifying only. No labels, drafts, or calendar events will be applied.{Style.RESET_ALL}\n")
        logger.info("MAILAI_DRY_RUN=true — write actions are disabled this run.")

    run_started = time.time()
    run_error: str | None = None

    try:
        license_status = require_valid_license()
        logger.info(f"License status: tier={license_status.tier}, valid={license_status.valid}")
    except RuntimeError as e:
        logger.critical(f"MailAI license check failed: {e}")
        print(f"{Fore.RED}MailAI license required: {e}{Style.RESET_ALL}")
        record_cycle(processed=0, drafts=0, calendar_events=0, errors=1, dry_run=dry_run, error=str(e))
        return

    # ── Authenticate ──────────────────────────────────────────────────────────
    print(f"{Fore.CYAN}🔑  Authenticating with Gmail...{Style.RESET_ALL}")
    try:
        service = get_gmail_service()
        print(f"{Fore.GREEN}✅  Connected to Gmail\n{Style.RESET_ALL}")
    except FileNotFoundError as e:
        print(e)
        logger.critical(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}❌  Auth failed: {e}{Style.RESET_ALL}")
        logger.exception("Auth failed")
        raise

    # ── Ensure Labels Exist ───────────────────────────────────────────────────
    print(f"{Fore.CYAN}🏷️   Setting up Gmail labels...{Style.RESET_ALL}")
    calendar_service = None
    if os.getenv("ENABLE_CALENDAR_EVENTS", "true").strip().lower() not in {"0", "false", "no", "off"}:
        try:
            calendar_service = get_calendar_service()
            print(f"{Fore.GREEN}Calendar connected for important job dates\n{Style.RESET_ALL}")
        except Exception as e:
            logger.warning(f"Calendar integration disabled for this run: {e}")
            print(f"{Fore.YELLOW}Calendar disabled: re-authorize at /login to grant Calendar access.{Style.RESET_ALL}\n")

    label_ids = {}
    for category, label_name in LABEL_MAP.items():
        label_ids[category] = get_or_create_label(service, label_name)
    print()

    # ── Fetch Emails ──────────────────────────────────────────────────────────
    days = int(os.getenv("SCAN_DAYS", "").strip() or 1)
    print(f"{Fore.CYAN}📬  Fetching emails from last {days} day(s)...{Style.RESET_ALL}")
    emails = fetch_recent_emails(service, days=days)
    print(f"    Found {len(emails)} emails\n")
    logger.info(f"Fetched {len(emails)} emails for last {days} day(s)")

    if not emails:
        print(f"{Fore.YELLOW}    No emails to process.{Style.RESET_ALL}\n")
        logger.info("No emails to process. Exiting run.")
        return

    # ── Load Processed Set ────────────────────────────────────────────────────
    processed_ids = load_processed()

    # ── Process Each Email ────────────────────────────────────────────────────
    print(f"{Fore.CYAN}🧠  Processing emails...{Style.RESET_ALL}\n  {'─' * 56}")

    stats = {k: 0 for k in ["REJECTION", "INTERVIEW", "HOLD", "FOLLOW_UP", "APPLIED", "IRRELEVANT"]}
    drafts_created = 0
    calendar_events_created = 0
    skipped = 0
    errors = 0

    for email in emails:
        # ── Skip already processed ────────────────────────────────────────────
        if email["id"] in processed_ids:
            skipped += 1
            continue

        # ── Run agent with retry + rate-limit handling ────────────────────────
        max_retries = 3
        result = None

        for attempt in range(max_retries):
            try:
                result = process_email(email)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str:
                    wait = 60 * (attempt + 1)   # 60s, 120s, 180s
                    print(f"\n{Fore.YELLOW}⚠️  Rate limit hit. Waiting {wait}s...{Style.RESET_ALL}")
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}). Sleeping {wait}s.")
                    time.sleep(wait)
                else:
                    logger.error(f"Error on attempt {attempt + 1} for email '{email['subject']}': {e}")
                    print(f"\n{Fore.RED}❌ Error (attempt {attempt + 1}): {e}{Style.RESET_ALL}")
                    if attempt < max_retries - 1:
                        time.sleep(5)

        if not result:
            errors += 1
            # Mark as processed so we don't retry endlessly on a broken email
            processed_ids.add(email["id"])
            continue

        category = result.get("category", "IRRELEVANT")
        action   = result.get("action",   "SKIP")
        stats[category] = stats.get(category, 0) + 1

        # ── Apply Gmail label ─────────────────────────────────────────────────
        applied_label_id: str | None = None
        applied_label_name: str | None = None
        if category in label_ids and label_ids[category]:
            target_label_id = label_ids[category]
            target_label_name = LABEL_MAP.get(category)
            if dry_run:
                applied_label_id = target_label_id
                applied_label_name = target_label_name
            elif apply_label(service, email["id"], target_label_id):
                applied_label_id = target_label_id
                applied_label_name = target_label_name

        # ── Save draft if needed ──────────────────────────────────────────────
        draft_saved = False
        draft_id_value: str | None = None
        calendar_saved = False
        calendar_event_id: str | None = None
        should_draft = (
            action in {"DRAFT_FEEDBACK", "DRAFT_CONFIRM", "DRAFT_RESPONSE"}
            and result.get("draft_body")
            and not _email_has_noreply_details(email)
        )

        if should_draft:
            if _thread_has_draft(service, email.get("thread_id")):
                logger.info(f"Draft already exists for thread, skipping: '{email['subject'][:50]}'")
                print(f"  {Fore.YELLOW}  ↳ Draft already exists for this thread, skipping{Style.RESET_ALL}")
            elif dry_run:
                draft_saved = True
                drafts_created += 1
            else:
                reply_to = email.get("reply_to") or email.get("sender")
                draft_id_value = save_draft(
                    service=service,
                    to=reply_to,
                    subject=result.get("draft_subject", f"Re: {email['subject']}"),
                    body=result.get("draft_body", ""),
                    thread_id=email.get("thread_id"),
                )
                if draft_id_value:
                    draft_saved = True
                    drafts_created += 1

        # ── Calendar event for important dates ───────────────────────────────
        if calendar_service or dry_run:
            try:
                calendar_event = extract_calendar_event(email, result)
                if calendar_event:
                    if dry_run:
                        calendar_saved = True
                        calendar_events_created += 1
                    else:
                        created, event_id = create_calendar_event_once(calendar_service, email, calendar_event)
                        calendar_saved = bool(created)
                        calendar_event_id = event_id
                        if created:
                            calendar_events_created += 1
                            logger.info(f"Calendar event created for '{email['subject'][:60]}' | event_id={event_id}")
            except Exception as e:
                logger.warning(f"Calendar event extraction failed for '{email['subject'][:60]}': {e}")

        record_action(
            email=email,
            category=category,
            action=action,
            label_id=applied_label_id,
            label_name=applied_label_name,
            draft_id=draft_id_value,
            calendar_event_id=calendar_event_id,
            dry_run=dry_run,
            rule_id=result.get("rule_id"),
        )

        processed_ids.add(email["id"])
        save_processed(processed_ids)

        print_result(email, result, draft_saved, calendar_saved)
        logger.info(
            f"[{category}] {action} | subject='{email['subject'][:60]}' | "
            f"draft={draft_saved} | calendar={calendar_saved} | dry_run={dry_run}"
        )

        # Pacing delay to avoid API burst limits
        time.sleep(1.5)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  {'─' * 56}\n")
    print(f"{Fore.CYAN}📊  Summary{Style.RESET_ALL}")

    total_processed = sum(stats.values())
    for cat, count in stats.items():
        if count > 0:
            color = CATEGORY_COLOR.get(cat, Fore.WHITE)
            print(f"    {color}{cat:<14}{Style.RESET_ALL}  {count}")

    draft_label = "Drafts would-save" if dry_run else "Drafts saved"
    cal_label = "Calendar events (planned)" if dry_run else "Calendar events added"
    print(f"\n    {Fore.GREEN}📝 {draft_label}:        {drafts_created}{Style.RESET_ALL}")
    print(f"    ⏭️  Skipped (already done): {skipped}")
    print(f"    {Fore.GREEN}{cal_label}:   {calendar_events_created}{Style.RESET_ALL}")
    if errors:
        print(f"    {Fore.RED}💥 Emails with errors:    {errors}{Style.RESET_ALL}")
    if dry_run:
        print(f"\n{Fore.YELLOW}Dry run only. Inbox was not modified.{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.CYAN}✅  Done — check Gmail Drafts before sending!{Style.RESET_ALL}\n")

    logger.info(
        f"Run complete. processed={total_processed}, drafts={drafts_created}, "
        f"calendar_events={calendar_events_created}, skipped={skipped}, errors={errors}, dry_run={dry_run}"
    )

    _save_daily_stats(stats, drafts_created, errors, calendar_events_created)
    record_cycle(
        processed=total_processed,
        drafts=drafts_created,
        calendar_events=calendar_events_created,
        errors=errors,
        dry_run=dry_run,
        error=run_error,
        duration_seconds=time.time() - run_started,
    )


if __name__ == "__main__":
    run()
