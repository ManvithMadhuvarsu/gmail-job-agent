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

STATS_FILE = Path("data/stats.json")

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
    """Persist processed email IDs, trimming oldest if set grows too large."""
    PROCESSED_LOG.parent.mkdir(exist_ok=True)
    id_list = list(ids)
    # Trim if too large — keep the most recent IDs (they're appended at the end)
    if len(id_list) > MAX_PROCESSED_IDS:
        id_list = id_list[-MAX_PROCESSED_IDS:]
    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(id_list, f, indent=2)


def _save_daily_stats(stats: dict, drafts: int, errors: int):
    """Append today's run statistics to a cumulative stats file."""
    today = datetime.now().strftime("%Y-%m-%d")
    all_stats = {}
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, encoding="utf-8") as f:
                all_stats = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    day_entry = all_stats.get(today, {"runs": 0, "emails": {}, "drafts": 0, "errors": 0})
    day_entry["runs"] += 1
    day_entry["drafts"] += drafts
    day_entry["errors"] += errors
    for cat, count in stats.items():
        day_entry["emails"][cat] = day_entry["emails"].get(cat, 0) + count
    all_stats[today] = day_entry

    # Keep only the last 90 days of stats
    if len(all_stats) > 90:
        sorted_keys = sorted(all_stats.keys())
        for key in sorted_keys[:-90]:
            del all_stats[key]

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)


def _thread_has_draft(service, thread_id: str) -> bool:
    """Check if a draft already exists for this thread to avoid duplicates."""
    if not thread_id:
        return False
    try:
        drafts = service.users().drafts().list(userId="me").execute()
        for draft in drafts.get("drafts", []):
            draft_detail = service.users().drafts().get(
                userId="me", id=draft["id"]
            ).execute()
            if draft_detail.get("message", {}).get("threadId") == thread_id:
                return True
    except Exception:
        pass  # If we can't check, proceed with creating the draft
    return False


def print_banner():
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    print(f"\n{Fore.CYAN}{'═' * 60}")
    print(f"  🤖  Job Email Agent")
    print(f"  📅  {now}")
    print(f"{'═' * 60}{Style.RESET_ALL}\n")


def print_result(email: dict, result: dict, draft_saved: bool):
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
    print("\n")


def _is_noreply(sender: str) -> bool:
    """Return True if the sender address looks like a no-reply address."""
    patterns = ["noreply", "no-reply", "donotreply", "do-not-reply", "notifications@", "mailer@", "automated@"]
    return any(p in sender.lower() for p in patterns)


def run():
    logger.info("=" * 50)
    logger.info("MailAI run starting")
    print_banner()

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
        if category in label_ids and label_ids[category]:
            apply_label(service, email["id"], label_ids[category])

        # ── Save draft if needed ──────────────────────────────────────────────
        draft_saved = False
        should_draft = (
            action in {"DRAFT_FEEDBACK", "DRAFT_CONFIRM", "DRAFT_RESPONSE"}
            and result.get("draft_body")
            and not _is_noreply(email.get("sender", ""))
        )

        if should_draft:
            # Duplicate detection: skip if a draft already exists for this thread
            if _thread_has_draft(service, email.get("thread_id")):
                logger.info(f"Draft already exists for thread, skipping: '{email['subject'][:50]}'")
                print(f"  {Fore.YELLOW}  ↳ Draft already exists for this thread, skipping{Style.RESET_ALL}")
            else:
                reply_to = email.get("reply_to") or email.get("sender")
                draft_id = save_draft(
                    service=service,
                    to=reply_to,
                    subject=result.get("draft_subject", f"Re: {email['subject']}"),
                    body=result.get("draft_body", ""),
                    thread_id=email.get("thread_id"),
                )
                if draft_id:
                    draft_saved = True
                    drafts_created += 1

        # ── Mark as processed immediately ─────────────────────────────────────
        processed_ids.add(email["id"])
        save_processed(processed_ids)   # Write after every email for crash-safety

        print_result(email, result, draft_saved)
        logger.info(f"[{category}] {action} | subject='{email['subject'][:60]}' | draft={draft_saved}")

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

    print(f"\n    {Fore.GREEN}📝 Drafts saved:          {drafts_created}{Style.RESET_ALL}")
    print(f"    ⏭️  Skipped (already done): {skipped}")
    if errors:
        print(f"    {Fore.RED}💥 Emails with errors:    {errors}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}✅  Done — check Gmail Drafts before sending!{Style.RESET_ALL}\n")

    logger.info(
        f"Run complete. processed={total_processed}, drafts={drafts_created}, "
        f"skipped={skipped}, errors={errors}"
    )

    # Persist daily stats
    _save_daily_stats(stats, drafts_created, errors)


if __name__ == "__main__":
    run()
