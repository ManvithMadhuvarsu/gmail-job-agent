import os
import threading
import time
import logging
from urllib.parse import parse_qs
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, JSONResponse
from dotenv import load_dotenv

from google_auth_oauthlib.flow import Flow

from main import run
from tools.gmail_tool import (
    SCOPES,
    CREDENTIALS_PATH,
    TOKEN_PATH,
    delete_local_google_token,
    google_token_status,
    save_token_pickle,
    materialize_token_pickle_from_env,
    _materialize_credentials_from_env,
)
from tools.s3_state import try_persist_file, try_restore_file
from tools.license_tool import license_required, save_license_key, verify_license
from tools.product_pages import (
    SETUP_STEPS,
    render_landing,
    render_license_page,
    render_profile,
    render_setup,
    render_status,
)
from tools.runtime_state import snapshot as runtime_snapshot, record_heartbeat
from tools.audit_log import entries_since, recent_entries, summary_counts
from tools.setup_config import (
    apply_to_env as apply_setup_env,
    load_config as load_setup_config,
    probe_llm,
    update_values as update_setup_values,
)


load_dotenv()
logger = logging.getLogger("railway_app")


def _public_base_url(request: Request) -> str:
    # Prefer explicit base URL (recommended on Railway)
    base = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    if base:
        return base.rstrip("/")
    # Fallback to request host
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}".rstrip("/")


def _oauth_callback_url(request: Request) -> str:
    return f"{_public_base_url(request)}/oauth/callback"


def _build_flow(request: Request) -> Flow:
    # Railway-friendly: allow providing OAuth client JSON via env var
    # (GMAIL_CREDENTIALS_JSON) rather than requiring a pre-mounted file.
    _materialize_credentials_from_env()
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            "credentials.json not found at config/credentials.json. "
            "Provide it either by setting GMAIL_CREDENTIALS_JSON (recommended) "
            "or by mounting /app/config with credentials.json."
        )
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_PATH),
        scopes=SCOPES,
        redirect_uri=_oauth_callback_url(request),
    )
    return flow


def _token_exists() -> bool:
    return TOKEN_PATH.exists()


def _app_status() -> dict:
    return {
        "authorized": _token_exists(),
        "poll_interval": os.getenv("POLL_INTERVAL_MINUTES", "180"),
        "license": verify_license().to_dict(),
    }


def _start_daemon_loop_once():
    # Prevent multiple background threads
    if getattr(_start_daemon_loop_once, "_started", False):
        return
    _start_daemon_loop_once._started = True  # type: ignore[attr-defined]

    def _loop():
        interval_minutes = int(os.getenv("POLL_INTERVAL_MINUTES", "").strip() or 180)
        interval_seconds = interval_minutes * 60
        logger.info(f"Daemon loop started. interval_minutes={interval_minutes}")
        consecutive_errors = 0
        while True:
            try:
                if _token_exists():
                    run()
                    consecutive_errors = 0
                else:
                    logger.warning("No token yet. Waiting for user OAuth at /login")
            except Exception:
                consecutive_errors += 1
                logger.exception(f"Error in daemon loop (consecutive: {consecutive_errors})")
                # Exponential backoff on repeated failures to avoid hammering APIs
                if consecutive_errors >= 3:
                    backoff = min(consecutive_errors * 120, 3600)
                    logger.warning(f"Backing off for {backoff}s after {consecutive_errors} consecutive errors")
                    time.sleep(backoff)
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, name="mailai-daemon", daemon=True)
    t.start()


app = FastAPI()


@app.on_event("startup")
def _startup():
    # Ensure dirs exist for volume mounts
    Path("data").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)
    apply_setup_env()
    _materialize_credentials_from_env()
    # If persistent volumes aren't available, optionally restore token from S3-compatible bucket.
    try_restore_file(TOKEN_PATH)
    # Fall back to env-backed token bootstrap when S3 restore is unavailable.
    if not TOKEN_PATH.exists():
        materialize_token_pickle_from_env()
    if os.getenv("MAILAI_DISABLE_DAEMON", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("Daemon loop disabled by MAILAI_DISABLE_DAEMON.")
        return
    _start_daemon_loop_once()



@app.get("/", response_class=HTMLResponse)
def home():
    return render_landing(_app_status())


@app.get("/status", response_class=HTMLResponse)
def status():
    return render_status(_app_status())


@app.get("/profile", response_class=HTMLResponse)
def profile(disconnected: str | None = None):
    config = load_setup_config()
    return render_profile(
        status=_app_status(),
        google=google_token_status(),
        values=config.values,
        disconnected=disconnected == "1",
    )


@app.get("/license", response_class=HTMLResponse)
def license_page(saved: str | None = None):
    return render_license_page(_app_status(), saved=saved == "1")


_VALID_STEP_KEYS = {key for key, _ in SETUP_STEPS}


def _current_setup_step(requested: str | None, completed: list[str]) -> str:
    if requested in _VALID_STEP_KEYS:
        return requested
    for key, _ in SETUP_STEPS:
        if key not in completed:
            return key
    return "done"


@app.get("/setup", response_class=HTMLResponse)
def setup_page(step: str | None = None):
    config = load_setup_config()
    current = _current_setup_step(step, config.completed_steps)
    status = _app_status()
    return render_setup(
        step=current,
        values=config.values,
        completed=config.completed_steps,
        auth=status["authorized"],
        license_status=status["license"],
    )


@app.post("/setup", response_class=HTMLResponse)
async def setup_save(request: Request):
    form = parse_qs((await request.body()).decode("utf-8", errors="replace"))
    step = (form.get("step") or [""])[0]
    if step not in _VALID_STEP_KEYS:
        return RedirectResponse(url="/setup", status_code=303)

    updates: dict[str, str] = {}
    for key, values in form.items():
        if key in {"step", "probe", "advance"}:
            continue
        # checkboxes only post when checked; treat absence as "off"
        updates[key] = values[0] if values else ""

    # Coerce checkbox-style toggles to true/false strings.
    for toggle_key in ("USE_OLLAMA", "ENABLE_CALENDAR_EVENTS", "MAILAI_DRY_RUN", "DISABLE_DRAFTS"):
        if toggle_key in updates:
            updates[toggle_key] = "true" if updates[toggle_key] else "false"

    # If the user didn't include a toggle in the form for this step, treat as off.
    step_toggles = {
        "llm": ["USE_OLLAMA"],
        "calendar": ["ENABLE_CALENDAR_EVENTS"],
        "safety": ["MAILAI_DRY_RUN", "DISABLE_DRAFTS"],
    }
    for key in step_toggles.get(step, []):
        updates.setdefault(key, "false")

    config = update_setup_values(updates, step=step)
    apply_setup_env(config)

    if step == "llm" and "probe" in form:
        status = _app_status()
        return render_setup(
            step="llm",
            values=config.values,
            completed=config.completed_steps,
            auth=status["authorized"],
            license_status=status["license"],
            llm_probe=probe_llm(),
        )

    next_step_map = {
        "identity": "gmail",
        "gmail":    "llm",
        "llm":      "calendar",
        "calendar": "labels",
        "labels":   "safety",
        "safety":   "done",
        "done":     "done",
    }
    return RedirectResponse(url=f"/setup?step={next_step_map[step]}", status_code=303)


@app.post("/license")
async def save_license(request: Request):
    body = (await request.body()).decode("utf-8", errors="replace")
    token = (parse_qs(body).get("license_key") or [""])[0].strip()
    if token:
        save_license_key(token)
    return RedirectResponse(url="/license?saved=1", status_code=303)


@app.post("/profile/disconnect")
def disconnect_profile_google():
    delete_local_google_token()
    return RedirectResponse(url="/profile?disconnected=1", status_code=303)


@app.get("/health")
def health():
    record_heartbeat()
    state = runtime_snapshot()
    last_24h = entries_since(86400)
    healthy = bool(state.get("last_run_at")) and int(state.get("consecutive_errors", 0)) < 3
    body = {
        "status": "ok" if healthy else "degraded",
        "authorized": _token_exists(),
        "dry_run": bool(state.get("dry_run", False)),
        "last_run_at": state.get("last_run_at"),
        "last_heartbeat_at": state.get("last_heartbeat_at"),
        "last_error": state.get("last_error"),
        "last_error_at": state.get("last_error_at"),
        "consecutive_errors": int(state.get("consecutive_errors", 0)),
        "last_run": {
            "processed": int(state.get("last_processed", 0)),
            "drafts": int(state.get("last_drafts", 0)),
            "calendar_events": int(state.get("last_calendar_events", 0)),
            "errors": int(state.get("last_errors", 0)),
            "duration_seconds": state.get("last_duration_seconds"),
        },
        "totals": {
            "runs": int(state.get("total_runs", 0)),
            "processed": int(state.get("total_processed", 0)),
            "drafts": int(state.get("total_drafts", 0)),
            "calendar_events": int(state.get("total_calendar_events", 0)),
        },
        "last_24h": {
            "actions": len(last_24h),
            "by_category": summary_counts(last_24h),
        },
        "poll_interval_minutes": int(os.getenv("POLL_INTERVAL_MINUTES", "180") or 180),
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


@app.get("/audit")
def audit(limit: int = 100):
    limit = max(1, min(int(limit or 100), 1000))
    return JSONResponse({"entries": recent_entries(limit=limit)})


@app.get("/login")
def login(request: Request):
    status = verify_license()
    if license_required() and not status.valid:
        return RedirectResponse(url="/license")
    flow = _build_flow(request)
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="select_account consent",
    )
    # Store state + PKCE verifier in cookies for callback validation/token exchange.
    resp = RedirectResponse(url=authorization_url)
    resp.set_cookie("oauth_state", state, httponly=True, secure=True, samesite="lax")
    if getattr(flow, "code_verifier", None):
        resp.set_cookie("oauth_code_verifier", flow.code_verifier, httponly=True, secure=True, samesite="lax")
    return resp


@app.get("/oauth/callback")
def oauth_callback(request: Request, code: str | None = None, state: str | None = None):
    if not code:
        return PlainTextResponse("Missing code", status_code=400)

    cookie_state = request.cookies.get("oauth_state")
    if state and cookie_state and state != cookie_state:
        return PlainTextResponse("Invalid OAuth state", status_code=400)
    code_verifier = request.cookies.get("oauth_code_verifier")

    flow = _build_flow(request)
    # Exchange code for tokens — pass code_verifier only if PKCE was used
    fetch_kwargs = {"code": code}
    if code_verifier:
        fetch_kwargs["code_verifier"] = code_verifier
    flow.fetch_token(**fetch_kwargs)
    creds = flow.credentials
    save_token_pickle(creds)
    # Persist token so redeploys don't require re-auth (S3-compatible bucket).
    try_persist_file(TOKEN_PATH)
    resp = RedirectResponse(url="/status")
    resp.delete_cookie("oauth_state")
    resp.delete_cookie("oauth_code_verifier")
    return resp
