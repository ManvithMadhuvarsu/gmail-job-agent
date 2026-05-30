import os
import threading
import time
import logging
from urllib.parse import parse_qs
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from dotenv import load_dotenv

from google_auth_oauthlib.flow import Flow

from main import run
from tools.gmail_tool import (
    SCOPES,
    CREDENTIALS_PATH,
    TOKEN_PATH,
    save_token_pickle,
    materialize_token_pickle_from_env,
    _materialize_credentials_from_env,
)
from tools.s3_state import try_persist_file, try_restore_file
from tools.license_tool import license_required, save_license_key, verify_license
from tools.product_pages import render_landing, render_license_page, render_status


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


@app.get("/license", response_class=HTMLResponse)
def license_page(saved: str | None = None):
    return render_license_page(_app_status(), saved=saved == "1")


@app.post("/license")
async def save_license(request: Request):
    body = (await request.body()).decode("utf-8", errors="replace")
    token = (parse_qs(body).get("license_key") or [""])[0].strip()
    if token:
        save_license_key(token)
    return RedirectResponse(url="/license?saved=1", status_code=303)


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


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
