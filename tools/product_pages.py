"""Polished HTML pages for MailAI's product and onboarding surface."""

from __future__ import annotations

import html
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


_LEVEL_COLORS = {
    "expired":  ("#fbe1d6", "#7c2410", "License expired"),
    "critical": ("#fbe1d6", "#7c2410", "License expires within 24 hours"),
    "warning":  ("#fff0d0", "#805d0d", "License expires within 7 days"),
    "notice":   ("#e5f0df", "#234b3b", "License expires within 14 days"),
}


def _expiry_banner(status: dict[str, Any]) -> str:
    lic = status.get("license") or {}
    level = lic.get("expiry_warning_level")
    if not level or level not in _LEVEL_COLORS:
        return ""
    bg, fg, label = _LEVEL_COLORS[level]
    days = lic.get("days_until_expiry")
    if days is None:
        when = ""
    elif days < 0:
        when = f" ({abs(days)} day(s) ago)"
    else:
        when = f" ({days} day(s) remaining)"
    return (
        f'<div style="background:{bg};color:{fg};border-radius:14px;padding:12px 16px;'
        f'margin:14px 0;font-weight:700;">{_esc(label)}{_esc(when)} — '
        f'<a href="/license" style="color:{fg};text-decoration:underline;">Update license</a></div>'
    )


def _shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_esc(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ink: #10201b;
      --muted: #5f6f68;
      --cream: #f5f0e6;
      --paper: rgba(255, 252, 244, 0.82);
      --edge: rgba(16, 32, 27, 0.16);
      --moss: #234b3b;
      --gold: #d5a642;
      --coral: #e86f54;
      --shadow: 0 24px 80px rgba(20, 41, 34, 0.18);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 13% 13%, rgba(216, 166, 66, 0.32), transparent 28rem),
        radial-gradient(circle at 84% 7%, rgba(184, 216, 216, 0.52), transparent 26rem),
        linear-gradient(135deg, #f5f0e6 0%, #e7efe6 48%, #f7dec9 100%);
      font-family: "Space Grotesk", sans-serif;
      min-height: 100vh;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.35;
      background-image:
        linear-gradient(rgba(16, 32, 27, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(16, 32, 27, 0.05) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, #000, transparent 82%);
    }}
    a {{ color: inherit; }}
    .wrap {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; position: relative; z-index: 1; }}
    .nav {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 26px 0;
    }}
    .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 700; letter-spacing: -0.04em; }}
    .mark {{
      width: 42px; height: 42px; border-radius: 16px;
      background: conic-gradient(from 210deg, var(--moss), var(--gold), var(--coral), var(--moss));
      box-shadow: inset 0 0 0 2px rgba(255,255,255,0.48), 0 14px 30px rgba(35,75,59,0.24);
    }}
    .navlinks {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .navlinks a, .button {{
      text-decoration: none;
      border: 1px solid var(--edge);
      background: rgba(255, 252, 244, 0.62);
      padding: 11px 16px;
      border-radius: 999px;
      font-weight: 700;
      color: var(--ink);
      backdrop-filter: blur(18px);
    }}
    .button.primary {{ background: var(--ink); color: #fffaf0; border-color: var(--ink); }}
    .button.gold {{ background: var(--gold); border-color: rgba(16,32,27,0.12); color: #1f2417; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 36px;
      align-items: center;
      padding: 52px 0 62px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--edge);
      background: rgba(255,255,255,0.45);
      border-radius: 999px;
      padding: 9px 13px;
      color: var(--moss);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    h1 {{
      font-family: "Instrument Serif", serif;
      font-size: clamp(58px, 10vw, 118px);
      line-height: 0.86;
      letter-spacing: -0.06em;
      margin: 24px 0 22px;
      max-width: 820px;
    }}
    h2 {{
      font-family: "Instrument Serif", serif;
      font-size: clamp(40px, 6vw, 70px);
      line-height: 0.95;
      letter-spacing: -0.045em;
      margin: 0 0 18px;
    }}
    h3 {{ margin: 0 0 10px; font-size: 20px; letter-spacing: -0.03em; }}
    p {{ color: var(--muted); line-height: 1.65; font-size: 17px; }}
    .lead {{ max-width: 700px; font-size: 20px; }}
    .actions {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 28px; }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--edge);
      box-shadow: var(--shadow);
      border-radius: 36px;
      padding: 24px;
      backdrop-filter: blur(20px);
    }}
    .inbox-card {{ transform: rotate(1.5deg); min-height: 520px; position: relative; overflow: hidden; }}
    .metric-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 22px 0; }}
    .metric {{ border: 1px solid var(--edge); border-radius: 24px; padding: 18px; background: rgba(255,255,255,0.42); }}
    .metric strong {{ display: block; font-size: 30px; letter-spacing: -0.05em; }}
    .mail {{ margin: 14px 0; padding: 17px; border: 1px solid rgba(16,32,27,0.12); border-radius: 22px; background: rgba(255,255,255,0.62); }}
    .tag {{ display: inline-flex; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 800; background: #e5f0df; color: var(--moss); }}
    .tag.warn {{ background: #fff0d0; color: #805d0d; }}
    .tag.stop {{ background: #ffe3dc; color: #944231; }}
    .section {{ padding: 44px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }}
    .card {{ background: rgba(255,252,244,0.68); border: 1px solid var(--edge); border-radius: 30px; padding: 24px; box-shadow: 0 18px 50px rgba(16,32,27,0.08); }}
    .price {{ font-size: 46px; letter-spacing: -0.07em; font-weight: 800; margin: 12px 0 4px; }}
    .fine {{ font-size: 13px; color: var(--muted); }}
    .checklist {{ list-style: none; padding: 0; margin: 18px 0 0; }}
    .checklist li {{ padding: 9px 0; color: var(--muted); border-top: 1px solid rgba(16,32,27,0.08); }}
    .status-line {{ display: flex; justify-content: space-between; gap: 16px; padding: 13px 0; border-bottom: 1px solid rgba(16,32,27,0.1); }}
    .status-line b {{ text-align: right; }}
    .formbox textarea {{
      width: 100%;
      min-height: 170px;
      border: 1px solid var(--edge);
      border-radius: 22px;
      padding: 16px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      background: rgba(255,255,255,0.7);
    }}
    .formbox button {{ border: 0; cursor: pointer; margin-top: 14px; }}
    code {{ background: rgba(16,32,27,0.08); padding: 3px 6px; border-radius: 7px; }}
    footer {{ padding: 34px 0 44px; color: var(--muted); }}
    @media (max-width: 880px) {{
      .hero, .grid {{ grid-template-columns: 1fr; }}
      .inbox-card {{ transform: none; min-height: auto; }}
      .metric-row {{ grid-template-columns: 1fr; }}
      .nav {{ align-items: flex-start; gap: 14px; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <nav class="nav">
      <div class="brand"><span class="mark"></span><span>MailAI</span></div>
      <div class="navlinks">
        <a href="/">Product</a>
        <a href="/setup">Setup</a>
        <a href="/profile">Profile</a>
        <a href="/status">Status</a>
        <a href="/license">License</a>
        <a href="/login">Connect Gmail</a>
      </div>
    </nav>
    {body}
    <footer>MailAI is approval-first. It labels, drafts, and schedules, but never auto-sends email.</footer>
  </div>
</body>
</html>"""


def render_landing(status: dict[str, Any]) -> str:
    auth = "Authorized" if status["authorized"] else "Needs OAuth"
    license_label = "Licensed" if status["license"]["valid"] else "License needed"
    body = f"""
<main>
  {_expiry_banner(status)}
  <section class="hero">
    <div>
      <div class="eyebrow">Trust-first inbox automation</div>
      <h1>The job-search email agent people can actually trust.</h1>
      <p class="lead">MailAI reads Gmail, labels job emails, queues safe drafts, and adds interviews, assessments, tests, and deadlines to Google Calendar without auto-sending anything.</p>
      <div class="actions">
        <a class="button primary" href="/login">Connect Gmail</a>
        <a class="button gold" href="/license">Install License</a>
        <a class="button" href="/status">Open Status</a>
      </div>
    </div>
    <div class="panel inbox-card">
      <span class="tag">Live agent surface</span>
      <div class="metric-row">
        <div class="metric"><strong>0</strong><span class="fine">Auto-sent emails</span></div>
        <div class="metric"><strong>1:1</strong><span class="fine">Gmail OAuth</span></div>
        <div class="metric"><strong>24h</strong><span class="fine">Reminder layer</span></div>
      </div>
      <div class="mail"><span class="tag">INTERVIEW</span><h3>Recruiter screen confirmed</h3><p>Label applied, reply drafted for review, Calendar event created with reminders.</p></div>
      <div class="mail"><span class="tag warn">ASSESSMENT</span><h3>Coding assessment due</h3><p>Deadline extracted from Gmail and added as a Google Calendar reminder.</p></div>
      <div class="mail"><span class="tag stop">NO-REPLY</span><h3>Transactional sender detected</h3><p>MailAI skips reply drafts when no-reply patterns appear anywhere in the email details.</p></div>
    </div>
  </section>
  <section class="section">
    <h2>Sell the safe version first.</h2>
    <div class="grid">
      <div class="card"><h3>MailAI Local</h3><p>Runs on the buyer's machine with their Gmail OAuth and their LLM keys. Best for launching fast without storing customer inboxes.</p></div>
      <div class="card"><h3>MailAI Pro</h3><p>Self-hosted Docker, license key, Calendar backfills, and priority setup support for serious job seekers and recruiters.</p></div>
      <div class="card"><h3>MailAI SaaS</h3><p>Hosted multi-user version later, after Google OAuth verification, policy pages, encrypted token storage, billing, and audits.</p></div>
    </div>
  </section>
  <section class="section">
    <h2>Pricing model</h2>
    <div class="grid">
      <div class="card"><h3>Local</h3><div class="price">$79</div><div class="fine">per year</div><ul class="checklist"><li>Gmail labels and safe drafts</li><li>Calendar reminders</li><li>Local/self-hosted install</li></ul></div>
      <div class="card"><h3>Pro</h3><div class="price">$199</div><div class="fine">per year</div><ul class="checklist"><li>Everything in Local</li><li>Calendar backfill command</li><li>Priority setup support</li></ul></div>
      <div class="card"><h3>Concierge</h3><div class="price">$999</div><div class="fine">setup plus support</div><ul class="checklist"><li>Done-for-you deployment</li><li>Ollama/Groq configuration</li><li>Calendar and Gmail verification</li></ul></div>
    </div>
  </section>
  <section class="section panel">
    <h2>Current install</h2>
    <div class="status-line"><span>Gmail</span><b>{_esc(auth)}</b></div>
    <div class="status-line"><span>License</span><b>{_esc(license_label)}</b></div>
    <div class="status-line"><span>Tier</span><b>{_esc(status["license"].get("tier"))}</b></div>
    <div class="status-line"><span>Polling</span><b>{_esc(status["poll_interval"])} minutes</b></div>
  </section>
</main>"""
    return _shell("MailAI - Trust-first Gmail agent", body)


def render_status(status: dict[str, Any]) -> str:
    license_status = status["license"]
    body = f"""
<main>
  {_expiry_banner(status)}
  <section class="section panel">
    <div class="eyebrow">Install status</div>
    <h2>MailAI Control Surface</h2>
    <p>This page shows whether the product build can process Gmail and Calendar events.</p>
    <div class="status-line"><span>Gmail OAuth</span><b>{'Authorized' if status["authorized"] else 'Not authorized'}</b></div>
    <div class="status-line"><span>License valid</span><b>{'Yes' if license_status["valid"] else 'No'}</b></div>
    <div class="status-line"><span>License required</span><b>{'Yes' if license_status["required"] else 'No'}</b></div>
    <div class="status-line"><span>Tier</span><b>{_esc(license_status["tier"])}</b></div>
    <div class="status-line"><span>Customer</span><b>{_esc(license_status["customer"])}</b></div>
    <div class="status-line"><span>Expires</span><b>{_esc(license_status["expires_at"] or 'Not set')}</b></div>
    <div class="status-line"><span>Reason</span><b>{_esc(license_status["reason"])}</b></div>
    <div class="actions">
      <a class="button primary" href="/login">Connect Gmail</a>
      <a class="button gold" href="/license">Update License</a>
    </div>
  </section>
</main>"""
    return _shell("MailAI Status", body)


def _scope_rows(google: dict[str, Any]) -> str:
    scopes = google.get("scopes") or []
    missing = set(google.get("missing_scopes") or [])
    if not scopes:
        return '<p class="fine">No granted scopes found yet. Connect Google to authorize Gmail and Calendar.</p>'
    rows = []
    for scope in scopes:
        marker = "Missing" if scope in missing else "Granted"
        rows.append(
            f'<div class="status-line"><span>{_esc(scope)}</span><b>{_esc(marker)}</b></div>'
        )
    for scope in sorted(missing.difference(scopes)):
        rows.append(
            f'<div class="status-line"><span>{_esc(scope)}</span><b>Missing</b></div>'
        )
    return "".join(rows)


def render_profile(
    *,
    status: dict[str, Any],
    google: dict[str, Any],
    values: dict[str, str],
    disconnected: bool = False,
) -> str:
    license_status = status["license"]
    google_label = "Connected" if google.get("valid") and not google.get("missing_scopes") else "Needs attention"
    if not google.get("configured"):
        google_label = "Not connected"
    disconnect_notice = (
        '<p><b>Google token removed.</b> Connect Google again before running MailAI.</p>'
        if disconnected else ""
    )
    env_note = (
        "<p class=\"fine\">This install also has GMAIL_TOKEN_PICKLE_B64 configured. "
        "If a hosted deploy keeps reconnecting automatically, remove that env variable too.</p>"
        if google.get("env_token") else ""
    )
    body = f"""
<main>
  {_expiry_banner(status)}
  <section class="hero">
    <div>
      <div class="eyebrow">Profile and connections</div>
      <h1>Connect the user, not a raw token.</h1>
      <p class="lead">MailAI should get Gmail and Calendar access through Google's OAuth consent screen. Users should not paste Gmail access tokens manually; those tokens expire, rotate, and are tied to the OAuth client.</p>
      <div class="actions">
        <a class="button primary" href="/login">Connect or re-authorize Google</a>
        <a class="button gold" href="/setup">Open setup wizard</a>
        <a class="button" href="/license">Install license</a>
      </div>
    </div>
    <div class="panel">
      {disconnect_notice}
      <h3>Connection summary</h3>
      <div class="status-line"><span>Google OAuth</span><b>{_esc(google_label)}</b></div>
      <div class="status-line"><span>Token source</span><b>{_esc(google.get("source") or "none")}</b></div>
      <div class="status-line"><span>Refresh token</span><b>{'Available' if google.get("has_refresh_token") else 'Not available'}</b></div>
      <div class="status-line"><span>Expires</span><b>{_esc(google.get("expiry") or "Unknown")}</b></div>
      <div class="status-line"><span>License</span><b>{_esc(license_status.get("tier"))} / {'valid' if license_status.get("valid") else 'invalid'}</b></div>
      {env_note}
      <form method="post" action="/profile/disconnect" style="margin-top:18px;">
        <button class="button" type="submit">Disconnect local Google token</button>
      </form>
    </div>
  </section>

  <section class="section">
    <div class="grid">
      <div class="card">
        <h3>User identity</h3>
        <div class="status-line"><span>Name</span><b>{_esc(values.get("YOUR_NAME") or "Not set")}</b></div>
        <div class="status-line"><span>Email</span><b>{_esc(values.get("YOUR_EMAIL") or "Not set")}</b></div>
        <div class="status-line"><span>Phone</span><b>{_esc(values.get("YOUR_PHONE") or "Not set")}</b></div>
        <div class="status-line"><span>LinkedIn</span><b>{_esc(values.get("YOUR_LINKEDIN") or "Not set")}</b></div>
        <div class="actions"><a class="button" href="/setup?step=identity">Edit identity</a></div>
      </div>
      <div class="card">
        <h3>Access model</h3>
        <p>Google access is automated after the user clicks Connect Google and approves the requested scopes. The profile page stores only the resulting OAuth credential file locally.</p>
        <p>License keys are different: those can be pasted safely at <a href="/license">/license</a> because they are signed MailAI product tokens, not Google tokens.</p>
      </div>
      <div class="card">
        <h3>OAuth scopes</h3>
        {_scope_rows(google)}
      </div>
    </div>
  </section>
</main>"""
    return _shell("MailAI Profile", body)


def render_license_page(status: dict[str, Any], saved: bool = False) -> str:
    notice = "<p><b>License saved.</b> Restart is not required.</p>" if saved else ""
    license_status = status["license"]
    body = f"""
<main>
  <section class="hero">
    <div>
      <div class="eyebrow">License gate</div>
      <h1>Turn MailAI into a paid local product.</h1>
      <p class="lead">Paste a signed MailAI license token here. The app verifies it with a public key, so the private issuing key never ships to customers.</p>
    </div>
    <div class="panel formbox">
      {notice}
      <form method="post" action="/license">
        <label for="license_key"><b>License key</b></label>
        <textarea id="license_key" name="license_key" placeholder="mailai_v1..."></textarea>
        <button class="button primary" type="submit">Save License</button>
      </form>
      <div class="status-line"><span>Current status</span><b>{'Valid' if license_status["valid"] else 'Invalid'}</b></div>
      <div class="status-line"><span>Tier</span><b>{_esc(license_status["tier"])}</b></div>
      <div class="status-line"><span>Reason</span><b>{_esc(license_status["reason"])}</b></div>
    </div>
  </section>
  <section class="section panel">
    <h2>Seller setup</h2>
    <p>Generate a private/public license keypair with <code>python scripts/generate_license.py init-keys</code>. Keep the private key secret. Put the public key into <code>MAILAI_LICENSE_PUBLIC_KEY</code> and set <code>MAILAI_LICENSE_REQUIRED=true</code> for paid builds.</p>
  </section>
</main>"""
    return _shell("MailAI License", body)


SETUP_STEPS = [
    ("identity", "Your details"),
    ("gmail",    "Connect Gmail"),
    ("llm",      "Choose an LLM"),
    ("calendar", "Calendar reminders"),
    ("labels",   "Label prefix"),
    ("safety",   "Safety defaults"),
    ("done",     "Finish"),
]


def _step_index(step: str) -> int:
    for i, (key, _label) in enumerate(SETUP_STEPS):
        if key == step:
            return i
    return 0


def _setup_progress(step: str, completed: list[str]) -> str:
    current_idx = _step_index(step)
    cells = []
    for i, (key, label) in enumerate(SETUP_STEPS):
        if key in completed:
            bg, fg = "#234b3b", "#fffaf0"
        elif i == current_idx:
            bg, fg = "#d5a642", "#1f2417"
        else:
            bg, fg = "rgba(255,255,255,0.55)", "#5f6f68"
        cells.append(
            f'<div style="flex:1;text-align:center;padding:8px 6px;border-radius:14px;'
            f'background:{bg};color:{fg};font-weight:700;font-size:13px;">'
            f'{i+1}. {_esc(label)}</div>'
        )
    return f'<div style="display:flex;gap:6px;margin:18px 0 24px;flex-wrap:wrap;">{"".join(cells)}</div>'


def _btn_row(prev: str | None, next_label: str = "Continue") -> str:
    prev_html = (
        f'<a class="button" href="/setup?step={_esc(prev)}">Back</a>' if prev else ""
    )
    return (
        f'<div class="actions" style="margin-top:18px;">'
        f'{prev_html}'
        f'<button class="button primary" type="submit">{_esc(next_label)}</button>'
        f'</div>'
    )


def _input(name: str, value: str, *, label: str, placeholder: str = "", type_: str = "text", helper: str = "") -> str:
    helper_html = f'<div class="fine">{_esc(helper)}</div>' if helper else ""
    return f"""
      <label for="{_esc(name)}" style="display:block;margin-top:14px;"><b>{_esc(label)}</b></label>
      <input id="{_esc(name)}" name="{_esc(name)}" type="{_esc(type_)}" value="{_esc(value)}"
             placeholder="{_esc(placeholder)}"
             style="width:100%;padding:12px 14px;border:1px solid var(--edge);border-radius:14px;
                    background:rgba(255,255,255,0.7);margin-top:6px;" />
      {helper_html}
    """


def _toggle(name: str, value: str, *, label: str, helper: str = "") -> str:
    checked = "checked" if str(value).strip().lower() in {"1", "true", "yes", "on"} else ""
    helper_html = f'<div class="fine">{_esc(helper)}</div>' if helper else ""
    return f"""
      <label style="display:flex;gap:10px;align-items:center;margin-top:14px;">
        <input type="checkbox" name="{_esc(name)}" value="true" {checked} />
        <span><b>{_esc(label)}</b></span>
      </label>
      {helper_html}
    """


def render_setup(
    *,
    step: str,
    values: dict[str, str],
    completed: list[str],
    auth: bool,
    license_status: dict[str, Any],
    llm_probe: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    progress = _setup_progress(step, completed)
    error_html = (
        f'<div style="background:#fbe1d6;color:#7c2410;border-radius:14px;padding:10px 14px;'
        f'margin-bottom:14px;font-weight:700;">{_esc(error)}</div>' if error else ""
    )

    if step == "identity":
        prev = None
        content = f"""
        <h2>Tell us who's running this inbox.</h2>
        <p>MailAI signs drafts with your name and contact details. None of this leaves your machine in Local mode.</p>
        <form method="post" action="/setup">
          <input type="hidden" name="step" value="identity" />
          {_input("YOUR_NAME",     values.get("YOUR_NAME", ""),     label="Full name",    placeholder="Ada Lovelace")}
          {_input("YOUR_EMAIL",    values.get("YOUR_EMAIL", ""),    label="Reply-to email", placeholder="ada@example.com", type_="email")}
          {_input("YOUR_PHONE",    values.get("YOUR_PHONE", ""),    label="Phone (optional)",   placeholder="+1 555 0100")}
          {_input("YOUR_LINKEDIN", values.get("YOUR_LINKEDIN", ""), label="LinkedIn (optional)", placeholder="linkedin.com/in/ada")}
          {_btn_row(prev)}
        </form>
        """
    elif step == "gmail":
        gmail_state = "connected" if auth else "not connected"
        action_label = "Re-connect Gmail" if auth else "Connect Gmail"
        content = f"""
        <h2>Connect your Gmail account.</h2>
        <p>MailAI uses your Gmail OAuth — your messages never reach a MailAI server. Current state: <b>{_esc(gmail_state)}</b>.</p>
        <div class="actions">
          <a class="button primary" href="/login">{_esc(action_label)}</a>
          <a class="button" href="/setup?step=llm">Skip for now</a>
        </div>
        <form method="post" action="/setup">
          <input type="hidden" name="step" value="gmail" />
          {_btn_row("identity", next_label="Continue")}
        </form>
        """
    elif step == "llm":
        use_ollama = str(values.get("USE_OLLAMA", "false")).lower() == "true"
        probe_html = ""
        if llm_probe:
            color = "#234b3b" if llm_probe.get("ok") else "#7c2410"
            bg = "#e5f0df" if llm_probe.get("ok") else "#fbe1d6"
            probe_html = (
                f'<div style="background:{bg};color:{color};border-radius:14px;padding:10px 14px;'
                f'margin-top:14px;font-weight:700;">{_esc(llm_probe.get("detail", ""))}</div>'
            )
        content = f"""
        <h2>Choose how MailAI thinks.</h2>
        <p>Local Ollama keeps every classification on your machine. Groq is fast and free for low volumes but the email body is sent to Groq's API.</p>
        <form method="post" action="/setup">
          <input type="hidden" name="step" value="llm" />
          {_toggle("USE_OLLAMA", "true" if use_ollama else "", label="Use local Ollama", helper="Recommended for privacy-focused buyers")}
          {_input("OLLAMA_MODEL",    values.get("OLLAMA_MODEL", "llama3"),                    label="Ollama model")}
          {_input("OLLAMA_BASE_URL", values.get("OLLAMA_BASE_URL", "http://localhost:11434"), label="Ollama base URL")}
          {_input("GROQ_API_KEY",    values.get("GROQ_API_KEY", ""),                          label="Groq API key (fallback)", type_="password", helper="Used when Ollama is off or unreachable.")}
          {probe_html}
          <div class="actions" style="margin-top:18px;">
            <a class="button" href="/setup?step=gmail">Back</a>
            <button class="button" type="submit" name="probe" value="1">Test connection</button>
            <button class="button primary" type="submit" name="advance" value="1">Continue</button>
          </div>
        </form>
        """
    elif step == "calendar":
        enable = str(values.get("ENABLE_CALENDAR_EVENTS", "true")).lower() == "true"
        content = f"""
        <h2>Calendar reminders.</h2>
        <p>MailAI can add interviews, assessments, and deadlines to Google Calendar. Skip this if you only want Gmail labels.</p>
        <form method="post" action="/setup">
          <input type="hidden" name="step" value="calendar" />
          {_toggle("ENABLE_CALENDAR_EVENTS", "true" if enable else "", label="Create Google Calendar events")}
          {_input("GOOGLE_CALENDAR_ID", values.get("GOOGLE_CALENDAR_ID", "primary"), label="Calendar ID", placeholder="primary")}
          {_input("CALENDAR_TIMEZONE",  values.get("CALENDAR_TIMEZONE", "Asia/Kolkata"), label="Default timezone", placeholder="Asia/Kolkata")}
          {_btn_row("llm")}
        </form>
        """
    elif step == "labels":
        content = f"""
        <h2>Pick a label prefix.</h2>
        <p>MailAI groups your job mail under labels. Defaults are <code>Job/Rejection</code>, <code>Job/Interview</code>, etc. Change the prefix below if you prefer a different folder.</p>
        <form method="post" action="/setup">
          <input type="hidden" name="step" value="labels" />
          {_input("LABEL_REJECTION", values.get("LABEL_REJECTION", "Job/Rejection"), label="Rejection")}
          {_input("LABEL_INTERVIEW", values.get("LABEL_INTERVIEW", "Job/Interview"), label="Interview")}
          {_input("LABEL_HOLD",      values.get("LABEL_HOLD",      "Job/On-Hold"),   label="On hold")}
          {_input("LABEL_FOLLOWUP",  values.get("LABEL_FOLLOWUP",  "Job/Follow-Up"), label="Follow-up")}
          {_input("LABEL_APPLIED",   values.get("LABEL_APPLIED",   "Job/Applied"),   label="Applied")}
          {_btn_row("calendar")}
        </form>
        """
    elif step == "safety":
        dry = str(values.get("MAILAI_DRY_RUN", "true")).lower() == "true"
        drafts_off = str(values.get("DISABLE_DRAFTS", "false")).lower() == "true"
        content = f"""
        <h2>Safety defaults.</h2>
        <p>We recommend starting in dry-run mode so you can watch MailAI for a few cycles before it touches your inbox.</p>
        <form method="post" action="/setup">
          <input type="hidden" name="step" value="safety" />
          {_toggle("MAILAI_DRY_RUN", "true" if dry else "", label="Start in dry-run mode", helper="Classify and audit-log only — no labels, drafts, or calendar events.")}
          {_toggle("DISABLE_DRAFTS", "true" if drafts_off else "", label="Never create reply drafts", helper="Labels and calendar events still work.")}
          {_input("POLL_INTERVAL_MINUTES", values.get("POLL_INTERVAL_MINUTES", "180"), label="Polling interval (minutes)", placeholder="180")}
          {_input("SCAN_DAYS",             values.get("SCAN_DAYS", "1"),               label="Look-back window (days)",     placeholder="1")}
          {_btn_row("labels", next_label="Finish setup")}
        </form>
        """
    else:  # done
        license_label = "Licensed" if license_status.get("valid") else "License needed"
        auth_label = "Connected" if auth else "Not connected"
        dry = str(values.get("MAILAI_DRY_RUN", "false")).lower() == "true"
        content = f"""
        <h2>You're set up.</h2>
        <p>{('MailAI is running in dry-run mode. Watch a few cycles, then return here to turn it off.' if dry else 'MailAI will start processing on the next daemon cycle.')}</p>
        <div class="status-line"><span>Gmail</span><b>{_esc(auth_label)}</b></div>
        <div class="status-line"><span>License</span><b>{_esc(license_label)}</b></div>
        <div class="status-line"><span>Dry run</span><b>{'On' if dry else 'Off'}</b></div>
        <div class="actions" style="margin-top:18px;">
          <a class="button primary" href="/status">Open status</a>
          <a class="button" href="/">Back to product</a>
        </div>
        """

    body = f"""
<main>
  <section class="section panel">
    <div class="eyebrow">Onboarding</div>
    <h1 style="font-size:clamp(36px,5vw,52px);margin:12px 0 4px;">MailAI setup wizard</h1>
    {progress}
    {error_html}
    {content}
  </section>
</main>"""
    return _shell("MailAI Setup", body)
