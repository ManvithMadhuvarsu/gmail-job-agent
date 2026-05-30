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
