from tools.product_pages import render_status


def _license_status():
    return {
        "valid": True,
        "required": False,
        "tier": "community",
        "customer": "Local user",
        "email": "",
        "license_id": "",
        "expires_at": "",
        "features": [],
        "reason": "ok",
        "days_until_expiry": None,
        "expiry_warning_level": None,
    }


def test_render_status_exposes_manual_run_control():
    html = render_status(
        {
            "authorized": True,
            "poll_interval": "180",
            "license": _license_status(),
            "manual_notice": "Manual run started.",
            "runtime": {
                "last_run_at": "2026-06-03T10:00:00+00:00",
                "last_processed": 12,
                "last_drafts": 2,
                "last_calendar_events": 1,
                "last_errors": 0,
                "last_duration_seconds": 45.2,
                "manual_run": {
                    "status": "running",
                    "dry_run": True,
                    "started_at": "2026-06-03T10:01:00+00:00",
                    "finished_at": None,
                    "error": None,
                },
            },
        }
    )

    assert 'action="/run-now"' in html
    assert 'name="dry_run"' in html
    assert "Run MailAI now" in html
    assert "Running now" in html
    assert "Manual run started." in html
    assert "45.2s" in html
