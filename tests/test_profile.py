from tools import gmail_tool
from tools.product_pages import render_profile


def test_google_token_status_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(gmail_tool, "TOKEN_PATH", tmp_path / "token.pickle")
    monkeypatch.delenv("GMAIL_TOKEN_PICKLE_B64", raising=False)

    status = gmail_tool.google_token_status()

    assert status["configured"] is False
    assert status["valid"] is False
    assert status["source"] == "none"


def test_render_profile_points_users_to_oauth_not_raw_tokens():
    html = render_profile(
        status={
            "authorized": False,
            "poll_interval": "180",
            "license": {
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
            },
        },
        google={
            "configured": False,
            "valid": False,
            "missing_scopes": [],
            "source": "none",
            "has_refresh_token": False,
            "expiry": "",
            "scopes": [],
        },
        values={"YOUR_NAME": "Ada"},
    )

    assert "Profile and connections" in html
    assert "Connect or re-authorize Google" in html
    assert "should not paste Gmail access tokens manually" in html
    assert "Ada" in html
