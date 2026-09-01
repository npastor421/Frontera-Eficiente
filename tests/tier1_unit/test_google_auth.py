"""
Unit Tests for Google OAuth Authentication and User Identity.
"""

import pytest
from src.auth.google_auth import (
    get_oauth_config,
    is_oauth_configured,
    get_google_auth_url,
    get_active_user_id,
)


def test_get_oauth_config_from_env(monkeypatch):
    """Verify OAuth configuration loads from environment variables."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id-123.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test-secret-456")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")

    cfg = get_oauth_config()
    assert cfg["client_id"] == "test-client-id-123.apps.googleusercontent.com"
    assert cfg["client_secret"] == "test-secret-456"
    assert cfg["redirect_uri"] == "http://localhost:8501"
    assert is_oauth_configured() is True

    url = get_google_auth_url()
    assert "https://accounts.google.com/o/oauth2/v2/auth" in url
    assert "test-client-id-123" in url
    assert "openid" in url


def test_get_active_user_id_guest():
    """Verify unauthenticated session returns 'guest'."""
    # When no session state user is set
    assert get_active_user_id() == "guest"
