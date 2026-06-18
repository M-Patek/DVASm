"""Tests for API authentication.

Validates the API key authentication system added to resolve
07-api known_gap "No authentication".
"""

from __future__ import annotations

import pytest


class TestAuthVerify:
    """Test the verify_api_key function directly."""

    @pytest.mark.asyncio
    async def test_verify_no_key_when_disabled(self):
        from dvas.api.auth import verify_api_key

        result = await verify_api_key(None)
        assert result is None  # Auth disabled

    @pytest.mark.asyncio
    async def test_verify_no_key_when_allowed(self, monkeypatch):
        from dvas.api.auth import verify_api_key
        from dvas.config import settings

        monkeypatch.setattr(settings, "API_KEY", "secret")
        monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", True)

        result = await verify_api_key(None)
        assert result is None  # Allowed without auth

    @pytest.mark.asyncio
    async def test_verify_wrong_key_raises(self, monkeypatch):
        from dvas.api.auth import verify_api_key
        from dvas.config import settings
        from fastapi import HTTPException

        monkeypatch.setattr(settings, "API_KEY", "secret")
        monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", False)

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key("wrong")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_valid_key_returns_key(self, monkeypatch):
        from dvas.api.auth import verify_api_key
        from dvas.config import settings

        monkeypatch.setattr(settings, "API_KEY", "secret")
        monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", False)

        result = await verify_api_key("secret")
        assert result == "secret"


class TestAuthConfig:
    """Test auth configuration."""

    def test_default_auth_disabled(self):
        from dvas.api.auth import get_auth_status
        from dvas.config import settings

        # Default should have no API_KEY
        status = get_auth_status()
        assert status["enabled"] == bool(settings.API_KEY)
        assert isinstance(status["header_name"], str)
        assert isinstance(status["allow_unauthenticated"], bool)

    def test_auth_status_with_config(self, monkeypatch):
        from dvas.api.auth import get_auth_status
        from dvas.config import settings

        monkeypatch.setattr(settings, "API_KEY", "test-key")
        monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", False)

        status = get_auth_status()
        assert status["enabled"] is True
        assert status["required"] is True
        assert status["allow_unauthenticated"] is False
