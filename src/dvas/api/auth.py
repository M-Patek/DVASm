"""API Authentication for DVAS.

Provides API key authentication using header-based tokens.

Configuration (via environment or .env):
    API_KEY=your-secret-key        # Required to enable auth
    API_KEY_HEADER=X-API-Key       # Default header name
    ALLOW_UNAUTHENTICATED=false    # Set false to require API key

Usage:
    from dvas.api.auth import require_auth

    @app.get("/protected", dependencies=[require_auth])
    async def protected_endpoint():
        return {"message": "This requires API key"}
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional

from dvas.config import settings
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# API Key header security scheme
api_key_header = APIKeyHeader(
    name=settings.API_KEY_HEADER,
    auto_error=False,  # We handle errors ourselves for better control
)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """Verify API key if authentication is enabled.

    Args:
        api_key: The API key from the request header

    Returns:
        The validated API key, or None if auth is disabled

    Raises:
        HTTPException: 401 if authentication is required but invalid/missing
    """
    # If no API_KEY is configured, auth is disabled
    if not settings.API_KEY:
        return None

    # If unauthenticated requests are allowed and no key provided
    if settings.ALLOW_UNAUTHENTICATED and not api_key:
        return None

    # Validate the provided key
    if api_key == settings.API_KEY:
        logger.debug("api_key_validated")
        return api_key

    # Invalid or missing key when auth is required
    logger.warning(
        "api_key_invalid",
        provided_key=api_key[:8] + "..." if api_key else None,
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": settings.API_KEY_HEADER},
    )


# Dependency to use in FastAPI endpoints
require_auth = Depends(verify_api_key)


# For stricter auth (always requires key when configured)
async def require_api_key_strict(
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    """Strict API key verification - always requires key if configured.

    Use this for admin endpoints that should always require authentication.
    """
    if not settings.API_KEY:
        # Auth not configured, allow anyway
        return "unconfigured"

    if api_key == settings.API_KEY:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="API key required",
        headers={"WWW-Authenticate": settings.API_KEY_HEADER},
    )


require_auth_strict = Depends(require_api_key_strict)


def get_auth_status() -> dict:
    """Get current authentication configuration status.

    Returns:
        Dict with auth enabled status and configuration
    """
    return {
        "enabled": bool(settings.API_KEY),
        "required": bool(settings.API_KEY) and not settings.ALLOW_UNAUTHENTICATED,
        "header_name": settings.API_KEY_HEADER,
        "allow_unauthenticated": settings.ALLOW_UNAUTHENTICATED,
    }
