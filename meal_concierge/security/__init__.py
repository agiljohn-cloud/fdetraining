"""Secure Secret Management via Google Cloud Secret Manager."""

from meal_concierge.security.secret_manager import (
    SecretManagerClient,
    SecretResolutionError,
    get_secret_manager,
    verify_no_hardcoded_secrets,
)

__all__ = [
    "SecretManagerClient",
    "SecretResolutionError",
    "get_secret_manager",
    "verify_no_hardcoded_secrets",
]
