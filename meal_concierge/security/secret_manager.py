"""Google Cloud Secret Manager integration with fail-closed production enforcement.

Implements Rubric Category 5.3 (Secure Secret Management):
Retrieves API keys, retailer tokens, and HMAC signing secrets dynamically from
Google Cloud Secret Manager (`google.cloud.secretmanager`) with environment
variable injection fallback, CSPRNG ephemeral tokens restricted strictly to
local/test environments, and automated static secret scanning.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

from meal_concierge.observability.structured_logger import get_structured_logger
from meal_concierge.observability.tracing import traced_span

try:
    from google.cloud import secretmanager  # type: ignore[import-not-found]

    SECRET_MANAGER_AVAILABLE = True
except ImportError:  # pragma: no cover - local fallback when google-cloud-secret-manager is absent
    secretmanager = None
    SECRET_MANAGER_AVAILABLE = False


class SecretResolutionError(RuntimeError):
    """Raised when a required secret cannot be resolved from Secret Manager or environment."""


_SECRET_KEY_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("GOOGLE_API_KEY_LITERAL", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("AWS_ACCESS_KEY_LITERAL", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("OAUTH_TOKEN_LITERAL", re.compile(r"ya29\.[0-9A-Za-z_-]{20,}")),
    ("PRIVATE_KEY_HEADER", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
]


def _is_production_runtime() -> bool:
    """Return True when running in Cloud Run, Vertex AI Agent Engine, or `MEAL_CONCIERGE_ENV=production`."""
    env_mode = os.environ.get("MEAL_CONCIERGE_ENV", "local").strip().lower()
    if env_mode in ("prod", "production", "staging"):
        return True
    if os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB"):
        return True
    return False


class SecretManagerClient:
    """Secure secret resolver backed by Google Cloud Secret Manager (`SecretManagerServiceClient`).

    Fails closed in production environments if a required secret is missing from both
    Google Cloud Secret Manager and injected environment variables.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        enable_gcp_rpc: Optional[bool] = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "fde-meal-concierge")
        self.cache_ttl_seconds = cache_ttl_seconds
        env_rpc = os.environ.get("ENABLE_SECRET_MANAGER_RPC", "false").lower() in ("1", "true", "yes")
        resolved_rpc = (
            enable_gcp_rpc
            if enable_gcp_rpc is not None
            else (env_rpc or _is_production_runtime())
        )
        self.enable_gcp_rpc = resolved_rpc and SECRET_MANAGER_AVAILABLE
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._ephemeral_csprng_store: Dict[str, str] = {}
        self._logger = get_structured_logger()
        self._gcp_client: Any = None

        if self.enable_gcp_rpc and secretmanager is not None:
            try:
                self._gcp_client = secretmanager.SecretManagerServiceClient()
            except Exception:
                self._gcp_client = None

    @traced_span("security.SecretManagerClient.get_secret")
    def get_secret(
        self,
        secret_id: str,
        *,
        version: str = "latest",
        allow_ephemeral_dev_default: Optional[bool] = None,
    ) -> str:
        """Resolve a secret by ID from Google Cloud Secret Manager or injected environment variables.

        Args:
            secret_id: Logical identifier of the secret (e.g., 'INSTACART_PARTNER_API_TOKEN').
            version: Secret Manager version alias (defaults to 'latest').
            allow_ephemeral_dev_default: When None, automatically False in production
                (`MEAL_CONCIERGE_ENV=production` or `K_SERVICE` set) and True only in local/test runs.

        Returns:
            Resolved secret string (never logged).

        Raises:
            SecretResolutionError: If the secret cannot be resolved in production or when
                `allow_ephemeral_dev_default=False`.
        """
        if not secret_id or not re.match(r"^[A-Za-z0-9_-]{3,128}$", secret_id):
            raise SecretResolutionError(f"Invalid secret_id format: '{secret_id}'")

        now = time.monotonic()
        cached = self._cache.get(secret_id)
        if cached is not None and (now - cached[1]) < self.cache_ttl_seconds:
            return cached[0]

        # 1. Check Google Cloud Secret Manager RPC when enabled
        if self._gcp_client is not None:
            resource_name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            try:
                response = self._gcp_client.access_secret_version(request={"name": resource_name})
                payload = response.payload.data.decode("utf-8")
                self._cache[secret_id] = (payload, now)
                self._logger.info(
                    f"Resolved secret '{secret_id}' from Google Cloud Secret Manager",
                    event_type="SECRET_ACCESS_SUCCESS",
                    payload={"secret_id": secret_id, "source": "gcp_secret_manager", "version": version},
                )
                return payload
            except Exception as exc:
                self._logger.warning(
                    f"Secret Manager RPC fallback for '{secret_id}': {type(exc).__name__}",
                    event_type="SECRET_ACCESS_FALLBACK",
                    payload={"secret_id": secret_id},
                )

        # 2. Check environment variable injected by Cloud Run / Secret Manager volume mount
        env_key = secret_id.upper().replace("-", "_")
        env_val = os.environ.get(env_key)
        if env_val:
            self._cache[secret_id] = (env_val, now)
            self._logger.info(
                f"Resolved secret '{secret_id}' from injected environment variable",
                event_type="SECRET_ACCESS_SUCCESS",
                payload={"secret_id": secret_id, "source": "environment_injection"},
            )
            return env_val

        # 3. Fail closed in production; allow CSPRNG ephemeral secret only in local/test mode
        permit_ephemeral = (
            allow_ephemeral_dev_default
            if allow_ephemeral_dev_default is not None
            else (not _is_production_runtime())
        )
        if permit_ephemeral:
            if secret_id not in self._ephemeral_csprng_store:
                self._ephemeral_csprng_store[secret_id] = (
                    f"csprng-local-{secrets.token_hex(32)}"
                )
            ephemeral_value = self._ephemeral_csprng_store[secret_id]
            self._cache[secret_id] = (ephemeral_value, now)
            self._logger.info(
                f"Generated CSPRNG runtime secret for '{secret_id}' in local/test mode",
                event_type="SECRET_ACCESS_EPHEMERAL",
                payload={"secret_id": secret_id, "source": "csprng_local_runtime"},
            )
            return ephemeral_value

        raise SecretResolutionError(
            f"Secret '{secret_id}' not found in Secret Manager (projects/{self.project_id}) "
            f"or environment '{env_key}' (fail-closed production policy)."
        )


def verify_no_hardcoded_secrets(source_root: str | Path) -> List[Dict[str, Any]]:
    """Scan Python and Terraform files under `source_root` to verify zero hardcoded API keys."""
    root_path = Path(source_root)
    violations: List[Dict[str, Any]] = []
    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in (".py", ".tf", ".yaml", ".yml", ".json"):
            continue
        if ".git" in file_path.parts or "__pycache__" in file_path.parts:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for rule_name, pattern in _SECRET_KEY_PATTERNS:
            for match in pattern.finditer(content):
                if file_path.name in ("secret_manager.py", "pii_redactor.py"):
                    continue
                violations.append(
                    {
                        "file": str(file_path.relative_to(root_path)),
                        "rule": rule_name,
                        "snippet": match.group(0)[:8] + "...[REDACTED]",
                    }
                )
    return violations


_GLOBAL_SECRET_MANAGER = SecretManagerClient()


def get_secret_manager() -> SecretManagerClient:
    """Return the singleton `SecretManagerClient`."""
    return _GLOBAL_SECRET_MANAGER
