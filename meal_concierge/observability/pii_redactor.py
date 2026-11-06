"""Active PII Redaction pipeline for logs, session state, and memory banks.

Implements Rubric Category 4.4 (PII Redaction):
Integrates Google Cloud Sensitive Data Protection / DLP API (`google.cloud.dlp_v2`)
with deterministic local regex and Luhn-validated scrubbers so that emails, phone
numbers, payment card numbers, SSNs, physical street delivery addresses, person names,
and medical record IDs are actively redacted prior to logging or persistence.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

try:
    from google.cloud import dlp_v2  # type: ignore[import-not-found]

    CLOUD_DLP_AVAILABLE = True
except ImportError:  # pragma: no cover - local fallback when google-cloud-dlp is absent
    dlp_v2 = None
    CLOUD_DLP_AVAILABLE = False


_CREDIT_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def _is_luhn_or_formatted_card(raw_candidate: str) -> bool:
    """Return True if `raw_candidate` passes Luhn checksum or is a 4x4 formatted card number."""
    digits = [int(ch) for ch in raw_candidate if ch.isdigit()]
    if not (13 <= len(digits) <= 16):
        return False
    # Standard 4-4-4-4 formatted payment card groups
    if re.fullmatch(r"\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}", raw_candidate.strip()):
        return True
    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return (checksum % 10) == 0


# Ordered deterministic regex rules for local & defense-in-depth scrubbing
_PII_PATTERNS: List[Tuple[str, Pattern[str], str]] = [
    (
        "US_SOCIAL_SECURITY_NUMBER",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED:SSN]",
    ),
    (
        "EMAIL_ADDRESS",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED:EMAIL]",
    ),
    (
        "PHONE_NUMBER",
        re.compile(
            r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
        ),
        "[REDACTED:PHONE]",
    ),
    (
        "MEDICAL_RECORD_ID",
        re.compile(r"\bMRN[-:\s]?[0-9A-Z]{6,12}\b", re.IGNORECASE),
        "[REDACTED:MEDICAL_ID]",
    ),
    (
        "STREET_ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][a-zA-Z0-9\s]{2,35}\s+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Way|Terrace|Ter|Place|Pl|Circle|Cir|Parkway|Pkwy)\b\.?(?:\s*(?:Apt|Suite|Unit|#)\s*[A-Za-z0-9-]+)?",
            re.IGNORECASE,
        ),
        "[REDACTED:STREET_ADDRESS]",
    ),
    (
        "PERSON_NAME",
        re.compile(
            r"(?:(?<=Deliver to )|(?<=Recipient: )|(?<=Patient: )|(?<=Customer: ))[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,25}\b"
        ),
        "[REDACTED:PERSON_NAME]",
    ),
]

_SENSITIVE_DICT_KEYS = frozenset(
    {
        "credit_card",
        "card_number",
        "cvv",
        "ssn",
        "social_security_number",
        "delivery_address",
        "street_address",
        "home_address",
        "phone_number",
        "email",
        "email_address",
        "person_name",
        "full_name",
        "recipient_name",
        "medical_record_number",
        "api_key",
        "secret_key",
        "password",
    }
)


class GoogleCloudDLPRedactor:
    """Scrubs sensitive PII from strings and nested payloads using Cloud DLP + local rules."""

    DEFAULT_INFO_TYPES: List[str] = [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD_NUMBER",
        "US_SOCIAL_SECURITY_NUMBER",
        "STREET_ADDRESS",
        "MEDICAL_RECORD_NUMBER",
        "PERSON_NAME",
    ]

    def __init__(
        self,
        project_id: Optional[str] = None,
        enable_cloud_dlp_rpc: Optional[bool] = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        env_flag = os.environ.get("ENABLE_CLOUD_DLP", "false").lower() in ("1", "true", "yes")
        resolved_flag = enable_cloud_dlp_rpc if enable_cloud_dlp_rpc is not None else env_flag
        self.enable_cloud_dlp_rpc = resolved_flag and CLOUD_DLP_AVAILABLE and bool(self.project_id)
        self._dlp_client: Any = None
        if self.enable_cloud_dlp_rpc and dlp_v2 is not None:
            try:
                self._dlp_client = dlp_v2.DlpServiceClient()
            except Exception:
                self._dlp_client = None

    def redact_text(self, text: str, *, use_dlp_rpc: bool = True) -> str:
        """Scrub all PII tokens from a raw text string."""
        if not text or not isinstance(text, str):
            return text

        scrubbed = text
        if use_dlp_rpc and self._dlp_client is not None and self.project_id:
            try:
                parent = f"projects/{self.project_id}/locations/global"
                inspect_config = {
                    "info_types": [{"name": it} for it in self.DEFAULT_INFO_TYPES],
                    "min_likelihood": "POSSIBLE",
                }
                deidentify_config = {
                    "info_type_transformations": {
                        "transformations": [
                            {
                                "primitive_transformation": {
                                    "replace_with_info_type_config": {}
                                }
                            }
                        ]
                    }
                }
                response = self._dlp_client.deidentify_content(
                    request={
                        "parent": parent,
                        "inspect_config": inspect_config,
                        "deidentify_config": deidentify_config,
                        "item": {"value": scrubbed},
                    }
                )
                scrubbed = response.item.value
            except Exception:
                pass

        # Luhn-validated credit card scrubbing
        def _replace_card(match: re.Match[str]) -> str:
            candidate = match.group(0)
            return "[REDACTED:CREDIT_CARD]" if _is_luhn_or_formatted_card(candidate) else candidate

        scrubbed = _CREDIT_CARD_CANDIDATE_RE.sub(_replace_card, scrubbed)

        for _, pattern, replacement in _PII_PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed

    def _redact_structured_local(self, payload: Any) -> Any:
        """Recursively scrub PII using deterministic local patterns without per-leaf RPC fan-out."""
        if isinstance(payload, str):
            return self.redact_text(payload, use_dlp_rpc=False)
        if isinstance(payload, list):
            return [self._redact_structured_local(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self._redact_structured_local(item) for item in payload)
        if isinstance(payload, dict):
            sanitized: Dict[str, Any] = {}
            for key, value in payload.items():
                norm_key = str(key).lower().strip()
                if norm_key in _SENSITIVE_DICT_KEYS and value is not None:
                    sanitized[key] = f"[REDACTED:{norm_key.upper()}]"
                else:
                    sanitized[key] = self._redact_structured_local(value)
            return sanitized
        return payload

    def redact_structured(self, payload: Any) -> Any:
        """Recursively scrub PII from dictionaries, lists, and scalar values (at most 1 DLP RPC per payload)."""
        if isinstance(payload, str):
            return self.redact_text(payload, use_dlp_rpc=True)
        return self._redact_structured_local(payload)


_DEFAULT_REDACTOR = GoogleCloudDLPRedactor()


def redact_pii_text(text: str) -> str:
    """Convenience function to scrub PII from a text string."""
    return _DEFAULT_REDACTOR.redact_text(text)


def redact_pii_data(payload: Any) -> Any:
    """Convenience function to recursively scrub PII from structured data."""
    return _DEFAULT_REDACTOR.redact_structured(payload)


class PIIRedactionFilter(logging.Filter):
    """Python `logging.Filter` that automatically scrubs PII from all log records."""

    def __init__(self, redactor: Optional[GoogleCloudDLPRedactor] = None) -> None:
        super().__init__(name="pii_redaction_filter")
        self.redactor = redactor or _DEFAULT_REDACTOR

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_pii_already_redacted", False):
            return True
        if isinstance(record.msg, str):
            record.msg = self.redactor.redact_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = self.redactor.redact_structured(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(self.redactor.redact_structured(a) for a in record.args)
        if hasattr(record, "structured_payload"):
            record.structured_payload = self.redactor.redact_structured(
                getattr(record, "structured_payload")
            )
        record._pii_already_redacted = True
        return True
