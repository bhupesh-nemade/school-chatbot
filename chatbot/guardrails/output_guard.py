from __future__ import annotations

import re
from dataclasses import dataclass


SAFE_FALLBACK = "I cannot provide that information."

DEFAULT_FALLBACK = (
    "I do not have information related to your question."
)


@dataclass(frozen=True)
class OutputGuardResult:
    allowed: bool
    answer: str
    reason: str
    category: str


SENSITIVE_PATTERNS = [
    # Explicit secret assignments
    r"\bapi[_\s-]?key\s*[:=]\s*\S+",
    r"\bsecret[_\s-]?key\s*[:=]\s*\S+",
    r"\baccess[_\s-]?token\s*[:=]\s*\S+",
    r"\bpassword\s*[:=]\s*\S+",

    # Common secret/key prefixes
    r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{5,}\b",
    r"\bpk-[A-Za-z0-9][A-Za-z0-9._-]{5,}\b",

    # Bearer tokens
    r"\bbearer\s+[A-Za-z0-9._+/=-]{20,}\b",

    # Prompt/config disclosure
    r"\b(?:system|developer)\s+prompt\s*[:=]",
]


def _contains_sensitive_information(
    text: str,
) -> bool:
    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in SENSITIVE_PATTERNS
    )


def validate_output(
    answer: str | None,
) -> str:
    if answer is None:
        return DEFAULT_FALLBACK

    if not isinstance(answer, str):
        return DEFAULT_FALLBACK

    cleaned = answer.strip()

    if not cleaned:
        return DEFAULT_FALLBACK

    if _contains_sensitive_information(cleaned):
        return SAFE_FALLBACK

    return cleaned


def validate_output_detailed(
    answer: str | None,
) -> OutputGuardResult:
    if answer is None:
        return OutputGuardResult(
            allowed=False,
            answer=DEFAULT_FALLBACK,
            reason="empty_output",
            category="empty_output",
        )

    if not isinstance(answer, str):
        return OutputGuardResult(
            allowed=False,
            answer=DEFAULT_FALLBACK,
            reason="invalid_output_type",
            category="invalid_output",
        )

    cleaned = answer.strip()

    if not cleaned:
        return OutputGuardResult(
            allowed=False,
            answer=DEFAULT_FALLBACK,
            reason="empty_output",
            category="empty_output",
        )

    if _contains_sensitive_information(cleaned):
        return OutputGuardResult(
            allowed=False,
            answer=SAFE_FALLBACK,
            reason="Sensitive information detected.",
            category="sensitive_information",
        )

    return OutputGuardResult(
        allowed=True,
        answer=cleaned,
        reason="Allowed",
        category="safe",
    )