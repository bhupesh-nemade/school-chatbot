from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from config import MAX_QUESTION_LENGTH


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str
    category: str


# These are deliberately focused on attempts to manipulate the assistant,
# reveal hidden instructions, or execute clearly unsafe behavior.
PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bforget\s+(all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
    r"\breveal\s+(the\s+)?system\s+prompt\b",
    r"\bshow\s+(the\s+)?system\s+prompt\b",
    r"\bprint\s+(the\s+)?system\s+prompt\b",
    r"\breveal\s+developer\s+instructions\b",
    r"\bshow\s+developer\s+instructions\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"\bprompt\s+injection\b",
]

SECURITY_REQUEST_PATTERNS = [
    r"\bsteal\s+(the\s+)?api\s+key\b",
    r"\bextract\s+(the\s+)?api\s+key\b",
    r"\bshow\s+(me\s+)?the\s+api\s+key\b",
    r"\breveal\s+(the\s+)?secret\b",
]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = normalized.replace(
        "\x00",
        " ",
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def _matches_any(
    text: str,
    patterns: list[str],
) -> bool:
    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def validate_input(question: str) -> GuardResult:
    if not isinstance(question, str):
        return GuardResult(
            allowed=False,
            reason="Invalid question.",
            category="invalid_type",
        )

    normalized = _normalize(question)

    if not normalized:
        return GuardResult(
            allowed=False,
            reason="Please enter a question.",
            category="empty_input",
        )

    if len(normalized) > MAX_QUESTION_LENGTH:
        return GuardResult(
            allowed=False,
            reason="Question is too long.",
            category="length_limit",
        )

    if _matches_any(
        normalized,
        PROMPT_INJECTION_PATTERNS,
    ):
        return GuardResult(
            allowed=False,
            reason="This request violates chatbot policy.",
            category="prompt_injection",
        )

    if _matches_any(
        normalized,
        SECURITY_REQUEST_PATTERNS,
    ):
        return GuardResult(
            allowed=False,
            reason="This request violates chatbot policy.",
            category="security",
        )

    return GuardResult(
        allowed=True,
        reason="Allowed",
        category="school",
    )