from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================================
# Helpers
# ============================================================================

def _get_int(
    name: str,
    default: int,
    minimum: int = 0,
) -> int:
    raw_value = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer. "
            f"Received: {raw_value!r}"
        ) from exc

    if value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}. "
            f"Received: {value}"
        )

    return value


def _get_float(
    name: str,
    default: float,
) -> float:
    raw_value = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number. "
            f"Received: {raw_value!r}"
        ) from exc


# ============================================================================
# Application
# ============================================================================

APP_ENV = (
    os.getenv(
        "APP_ENV",
        "development",
    )
    .strip()
    .lower()
)

DEFAULT_USER_ID = (
    os.getenv(
        "DEFAULT_USER_ID",
        "local-development-user",
    )
    .strip()
)

MAX_USER_ID_LENGTH = _get_int(
    "MAX_USER_ID_LENGTH",
    128,
    minimum=1,
)


# ============================================================================
# LLM
# ============================================================================

LLM_PROVIDER = (
    os.getenv(
        "LLM_PROVIDER",
        "mistral",
    )
    .strip()
    .lower()
)

DEFAULT_MODEL = (
    os.getenv(
        "DEFAULT_MODEL",
        "mistral-small-latest",
    )
    .strip()
)

LLM_TEMPERATURE = _get_float(
    "LLM_TEMPERATURE",
    0.0,
)

LLM_MAX_TOKENS = _get_int(
    "LLM_MAX_TOKENS",
    1024,
    minimum=1,
)

LLM_TIMEOUT_SECONDS = _get_int(
    "LLM_TIMEOUT_SECONDS",
    60,
    minimum=1,
)

LLM_MAX_RETRIES = _get_int(
    "LLM_MAX_RETRIES",
    2,
    minimum=0,
)

MISTRAL_API_KEY = os.getenv(
    "MISTRAL_API_KEY"
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

HF_API_KEY = os.getenv(
    "HF_API_KEY"
)

MISTRAL_BASE_URL = (
    "https://api.mistral.ai/v1"
)

OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)

HF_BASE_URL = (
    "https://router.huggingface.co/v1"
)


# ============================================================================
# Pinecone
# ============================================================================

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME"
)

# ---------------------------------------------------------------------------
# Versioned vector namespace
#
# The existing empty namespace contains the old 2,160-vector corpus.
# The new production corpus will be written to this separate namespace.
# ---------------------------------------------------------------------------

PINECONE_NAMESPACE = (
    os.getenv(
        "PINECONE_NAMESPACE",
        "school_chatbot_v2",
    )
    .strip()
)

# Explicit version identifiers allow us to know exactly how the index
# was produced.
INDEX_VERSION = (
    os.getenv(
        "INDEX_VERSION",
        "v2",
    )
    .strip()
)

CHUNKING_VERSION = (
    os.getenv(
        "CHUNKING_VERSION",
        "v1",
    )
    .strip()
)

EMBEDDING_MODEL_NAME = (
    os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-m3",
    )
    .strip()
)


# ============================================================================
# RAG
# ============================================================================

RAG_SEARCH_TYPE = (
    os.getenv(
        "RAG_SEARCH_TYPE",
        "similarity",
    )
    .strip()
    .lower()
)

RAG_TOP_K = _get_int(
    "RAG_TOP_K",
    6,
    minimum=1,
)

RAG_FETCH_K = _get_int(
    "RAG_FETCH_K",
    20,
    minimum=1,
)

RAG_SIMILARITY_DEBUG_K = _get_int(
    "RAG_SIMILARITY_DEBUG_K",
    20,
    minimum=1,
)

RAG_NEIGHBOR_PAGE_WINDOW = _get_int(
    "RAG_NEIGHBOR_PAGE_WINDOW",
    1,
    minimum=0,
)

RAG_NEIGHBOR_CHUNKS_PER_PAGE = _get_int(
    "RAG_NEIGHBOR_CHUNKS_PER_PAGE",
    5,
    minimum=1,
)

RAG_MIN_RELEVANCE_SCORE = _get_float(
    "RAG_MIN_RELEVANCE_SCORE",
    0.0,
)

# Context limits.
RAG_MAX_INITIAL_DOCS = _get_int(
    "RAG_MAX_INITIAL_DOCS",
    12,
    minimum=1,
)

RAG_MAX_CONTEXT_DOCS = _get_int(
    "RAG_MAX_CONTEXT_DOCS",
    20,
    minimum=1,
)

RAG_MAX_CONTEXT_CHARS = _get_int(
    "RAG_MAX_CONTEXT_CHARS",
    30000,
    minimum=1000,
)


# ============================================================================
# Memory
# ============================================================================

MEM0_API_KEY = os.getenv(
    "MEM0_API_KEY"
)

MEM0_TOP_K = _get_int(
    "MEM0_TOP_K",
    5,
    minimum=1,
)


# ============================================================================
# Input / API
# ============================================================================

MAX_QUESTION_LENGTH = _get_int(
    "MAX_QUESTION_LENGTH",
    2000,
    minimum=1,
)

MAX_CHAT_HISTORY_TURNS = _get_int(
    "MAX_CHAT_HISTORY_TURNS",
    12,
    minimum=1,
)

CORS_ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        (
            "http://localhost:8501,"
            "http://127.0.0.1:8501"
        ),
    ).split(",")
    if item.strip()
]


# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL = (
    os.getenv(
        "LOG_LEVEL",
        "INFO",
    )
    .strip()
    .upper()
)


# ============================================================================
# Validation
# ============================================================================

def validate_env() -> None:
    missing: list[str] = []

    if not PINECONE_API_KEY:
        missing.append(
            "PINECONE_API_KEY"
        )

    if not PINECONE_INDEX_NAME:
        missing.append(
            "PINECONE_INDEX_NAME"
        )

    if not PINECONE_NAMESPACE:
        missing.append(
            "PINECONE_NAMESPACE"
        )

    if not MEM0_API_KEY:
        missing.append(
            "MEM0_API_KEY"
        )

    if LLM_PROVIDER == "mistral":
        if not MISTRAL_API_KEY:
            missing.append(
                "MISTRAL_API_KEY"
            )

    elif LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            missing.append(
                "OPENROUTER_API_KEY"
            )

    elif LLM_PROVIDER in {
        "huggingface",
        "hf",
    }:
        if not HF_API_KEY:
            missing.append(
                "HF_API_KEY"
            )

    else:
        raise ValueError(
            "Unsupported LLM_PROVIDER. "
            "Use: mistral, openrouter, or huggingface."
        )

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(
                sorted(set(missing))
            )
        )


def validate_runtime_config() -> None:
    if not DEFAULT_MODEL:
        raise ValueError(
            "DEFAULT_MODEL cannot be empty."
        )

    if not DEFAULT_USER_ID:
        raise ValueError(
            "DEFAULT_USER_ID cannot be empty."
        )

    if not PINECONE_NAMESPACE:
        raise ValueError(
            "PINECONE_NAMESPACE cannot be empty."
        )

    if not INDEX_VERSION:
        raise ValueError(
            "INDEX_VERSION cannot be empty."
        )

    if not CHUNKING_VERSION:
        raise ValueError(
            "CHUNKING_VERSION cannot be empty."
        )

    if not EMBEDDING_MODEL_NAME:
        raise ValueError(
            "EMBEDDING_MODEL_NAME cannot be empty."
        )

    if not 0.0 <= LLM_TEMPERATURE <= 2.0:
        raise ValueError(
            "LLM_TEMPERATURE must be between 0 and 2."
        )

    if RAG_SEARCH_TYPE not in {
        "similarity",
        "mmr",
    }:
        raise ValueError(
            "RAG_SEARCH_TYPE must be "
            "'similarity' or 'mmr'."
        )

    if RAG_FETCH_K < RAG_TOP_K:
        raise ValueError(
            "RAG_FETCH_K must be >= RAG_TOP_K."
        )

    if RAG_MIN_RELEVANCE_SCORE < 0:
        raise ValueError(
            "RAG_MIN_RELEVANCE_SCORE cannot be negative."
        )

    if RAG_MAX_CONTEXT_DOCS < RAG_MAX_INITIAL_DOCS:
        raise ValueError(
            "RAG_MAX_CONTEXT_DOCS must be >= "
            "RAG_MAX_INITIAL_DOCS."
        )


validate_runtime_config()