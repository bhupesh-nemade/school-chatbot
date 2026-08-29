from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import CustomTextVectorizer

from config import (
    EMBEDDING_MODEL_NAME,
    INDEX_VERSION,
    CHUNKING_VERSION,
    REDIS_URL,
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_TTL_SECONDS,
    SEMANTIC_CACHE_DISTANCE_THRESHOLD,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_NAME = "school_chatbot_semantic_cache"

# Separate Redis cache for exact question -> embedding mappings.
EMBEDDING_CACHE_NAME = "school_chatbot_embedding_cache"

VECTOR_DIM = 1024
VECTOR_DTYPE = "float32"

OPENROUTER_EMBEDDINGS_URL = (
    f"{OPENROUTER_BASE_URL.rstrip('/')}/embeddings"
)

OPENROUTER_TIMEOUT_SECONDS = 30

# Keep embeddings somewhat longer than response entries so repeated
# questions can reuse their vectors even if a response has expired.
EMBEDDING_CACHE_TTL_SECONDS = max(
    SEMANTIC_CACHE_TTL_SECONDS,
    3600,
)


# ---------------------------------------------------------------------------
# Cached clients
# ---------------------------------------------------------------------------

_cached_semantic_cache: Any = None
_cached_embedding_cache: Optional[EmbeddingsCache] = None
_cached_vectorizer: Optional[CustomTextVectorizer] = None
_cached_http_session: Optional[requests.Session] = None


# ---------------------------------------------------------------------------
# OpenRouter HTTP session
# ---------------------------------------------------------------------------

def _get_http_session() -> requests.Session:
    """
    Return one reusable HTTP session for OpenRouter requests.

    A persistent session allows HTTP connection reuse and avoids creating
    a new TCP/TLS connection for every embedding request.
    """

    global _cached_http_session

    if _cached_http_session is None:

        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured."
            )

        session = requests.Session()

        session.headers.update(
            {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
        )

        _cached_http_session = session

    return _cached_http_session


# ---------------------------------------------------------------------------
# OpenRouter single embedding
# ---------------------------------------------------------------------------

def _embed_with_openrouter(
    text: str,
    **kwargs: Any,
) -> list[float]:
    """
    Generate one BGE-M3 embedding through OpenRouter.

    This function is only called when RedisVL's EmbeddingsCache does not
    already contain the exact embedding for the requested text.
    """

    if not text or not text.strip():
        raise ValueError(
            "Cannot generate an embedding for empty text."
        )

    session = _get_http_session()

    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": text,
        "encoding_format": "float",
    }

    start_time = time.perf_counter()

    try:
        response = session.post(
            OPENROUTER_EMBEDDINGS_URL,
            json=payload,
            timeout=OPENROUTER_TIMEOUT_SECONDS,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        response.raise_for_status()

    except requests.RequestException as exc:

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.error(
            "OpenRouter embedding request failed "
            "model=%s latency=%.2fms error=%s",
            EMBEDDING_MODEL_NAME,
            latency_ms,
            exc,
        )

        raise RuntimeError(
            "OpenRouter embedding request failed."
        ) from exc

    try:
        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "OpenRouter returned invalid JSON."
        ) from exc

    if "error" in data:

        raise RuntimeError(
            "OpenRouter embedding error: "
            f"{data['error']}"
        )

    embedding_data = data.get("data")

    if not embedding_data:

        raise RuntimeError(
            "OpenRouter response did not contain "
            "embedding data."
        )

    embedding = embedding_data[0].get("embedding")

    if not isinstance(embedding, list):

        raise RuntimeError(
            "OpenRouter returned an invalid embedding."
        )

    embedding = [
        float(value)
        for value in embedding
    ]

    if len(embedding) != VECTOR_DIM:

        raise RuntimeError(
            "Embedding dimension mismatch. "
            f"Expected {VECTOR_DIM}, "
            f"received {len(embedding)}."
        )

    logger.debug(
        "OpenRouter embedding generated "
        "model=%s dimensions=%d latency=%.2fms",
        EMBEDDING_MODEL_NAME,
        len(embedding),
        latency_ms,
    )

    return embedding


# ---------------------------------------------------------------------------
# OpenRouter batch embedding
# ---------------------------------------------------------------------------

def _embed_many_with_openrouter(
    texts: list[str],
    **kwargs: Any,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts through OpenRouter.
    """

    if not texts:
        return []

    for text in texts:

        if not text or not text.strip():
            raise ValueError(
                "Cannot generate an embedding for empty text."
            )

    session = _get_http_session()

    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": texts,
        "encoding_format": "float",
    }

    start_time = time.perf_counter()

    try:
        response = session.post(
            OPENROUTER_EMBEDDINGS_URL,
            json=payload,
            timeout=OPENROUTER_TIMEOUT_SECONDS,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        response.raise_for_status()

    except requests.RequestException as exc:

        logger.error(
            "OpenRouter batch embedding request failed "
            "model=%s error=%s",
            EMBEDDING_MODEL_NAME,
            exc,
        )

        raise RuntimeError(
            "OpenRouter batch embedding request failed."
        ) from exc

    try:
        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "OpenRouter returned invalid JSON."
        ) from exc

    if "error" in data:

        raise RuntimeError(
            "OpenRouter embedding error: "
            f"{data['error']}"
        )

    embedding_data = data.get("data")

    if not embedding_data:

        raise RuntimeError(
            "OpenRouter response did not contain "
            "embedding data."
        )

    # OpenRouter returns an index for each embedding.
    embedding_data = sorted(
        embedding_data,
        key=lambda item: item.get("index", 0),
    )

    embeddings: list[list[float]] = []

    for item in embedding_data:

        embedding = item.get("embedding")

        if not isinstance(embedding, list):

            raise RuntimeError(
                "OpenRouter returned an invalid embedding."
            )

        embedding = [
            float(value)
            for value in embedding
        ]

        if len(embedding) != VECTOR_DIM:

            raise RuntimeError(
                "Embedding dimension mismatch. "
                f"Expected {VECTOR_DIM}, "
                f"received {len(embedding)}."
            )

        embeddings.append(embedding)

    if len(embeddings) != len(texts):

        raise RuntimeError(
            "OpenRouter returned an unexpected number "
            "of embeddings. "
            f"Expected {len(texts)}, "
            f"received {len(embeddings)}."
        )

    logger.debug(
        "OpenRouter batch embeddings generated "
        "model=%s count=%d latency=%.2fms",
        EMBEDDING_MODEL_NAME,
        len(embeddings),
        latency_ms,
    )

    return embeddings


# ---------------------------------------------------------------------------
# RedisVL embedding cache
# ---------------------------------------------------------------------------

def _get_embedding_cache() -> Optional[EmbeddingsCache]:
    """
    Initialize the exact embedding cache.

    This cache stores:

        question + embedding model
                    ↓
              embedding vector

    Therefore, the same question does not need to call OpenRouter again.
    """

    global _cached_embedding_cache

    if _cached_embedding_cache is not None:
        return _cached_embedding_cache

    try:

        _cached_embedding_cache = EmbeddingsCache(
            name=EMBEDDING_CACHE_NAME,
            ttl=EMBEDDING_CACHE_TTL_SECONDS,
            redis_url=REDIS_URL,
        )

        logger.info(
            "Embedding cache initialized "
            "name=%s model=%s ttl=%ds",
            EMBEDDING_CACHE_NAME,
            EMBEDDING_MODEL_NAME,
            EMBEDDING_CACHE_TTL_SECONDS,
        )

        return _cached_embedding_cache

    except Exception as exc:

        logger.warning(
            "Embedding cache initialization failed: %s. "
            "Continuing without embedding cache.",
            exc,
        )

        return None


# ---------------------------------------------------------------------------
# RedisVL vectorizer
# ---------------------------------------------------------------------------

def _get_vectorizer() -> CustomTextVectorizer:
    """
    Return one reusable RedisVL vectorizer.

    The vectorizer uses:
        RedisVL EmbeddingsCache
                    ↓
              exact lookup
                    ↓
        OpenRouter only on miss
    """

    global _cached_vectorizer

    if _cached_vectorizer is not None:
        return _cached_vectorizer

    embedding_cache = _get_embedding_cache()

    _cached_vectorizer = CustomTextVectorizer(
        embed=_embed_with_openrouter,
        embed_many=_embed_many_with_openrouter,
        dtype=VECTOR_DTYPE,
        cache=embedding_cache,
    )

    logger.info(
        "RedisVL vectorizer initialized "
        "model=%s dimensions=%d embedding_cache=%s",
        EMBEDDING_MODEL_NAME,
        VECTOR_DIM,
        embedding_cache is not None,
    )

    return _cached_vectorizer


# ---------------------------------------------------------------------------
# Redis semantic cache
# ---------------------------------------------------------------------------

def _get_cache(
    redis_url: str | None = None,
) -> Any:
    """
    Initialize the Redis semantic response cache.

    The existing Redis semantic-cache index is preserved.
    """

    global _cached_semantic_cache

    if _cached_semantic_cache is not None:
        return _cached_semantic_cache

    if not SEMANTIC_CACHE_ENABLED:

        logger.info(
            "Semantic cache is disabled via configuration."
        )

        return None

    try:

        vectorizer = _get_vectorizer()

        _cached_semantic_cache = SemanticCache(
            name=CACHE_NAME,
            distance_threshold=(
                SEMANTIC_CACHE_DISTANCE_THRESHOLD
            ),
            ttl=SEMANTIC_CACHE_TTL_SECONDS,
            vectorizer=vectorizer,
            redis_url=(
                redis_url
                or REDIS_URL
            ),
            create_index=True,
        )

        logger.info(
            "Semantic cache initialized "
            "name=%s model=%s threshold=%s "
            "ttl=%ds dimensions=%d",
            CACHE_NAME,
            EMBEDDING_MODEL_NAME,
            SEMANTIC_CACHE_DISTANCE_THRESHOLD,
            SEMANTIC_CACHE_TTL_SECONDS,
            VECTOR_DIM,
        )

        return _cached_semantic_cache

    except Exception as exc:

        logger.warning(
            "Semantic cache initialization failed: %s. "
            "Cache will be disabled for this run.",
            exc,
        )

        return None


# ---------------------------------------------------------------------------
# Cache availability
# ---------------------------------------------------------------------------

def is_cache_available() -> bool:
    """
    Check whether semantic caching is enabled and available.
    """

    if not SEMANTIC_CACHE_ENABLED:
        return False

    if _cached_semantic_cache is None:

        cache = _get_cache()

        if cache is None:
            return False

    return True


# ---------------------------------------------------------------------------
# Cache eligibility
# ---------------------------------------------------------------------------

def is_cache_eligible(
    question: str,
    chat_history: list,
    user_id: str | None,
    memory_layer=None,
) -> bool:
    """
    Determine whether a request is eligible for semantic caching.

    Cache is shared between users only for non-personalized,
    history-free informational questions.
    """

    # -----------------------------------------------------------------------
    # Conversation history
    # -----------------------------------------------------------------------

    if chat_history:
        return False

    # -----------------------------------------------------------------------
    # User-specific Mem0 memories
    # -----------------------------------------------------------------------

    if user_id:

        try:

            if memory_layer is not None:

                relevant_memories = (
                    memory_layer.get_relevant_memories(
                        question
                    )
                )

                if relevant_memories:

                    logger.debug(
                        "Cache bypassed: user %s has "
                        "relevant Mem0 memories",
                        user_id,
                    )

                    return False

        except Exception as exc:

            logger.warning(
                "Mem0 check failed during cache "
                "eligibility: %s",
                exc,
            )

    # -----------------------------------------------------------------------
    # Block personalized/account/auth questions
    # -----------------------------------------------------------------------

    question_lower = (
        question
        .lower()
        .strip()
    )

    blocked_prefixes = [
        "ignore ",
        "ignore previous",
        "forget ",
        "disregard",
        "reveal ",
        "show ",
        "what is my",
        "what are my",
        "account ",
        "login ",
        "password ",
        "change password",
        "delete my",
        "delete account",
        "authenticate",
        "login with",
        "who am i",
        "who are",
    ]

    for prefix in blocked_prefixes:

        if question_lower.startswith(prefix):

            logger.debug(
                "Cache bypassed: question starts "
                "with blocked prefix '%s'",
                prefix,
            )

            return False

    # -----------------------------------------------------------------------
    # Empty question
    # -----------------------------------------------------------------------

    if not question.strip():
        return False

    return True


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------

def get_cached_response(
    question: str,
    user_id: str | None = None,
    chat_history: list | None = None,
    memory_layer=None,
) -> tuple[Optional[str], dict[str, Any]]:
    """
    Look up a semantically similar response in Redis.

    RedisVL first obtains the question embedding through the cached
    vectorizer:

        EmbeddingsCache HIT
            -> reuse vector

        EmbeddingsCache MISS
            -> OpenRouter BGE-M3
            -> store embedding

    Then Redis performs semantic vector search.
    """

    if not is_cache_available():

        return None, {
            "cache_hit": False,
            "cache_source": "disabled",
            "cache_lookup_latency_ms": 0,
        }

    start_time = time.perf_counter()

    try:

        cache = _get_cache()

        if cache is None:

            return None, {
                "cache_hit": False,
                "cache_source": "initialization_failed",
                "cache_lookup_latency_ms": 0,
            }

        cache_hits = cache.check(
            prompt=question,
            num_results=1,
            distance_threshold=(
                SEMANTIC_CACHE_DISTANCE_THRESHOLD
            ),
        )

        lookup_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # -------------------------------------------------------------------
        # Cache HIT
        # -------------------------------------------------------------------

        if cache_hits:

            hit = cache_hits[0]

            answer = str(
                hit.get("response")
                or ""
            ).strip()

            metadata = {
                "cache_hit": True,
                "cache_source": "semantic",
                "cache_lookup_latency_ms": round(
                    lookup_latency_ms,
                    2,
                ),
                "cache_entry_id": hit.get(
                    "entry_id",
                    hit.get("id", ""),
                ),
                "cache_entry_prompt": hit.get(
                    "prompt",
                    question,
                ),
                "cache_entry_metadata": hit.get(
                    "metadata",
                    {},
                ),
                "cache_vector_distance": hit.get(
                    "vector_distance"
                ),
            }

            logger.info(
                "Semantic cache HIT "
                "latency=%.2fms distance=%s",
                lookup_latency_ms,
                hit.get("vector_distance"),
            )

            return answer, metadata

        # -------------------------------------------------------------------
        # Cache MISS
        # -------------------------------------------------------------------

        logger.info(
            "Semantic cache MISS "
            "latency=%.2fms",
            lookup_latency_ms,
        )

        return None, {
            "cache_hit": False,
            "cache_source": "semantic_miss",
            "cache_lookup_latency_ms": round(
                lookup_latency_ms,
                2,
            ),
        }

    except Exception as exc:

        lookup_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.warning(
            "Semantic cache lookup failed: %s. "
            "Treating as cache miss.",
            exc,
        )

        return None, {
            "cache_hit": False,
            "cache_source": "semantic_error",
            "cache_lookup_latency_ms": round(
                lookup_latency_ms,
                2,
            ),
        }


# ---------------------------------------------------------------------------
# Cache storage
# ---------------------------------------------------------------------------

def set_cached_response(
    question: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
    ttl: int | None = None,
) -> str | None:
    """
    Store a successful chatbot response in Redis.

    The response is shared between users because only non-personalized,
    cache-eligible requests should reach this function.

    The embedding is automatically reused from EmbeddingsCache when
    the same question has already been embedded.
    """

    if not is_cache_available():
        return None

    start_time = time.perf_counter()

    try:

        cache = _get_cache()

        if cache is None:
            return None

        cache_entry_metadata = {
            "cache_version": "1.0",
            "model_name": EMBEDDING_MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "index_version": INDEX_VERSION,
            "chunking_version": CHUNKING_VERSION,
            **(metadata or {}),
        }

        key = cache.store(
            prompt=question,
            response=answer,
            metadata=cache_entry_metadata,
            filters={},
            ttl=(
                ttl
                if ttl is not None
                else SEMANTIC_CACHE_TTL_SECONDS
            ),
        )

        store_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "Semantic cache STORE completed "
            "latency=%.2fms key=%s",
            store_latency_ms,
            key,
        )

        return key

    except Exception as exc:

        store_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.warning(
            "Semantic cache store failed "
            "latency=%.2fms error=%s. "
            "RAG will continue normally.",
            store_latency_ms,
            exc,
        )

        return None