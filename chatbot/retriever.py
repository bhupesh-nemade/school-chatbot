from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

import requests
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import (
    EMBEDDING_MODEL_NAME,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    RAG_FETCH_K,
    RAG_SEARCH_TYPE,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_EMBEDDINGS_URL = (
    f"{OPENROUTER_BASE_URL.rstrip('/')}/embeddings"
)

EMBEDDING_DIMENSIONS = 1024

EMBEDDING_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is not configured."
    )

if not PINECONE_INDEX_NAME:
    raise RuntimeError(
        "PINECONE_INDEX_NAME is not configured."
    )

if not PINECONE_NAMESPACE:
    raise RuntimeError(
        "PINECONE_NAMESPACE is not configured."
    )

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not configured."
    )

if not EMBEDDING_MODEL_NAME:
    raise RuntimeError(
        "EMBEDDING_MODEL_NAME is not configured."
    )


# ---------------------------------------------------------------------------
# OpenRouter Embeddings
# ---------------------------------------------------------------------------

class OpenRouterEmbeddings(Embeddings):
    """
    Production-oriented OpenRouter embedding implementation.

    Uses the OpenRouter embeddings API instead of loading the
    embedding model locally.

    Expected model:
        BAAI/bge-m3

    Expected dimensions:
        1024
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: int = EMBEDDING_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._session = requests.Session()

        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _embed(
        self,
        inputs: list[str],
    ) -> list[list[float]]:
        """
        Send embedding request to OpenRouter.
        """

        if not inputs:
            return []

        payload = {
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        }

        start = time.perf_counter()

        try:
            response = self._session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                timeout=self.timeout,
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            response.raise_for_status()

        except requests.RequestException:
            logger.exception(
                "OpenRouter embedding request failed "
                "model=%s",
                self.model,
            )
            raise

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "OpenRouter returned invalid JSON "
                "for embedding request."
            ) from exc

        if "data" not in data:
            error = data.get("error")

            raise RuntimeError(
                "OpenRouter embedding response does not "
                f"contain 'data'. error={error!r}"
            )

        embeddings = data["data"]

        if not isinstance(embeddings, list):
            raise RuntimeError(
                "Invalid embedding response format."
            )

        # OpenRouter normally returns items with an index.
        embeddings = sorted(
            embeddings,
            key=lambda item: item.get("index", 0),
        )

        vectors: list[list[float]] = []

        for item in embeddings:
            vector = item.get("embedding")

            if not isinstance(vector, list):
                raise RuntimeError(
                    "Invalid embedding vector returned "
                    "by OpenRouter."
                )

            vector = [float(value) for value in vector]

            if len(vector) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    "Embedding dimension mismatch. "
                    f"Expected {EMBEDDING_DIMENSIONS}, "
                    f"received {len(vector)}."
                )

            vectors.append(vector)

        logger.debug(
            "OpenRouter embeddings generated "
            "model=%s count=%d latency=%.2fms",
            self.model,
            len(vectors),
            latency_ms,
        )

        return vectors

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """

        if not texts:
            return []

        return self._embed(texts)

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate embedding for a single query.
        """

        vectors = self._embed([text])

        if not vectors:
            raise RuntimeError(
                "OpenRouter returned no embedding."
            )

        return vectors[0]


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model() -> OpenRouterEmbeddings:
    """
    Create one reusable OpenRouter embedding client.

    lru_cache prevents creating a new HTTP session for
    every request.
    """

    logger.info(
        "Initializing OpenRouter embedding model=%s",
        EMBEDDING_MODEL_NAME,
    )

    return OpenRouterEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Pinecone vector store
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    """
    Initialize Pinecone vector store.

    The same OpenRouter BGE-M3 embedding model is used
    for both query embeddings and the vectors stored
    in Pinecone.
    """

    logger.info(
        "Initializing Pinecone index=%s namespace=%s",
        PINECONE_INDEX_NAME,
        PINECONE_NAMESPACE,
    )

    embeddings = get_embedding_model()

    pinecone_client = Pinecone(
        api_key=PINECONE_API_KEY,
    )

    index = pinecone_client.Index(
        PINECONE_INDEX_NAME,
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_retriever():
    """
    Create and cache the Pinecone retriever.
    """

    if RAG_TOP_K <= 0:
        raise ValueError(
            "RAG_TOP_K must be greater than zero."
        )

    if RAG_SEARCH_TYPE == "mmr":

        if RAG_FETCH_K < RAG_TOP_K:
            raise ValueError(
                "RAG_FETCH_K must be greater than or equal "
                "to RAG_TOP_K."
            )

        search_kwargs = {
            "k": RAG_TOP_K,
            "fetch_k": RAG_FETCH_K,
        }

    elif RAG_SEARCH_TYPE == "similarity":

        search_kwargs = {
            "k": RAG_TOP_K,
        }

    else:

        raise ValueError(
            "RAG_SEARCH_TYPE must be either "
            "'mmr' or 'similarity'. "
            f"Received: {RAG_SEARCH_TYPE!r}"
        )

    logger.info(
        "Creating retriever "
        "namespace=%s search_type=%s kwargs=%s",
        PINECONE_NAMESPACE,
        RAG_SEARCH_TYPE,
        search_kwargs,
    )

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type=RAG_SEARCH_TYPE,
        search_kwargs=search_kwargs,
    )