from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from transformers.utils import logging as transformers_logging

from config import (
    EMBEDDING_MODEL_NAME,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    RAG_FETCH_K,
    RAG_SEARCH_TYPE,
    RAG_TOP_K,
)


logger = logging.getLogger(__name__)


os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers_logging.set_verbosity_error()


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


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL_NAME,
    )

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )


# ---------------------------------------------------------------------------
# Pinecone vector store
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
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