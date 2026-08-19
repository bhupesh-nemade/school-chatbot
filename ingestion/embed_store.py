from __future__ import annotations

import logging
import time
from typing import Iterable

from langchain_core.documents import Document
from pinecone.core.openapi.shared.exceptions import PineconeApiException

from chatbot.retriever import get_vectorstore
from config import (
    CHUNKING_VERSION,
    EMBEDDING_MODEL_NAME,
    INDEX_VERSION,
    PINECONE_NAMESPACE,
)


logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 8
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 2.0


def _batches(
    documents: list[Document],
    batch_size: int,
) -> Iterable[tuple[int, list[Document]]]:
    for start in range(0, len(documents), batch_size):
        yield start, documents[start:start + batch_size]


def _validate_chunks(
    chunks: list[Document],
) -> None:
    for index, document in enumerate(chunks):
        chunk_id = document.metadata.get("chunk_id")

        if not chunk_id:
            raise ValueError(
                f"Chunk at index {index} is missing chunk_id."
            )

        if not document.page_content.strip():
            raise ValueError(
                f"Chunk at index {index} has empty content."
            )


def _prepare_metadata(
    document: Document,
) -> dict:
    metadata = dict(document.metadata)

    metadata.update(
        {
            "index_version": INDEX_VERSION,
            "chunking_version": CHUNKING_VERSION,
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
    )

    return metadata


def _is_retryable_error(exc: Exception) -> bool:
    """
    Retry only errors that are plausibly temporary.

    HTTP 4xx errors generally indicate that the request itself is invalid,
    so retrying them repeatedly is not useful.
    """

    if isinstance(exc, PineconeApiException):
        status_code = getattr(
            exc,
            "status",
            None,
        )

        if status_code is None:
            response = getattr(
                exc,
                "response",
                None,
            )
            status_code = getattr(
                response,
                "status_code",
                None,
            )

        if status_code is not None:
            return status_code >= 500

        message = str(exc).lower()

        permanent_markers = (
            "bad request",
            "not allowed",
            "invalid",
            "missing",
            "namespace",
            "dimension",
            "unauthorized",
            "forbidden",
        )

        if any(
            marker in message
            for marker in permanent_markers
        ):
            return False

        return True

    # Network/timeout-style exceptions are potentially transient.
    transient_markers = (
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "temporary failure",
        "503",
        "502",
        "504",
    )

    message = str(exc).lower()

    return any(
        marker in message
        for marker in transient_markers
    )


def store_chunks_in_pinecone(
    chunks: list[Document],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """
    Store chunks in the configured Pinecone namespace.

    Deterministic chunk IDs make repeated ingestion safe:
    the same chunk is written to the same vector ID rather than creating
    another copy.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if not chunks:
        logger.warning(
            "No chunks provided for Pinecone indexing."
        )
        return 0

    _validate_chunks(chunks)

    prepared_documents = [
        Document(
            page_content=document.page_content,
            metadata=_prepare_metadata(document),
        )
        for document in chunks
    ]

    vectorstore = get_vectorstore()

    batches = list(
        _batches(
            prepared_documents,
            batch_size,
        )
    )

    total_batches = len(batches)
    indexed_count = 0

    logger.info(
        "Starting Pinecone ingestion "
        "chunks=%d batches=%d batch_size=%d namespace=%s",
        len(prepared_documents),
        total_batches,
        batch_size,
        PINECONE_NAMESPACE,
    )

    for batch_number, (start_index, batch) in enumerate(
        batches,
        start=1,
    ):
        ids = [
            str(
                document.metadata["chunk_id"]
            )
            for document in batch
        ]

        logger.info(
            "[%d/%d] Indexing %d chunks.",
            batch_number,
            total_batches,
            len(batch),
        )

        last_exception: Exception | None = None

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):
            try:
                started = time.perf_counter()

                vectorstore.add_documents(
                    documents=batch,
                    ids=ids,
                )

                elapsed = (
                    time.perf_counter()
                    - started
                )

                indexed_count += len(batch)

                logger.info(
                    "[%d/%d] Completed %d chunks in %.2fs.",
                    batch_number,
                    total_batches,
                    len(batch),
                    elapsed,
                )

                last_exception = None
                break

            except Exception as exc:
                last_exception = exc

                retryable = _is_retryable_error(
                    exc
                )

                logger.exception(
                    "[%d/%d] Pinecone indexing failed "
                    "attempt=%d/%d retryable=%s",
                    batch_number,
                    total_batches,
                    attempt,
                    MAX_RETRIES,
                    retryable,
                )

                if not retryable:
                    raise RuntimeError(
                        f"Permanent failure while indexing "
                        f"batch {batch_number}/{total_batches}."
                    ) from exc

                if attempt < MAX_RETRIES:
                    delay = (
                        RETRY_BASE_SECONDS
                        * (2 ** (attempt - 1))
                    )

                    logger.info(
                        "Retrying batch %d in %.1f seconds.",
                        batch_number,
                        delay,
                    )

                    time.sleep(delay)

        if last_exception is not None:
            raise RuntimeError(
                f"Failed to index batch "
                f"{batch_number}/{total_batches} "
                f"after {MAX_RETRIES} attempts."
            ) from last_exception

    logger.info(
        "Pinecone ingestion completed "
        "indexed=%d namespace=%s",
        indexed_count,
        PINECONE_NAMESPACE,
    )

    return indexed_count