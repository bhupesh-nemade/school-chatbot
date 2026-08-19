from __future__ import annotations

import hashlib
import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


logger = logging.getLogger(__name__)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _chunk_hash(
    document_id: str,
    page_number: int,
    chunk_index: int,
    content: str,
) -> str:
    """
    Deterministic identity for a chunk.

    The same source document + page + chunk content generates
    the same ID across repeated ingestion runs.
    """
    payload = (
        f"{document_id}:"
        f"{page_number}:"
        f"{chunk_index}:"
        f"{content}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def chunk_documents(
    documents: list[Document],
) -> list[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n",
            "\t",
            "|",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunked_documents = (
        splitter.split_documents(
            documents
        )
    )

    result: list[Document] = []

    # Track chunk positions per source page/document.
    counters: dict[
        tuple[str, int],
        int,
    ] = {}

    for document in chunked_documents:
        metadata = dict(
            document.metadata
        )

        document_id = str(
            metadata.get(
                "document_id",
                metadata.get(
                    "document_hash",
                    metadata.get(
                        "source",
                        "unknown",
                    ),
                ),
            )
        )

        try:
            page_number = int(
                metadata.get(
                    "page_number",
                    metadata.get(
                        "page",
                        0,
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            page_number = 0

        counter_key = (
            document_id,
            page_number,
        )

        chunk_index = counters.get(
            counter_key,
            0,
        )

        counters[counter_key] = (
            chunk_index + 1
        )

        content = (
            document.page_content
            or ""
        ).strip()

        if not content:
            continue

        chunk_id = _chunk_hash(
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            content=content,
        )

        metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_size": len(content),
            }
        )

        result.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    logger.info(
        "Created %d chunks from %d source documents.",
        len(result),
        len(documents),
    )

    return result