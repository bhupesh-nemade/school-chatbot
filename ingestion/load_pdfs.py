from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
)
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_FOLDER = BASE_DIR / "data" / "raw_pdfs"


def _document_hash(
    file_path: Path,
) -> str:
    """
    SHA-256 hash of the source PDF.

    This gives the ingestion pipeline a stable identity for a document.
    """
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_all_pdfs(
    pdf_folder: Path | str = PDF_FOLDER,
) -> list[Document]:
    folder = Path(pdf_folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"PDF folder does not exist: {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"PDF path is not a directory: {folder}"
        )

    pdf_files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.lower(),
    )

    if not pdf_files:
        logger.warning(
            "No PDF files found in %s",
            folder,
        )
        return []

    documents: list[Document] = []

    for file_path in pdf_files:
        logger.info(
            "Loading PDF: %s",
            file_path.name,
        )

        try:
            document_hash = _document_hash(
                file_path
            )

            loader = PyPDFLoader(
                str(file_path)
            )

            pdf_pages = loader.load()

            for page_index, document in enumerate(
                pdf_pages
            ):
                metadata = dict(
                    document.metadata
                )

                metadata.update(
                    {
                        "source": file_path.name,
                        "source_path": str(
                            file_path
                        ),
                        "document_hash": (
                            document_hash
                        ),
                        "document_id": (
                            document_hash
                        ),
                        "page": (
                            metadata.get(
                                "page",
                                page_index,
                            )
                        ),
                        "page_number": page_index
                        + 1,
                    }
                )

                documents.append(
                    Document(
                        page_content=(
                            document.page_content
                        ),
                        metadata=metadata,
                    )
                )

            logger.info(
                "Loaded %d pages from %s",
                len(pdf_pages),
                file_path.name,
            )

        except Exception:
            # One bad PDF should not silently corrupt the entire run.
            # We log the failure and continue with the remaining files.
            logger.exception(
                "Failed to load PDF: %s",
                file_path,
            )

    logger.info(
        "Total PDF pages loaded: %d",
        len(documents),
    )

    return documents