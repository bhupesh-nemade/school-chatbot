from __future__ import annotations

import logging
from pathlib import Path

from chatbot.chain import ask_question
from config import DEFAULT_MODEL


logger = logging.getLogger(__name__)


FALLBACK_MESSAGE = (
    "I do not have information related to your question."
)


def format_source(doc):
    source = doc.metadata.get(
        "source",
        "Unknown source",
    )

    page = doc.metadata.get(
        "page",
        "Unknown",
    )

    return {
        "source": Path(
            str(source)
        ).name,
        "page": str(page),
        "preview": (
            doc.page_content[
                :350
            ].strip()
        ),
    }


def answer_question(
    question: str,
    model_name: str = DEFAULT_MODEL,
    chat_history=None,
    user_id: str | None = None,
    conversation_id: str | None = None,
):
    if chat_history is None:
        chat_history = []

    answer, docs = ask_question(
        question=question,
        model_name=model_name,
        chat_history=chat_history,
        user_id=user_id,
        conversation_id=conversation_id,
    )

    sources = []

    if answer.strip() != FALLBACK_MESSAGE:
        sources = [
            format_source(doc)
            for doc in docs
        ]

    logger.info(
        "Answer generated. sources=%d",
        len(sources),
    )

    return {
        "answer": answer,
        "sources": sources,
    }