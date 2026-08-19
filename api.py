from __future__ import annotations

import logging
from typing import Annotated

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot.conversation_store import (
    get_conversation_service,
)
from chatbot.rag_service import (
    DEFAULT_MODEL,
    answer_question,
)
from config import (
    CORS_ALLOWED_ORIGINS,
    MAX_QUESTION_LENGTH,
)


logger = logging.getLogger(__name__)


app = FastAPI(
    title="School Chatbot API",
    version="1.0.0",
    description=(
        "User-scoped school document RAG assistant."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Content-Type",
        "X-User-ID",
    ],
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
    )
    conversation_id: str | None = None
    model_name: str | None = None


class Source(BaseModel):
    source: str
    page: str
    preview: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[Source]


class ConversationResponse(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    timestamp: str
    sources: list[Source]


UserIdHeader = Annotated[
    str | None,
    Header(
        alias="X-User-ID",
        max_length=128,
    ),
]


def require_user_id(
    user_id: UserIdHeader,
) -> str:
    normalized = (
        user_id or ""
    ).strip()

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-ID header is required.",
        )

    return normalized


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "school-chatbot",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    user_id: UserIdHeader,
):
    effective_user_id = require_user_id(
        user_id
    )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    service = get_conversation_service()

    try:
        if request.conversation_id:
            conversation = service.get_conversation(
                effective_user_id,
                request.conversation_id,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found.",
                )
        else:
            conversation = service.create_conversation(
                effective_user_id
            )

        chat_history = service.get_chat_history(
            effective_user_id,
            conversation.conversation_id,
        )

        service.add_user_message(
            effective_user_id,
            conversation.conversation_id,
            question,
        )

        result = await run_in_threadpool(
            answer_question,
            question,
            request.model_name or DEFAULT_MODEL,
            chat_history,
            effective_user_id,
            conversation.conversation_id,
        )

        service.add_assistant_message(
            effective_user_id,
            conversation.conversation_id,
            result["answer"],
            result["sources"],
        )

        return {
            "conversation_id": (
                conversation.conversation_id
            ),
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid chat request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat request.",
        ) from exc

    except Exception:
        logger.exception(
            "Unhandled chat request failure. "
            "user_id=%s",
            effective_user_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to process the request right now."
            ),
        )


@app.post(
    "/conversations",
    response_model=ConversationResponse,
)
async def create_conversation(
    user_id: UserIdHeader,
):
    effective_user_id = require_user_id(
        user_id
    )

    conversation = (
        get_conversation_service()
        .create_conversation(
            effective_user_id
        )
    )

    return conversation


@app.get(
    "/conversations",
    response_model=list[ConversationResponse],
)
async def list_conversations(
    user_id: UserIdHeader,
):
    effective_user_id = require_user_id(
        user_id
    )

    conversations = (
        get_conversation_service()
        .list_conversations(
            effective_user_id
        )
    )

    return conversations


@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def list_messages(
    conversation_id: str,
    user_id: UserIdHeader,
):
    effective_user_id = require_user_id(
        user_id
    )

    service = get_conversation_service()

    conversation = service.get_conversation(
        effective_user_id,
        conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return service.get_messages(
        effective_user_id,
        conversation_id,
    )