from __future__ import annotations

import logging

from functools import lru_cache
from typing import Any, Literal

from mem0 import MemoryClient

from config import MEM0_API_KEY, MEM0_TOP_K


logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


class Mem0MemoryLayer:
    """
    User-scoped long-term memory.

    SQLite:
        exact conversation transcript

    Mem0:
        semantic memories about that user
    """

    def __init__(
        self,
        user_id: str,
    ):
        normalized_user_id = (
            user_id or ""
        ).strip()

        if not normalized_user_id:
            raise ValueError(
                "user_id is required."
            )

        self.user_id = normalized_user_id
        self.client = self._build_client()

    @staticmethod
    def _build_client() -> MemoryClient | None:
        if not MEM0_API_KEY:
            logger.warning(
                "MEM0_API_KEY is missing. "
                "Long-term memory disabled."
            )
            return None

        try:
            return MemoryClient(
                api_key=MEM0_API_KEY
            )

        except Exception:
            logger.exception(
                "Failed to initialize Mem0 client."
            )
            return None

    def add_message(
        self,
        conversation_id: str,
        role: Role,
        content: str,
        timestamp: str,
    ) -> None:
        if self.client is None:
            return

        if not conversation_id:
            return

        if not content or not content.strip():
            return

        try:
            self.client.add(
                messages=[
                    {
                        "role": role,
                        "content": content.strip(),
                    }
                ],
                user_id=self.user_id,
                run_id=conversation_id,
                metadata={
                    "conversation_id": conversation_id,
                    "role": role,
                    "timestamp": timestamp,
                    "source": "school_chatbot",
                },
                infer=False,
            )

        except Exception:
            logger.exception(
                "Mem0 write failed "
                "user_id=%s conversation_id=%s",
                self.user_id,
                conversation_id,
            )

    def get_conversation_memories(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        if self.client is None:
            return []

        if not conversation_id:
            return []

        filters = {
            "AND": [
                {
                    "user_id": self.user_id,
                },
                {
                    "run_id": conversation_id,
                },
            ]
        }

        try:
            result = self.client.get_all(
                filters=filters
            )

            return self._normalize(
                result
            )

        except Exception:
            logger.exception(
                "Mem0 conversation retrieval failed "
                "user_id=%s conversation_id=%s",
                self.user_id,
                conversation_id,
            )
            return []

    def get_relevant_memories(
        self,
        query: str,
        top_k: int = MEM0_TOP_K,
    ) -> list[dict[str, Any]]:
        if self.client is None:
            return []

        if not query or not query.strip():
            return []

        try:
            result = self.client.search(
                query.strip(),
                filters={
                    "user_id": self.user_id,
                },
                limit=top_k,
            )

            return self._normalize(
                result
            )

        except Exception:
            logger.exception(
                "Mem0 semantic search failed "
                "user_id=%s",
                self.user_id,
            )
            return []

    @staticmethod
    def _normalize(
        result: Any,
    ) -> list[dict[str, Any]]:
        if isinstance(
            result,
            dict,
        ):
            result = (
                result.get("results")
                or []
            )

        if not isinstance(
            result,
            list,
        ):
            return []

        return [
            item
            for item in result
            if isinstance(
                item,
                dict,
            )
        ]


@lru_cache(maxsize=256)
def get_memory_layer(
    user_id: str,
) -> Mem0MemoryLayer:
    normalized = (
        user_id or ""
    ).strip()

    if not normalized:
        raise ValueError(
            "user_id is required."
        )

    return Mem0MemoryLayer(
        user_id=normalized
    )