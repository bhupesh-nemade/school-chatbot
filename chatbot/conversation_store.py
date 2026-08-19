from __future__ import annotations

import json
import logging
import sqlite3
import threading

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from uuid import uuid4

from config import DATA_DIR, MAX_USER_ID_LENGTH


logger = logging.getLogger(__name__)


DB_PATH = DATA_DIR / "conversations.db"
DEFAULT_TITLE = "New chat"

_DB_LOCK = threading.RLock()


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    email: str | None
    password_hash: str | None
    auth_provider: str
    provider_user_id: str | None
    is_verified: bool
    created_at: str
    updated_at: str
    last_login_at: str | None


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    sources: list[dict[str, Any]] = field(
        default_factory=list
    )
    timestamp: str = ""


# ============================================================================
# DATABASE
# ============================================================================

def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    conn.execute(
        "PRAGMA journal_mode = WAL"
    )

    conn.execute(
        "PRAGMA synchronous = NORMAL"
    )

    conn.execute(
        "PRAGMA busy_timeout = 30000"
    )

    return conn


def _get_columns(
    conn: sqlite3.Connection,
    table: str,
) -> set[str]:
    rows = conn.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    return {
        str(row["name"])
        for row in rows
    }


def _init_db(
    conn: sqlite3.Connection,
) -> None:
    """
    Initialize and migrate the local SQLite database.

    Existing conversations/messages are preserved.
    Existing conversation users are migrated into users.
    """

    with _DB_LOCK:

        # ------------------------------------------------------------------
        # Users
        # ------------------------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT,
                auth_provider TEXT NOT NULL DEFAULT 'local',
                provider_user_id TEXT,
                is_verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_users_email
            ON users(email)
            """
        )

        # ------------------------------------------------------------------
        # User identities
        #
        # One application user can have multiple login methods:
        #
        #   local  -> email/password
        #   google -> Google subject ID
        #   microsoft -> Microsoft subject ID, later if needed
        # ------------------------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_identities (
                identity_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                provider_user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login_at TEXT,
                FOREIGN KEY (user_id)
                    REFERENCES users(user_id)
                    ON DELETE CASCADE,
                UNIQUE(provider, provider_user_id)
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_user_identities_user
            ON user_identities(user_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_user_identities_provider
            ON user_identities(provider, provider_user_id)
            """
        )

        # ------------------------------------------------------------------
        # Conversations
        # ------------------------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conversation_columns = _get_columns(
            conn,
            "conversations",
        )

        if "user_id" not in conversation_columns:
            conn.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN user_id TEXT
                NOT NULL
                DEFAULT 'legacy-user'
                """
            )

        # ------------------------------------------------------------------
        # Messages
        # ------------------------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            )
            """
        )

        # ------------------------------------------------------------------
        # Indexes
        # ------------------------------------------------------------------

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_conversations_user_updated
            ON conversations(user_id, updated_at DESC)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_conversation_timestamp
            ON messages(conversation_id, timestamp ASC)
            """
        )

        # ------------------------------------------------------------------
        # Migrate existing conversation user IDs
        # ------------------------------------------------------------------

        existing_users = conn.execute(
            """
            SELECT DISTINCT user_id
            FROM conversations
            WHERE user_id IS NOT NULL
              AND TRIM(user_id) != ''
            """
        ).fetchall()

        for row in existing_users:
            user_id = str(
                row["user_id"]
            ).strip()

            if not user_id:
                continue

            exists = conn.execute(
                """
                SELECT 1
                FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

            if exists:
                continue

            now = datetime.now(
                timezone.utc
            ).isoformat()

            conn.execute(
                """
                INSERT INTO users (
                    user_id,
                    name,
                    email,
                    password_hash,
                    auth_provider,
                    provider_user_id,
                    is_verified,
                    created_at,
                    updated_at,
                    last_login_at
                )
                VALUES (?, ?, NULL, NULL, 'legacy', NULL, 0, ?, ?, NULL)
                """,
                (
                    user_id,
                    user_id,
                    now,
                    now,
                ),
            )

        conn.commit()


# ============================================================================
# VALIDATION
# ============================================================================

def _require_user_id(
    user_id: str,
) -> str:
    normalized = (
        user_id or ""
    ).strip()

    if not normalized:
        raise ValueError(
            "user_id is required."
        )

    if len(normalized) > MAX_USER_ID_LENGTH:
        raise ValueError(
            "user_id is too long."
        )

    return normalized


def _normalize_email(
    email: str,
) -> str:
    normalized = (
        email or ""
    ).strip().lower()

    if not normalized:
        raise ValueError(
            "email is required."
        )

    if len(normalized) > 320:
        raise ValueError(
            "email is too long."
        )

    if "@" not in normalized:
        raise ValueError(
            "Invalid email address."
        )

    return normalized


def _normalize_name(
    name: str,
) -> str:
    normalized = (
        name or ""
    ).strip()

    if not normalized:
        raise ValueError(
            "name is required."
        )

    if len(normalized) > 200:
        raise ValueError(
            "name is too long."
        )

    return normalized


def _row_to_user(
    row: sqlite3.Row,
) -> User:
    return User(
        user_id=str(row["user_id"]),
        name=str(row["name"]),
        email=(
            str(row["email"])
            if row["email"] is not None
            else None
        ),
        password_hash=(
            str(row["password_hash"])
            if row["password_hash"] is not None
            else None
        ),
        auth_provider=str(
            row["auth_provider"]
        ),
        provider_user_id=(
            str(row["provider_user_id"])
            if row["provider_user_id"] is not None
            else None
        ),
        is_verified=bool(
            row["is_verified"]
        ),
        created_at=str(
            row["created_at"]
        ),
        updated_at=str(
            row["updated_at"]
        ),
        last_login_at=(
            str(row["last_login_at"])
            if row["last_login_at"] is not None
            else None
        ),
    )


# ============================================================================
# SERVICE
# ============================================================================

class ConversationMemoryService:
    """
    SQLite application database.

    Stores:
        - users
        - user identities
        - conversations
        - messages

    Mem0 remains responsible for long-term semantic memory.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
    ):
        self._conn = conn

    # ========================================================================
    # USER OPERATIONS
    # ========================================================================

    def create_user(
        self,
        name: str,
        email: str,
        password_hash: str | None = None,
        auth_provider: str = "local",
        provider_user_id: str | None = None,
        is_verified: bool = False,
        user_id: str | None = None,
    ) -> User:
        name = _normalize_name(name)
        email = _normalize_email(email)

        auth_provider = (
            auth_provider or "local"
        ).strip().lower()

        if not auth_provider:
            raise ValueError(
                "auth_provider is required."
            )

        if provider_user_id is not None:
            provider_user_id = (
                provider_user_id.strip()
            )

            if not provider_user_id:
                provider_user_id = None

        if user_id is None:
            user_id = str(uuid4())

        user_id = _require_user_id(
            user_id
        )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with _DB_LOCK:
            try:
                self._conn.execute(
                    """
                    INSERT INTO users (
                        user_id,
                        name,
                        email,
                        password_hash,
                        auth_provider,
                        provider_user_id,
                        is_verified,
                        created_at,
                        updated_at,
                        last_login_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        user_id,
                        name,
                        email,
                        password_hash,
                        auth_provider,
                        provider_user_id,
                        int(is_verified),
                        now,
                        now,
                    ),
                )

                self._conn.commit()

            except sqlite3.IntegrityError as exc:
                self._conn.rollback()

                raise ValueError(
                    "A user with this email already exists."
                ) from exc

        return User(
            user_id=user_id,
            name=name,
            email=email,
            password_hash=password_hash,
            auth_provider=auth_provider,
            provider_user_id=provider_user_id,
            is_verified=is_verified,
            created_at=now,
            updated_at=now,
            last_login_at=None,
        )

    def get_user_by_id(
        self,
        user_id: str,
    ) -> User | None:
        user_id = _require_user_id(
            user_id
        )

        row = self._conn.execute(
            """
            SELECT *
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_user(row)

    def get_user_by_email(
        self,
        email: str,
    ) -> User | None:
        email = _normalize_email(
            email
        )

        row = self._conn.execute(
            """
            SELECT *
            FROM users
            WHERE LOWER(email) = ?
            """,
            (email,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_user(row)

    def get_user_by_identity(
        self,
        provider: str,
        provider_user_id: str,
    ) -> User | None:
        provider = (
            provider or ""
        ).strip().lower()

        provider_user_id = (
            provider_user_id or ""
        ).strip()

        if not provider:
            raise ValueError(
                "provider is required."
            )

        if not provider_user_id:
            raise ValueError(
                "provider_user_id is required."
            )

        row = self._conn.execute(
            """
            SELECT u.*
            FROM users u
            INNER JOIN user_identities ui
                ON ui.user_id = u.user_id
            WHERE ui.provider = ?
              AND ui.provider_user_id = ?
            """,
            (
                provider,
                provider_user_id,
            ),
        ).fetchone()

        if row is None:
            return None

        return _row_to_user(row)

    def update_user(
        self,
        user_id: str,
        *,
        name: str | None = None,
        email: str | None = None,
        password_hash: str | None = None,
        auth_provider: str | None = None,
        provider_user_id: str | None = None,
        is_verified: bool | None = None,
    ) -> User:
        user_id = _require_user_id(
            user_id
        )

        current = self.get_user_by_id(
            user_id
        )

        if current is None:
            raise ValueError(
                "User not found."
            )

        new_name = (
            _normalize_name(name)
            if name is not None
            else current.name
        )

        new_email = (
            _normalize_email(email)
            if email is not None
            else current.email
        )

        new_password_hash = (
            password_hash
            if password_hash is not None
            else current.password_hash
        )

        new_auth_provider = (
            auth_provider.strip().lower()
            if auth_provider is not None
            else current.auth_provider
        )

        new_provider_user_id = (
            provider_user_id.strip()
            if provider_user_id is not None
            else current.provider_user_id
        )

        new_is_verified = (
            is_verified
            if is_verified is not None
            else current.is_verified
        )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with _DB_LOCK:
            try:
                self._conn.execute(
                    """
                    UPDATE users
                    SET
                        name = ?,
                        email = ?,
                        password_hash = ?,
                        auth_provider = ?,
                        provider_user_id = ?,
                        is_verified = ?,
                        updated_at = ?
                    WHERE user_id = ?
                    """,
                    (
                        new_name,
                        new_email,
                        new_password_hash,
                        new_auth_provider,
                        new_provider_user_id,
                        int(new_is_verified),
                        now,
                        user_id,
                    ),
                )

                self._conn.commit()

            except sqlite3.IntegrityError as exc:
                self._conn.rollback()

                raise ValueError(
                    "A user with this email already exists."
                ) from exc

        updated = self.get_user_by_id(
            user_id
        )

        if updated is None:
            raise RuntimeError(
                "User could not be loaded after update."
            )

        return updated

    def update_last_login(
        self,
        user_id: str,
    ) -> None:
        user_id = _require_user_id(
            user_id
        )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with _DB_LOCK:
            cursor = self._conn.execute(
                """
                UPDATE users
                SET
                    last_login_at = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (
                    now,
                    now,
                    user_id,
                ),
            )

            self._conn.commit()

        if cursor.rowcount == 0:
            raise ValueError(
                "User not found."
            )

    # ========================================================================
    # USER IDENTITY OPERATIONS
    # ========================================================================

    def create_user_identity(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
    ) -> None:
        user_id = _require_user_id(
            user_id
        )

        provider = (
            provider or ""
        ).strip().lower()

        provider_user_id = (
            provider_user_id or ""
        ).strip()

        if not provider:
            raise ValueError(
                "provider is required."
            )

        if not provider_user_id:
            raise ValueError(
                "provider_user_id is required."
            )

        if self.get_user_by_id(
            user_id
        ) is None:
            raise ValueError(
                "User not found."
            )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with _DB_LOCK:
            try:
                self._conn.execute(
                    """
                    INSERT INTO user_identities (
                        identity_id,
                        user_id,
                        provider,
                        provider_user_id,
                        created_at,
                        last_login_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        user_id,
                        provider,
                        provider_user_id,
                        now,
                        now,
                    ),
                )

                self._conn.commit()

            except sqlite3.IntegrityError as exc:
                self._conn.rollback()

                raise ValueError(
                    "This provider identity is already linked."
                ) from exc

    def get_user_by_provider(
        self,
        provider: str,
        provider_user_id: str,
    ) -> User | None:
        """
        Backward-compatible alias.
        """

        return self.get_user_by_identity(
            provider,
            provider_user_id,
        )

    def update_identity_last_login(
        self,
        provider: str,
        provider_user_id: str,
    ) -> None:
        provider = (
            provider or ""
        ).strip().lower()

        provider_user_id = (
            provider_user_id or ""
        ).strip()

        if not provider or not provider_user_id:
            return

        now = datetime.now(
            timezone.utc
        ).isoformat()

        with _DB_LOCK:
            self._conn.execute(
                """
                UPDATE user_identities
                SET last_login_at = ?
                WHERE provider = ?
                  AND provider_user_id = ?
                """,
                (
                    now,
                    provider,
                    provider_user_id,
                ),
            )

            self._conn.commit()

    def list_user_identities(
        self,
        user_id: str,
    ) -> list[dict[str, Any]]:
        user_id = _require_user_id(
            user_id
        )

        rows = self._conn.execute(
            """
            SELECT
                identity_id,
                provider,
                provider_user_id,
                created_at,
                last_login_at
            FROM user_identities
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================================
    # CONVERSATIONS
    # ========================================================================

    def create_conversation(
        self,
        user_id: str,
        title: str = DEFAULT_TITLE,
    ) -> Conversation:
        user_id = _require_user_id(
            user_id
        )

        if self.get_user_by_id(
            user_id
        ) is None:
            raise ValueError(
                "User not found."
            )

        title = (
            title or DEFAULT_TITLE
        ).strip()

        if not title:
            title = DEFAULT_TITLE

        now = datetime.now(
            timezone.utc
        ).isoformat()

        conversation_id = str(
            uuid4()
        )

        with _DB_LOCK:
            self._conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id,
                    user_id,
                    title,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    user_id,
                    title,
                    now,
                    now,
                ),
            )

            self._conn.commit()

        return Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )

    def get_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> Conversation | None:
        user_id = _require_user_id(
            user_id
        )

        if not conversation_id:
            return None

        row = self._conn.execute(
            """
            SELECT *
            FROM conversations
            WHERE conversation_id = ?
              AND user_id = ?
            """,
            (
                conversation_id,
                user_id,
            ),
        ).fetchone()

        if row is None:
            return None

        return Conversation(
            **dict(row)
        )

    def ensure_latest_conversation(
        self,
        user_id: str,
    ) -> Conversation:
        conversations = self.list_conversations(
            user_id
        )

        if conversations:
            return conversations[0]

        return self.create_conversation(
            user_id
        )

    def list_conversations(
        self,
        user_id: str,
    ) -> list[Conversation]:
        user_id = _require_user_id(
            user_id
        )

        rows = self._conn.execute(
            """
            SELECT *
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()

        return [
            Conversation(
                **dict(row)
            )
            for row in rows
        ]

    # ========================================================================
    # MESSAGES
    # ========================================================================

    def get_messages(
        self,
        user_id: str,
        conversation_id: str,
    ) -> list[Message]:
        if (
            self.get_conversation(
                user_id,
                conversation_id,
            )
            is None
        ):
            raise ValueError(
                "Conversation not found."
            )

        rows = self._conn.execute(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            """,
            (conversation_id,),
        ).fetchall()

        result: list[Message] = []

        for row in rows:
            data = dict(row)

            try:
                sources = json.loads(
                    data.get(
                        "sources",
                        "[]",
                    )
                    or "[]"
                )
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid source JSON for message=%s",
                    data.get("id"),
                )
                sources = []

            if not isinstance(
                sources,
                list,
            ):
                sources = []

            data["sources"] = sources

            result.append(
                Message(
                    **data
                )
            )

        return result

    def get_chat_history(
        self,
        user_id: str,
        conversation_id: str,
        max_turns: int = 12,
    ) -> list[tuple[str, str]]:
        messages = self.get_messages(
            user_id,
            conversation_id,
        )

        history: list[
            tuple[str, str]
        ] = []

        pending_question: str | None = None

        for message in messages:
            if message.role == "user":
                pending_question = (
                    message.content
                )

            elif (
                message.role == "assistant"
                and pending_question is not None
            ):
                history.append(
                    (
                        pending_question,
                        message.content,
                    )
                )

                pending_question = None

        if max_turns <= 0:
            return []

        return history[-max_turns:]

    def add_user_message(
        self,
        user_id: str,
        conversation_id: str,
        content: str,
    ) -> Message:
        return self._add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role="user",
            content=content,
            sources=[],
        )

    def add_assistant_message(
        self,
        user_id: str,
        conversation_id: str,
        content: str,
        sources: list[
            dict[str, Any]
        ] | None = None,
    ) -> Message:
        return self._add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            sources=sources or [],
        )

    def _add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[
            dict[str, Any]
        ],
    ) -> Message:
        user_id = _require_user_id(
            user_id
        )

        if not content or not content.strip():
            raise ValueError(
                "Message content cannot be empty."
            )

        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                f"Invalid message role: {role}"
            )

        if (
            self.get_conversation(
                user_id,
                conversation_id,
            )
            is None
        ):
            raise ValueError(
                "Conversation not found."
            )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        message_id = str(
            uuid4()
        )

        sources_json = json.dumps(
            sources or [],
            ensure_ascii=False,
        )

        with _DB_LOCK:
            self._conn.execute(
                """
                INSERT INTO messages (
                    id,
                    conversation_id,
                    role,
                    content,
                    sources,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content.strip(),
                    sources_json,
                    now,
                ),
            )

            self._conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE conversation_id = ?
                  AND user_id = ?
                """,
                (
                    now,
                    conversation_id,
                    user_id,
                ),
            )

            if role == "user":
                self._maybe_set_title(
                    user_id,
                    conversation_id,
                    content,
                )

            self._conn.commit()

        return Message(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content.strip(),
            sources=sources or [],
            timestamp=now,
        )

    def _maybe_set_title(
        self,
        user_id: str,
        conversation_id: str,
        content: str,
    ) -> None:
        row = self._conn.execute(
            """
            SELECT title
            FROM conversations
            WHERE conversation_id = ?
              AND user_id = ?
            """,
            (
                conversation_id,
                user_id,
            ),
        ).fetchone()

        if row is None:
            return

        if row["title"] != DEFAULT_TITLE:
            return

        first_line = (
            content
            .strip()
            .split("\n")[0]
        )

        title = first_line[:60]

        if len(first_line) > 60:
            title += "..."

        self._conn.execute(
            """
            UPDATE conversations
            SET title = ?
            WHERE conversation_id = ?
              AND user_id = ?
            """,
            (
                title or DEFAULT_TITLE,
                conversation_id,
                user_id,
            ),
        )


# ============================================================================
# SERVICE FACTORY
# ============================================================================

@lru_cache(maxsize=1)
def get_conversation_service() -> ConversationMemoryService:
    conn = _connect()

    _init_db(
        conn
    )

    return ConversationMemoryService(
        conn
    )