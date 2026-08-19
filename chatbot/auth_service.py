from __future__ import annotations

import re

import bcrypt

from chatbot.conversation_store import (
    User,
    get_conversation_service,
)


EMAIL_PATTERN = re.compile(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
)


MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


def validate_email(email: str) -> str:
    email = (email or "").strip().lower()

    if not email:
        raise ValueError(
            "Email is required."
        )

    if len(email) > 320:
        raise ValueError(
            "Email is too long."
        )

    if not EMAIL_PATTERN.match(email):
        raise ValueError(
            "Invalid email address."
        )

    return email


def validate_password(
    password: str,
) -> str:
    if not password:
        raise ValueError(
            "Password is required."
        )

    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            "Password must contain at least "
            f"{MIN_PASSWORD_LENGTH} characters."
        )

    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(
            "Password is too long."
        )

    return password


def hash_password(
    password: str,
) -> str:
    password = validate_password(
        password
    )

    hashed = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    )

    return hashed.decode("utf-8")


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    if not password or not password_hash:
        return False

    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )

    except (ValueError, TypeError):
        return False


def register_user(
    name: str,
    email: str,
    password: str,
) -> User:
    service = get_conversation_service()

    email = validate_email(
        email
    )

    password = validate_password(
        password
    )

    if service.get_user_by_email(
        email
    ) is not None:
        raise ValueError(
            "An account with this email already exists."
        )

    password_hash = hash_password(
        password
    )

    user = service.create_user(
        name=name,
        email=email,
        password_hash=password_hash,
        auth_provider="local",
        provider_user_id=None,
        is_verified=False,
    )

    return user


def authenticate_user(
    email: str,
    password: str,
) -> User | None:
    service = get_conversation_service()

    email = validate_email(
        email
    )

    user = service.get_user_by_email(
        email
    )

    if user is None:
        return None

    if user.auth_provider != "local":
        return None

    if not user.password_hash:
        return None

    if not verify_password(
        password,
        user.password_hash,
    ):
        return None

    service.update_last_login(
        user.user_id
    )

    return service.get_user_by_id(
        user.user_id
    )


def authenticate_google_user(
    *,
    provider_user_id: str,
    email: str,
    name: str,
    email_verified: bool,
) -> User:
    """
    Create or retrieve a Google-authenticated user.

    Google/OIDC validation itself will be handled by Streamlit's
    authentication layer. This function only maps the verified identity
    into our application users table.
    """

    service = get_conversation_service()

    provider_user_id = (
        provider_user_id or ""
    ).strip()

    if not provider_user_id:
        raise ValueError(
            "Google provider user ID is required."
        )

    email = validate_email(
        email
    )

    name = (
        name or ""
    ).strip()

    if not name:
        name = email.split("@")[0]

    user = service.get_user_by_identity(
    provider="google",
    provider_user_id=provider_user_id,
)

    if user is not None:
        service.update_last_login(
            user.user_id
        )

        return service.get_user_by_id(
            user.user_id
        )

    # Link an existing local account if the verified
    # Google email already exists.
    existing_user = (
        service.get_user_by_email(email)
    )

    if existing_user is not None:
        updated_user = service.update_user(
            existing_user.user_id,
            name=name,
            auth_provider="google",
            provider_user_id=provider_user_id,
            is_verified=email_verified,
        )

        service.update_last_login(
            updated_user.user_id
        )

        return service.get_user_by_id(
            updated_user.user_id
        )

    return service.create_user(
        name=name,
        email=email,
        password_hash=None,
        auth_provider="google",
        provider_user_id=provider_user_id,
        is_verified=email_verified,
    )