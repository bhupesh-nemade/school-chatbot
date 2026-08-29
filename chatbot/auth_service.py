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

    # Name validation: reuse store's stricter check via _normalize_name logic
    name = (name or "").strip()
    if not name:
        raise ValueError("name is required.")
    if len(name) > 200:
        raise ValueError("name is too long.")

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

    # Best-effort: create local identity mapping for the application user.
    # This satisfies the user_identities design without breaking existing
    # installations where the table may already be populated.
    try:
        service.create_user_identity(
            user.user_id,
            provider="local",
            provider_user_id=user.user_id,
        )
    except Exception:
        pass

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

    # Do not reject non-local auth_provider users; a linked account
    # retains its local password_hash and must remain usable via
    # email/password even after Google linking. Require a password_hash
    # instead of checking provider type.
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

    # Best-effort: also update local identity last_login if present
    try:
        service.update_identity_last_login(
            "local", user.user_id
        )
    except Exception:
        pass

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

    # 1. Stable Google identity lookup (sub) via user_identities
    user = service.get_user_by_identity(
        provider="google",
        provider_user_id=provider_user_id,
    )

    if user is not None:
        service.update_last_login(
            user.user_id
        )
        try:
            service.update_identity_last_login(
                "google", provider_user_id
            )
        except Exception:
            pass
        return service.get_user_by_id(
            user.user_id
        )

    # 2. Link an existing account only when the Google email is verified.
    #    This prevents unverified Google emails from hijacking local accounts
    #    and satisfies the "same verified email does not create duplicate"
    #    requirement.
    existing_user = (
        service.get_user_by_email(email)
    )

    if existing_user is not None:
        if not email_verified:
            # Do not auto-link unverified emails; avoid duplicate users
            # but users.email has UNIQUE constraint so we cannot create a
            # second user with the same email. Raise a clear error.
            raise ValueError(
                "An account with this email already exists. "
                "Please sign in with email/password to link."
            )
        # Verified email: link Google identity to existing application user
        try:
            service.create_user_identity(
                existing_user.user_id,
                provider="google",
                provider_user_id=provider_user_id,
            )
        except ValueError:
            # Already linked concurrently
            pass
        # Preserve original local password_hash; only update verification flag
        # if Google confirms the email; do not overwrite auth_provider lossily.
        if email_verified and not existing_user.is_verified:
            try:
                service.update_user(
                    existing_user.user_id,
                    is_verified=email_verified,
                )
            except Exception:
                pass
        service.update_last_login(
            existing_user.user_id
        )
        try:
            service.update_identity_last_login(
                "google", provider_user_id
            )
        except Exception:
            pass
        return service.get_user_by_id(
            existing_user.user_id
        )

    # 3. No existing user: create new application user + Google identity
    new_user = service.create_user(
        name=name,
        email=email,
        password_hash=None,
        auth_provider="google",
        provider_user_id=provider_user_id,
        is_verified=email_verified,
    )
    try:
        service.create_user_identity(
            new_user.user_id,
            provider="google",
            provider_user_id=provider_user_id,
        )
    except ValueError:
        pass
    return service.get_user_by_id(new_user.user_id)