import pytest

from chatbot.auth_service import (
    hash_password,
    validate_email,
    validate_password,
    verify_password,
)


def test_valid_email():
    assert (
        validate_email("User@Example.com")
        == "user@example.com"
    )


def test_invalid_email():
    with pytest.raises(ValueError):
        validate_email("invalid-email")


def test_password_minimum_length():
    with pytest.raises(ValueError):
        validate_password("short")


def test_password_hash_verifies():
    password = "StrongPassword123"

    password_hash = hash_password(
        password
    )

    assert password_hash != password

    assert verify_password(
        password,
        password_hash,
    )


def test_wrong_password_fails():
    password_hash = hash_password(
        "StrongPassword123"
    )

    assert not verify_password(
        "WrongPassword123",
        password_hash,
    )


def test_password_hash_is_not_reversible_text():
    password = "StrongPassword123"

    password_hash = hash_password(
        password
    )

    assert password not in password_hash