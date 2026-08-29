from chatbot.conversation_store import get_conversation_service

import pytest
def test_user_can_access_own_conversation():
    service = get_conversation_service()

    user_id = "test-user-own"
    conversation = service.create_conversation(user_id)

    result = service.get_conversation(
        user_id,
        conversation.conversation_id,
    )

    assert result is not None
    assert result.conversation_id == conversation.conversation_id
    assert result.user_id == user_id


def test_user_cannot_access_another_users_conversation():
    service = get_conversation_service()

    owner_id = "test-user-owner"
    attacker_id = "test-user-attacker"

    conversation = service.create_conversation(owner_id)

    result = service.get_conversation(
        attacker_id,
        conversation.conversation_id,
    )

    assert result is None


def test_user_only_sees_own_conversations():
    service = get_conversation_service()

    user_a = "test-user-a"
    user_b = "test-user-b"

    conversation_a = service.create_conversation(user_a)
    conversation_b = service.create_conversation(user_b)

    conversations_a = service.list_conversations(user_a)
    conversations_b = service.list_conversations(user_b)

    ids_a = {
        conversation.conversation_id
        for conversation in conversations_a
    }

    ids_b = {
        conversation.conversation_id
        for conversation in conversations_b
    }

    assert conversation_a.conversation_id in ids_a
    assert conversation_b.conversation_id not in ids_a

    assert conversation_b.conversation_id in ids_b
    assert conversation_a.conversation_id not in ids_b


def test_user_can_read_own_messages():
    service = get_conversation_service()

    user_id = "test-message-reader"

    conversation = service.create_conversation(
        user_id
    )

    service.add_user_message(
        user_id,
        conversation.conversation_id,
        "What is the admission process?",
    )

    messages = service.get_messages(
        user_id,
        conversation.conversation_id,
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert (
        messages[0].content
        == "What is the admission process?"
    )
    



def test_user_cannot_read_another_users_messages():
    service = get_conversation_service()

    owner_id = "test-message-owner"
    attacker_id = "test-message-attacker"

    conversation = service.create_conversation(
        owner_id
    )

    service.add_user_message(
        owner_id,
        conversation.conversation_id,
        "What is the admission process?",
    )

    with pytest.raises(ValueError, match="Conversation not found"):
        service.get_messages(
            attacker_id,
            conversation.conversation_id,
        )