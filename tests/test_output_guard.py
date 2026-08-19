from chatbot.guardrails.output_guard import validate_output


def test_safe_output_is_allowed():
    answer = "The school reopens on Tuesday."

    result = validate_output(answer)

    assert result == answer


def test_empty_output_gets_fallback():
    result = validate_output("")

    assert result == (
        "I do not have information related to your question."
    )


def test_none_output_gets_fallback():
    result = validate_output(None)

    assert result == (
        "I do not have information related to your question."
    )


def test_api_key_is_blocked():
    result = validate_output(
        "The API key is: sk-test-secret-value"
    )

    assert result == "I cannot provide that information."


def test_secret_key_is_blocked():
    result = validate_output(
        "secret key: abc123"
    )

    assert result == "I cannot provide that information."


def test_password_is_blocked():
    result = validate_output(
        "password: school123"
    )

    assert result == "I cannot provide that information."


def test_system_prompt_disclosure_is_blocked():
    result = validate_output(
        "system prompt: you are a school assistant"
    )

    assert result == "I cannot provide that information."
    
    
def test_normal_school_url_is_allowed():
    answer = (
        "Students can access the portal through "
        "https://student.ABC.edunxt.com."
    )

    result = validate_output(answer)

    assert result == answer