from unittest.mock import MagicMock, Mock

import openai
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from openai.types import chat as chat_types

from chatbot.utils.chat import stream_response


@pytest.fixture
def mock_client() -> MagicMock:
    """Fixture for a mock OpenAI client."""
    return MagicMock(spec=openai.OpenAI)


def test_stream_response_valid_messages(mock_client: MagicMock) -> None:
    """Test stream_response with valid user message."""
    # Mock OpenAI client and stream response
    mock_stream = MagicMock()

    # Mock stream chunks
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
        Mock(choices=[Mock(delta=Mock(content="!"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),  # End of stream
    ]
    mock_stream.__iter__.return_value = iter(mock_chunks)
    mock_client.chat.completions.create.return_value = mock_stream

    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    # Collect all chunks from the generator
    response_chunks = list(stream_response(messages, mock_client))

    assert response_chunks == ["Hello", " world", "!", ""]
    mock_client.chat.completions.create.assert_called_once()


def test_stream_response_empty_messages(mock_client: MagicMock) -> None:
    """Test stream_response with empty messages list."""
    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(stream_response([], mock_client))


def test_stream_response_no_user_message(mock_client: MagicMock) -> None:
    """Test stream_response when last message is not from user."""
    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "assistant", "content": "Hello"}]

    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(stream_response(messages, mock_client))


def test_stream_response_empty_user_content(mock_client: MagicMock) -> None:
    """Test stream_response with empty user message content."""
    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": ""}]

    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(stream_response(messages, mock_client))


def test_stream_response_whitespace_only_content(mock_client: MagicMock) -> None:
    """Test stream_response with whitespace-only user message content."""
    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": "   \n\t  "}]

    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(stream_response(messages, mock_client))


def test_stream_response_non_string_content(mock_client: MagicMock) -> None:
    """Test stream_response with non-string content."""
    # Content is not a string (could be list for multimodal)
    messages: list[chat_types.ChatCompletionMessageParam] = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}  # type: ignore
    ]

    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(stream_response(messages, mock_client))


def test_stream_response_multiple_messages(mock_client: MagicMock) -> None:
    """Test stream_response with multiple messages."""
    mock_stream = MagicMock()

    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Response"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),
    ]
    mock_stream.__iter__.return_value = iter(mock_chunks)
    mock_client.chat.completions.create.return_value = mock_stream

    messages: list[chat_types.ChatCompletionMessageParam] = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "Assistant response"},
        {"role": "user", "content": "Second message"},
    ]

    response_chunks = list(stream_response(messages, mock_client))

    assert response_chunks == ["Response", ""]

    # Verify that all messages were passed to the API
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["messages"] == messages


def test_stream_response_api_parameters(mock_client: MagicMock) -> None:
    """Test that stream_response passes correct parameters to OpenAI API."""
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = iter([Mock(choices=[Mock(delta=Mock(content=None))])])
    mock_client.chat.completions.create.return_value = mock_stream

    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": "Test message"}]

    list(stream_response(messages, mock_client))

    # Verify API call parameters
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["stream"] is True
    assert "model" in call_args[1]
    assert "temperature" in call_args[1]
    assert "max_tokens" in call_args[1]
    assert "seed" in call_args[1]


@given(content=st.text(min_size=1))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_stream_response_with_arbitrary_user_content(mock_client: MagicMock, content: str) -> None:
    """Test stream_response with arbitrary user content."""
    # Skip whitespace-only strings as they're handled by validation
    if not content.strip():
        pytest.skip("Whitespace-only content is invalid")

    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = iter([Mock(choices=[Mock(delta=Mock(content="OK"))])])
    mock_client.chat.completions.create.return_value = mock_stream

    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": content}]

    # Should not raise an exception
    response_chunks = list(stream_response(messages, mock_client))
    assert isinstance(response_chunks, list)


def test_stream_response_handles_none_content_chunks(mock_client: MagicMock) -> None:
    """Test that stream_response properly handles None content in chunks."""
    mock_stream = MagicMock()

    # Mix of content and None chunks
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),  # None content
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),  # None content
    ]
    mock_stream.__iter__.return_value = iter(mock_chunks)
    mock_client.chat.completions.create.return_value = mock_stream

    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": "Test"}]

    response_chunks = list(stream_response(messages, mock_client))

    # None content should be converted to empty strings
    assert response_chunks == ["Hello", "", " world", ""]


def test_stream_response_conversation_history(mock_client: MagicMock) -> None:
    """Test stream_response with conversation history."""
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = iter([Mock(choices=[Mock(delta=Mock(content="Response"))])])
    mock_client.chat.completions.create.return_value = mock_stream

    # Simulate a conversation with history
    messages: list[chat_types.ChatCompletionMessageParam] = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
    ]

    response_chunks = list(stream_response(messages, mock_client))

    assert response_chunks == ["Response"]

    # Verify the entire conversation history was sent
    call_args = mock_client.chat.completions.create.call_args
    assert len(call_args[1]["messages"]) == 3
    assert call_args[1]["messages"][-1]["content"] == "What about 3+3?"
