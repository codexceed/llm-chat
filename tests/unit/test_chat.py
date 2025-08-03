import json
import random
from typing import Any, Final

import httpx
import hypothesis
import openai
import pytest
import pytest_httpx
from hypothesis import strategies as strat
from openai.types import chat as chat_types

from chatbot import settings
from chatbot.utils import chat
from tests.unit import utils as test_utils

TEST_SERVER_BASE_URL: Final[str] = "http://testserver/v1"
TEST_SERVER_CHAT_COMPLETIONS_URL: Final[str] = TEST_SERVER_BASE_URL + "/chat/completions"
TEST_API_KEY: Final[str] = "test-key"


@pytest.fixture(scope="module")
def openai_client() -> openai.OpenAI:
    """Fixture for OpenAI client.

    Returns:
        An OpenAI client instance.
    """
    return openai.OpenAI(base_url=TEST_SERVER_BASE_URL, api_key=TEST_API_KEY)


@hypothesis.given(
    messages=test_utils.chat_completion_messages_strategy(),
    expected_chunks=test_utils.chat_completion_chunks_strategy(num_chunks=10),
)
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_stream_response_valid_messages(
    httpx_mock: pytest_httpx.HTTPXMock,
    openai_client: openai.OpenAI,
    messages: list[chat_types.ChatCompletionMessageParam],
    expected_chunks: list[str],
) -> None:
    """Test stream_response with valid user message using generated streaming responses.

    Args:
        httpx_mock: The mock HTTPX client.
        openai_client: The OpenAI client.
        messages: The user messages to send to the language model.
        expected_chunks: The expected chunks of the streaming response.
    """
    captured_request: dict[str, Any] = {}

    def callback(request: httpx.Request) -> httpx.Response:
        """Callback to capture and validate the request payload.

        Args:
            request: The incoming httpx.Request.

        Returns:
            A mock httpx.Response with streaming data.
        """
        captured_request["method"] = request.method
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["content"] = json.loads(request.content)

        return httpx.Response(
            status_code=200,
            stream=pytest_httpx.IteratorStream([chunk.encode("utf-8") for chunk in expected_chunks]),
        )

    httpx_mock.add_callback(callback, method="POST", url=TEST_SERVER_CHAT_COMPLETIONS_URL)

    # Validate chunks from the stream
    idx = 0
    for response_chunk in chat.stream_response(messages, openai_client):
        assert response_chunk == expected_chunks[idx]

    # Validate the captured request
    assert captured_request["method"] == "POST"
    headers = captured_request["headers"]
    assert "authorization" in headers
    assert headers["authorization"] == f"Bearer {TEST_API_KEY}"

    # Validate the request payload structure
    payload = captured_request["content"]
    assert "model" in payload
    assert "messages" in payload
    assert "stream" in payload
    assert "temperature" in payload
    assert "max_tokens" in payload
    assert "seed" in payload

    # Validate specific values
    assert payload["messages"] == messages
    assert payload["stream"] is True
    assert isinstance(payload["temperature"], (int, float))
    assert isinstance(payload["max_tokens"], int)
    assert isinstance(payload["seed"], (int, type(None)))


def test_stream_response_empty_messages(openai_client: openai.OpenAI) -> None:
    """Test stream_response with empty messages list.

    Args:
        openai_client: The OpenAI client.
    """
    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(chat.stream_response([], openai_client))


def test_stream_response_no_user_message(openai_client: openai.OpenAI) -> None:
    """Test stream_response when last message is not from user.

    Args:
        openai_client: The OpenAI client.
    """
    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "assistant", "content": "Hello"}]

    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(chat.stream_response(messages, openai_client))


def test_stream_response_empty_user_content(openai_client: openai.OpenAI) -> None:
    """Test stream_response with empty user message content.

    Args:
        openai_client: The OpenAI client.
    """
    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": ""}]

    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(chat.stream_response(messages, openai_client))


def test_stream_response_whitespace_only_content(openai_client: openai.OpenAI) -> None:
    """Test stream_response with whitespace-only user message content.

    Args:
        openai_client: The OpenAI client.
    """
    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": "   \n\t  "}]

    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(chat.stream_response(messages, openai_client))


def test_stream_response_non_string_content(openai_client: openai.OpenAI) -> None:
    """Test stream_response with non-string content.

    Args:
        openai_client: The OpenAI client.
    """
    # Content is not a string (could be list for multimodal)
    messages: list[chat_types.ChatCompletionMessageParam] = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}  # type: ignore
    ]

    with pytest.raises(ValueError, match="No messages provided for response generation"):
        list(chat.stream_response(messages, openai_client))


@hypothesis.given(
    temperature=strat.floats(min_value=0.0, max_value=1.0),
    max_tokens=strat.integers(min_value=1, max_value=100),
    model_name=strat.text(min_size=1, max_size=10),
)
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_stream_response_uses_chat_settings(
    httpx_mock: pytest_httpx.HTTPXMock,
    openai_client: openai.OpenAI,
    chatbot_settings: settings.Settings,
    temperature: float,
    max_tokens: int,
    model_name: str,
) -> None:
    """Test that stream_response uses chat settings from configuration.

    Args:
        httpx_mock: The mock HTTPX client.
        openai_client: The OpenAI client.
        chatbot_settings: A copy of default chatbot settings.
        temperature: The temperature value to be used in the test.
        max_tokens: The maximum number of tokens to generate.
        model_name: The name of the model to be used in the test.
    """
    seed = random.randint(0, 1000000)
    chatbot_settings.temperature = temperature
    chatbot_settings.max_tokens = max_tokens
    chatbot_settings.llm_model_name = model_name
    chatbot_settings.seed = seed
    messages: list[chat_types.ChatCompletionMessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    expected_chunks = ["Hello", " there", "!"]
    captured_request: dict[str, Any] = {}

    def callback(request: httpx.Request) -> httpx.Response:
        """Callback to capture the request payload.

        Args:
            request: The incoming httpx.Request.

        Returns:
            A mock httpx.Response with streaming data.
        """
        captured_request["content"] = json.loads(request.content)

        return httpx.Response(
            status_code=200,
            stream=pytest_httpx.IteratorStream([chunk.encode("utf-8") for chunk in expected_chunks]),
        )

    httpx_mock.add_callback(callback, method="POST", url=TEST_SERVER_CHAT_COMPLETIONS_URL)

    # Execute the stream_response function
    list(chat.stream_response(messages, openai_client))

    # Validate that chat settings are properly sent in the request
    payload = captured_request["content"]

    # Validate specific settings values are used
    assert payload["model"] == model_name
    assert payload["temperature"] == temperature
    assert payload["max_tokens"] == max_tokens
    assert payload["seed"] == seed
    assert payload["stream"] is True
