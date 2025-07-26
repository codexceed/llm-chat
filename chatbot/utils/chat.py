from collections.abc import Iterator

import openai
from openai.types import chat as chat_types
from streamlit import logger

from chatbot import resources, settings

LOGGER = logger.get_logger(__name__)
RAG_PROCESSOR = resources.get_rag_processor()


def stream_response(messages: list[chat_types.ChatCompletionMessageParam], openai_client: openai.OpenAI) -> Iterator[str]:
    """Streams the response from the language model.

    Args:
        messages: List of chat messages to send to the language model.
        openai_client: OpenAI client instance for making API calls.

    Yields:
        Response chunks from the language model as they are generated.
    """
    if (
        not messages
        or messages[-1]["role"] != "user"
        or not isinstance(messages[-1]["content"], str)
        or not messages[-1]["content"].strip()
    ):
        raise ValueError("No messages provided for response generation.")

    stream = openai_client.chat.completions.create(
        model=settings.settings.llm_model_name,
        messages=messages,  # type: ignore
        stream=True,
        temperature=settings.settings.temperature,
        max_tokens=settings.settings.max_tokens,
        seed=settings.settings.seed,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""  # type: ignore
