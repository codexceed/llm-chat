from collections.abc import Iterator

import openai
from openai.types import chat as chat_types
from streamlit import logger

from chatbot import resources, settings

LOGGER = logger.get_logger("streamlit")
RAG_PROCESSOR = resources.get_rag_processor()


def stream_response(
    messages: list[chat_types.ChatCompletionMessageParam],
    openai_client: openai.OpenAI,
) -> Iterator[str]:
    """Streams the response from the language model.

    Args:
        messages: A list of chat messages.
        openai_client: The OpenAI client instance.

    Yields:
        A string containing the next chunk of the response.

    Raises:
        ValueError: If no messages are provided for response generation.
    """
    if (
        not messages
        or messages[-1]["role"] != "user"
        or not isinstance(messages[-1]["content"], str)
        or not messages[-1]["content"].strip()
    ):
        raise ValueError("No messages provided for response generation.")

    stream = openai_client.chat.completions.create(
        model=settings.CHATBOT_SETTINGS.llm_model_name,
        messages=messages,
        stream=True,
        temperature=settings.CHATBOT_SETTINGS.temperature,
        max_tokens=settings.CHATBOT_SETTINGS.max_tokens,
        seed=settings.CHATBOT_SETTINGS.seed,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""
