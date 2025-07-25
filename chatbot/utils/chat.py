from collections.abc import Iterator
import logging

import httpx
import openai
from openai.types import chat as chat_types
from streamlit.runtime import uploaded_file_manager

from chatbot import config, resources

LOGGER = logging.getLogger(__name__)
RAG_PROCESSOR = resources.get_rag_processor()


async def process_web_urls_in_prompt(prompt: str, client: httpx.AsyncClient) -> None:
    """Extract web URLs from the user prompt and index their content for RAG.

    Args:
        prompt: User input containing potential web URLs.
        client: HTTP client for making requests to fetch URL content.
    """
    await RAG_PROCESSOR.process_web_urls(prompt, client)


def process_uploaded_files(
    uploaded_files: list[uploaded_file_manager.UploadedFile],
) -> None:
    """Process and index uploaded files for RAG

    Args:
        uploaded_file: List of uploaded files from the chat input.
    """
    RAG_PROCESSOR.process_uploaded_files(uploaded_files)


def stream_response(
    messages: list[chat_types.ChatCompletionMessageParam], openai_client: openai.OpenAI
) -> Iterator[str]:
    """Streams the response from the language model.

    Args:
        messages: List of chat messages to send to the language model.

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

    # Get RAG context if available
    user_query = messages[-1]["content"]
    rag_context_chunks = RAG_PROCESSOR.retrieve(user_query)

    # If we have RAG context, append it to the most recent user message
    if rag_context_chunks and messages:
        context_text = "\n\n".join(rag_context_chunks)
        messages[-1]["content"] = f"{messages[-1]['content']}\n\nRelevant context:\n{context_text}"

    stream = openai_client.chat.completions.create(
        model=config.settings.llm_model_name,
        messages=messages,  # type: ignore
        stream=True,
        temperature=config.settings.temperature,
        max_tokens=config.settings.max_tokens,
        seed=config.settings.seed,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""  # type: ignore
