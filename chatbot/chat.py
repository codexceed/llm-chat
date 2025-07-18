from collections.abc import Iterator

import openai
from openai.types import chat as chat_types
import streamlit as st
from streamlit.runtime import uploaded_file_manager

from chatbot import config
from chatbot.rag import RAG
from chatbot.types import Message

client = openai.OpenAI(
    api_key=config.settings.openai_api_key,
    base_url=config.settings.openai_api_base,
)

# Initialize RAG processor
rag_processor = RAG()


def process_uploaded_files(
    uploaded_files: list[uploaded_file_manager.UploadedFile],
) -> None:
    """Process and index uploaded files for RAG

    Args:
        uploaded_file: List of uploaded files from the chat input.
    """
    rag_processor.process_uploaded_files(uploaded_files)


def stream_response(
    messages: list[chat_types.ChatCompletionMessageParam],
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
    rag_context_chunks = rag_processor.retrieve(user_query)

    # If we have RAG context, append it to the most recent user message
    if rag_context_chunks and messages:
        context_text = "\n\n".join(rag_context_chunks)
        messages[-1]["content"] = f"{messages[-1]['content']}\n\nRelevant context:\n{context_text}"

    stream = client.chat.completions.create(
        model=config.settings.llm_model_name,
        messages=messages,  # type: ignore
        stream=True,
        temperature=config.settings.temperature,
        max_tokens=config.settings.max_tokens,
        seed=config.settings.seed,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""  # type: ignore


def generate_response(messages: list[chat_types.ChatCompletionMessageParam]) -> None:
    """Generates a response from the language model and displays it."""
    response = st.write_stream(stream_response(messages))
    if isinstance(response, str):
        st.session_state.messages.append(Message(role="assistant", content=response))
    else:
        raise TypeError("Expected response to be str, got %s", type(response).__name__)
