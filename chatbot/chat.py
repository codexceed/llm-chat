from collections.abc import Iterator

import openai
import streamlit as st
from streamlit.runtime import uploaded_file_manager

from chatbot.config import settings
from chatbot.types import Message

client = openai.OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_api_base,
)


def process_uploaded_file(
    uploaded_file: uploaded_file_manager.UploadedFile | None,
) -> None:
    """Reads the content of the uploaded file and adds it to the chat history."""
    if uploaded_file is not None:
        try:
            string_data: str = uploaded_file.getvalue().decode("utf-8")
            st.session_state.messages.append(
                Message(
                    role="user",
                    content=f"Uploaded file: {uploaded_file.name}\n\n{string_data}",
                )
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")


def stream_response() -> Iterator[str]:
    """Streams the response from the language model.

    Yields:
        Response chunks from the language model as they are generated.
    """
    stream = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        seed=settings.seed,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""


def generate_response() -> None:
    """Generates a response from the language model and displays it."""
    with st.chat_message("assistant"):
        response = st.write_stream(stream_response)
    if isinstance(response, str):
        st.session_state.messages.append(Message(role="assistant", content=response))
    else:
        raise TypeError("Expected response to be str, got %s", type(response).__name__)
