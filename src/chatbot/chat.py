import streamlit as st
from openai import OpenAI
from chatbot.config import settings

client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_api_base,
)


def process_uploaded_file(uploaded_file):
    """Reads the content of the uploaded file and adds it to the chat history."""
    if uploaded_file is not None:
        try:
            string_data = uploaded_file.getvalue().decode("utf-8")
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": f"Uploaded file: {uploaded_file.name}\n\n{string_data}",
                }
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")


def stream_response():
    """Streams the response from the language model."""
    stream = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""


def generate_response():
    """Generates a response from the language model and displays it."""
    with st.chat_message("assistant"):
        response = st.write_stream(stream_response)
    st.session_state.messages.append({"role": "assistant", "content": response})
