import streamlit as st
from streamlit.runtime import uploaded_file_manager

from chatbot.config import settings
from chatbot.types import Message


def render_sidebar() -> uploaded_file_manager.UploadedFile | None:
    """Renders the sidebar with model settings, file uploader, and chat controls."""
    st.sidebar.title("Model Settings")
    settings.model_name = st.sidebar.text_input("Model", settings.model_name)
    settings.temperature = st.sidebar.slider(
        "Temperature", 0.0, 1.0, settings.temperature
    )
    settings.max_tokens = st.sidebar.slider("Max Tokens", 1, 4096, settings.max_tokens)
    settings.repetition_penalty = st.sidebar.slider(
        "Repetition Penalty", 1.0, 2.0, settings.repetition_penalty
    )
    settings.seed = st.sidebar.number_input("Seed", 0, 1000000, settings.seed)

    st.sidebar.title("File Upload")
    uploaded_file: uploaded_file_manager.UploadedFile | None = st.sidebar.file_uploader(
        "Upload a file", type=["txt", "md", "py", "json", "csv"]
    )

    st.sidebar.title("Chat Controls")
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []

    return uploaded_file


def render_chat_interface() -> str | None:
    """Renders the chat interface, including the chat history and input."""
    st.title(settings.page_title)

    if "messages" not in st.session_state:
        messages_init: list[Message] = []
        st.session_state.messages = messages_init

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    return st.chat_input("What is up?")
