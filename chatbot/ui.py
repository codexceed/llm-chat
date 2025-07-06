import streamlit as st

from chatbot.config import settings


def render_sidebar():
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

    st.sidebar.title("File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file", type=["txt", "md", "py", "json", "csv"]
    )

    st.sidebar.title("Chat Controls")
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []

    return uploaded_file


def render_chat_interface():
    """Renders the chat interface, including the chat history and input."""
    st.title(settings.page_title)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    return st.chat_input("What is up?")
