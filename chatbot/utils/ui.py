import streamlit as st
from streamlit.elements.widgets import chat

from chatbot.config import settings


def render_sidebar() -> None:
    """Renders the sidebar for the chatbot application, including model settings and chat controls."""
    with st.sidebar:
        st.title("Model Settings")
        st.code(settings.llm_model_name, language="bash")
        settings.temperature = st.slider("Temperature", 0.0, 1.0, settings.temperature)
        settings.max_tokens = st.slider("Max Tokens", 1, 4096, settings.max_tokens)
        settings.repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, settings.repetition_penalty)
        settings.seed = st.number_input("Seed", 0, 1000000, settings.seed)

        st.title("Chat Controls")
        if st.sidebar.button("Clear Chat"):
            st.session_state.messages = []


def render_chat_interface() -> chat.ChatInputValue | None:
    """Renders the chat interface, including the chat history and input."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    return st.chat_input("What is up?", accept_file=True)
