import streamlit as st
from streamlit.elements.widgets import chat

from chatbot import settings


def render_sidebar() -> None:
    """Renders the sidebar for the chatbot application, including model settings and chat controls."""
    with st.sidebar:
        st.title("Model Settings")
        st.code(settings.CHATBOT_SETTINGS.llm_model_name, language="bash")
        settings.CHATBOT_SETTINGS.temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            settings.CHATBOT_SETTINGS.temperature,
        )
        settings.CHATBOT_SETTINGS.max_tokens = st.slider("Max Tokens", 1, 7000, settings.CHATBOT_SETTINGS.max_tokens)
        settings.CHATBOT_SETTINGS.repetition_penalty = st.slider(
            "Repetition Penalty",
            1.0,
            2.0,
            settings.CHATBOT_SETTINGS.repetition_penalty,
        )
        settings.CHATBOT_SETTINGS.seed = st.number_input("Seed", 0, 1000000, settings.CHATBOT_SETTINGS.seed)

        st.title("Chat Controls")
        if st.sidebar.button("Clear Chat"):
            st.session_state.messages = []


def _switch_web_search_flag() -> None:
    """Switches the web search flag in session state."""
    st.session_state.force_web_search = not st.session_state.force_web_search


def _switch_deep_reasoning_flag() -> None:
    """Switches the deep reasoning flag in session state."""
    st.session_state.force_deep_reasoning = not st.session_state.force_deep_reasoning


@st.fragment
def _toggle_web_search() -> None:
    """Toggles the web search feature."""
    st.toggle(
        "🔍 Enable web search",
        value=st.session_state.get("force_web_search", False),
        help="Enable web search functionality",
        on_change=_switch_web_search_flag,
    )


@st.fragment
def _toggle_deep_reasoning() -> None:
    """Toggles the deep reasoning feature."""
    st.toggle(
        "🧠 Deep reasoning",
        value=st.session_state.get("force_deep_reasoning", False),
        help="Force multi-step reasoning for complex analysis",
        on_change=_switch_deep_reasoning_flag,
    )


def render_chat_interface() -> chat.ChatInputValue | None:
    """Renders the chat interface, including the chat history and input.

    Returns:
       The user's input as a `ChatInputValue` object, or `None` if no input
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Feature toggles above chat input
    with st._bottom.container():  # pylint: disable=protected-access
        col1, col2, _ = st.columns([0.2, 0.2, 0.8])
        with col1:
            _toggle_web_search()
        with col2:
            _toggle_deep_reasoning()

    return st.chat_input("What is up?", accept_file=True)
