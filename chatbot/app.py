import asyncio
import logging

import httpx
import openai
import streamlit as st

from chatbot import config, constants
from chatbot.utils import chat, ui

LOGGER = logging.getLogger(__name__)


def initialize_session_state() -> None:
    """Initializes the session state for the chatbot application."""
    LOGGER.info("Initializing session state.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "http_client" not in st.session_state:
        st.session_state.http_client = httpx.AsyncClient()
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = openai.OpenAI(
            api_key=config.settings.openai_api_key,
            base_url=config.settings.openai_api_base,
        )


def main() -> None:
    """Main function for the Streamlit chatbot app."""
    st.logo("assets/rand_logo.jpg")
    st.set_page_config(layout="wide")
    ui.render_sidebar()

    if chat_input := ui.render_chat_interface():
        prompt, uploaded_files = chat_input.text, chat_input.files

        asyncio.run(chat.process_web_urls_in_prompt(prompt, st.session_state.http_client))
        if uploaded_files:
            chat.process_uploaded_files(uploaded_files)

        st.session_state.messages.append(constants.Message(role="user", content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.write_stream(chat.stream_response(st.session_state.messages, st.session_state.openai_client))
            if isinstance(response, str):
                st.session_state.messages.append(constants.Message(role="assistant", content=response))
            else:
                raise TypeError("Expected response to be str, got %s", type(response).__name__)


if __name__ == "__main__":
    initialize_session_state()
    main()
