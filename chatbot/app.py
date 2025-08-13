"""Main entry point for the chatbot application."""

import asyncio
import copy

import httpx
import openai
import streamlit as st
from streamlit import logger

from chatbot import constants, resources, settings
from chatbot.utils import chat, ui

LOGGER = logger.get_logger(__name__)
RAG_PROCESSOR = resources.get_rag_processor()
PROMPT_TEMPLATE = """
You are a helpful assistant that answers based on the context.

Context:
{context}

Question:
{prompt}
"""


def initialize_session_state() -> None:
    """Initializes the session state for the chatbot application."""
    LOGGER.info("Initializing session state.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "http_client" not in st.session_state:
        st.session_state.http_client = httpx.AsyncClient()
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = openai.OpenAI(
            api_key=settings.CHATBOT_SETTINGS.openai_api_key,
            base_url=settings.CHATBOT_SETTINGS.openai_api_base,
        )


def main() -> None:
    """Main function for the Streamlit chatbot app.

    Raises:
        TypeError: If an error occurs during the processing of the request.
    """
    st.logo("assets/rand_logo.jpg")
    st.set_page_config(layout="wide")
    ui.render_sidebar()

    if chat_input := ui.render_chat_interface():
        prompt, uploaded_files = chat_input.text, chat_input.files

        LOGGER.debug("Received prompt: %s", prompt)

        st.session_state.messages.append(constants.Message(role="user", content=prompt))
        contextualized_messages = copy.deepcopy(st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

            # If RAG is enabled, retrieve additional context
            if settings.CHATBOT_SETTINGS.rag.enabled:
                LOGGER.info("Initiating RAG processing.")
                context_text = ""
                with st.spinner("Retrieving additional context..."):
                    asyncio.run(RAG_PROCESSOR.process_web_urls(prompt, st.session_state.http_client))
                    if uploaded_files:
                        LOGGER.debug(
                            "Processing uploaded files:\n-%s", "- ".join([file.name for file in uploaded_files])
                        )
                        RAG_PROCESSOR.process_uploaded_files(uploaded_files)
                    rag_context_chunks = RAG_PROCESSOR.retrieve(prompt)
                    if rag_context_chunks:
                        context_text = "\n\n".join(rag_context_chunks)

                if context_text:
                    with st.expander("Relevant Context", expanded=False):
                        st.text(context_text[: settings.CHATBOT_SETTINGS.context_view_size] + "...")

                contextualized_prompt = PROMPT_TEMPLATE.format(context=context_text, prompt=prompt)
                contextualized_messages[-1]["content"] = contextualized_prompt

        # Stream the response from the LLM
        with st.chat_message("assistant"):
            response = st.write_stream(chat.stream_response(contextualized_messages, st.session_state.openai_client))
            if isinstance(response, str):
                st.session_state.messages.append(constants.Message(role="assistant", content=response))
            else:
                raise TypeError(f"Expected response to be str, got {type(response).__name__}")


if __name__ == "__main__":
    initialize_session_state()
    main()
