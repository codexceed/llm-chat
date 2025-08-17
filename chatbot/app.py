"""Main entry point for the chatbot application."""

import asyncio
import copy

import httpx
import openai
import streamlit as st
from streamlit import logger

from chatbot import constants, resources, settings
from chatbot.utils import chat, ui
from chatbot.web import context, search

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
    if "search_manager" not in st.session_state and settings.CHATBOT_SETTINGS.search.enabled:
        st.session_state.search_manager = search.SearchManager(
            provider=search.SearchProvider(settings.CHATBOT_SETTINGS.search.provider),
            api_key=settings.CHATBOT_SETTINGS.search.api_key,
            trigger_words=settings.CHATBOT_SETTINGS.search.trigger_words,
            search_engine_id=settings.CHATBOT_SETTINGS.search.search_id,
        )
    if "web_context_pipeline" not in st.session_state:
        st.session_state.web_context_pipeline = context.WebContextPipeline(st.session_state.get("search_manager"))
    if "force_web_search" not in st.session_state:
        st.session_state.force_web_search = False


async def main() -> None:
    """Main function for the Streamlit chatbot app.

    Raises:
        TypeError: If an error occurs during the processing of the request.
    """
    st.logo("assets/rand_logo.jpg")
    st.set_page_config(layout="wide")
    ui.render_sidebar()
    web_context_pipeline: context.WebContextPipeline = st.session_state.web_context_pipeline

    if chat_input := ui.render_chat_interface():
        prompt, uploaded_files = chat_input.text, chat_input.files

        LOGGER.debug("Received prompt: %s", prompt)

        st.session_state.messages.append(constants.Message(role="user", content=prompt))
        contextualized_messages = copy.deepcopy(st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

            # Initialize context components
            context_text = ""
            rag_context = ""

            with st.spinner("Retrieving additional context..."):
                # Always get web context (independent of RAG)
                LOGGER.info("Gathering web context.")

                web_context_dict = await web_context_pipeline.gather_web_context(
                    prompt,
                    st.session_state.http_client,
                    enable_web_search=settings.CHATBOT_SETTINGS.search.enabled,
                    force_web_search=st.session_state.force_web_search,
                    search_num_results=settings.CHATBOT_SETTINGS.search.num_results,
                )

                # If RAG is enabled, retrieve RAG context
                if settings.CHATBOT_SETTINGS.rag.enabled:
                    LOGGER.info("Gathering RAG context.")
                    # Process files through RAG system (if any)
                    if uploaded_files:
                        LOGGER.debug(
                            "Processing uploaded files:\n-%s", "- ".join([file.name for file in uploaded_files])
                        )
                        RAG_PROCESSOR.process_uploaded_files(uploaded_files)

                    # Get RAG context from vector database
                    rag_context_chunks = RAG_PROCESSOR.retrieve(prompt)
                    rag_context = "\n\n".join(rag_context_chunks) if rag_context_chunks else ""

                # Merge all context sources (web + RAG)
                context_text = web_context_pipeline.merge_context(web_context_dict, rag_context)

            if context_text:
                with st.expander("Relevant Context", expanded=False):
                    st.text(context_text[: settings.CHATBOT_SETTINGS.context_view_size] + "...")

            # Apply context to prompt if any context was gathered
            if context_text:
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
    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(main())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
