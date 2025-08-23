"""Main entry point for the chatbot application."""

import asyncio
import copy
from typing import Any, Final

import httpx
import openai
import streamlit as st
from streamlit import logger

from chatbot import constants, resources, settings
from chatbot.reasoning import classifier, orchestrator
from chatbot.utils import chat, ui
from chatbot.web import context, search

LOGGER = logger.get_logger(__name__)
RAG_PROCESSOR = resources.get_rag_processor()
SIMPLE_CONTEXT_PROMPT_TEMPLATE: Final[str] = """
You are a helpful assistant that answers based on the context.

Context:
{context}

Question:
{prompt}
"""

REASONED_CONTEXT_PROMPT_TEMPLATE: Final[str] = """
Based on the multi-step reasoning results below, provide a comprehensive answer to the original question.

Original Question: {prompt}

Multi-Step Reasoning Results:
{reasoning_context}

Please synthesize this information into a clear, well-structured answer that directly addresses
the original question. Focus on the key insights and connections across the different research steps.

Answer:"""


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
    if "query_classifier" not in st.session_state:
        st.session_state.query_classifier = classifier.QueryComplexityClassifier()
    if "multi_step_orchestrator" not in st.session_state and settings.CHATBOT_SETTINGS.multi_step.enabled:
        st.session_state.multi_step_orchestrator = orchestrator.MultiStepOrchestrator(
            web_context_pipeline=st.session_state.web_context_pipeline,
            openai_client=st.session_state.openai_client,
            http_client=st.session_state.http_client,
            model_name=settings.CHATBOT_SETTINGS.llm_model_name,
            seed=settings.CHATBOT_SETTINGS.seed,
            max_steps=settings.CHATBOT_SETTINGS.multi_step.max_steps,
            planning_temperature=settings.CHATBOT_SETTINGS.multi_step.planning_temperature,
            step_timeout=settings.CHATBOT_SETTINGS.multi_step.step_timeout,
            search_top_k=settings.CHATBOT_SETTINGS.multi_step.search_top_k,
            max_reasoning_tokens=settings.CHATBOT_SETTINGS.multi_step.max_reasoning_tokens,
            max_context_tokens=settings.CHATBOT_SETTINGS.multi_step.max_context_length,
        )


async def _process_multi_step_query(prompt: str) -> str | None:
    """Process query using multi-step reasoning.

    Args:
        prompt: User query to process

    Returns:
        Multi-step reasoning context if successfully synthesized, else None
    """
    LOGGER.info("Using multi-step reasoning for complex query")

    with st.spinner("Processing complex query with multi-step reasoning..."):
        # Use multi-step orchestrator for complex queries
        multi_step_orchestrator: orchestrator.MultiStepOrchestrator = st.session_state.multi_step_orchestrator
        if reasoned_context := await multi_step_orchestrator.execute_complex_query(prompt):
            # Show reasoning context
            with st.expander("Multi-Step Reasoning Process", expanded=False):
                st.info("This query was processed using multi-step reasoning for more thorough analysis.")
        return reasoned_context


async def _gather_context_for_simple_query(
    prompt: str,
    uploaded_files: list[Any],
    web_context_pipeline: context.WebContextPipeline,
) -> str:
    """Gather context for simple query processing.

    Args:
        prompt: User query
        uploaded_files: List of uploaded files
        web_context_pipeline: Web context pipeline instance

    Returns:
        Combined context text
    """
    LOGGER.info("Using simple context injection approach")

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
                    "Processing uploaded files:\n-%s",
                    "- ".join([file.name for file in uploaded_files]),
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

    return context_text


async def main() -> None:
    """Main function for the Streamlit chatbot app.

    Raises:
        TypeError: If an error occurs during the processing of the request.
    """
    st.logo("assets/rand_logo.jpg")
    st.set_page_config(layout="wide")
    ui.render_sidebar()
    web_context_pipeline: context.WebContextPipeline = st.session_state.web_context_pipeline
    query_classifier: classifier.QueryComplexityClassifier = st.session_state.query_classifier

    if chat_input := ui.render_chat_interface():
        prompt, uploaded_files = chat_input.text, chat_input.files

        LOGGER.debug("Received prompt: %s", prompt)

        st.session_state.messages.append(constants.Message(role="user", content=prompt))
        contextualized_messages = copy.deepcopy(st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if multi-step reasoning is enabled and needed
        use_multi_step = (
            settings.CHATBOT_SETTINGS.multi_step.enabled
            and "multi_step_orchestrator" in st.session_state
            and query_classifier.classify_query(prompt) == classifier.QueryComplexity.COMPLEX
        )

        context_text: str | None = ""
        if use_multi_step:
            context_text = await _process_multi_step_query(prompt)
            if context_text:
                contextualized_prompt = REASONED_CONTEXT_PROMPT_TEMPLATE.format(
                    prompt=prompt,
                    reasoning_context=context_text,
                )
                contextualized_messages[-1]["content"] = contextualized_prompt
            else:
                LOGGER.warning("Multi-step reasoning did not yield any context, falling back to simple context.")

        if not context_text:
            context_text = await _gather_context_for_simple_query(prompt, uploaded_files, web_context_pipeline)
            if context_text:
                contextualized_prompt = SIMPLE_CONTEXT_PROMPT_TEMPLATE.format(context=context_text, prompt=prompt)
                contextualized_messages[-1]["content"] = contextualized_prompt

        # Stream the response from the LLM
        with st.chat_message("assistant"):
            response_content = st.write_stream(
                chat.stream_response(contextualized_messages, st.session_state.openai_client),
            )

        if isinstance(response_content, str):
            st.session_state.messages.append(constants.Message(role="assistant", content=response_content))
        else:
            error_message = f"Expected response to be str, got {type(response_content).__name__}"
            raise TypeError(error_message)


if __name__ == "__main__":
    initialize_session_state()
    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(main())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
