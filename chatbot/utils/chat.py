import openai
import streamlit as st
from openai.types import chat as chat_types
from streamlit import delta_generator, logger

from chatbot import resources, settings

LOGGER = logger.get_logger("streamlit")
RAG_PROCESSOR = resources.get_rag_processor()


async def stream_response(
    messages: list[chat_types.ChatCompletionMessageParam],
    openai_client: openai.AsyncOpenAI,
    chat_container: delta_generator.DeltaGenerator | None = None,
) -> str:
    """Streams the response from the language model.

    Args:
        messages: A list of chat messages.
        openai_client: The OpenAI client instance.
        chat_container: Optional streamlit chat container for display.

    Returns:
        Complete streamed text content.

    Raises:
        ValueError: If no messages are provided for response generation.
    """
    if (
        not messages
        or messages[-1]["role"] != "user"
        or not isinstance(messages[-1]["content"], str)
        or not messages[-1]["content"].strip()
    ):
        raise ValueError("No messages provided for response generation.")

    stream = await openai_client.chat.completions.create(
        model=settings.CHATBOT_SETTINGS.llm_model_name,
        messages=messages,
        stream=True,
        temperature=settings.CHATBOT_SETTINGS.temperature,
        max_tokens=settings.CHATBOT_SETTINGS.max_tokens,
        seed=settings.CHATBOT_SETTINGS.seed,
    )

    if chat_container:
        with chat_container:
            placeholder = st.empty()
            streamed_text = ""

            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                streamed_text += content
                with placeholder:
                    st.write(streamed_text)

            if not streamed_text:
                st.error("Empty response detected. Please try again.")
                raise ValueError("Empty response detected at the end of streaming.")

        return streamed_text

    content = ""
    async for chunk in stream:
        content += chunk.choices[0].delta.content or ""
    return content
