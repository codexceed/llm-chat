"""Test utilities for generating hypothesis strategies."""

import time
import uuid

from hypothesis import strategies as strat
from openai.types import chat as chat_types
from openai.types.chat import chat_completion_chunk as chunk_types


@strat.composite
def chat_completion_chunks_strategy(draw: strat.DrawFn, num_chunks: int) -> list[str]:
    """Strategy to generate a list of chat completion chunks.

    Args:
        draw: The strategy draw function.
        num_chunks: The number of chunks to generate.

    Returns:
       A list of chat completion chunks.
    """
    # Generate base content chunks (without the final empty string)
    model_name = "some_model"

    content_chunks: list[str] = []
    for _ in range(num_chunks):
        chunk_content = draw(
            strat.text(
                min_size=0,
                max_size=10,
                alphabet=strat.characters(
                    whitelist_categories=(
                        "Lu",
                        "Ll",
                        "Lt",
                        "Lm",
                        "Lo",
                        "Nd",
                        "Pc",
                        "Pd",
                        "Ps",
                        "Pe",
                        "Pi",
                        "Pf",
                        "Po",
                        "Sm",
                        "Sc",
                        "Sk",
                        "So",
                        "Zs",
                    ),
                ),
            ),
        )
        content_chunks.append(chunk_content)

    # Build the streaming response
    response_lines: list[str] = []

    # Add content chunks
    idx = 0
    for chunk in content_chunks:
        chunk_data = chat_types.ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[chunk_types.Choice(index=idx, delta=chunk_types.ChoiceDelta(content=chunk))],
            created=int(time.time()),
            model=model_name,
            object="chat.completion.chunk",
        )
        response_lines.append(f"data: {chunk_data.to_json()}")
        idx += 1

    # Add final null content chunk
    final_chunk_data = chunk_data = chat_types.ChatCompletionChunk(
        id=str(uuid.uuid4()),
        choices=[chunk_types.Choice(index=idx, delta=chunk_types.ChoiceDelta(content=None))],
        created=int(time.time()),
        model=model_name,
        object="chat.completion.chunk",
    )
    response_lines.append(f"data: {final_chunk_data.to_json()}")

    # Add [DONE] marker
    response_lines.append("data: [DONE]")

    # Expected chunks include the original chunks plus empty string for null content
    expected_chunks = [*content_chunks, ""]

    return expected_chunks


@strat.composite
def chat_completion_messages_strategy(draw: strat.DrawFn) -> list[chat_types.ChatCompletionMessageParam]:
    """Generate a list of valid chat completion messages.

    Args:
        draw: A function to draw values from strategies.

    Returns:
        A list of chat completion messages that will pass validation.
    """
    # Generate 1-5 messages before the final user message
    num_messages = draw(strat.integers(min_value=0, max_value=4))

    messages: list[chat_types.ChatCompletionMessageParam] = []

    # Add optional system/assistant messages
    for _ in range(num_messages):
        role = draw(strat.sampled_from(["system", "assistant", "user"]))
        content = draw(
            strat.text(
                min_size=1,
                max_size=20,
                alphabet=strat.characters(
                    whitelist_categories=(
                        "Lu",
                        "Ll",
                        "Lt",
                        "Lm",
                        "Lo",  # Letters
                        "Nd",  # Numbers
                        "Pc",
                        "Pd",
                        "Ps",
                        "Pe",
                        "Pi",
                        "Pf",
                        "Po",  # Punctuation
                        "Sm",
                        "Sc",
                        "Sk",
                        "So",  # Symbols
                        "Zs",  # Spaces
                    ),
                ),
            ),
        ).strip()

        # Ensure content is not empty after stripping
        if content:
            messages.append({"role": role, "content": content})  # type: ignore

    # Always end with a valid user message (required by stream_response)
    user_content = draw(
        strat.text(
            min_size=1,
            max_size=20,
            alphabet=strat.characters(
                whitelist_categories=(
                    "Lu",
                    "Ll",
                    "Lt",
                    "Lm",
                    "Lo",  # Letters
                    "Nd",  # Numbers
                    "Pc",
                    "Pd",
                    "Ps",
                    "Pe",
                    "Pi",
                    "Pf",
                    "Po",  # Punctuation
                    "Sm",
                    "Sc",
                    "Sk",
                    "So",  # Symbols
                    "Zs",  # Spaces
                ),
            ),
        ),
    ).strip()

    # Ensure the user content is not empty
    if not user_content:
        user_content = "Hello"  # Fallback to ensure valid content

    messages.append({"role": "user", "content": user_content})

    return messages
