from typing import Literal, TypedDict


class Message(TypedDict):
    """A dictionary representing a chat message."""

    role: Literal["user", "assistant"]
    content: str
