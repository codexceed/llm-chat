"""Common constants for the chatbot application."""

import enum
from typing import Final, Literal, TypedDict


class Message(TypedDict):
    """A dictionary representing a chat message."""

    role: Literal["user", "assistant"]
    content: str


class FileTypes(enum.Enum):
    """Supported file types for RAG processing."""

    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    UNKNOWN = "unknown"


FILE_EXTENSION_TYPE_MAPPING: Final[dict[FileTypes, set[str]]] = {
    FileTypes.CODE: {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".cs",
        ".php",
        ".rb",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".sql",
        ".r",
        ".m",
        ".pl",
    },
    FileTypes.MARKDOWN: {".md"},
    FileTypes.HTML: {".html", ".htm"},
    FileTypes.TEXT: {".txt"},
}

# Map file extensions to CodeSplitter language identifiers
EXTENSION_TO_LANGUAGE_MAPPING: Final[dict[str, str]] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "c_sharp",
    ".php": "php",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
    ".r": "r",
    ".m": "objective_c",
    ".pl": "perl",
}
