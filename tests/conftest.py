"""Shared test configuration and fixtures."""

import copy
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import settings as hypothesis_settings
from streamlit.runtime.uploaded_file_manager import UploadedFile

from chatbot import settings
from chatbot.settings import QdrantSettings, RAGSettings, Settings

# Configure Hypothesis settings for faster test runs
hypothesis_settings.register_profile("default", max_examples=50, deadline=5000)
hypothesis_settings.load_profile("default")


class MockUploadedFile(UploadedFile):
    """Mock UploadedFile for testing file upload functionality."""

    def __init__(self, name: str, content: bytes, file_id: str = "test_file", mimetype: str = "text/plain"):
        self.name = name
        self._content = content
        self.file_id = file_id
        self.size = len(content)
        self.type = mimetype

    def getvalue(self) -> bytes:
        """Return the file content as bytes."""
        return self._content

    def read(self, size: int | None = -1) -> bytes:
        """Read file content.

        Returns:
            The content up to `size` bytes, or the entire content if `size` is not specified.
        """
        if size is None or size == -1:
            return self._content
        return self._content[:size]

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position (no-op for mock).

        Returns:
            The new position.
        """
        return 0

    def tell(self) -> int:
        """Return current position (always 0 for mock)."""
        return 0


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        The path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_files(temp_directory: Path) -> dict[str, Path]:
    """Create sample files for testing.

    Returns:
        A dictionary mapping file types to their paths.
    """
    files = {}

    # Python file
    python_file = temp_directory / "sample.py"
    python_file.write_text(
        """
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "Hello, World!"

class Calculator:
    '''A simple calculator class.'''

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
"""
    )
    files["python"] = python_file

    # Markdown file
    markdown_file = temp_directory / "README.md"
    markdown_file.write_text(
        """
# Test Project

This is a test project for the chatbot application.

## Features

- RAG (Retrieval-Augmented Generation)
- Multiple file format support
- Web URL processing
- Streamlit interface

## Usage

1. Upload files
2. Ask questions
3. Get intelligent responses

### Code Example

```python
from chatbot import RAG

rag = RAG()
context = rag.retrieve("How to use this?")
```
"""
    )
    files["markdown"] = markdown_file

    # HTML file
    html_file = temp_directory / "index.html"
    html_file.write_text(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Welcome to Test Page</h1>
    <p>This is a sample HTML document for testing.</p>
    <ul>
        <li>Feature 1: Fast processing</li>
        <li>Feature 2: Multiple formats</li>
        <li>Feature 3: AI-powered search</li>
    </ul>
    <div class="code-block">
        <pre><code>console.log("Hello from HTML!");</code></pre>
    </div>
</body>
</html>
"""
    )
    files["html"] = html_file

    # Text file
    text_file = temp_directory / "notes.txt"
    text_file.write_text(
        """
Project Notes
=============

Important considerations for the chatbot:

1. Security: Never expose API keys or sensitive data
2. Performance: Implement caching and efficient retrieval
3. User Experience: Provide clear error messages and feedback
4. Scalability: Design for multiple concurrent users
5. Testing: Comprehensive unit and integration tests

Technical debt items:
- Refactor RAG processing pipeline
- Add more embedding model options
- Improve error handling in web scraping
- Add support for more file formats

Ideas for future features:
- Voice input/output
- Multi-language support
- Document summarization
- Citation tracking
"""
    )
    files["text"] = text_file

    # JavaScript file
    js_file = temp_directory / "app.js"
    js_file.write_text(
        """
// Main application JavaScript

class ChatApp {
    constructor() {
        this.messages = [];
        this.isLoading = false;
    }

    async sendMessage(message) {
        this.isLoading = true;
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();
            this.messages.push({ role: 'user', content: message });
            this.messages.push({ role: 'assistant', content: data.response });

            this.renderMessages();
        } catch (error) {
            console.error('Error sending message:', error);
        } finally {
            this.isLoading = false;
        }
    }

    renderMessages() {
        const container = document.getElementById('messages');
        container.innerHTML = '';

        this.messages.forEach(msg => {
            const div = document.createElement('div');
            div.className = `message message-${msg.role}`;
            div.textContent = msg.content;
            container.appendChild(div);
        });
    }
}

const app = new ChatApp();
"""
    )
    files["javascript"] = js_file

    return files


@pytest.fixture
def mock_uploaded_files(sample_files: dict[str, Path]) -> list[MockUploadedFile]:
    """Create mock uploaded files from sample files.

    Returns:
        A list of mock uploaded files.
    """
    uploaded_files = []

    for file_type, file_path in sample_files.items():
        content = file_path.read_bytes()
        mock_file = MockUploadedFile(
            name=file_path.name, content=content, file_id=f"test_{file_type}", mimetype=_get_mimetype(file_path.suffix)
        )
        uploaded_files.append(mock_file)

    return uploaded_files


def _get_mimetype(suffix: str) -> str:
    """Get MIME type for file extension.

    Returns:
        The MIME type for the given file extension.
    """
    mimetypes = {
        ".py": "text/x-python",
        ".js": "application/javascript",
        ".ts": "application/typescript",
        ".jsx": "application/javascript",
        ".tsx": "application/typescript",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".txt": "text/plain",
        ".json": "application/json",
        ".xml": "application/xml",
        ".css": "text/css",
    }
    return mimetypes.get(suffix.lower(), "text/plain")


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults.

    Returns:
        A Settings object with test-safe values.
    """
    return Settings(
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="test-key",
        llm_model_name="test-model",
        temperature=0.5,
        max_tokens=100,
        repetition_penalty=1.0,
        seed=42,
        host="127.0.0.1",
        port=8080,
        debug=True,
        qdrant=QdrantSettings(
            url="http://localhost:6333", api_key=None, collection_name="test_collection", vector_size=384
        ),
        rag=RAGSettings(
            enabled=True,
            embedding_model="test-embedding-model",
            chunk_size=512,
            chunk_overlap=50,
            top_k=3,
            deduplication_similarity_threshold=0.9,
            use_adaptive_parsing=True,
            code_chunk_lines=20,
            code_chunk_overlap_lines=5,
            semantic_breakpoint_threshold=95,
            device="cpu",
            use_hybrid_retrieval=False,  # Disable for testing
            sparse_model="test-sparse-model",
            hybrid_top_k=50,
            enable_relevance_filtering=True,
            relevance_threshold=0.7,
        ),
        context_view_size=500,
    )


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client for testing.

    Returns:
        A mock OpenAI client.
    """
    mock_client = MagicMock()

    # Mock streaming response
    mock_stream = MagicMock()
    mock_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Test"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=" response"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
    ]
    mock_stream.__iter__.return_value = iter(mock_chunks)
    mock_client.chat.completions.create.return_value = mock_stream

    return mock_client


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Create a mock Qdrant client for testing.

    Returns:
        A mock Qdrant client.
    """
    mock_client = MagicMock()

    # Mock collection operations
    mock_client.get_collection.side_effect = Exception("Collection not found")
    mock_client.create_collection.return_value = None
    mock_client.upsert.return_value = None
    mock_client.search.return_value = []

    return mock_client


@pytest.fixture
def mock_embedding_model() -> MagicMock:
    """Create a mock embedding model for testing.

    Returns:
        A mock embedding model.
    """
    mock_model = MagicMock()

    # Mock embedding generation
    mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_model.get_text_embedding_batch.return_value = [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
    ]

    return mock_model


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock vector store for testing.

    Returns:
        A mock vector store.
    """
    mock_store = MagicMock()

    # Mock vector operations
    mock_store.add.return_value = None
    mock_store.query.return_value = []
    mock_store.delete.return_value = None

    return mock_store


@pytest.fixture
def mock_vector_index() -> MagicMock:
    """Create a mock vector index for testing.

    Returns:
        A mock vector index.
    """
    mock_index = MagicMock()

    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_index.as_retriever.return_value = mock_retriever

    # Mock node operations
    mock_index.insert_nodes.return_value = None
    mock_index.delete_nodes.return_value = None

    return mock_index


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """Create a mock httpx async client for testing.

    Returns:
        A mock httpx async client.
    """
    mock_client = MagicMock()

    # Mock successful response
    mock_response = MagicMock()
    mock_response.text = "Mock web content"
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_client.get.return_value = mock_response

    return mock_client


@pytest.fixture(autouse=True)
def isolate_environment() -> Generator[None, None, None]:
    """Isolate environment variables for each test."""
    original_env = os.environ.copy()

    # Clear chatbot-related environment variables
    chatbot_env_vars = [key for key in os.environ if key.startswith("CHATBOT_")]
    for key in chatbot_env_vars:
        del os.environ[key]

    try:
        yield
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


@pytest.fixture
def patch_llama_index() -> Generator[dict[str, MagicMock], None, None]:
    """Patch LlamaIndex components for testing.

    Yields:
        A dictionary of patched LlamaIndex components.
    """
    patches = {}

    with (
        patch("llama_index.core.SimpleDirectoryReader") as mock_reader,
        patch("llama_index.core.VectorStoreIndex.from_vector_store") as mock_index,
        patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding") as mock_embedding,
        patch("llama_index.vector_stores.qdrant.QdrantVectorStore") as mock_store,
        patch("llama_index.core.Settings") as mock_settings,
    ):
        patches["reader"] = mock_reader
        patches["index"] = mock_index
        patches["embedding"] = mock_embedding
        patches["store"] = mock_store
        patches["settings"] = mock_settings

        yield patches


@pytest.fixture
def sample_chat_messages() -> list[dict[str, str]]:
    """Create sample chat messages for testing.

    Returns:
        A list of sample chat messages.
    """
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "Can you explain what RAG is?"},
        {
            "role": "assistant",
            "content": "RAG stands for Retrieval-Augmented Generation. It's a technique that combines information retrieval with text generation to provide more accurate and contextual responses.",
        },
        {"role": "user", "content": "How does it work in this chatbot?"},
    ]


@pytest.fixture
def sample_web_urls() -> list[str]:
    """Create sample web URLs for testing.

    Returns:
        A list of sample web URLs.
    """
    return [
        "https://example.com",
        "http://test.org/page",
        "https://api.example.com/v1/docs",
        "http://localhost:3000/dashboard",
        "https://docs.python.org/3/library/os.html",
    ]


@pytest.fixture
def sample_web_content() -> dict[str, str]:
    """Create sample web content for testing.

    Returns:
        A dictionary of sample web content.
    """
    return {
        "https://example.com": """
        <html>
        <head><title>Example</title></head>
        <body>
            <h1>Welcome to Example</h1>
            <p>This is a sample webpage for testing.</p>
            <ul>
                <li>Feature 1</li>
                <li>Feature 2</li>
                <li>Feature 3</li>
            </ul>
        </body>
        </html>
        """,
        "http://test.org/page": """
        <html>
        <body>
            <h2>Test Page</h2>
            <p>This page contains test content for the chatbot.</p>
            <div class="content">
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                <p>Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
            </div>
        </body>
        </html>
        """,
        "https://api.example.com/v1/docs": """
        <html>
        <body>
            <h1>API Documentation</h1>
            <h2>Endpoints</h2>
            <ul>
                <li>GET /api/users</li>
                <li>POST /api/users</li>
                <li>PUT /api/users/{id}</li>
                <li>DELETE /api/users/{id}</li>
            </ul>
            <h2>Authentication</h2>
            <p>Use Bearer token authentication.</p>
        </body>
        </html>
        """,
    }


@pytest.fixture(scope="function")
def chatbot_settings() -> Generator[settings.Settings, None, None]:
    """Fixture to temporarily modify and restore chatbot settings during tests.

    Yields:
        Reference to chatbot settings.
    """
    original_settings = copy.deepcopy(settings.CHATBOT_SETTINGS)
    yield settings.CHATBOT_SETTINGS
    settings.CHATBOT_SETTINGS = original_settings
