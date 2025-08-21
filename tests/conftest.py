"""Shared test configuration and fixtures."""

import contextlib
import copy
import os
import pathlib
import subprocess
import tempfile
import time
import unittest.mock
from collections.abc import Generator
from typing import Final

import httpx
import hypothesis
import openai
import pytest
import qdrant_client

from chatbot import rag, settings

# pylint: disable=redefined-outer-name

# Configure Hypothesis settings for faster test runs
hypothesis.settings.register_profile("default", max_examples=50, deadline=5000)
hypothesis.settings.load_profile("default")


TEST_SERVER_BASE_URL: Final[str] = "http://testserver/v1"
TEST_SERVER_CHAT_COMPLETIONS_URL: Final[str] = TEST_SERVER_BASE_URL + "/chat/completions"
TEST_API_KEY: Final[str] = "test-key"


@pytest.fixture
def temp_directory() -> Generator[pathlib.Path]:
    """Create a temporary directory for test files.

    Yields:
        The path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def sample_files(temp_directory: pathlib.Path) -> dict[str, pathlib.Path]:
    """Create sample files for testing.

    Args:
        temp_directory: The temporary directory to create files in.

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

rag = chatbot.rag.RAG()
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
def test_settings() -> settings.Settings:
    """Create test settings with safe defaults.

    Returns:
        A chatbot.settings.Settings object with test-safe values.
    """
    return settings.Settings(
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
        qdrant=settings.QdrantSettings(
            url="http://localhost:6333", api_key=None, collection_name="test_collection", vector_size=384
        ),
        rag=settings.RAGSettings(
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
def mock_openai_client() -> unittest.mock.MagicMock:
    """Create a mock OpenAI client for testing.

    Returns:
        A mock OpenAI client.
    """
    mock_client = unittest.mock.MagicMock()

    # Mock streaming response
    mock_stream = unittest.mock.MagicMock()
    mock_chunks = [
        unittest.mock.MagicMock(choices=[unittest.mock.MagicMock(delta=unittest.mock.MagicMock(content="Test"))]),
        unittest.mock.MagicMock(choices=[unittest.mock.MagicMock(delta=unittest.mock.MagicMock(content=" response"))]),
        unittest.mock.MagicMock(choices=[unittest.mock.MagicMock(delta=unittest.mock.MagicMock(content=None))]),
    ]
    mock_stream.__iter__.return_value = iter(mock_chunks)
    mock_client.chat.completions.create.return_value = mock_stream

    return mock_client


@pytest.fixture
def mock_qdrant_client() -> unittest.mock.MagicMock:
    """Create a mock Qdrant client for testing.

    Returns:
        A mock Qdrant client.
    """
    mock_client = unittest.mock.MagicMock()

    # Mock collection operations
    mock_client.get_collection.side_effect = Exception("Collection not found")
    mock_client.create_collection.return_value = None
    mock_client.upsert.return_value = None
    mock_client.search.return_value = []

    return mock_client


@pytest.fixture
def mock_embedding_model() -> unittest.mock.MagicMock:
    """Create a mock embedding model for testing.

    Returns:
        A mock embedding model.
    """
    mock_model = unittest.mock.MagicMock()

    # Mock embedding generation
    mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_model.get_text_embedding_batch.return_value = [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
    ]

    return mock_model


@pytest.fixture
def mock_vector_store() -> unittest.mock.MagicMock:
    """Create a mock vector store for testing.

    Returns:
        A mock vector store.
    """
    mock_store = unittest.mock.MagicMock()

    # Mock vector operations
    mock_store.add.return_value = None
    mock_store.query.return_value = []
    mock_store.delete.return_value = None

    return mock_store


@pytest.fixture
def mock_vector_index() -> unittest.mock.MagicMock:
    """Create a mock vector index for testing.

    Returns:
        A mock vector index.
    """
    mock_index = unittest.mock.MagicMock()

    # Mock retriever
    mock_retriever = unittest.mock.MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_index.as_retriever.return_value = mock_retriever

    # Mock node operations
    mock_index.insert_nodes.return_value = None
    mock_index.delete_nodes.return_value = None

    return mock_index


@pytest.fixture
def mock_httpx_client() -> unittest.mock.MagicMock:
    """Create a mock httpx async client for testing.

    Returns:
        A mock httpx async client.
    """
    mock_client = unittest.mock.MagicMock()

    # Mock successful response
    mock_response = unittest.mock.MagicMock()
    mock_response.text = "Mock web content"
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_client.get.return_value = mock_response

    return mock_client


@pytest.fixture(autouse=True)
def isolate_environment() -> Generator[None]:
    """Isolate environment variables for each test.

    Yields:
        Nothing, but ensures environment isolation.
    """
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
def chatbot_settings() -> Generator[settings.Settings]:
    """Fixture to temporarily modify and restore chatbot settings during tests.

    Yields:
        Reference to chatbot settings.
    """
    original_settings = copy.deepcopy(settings.CHATBOT_SETTINGS)
    yield settings.CHATBOT_SETTINGS
    settings.CHATBOT_SETTINGS = original_settings


def _is_qdrant_running() -> bool:
    """Check if Qdrant service is already running.

    Returns:
        True if Qdrant service is running and accessible, False otherwise.
    """
    try:
        response = httpx.get(f"{settings.CHATBOT_SETTINGS.qdrant.url}/collections", timeout=2.0)
        return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


@pytest.fixture(scope="module")
def qdrant_service() -> Generator[None]:  # pylint: disable=missing-yield-doc
    """Fixture to start and stop the Qdrant Docker container."""
    # Check if Qdrant is already running
    was_running = _is_qdrant_running()

    if not was_running:
        # Start Qdrant service
        subprocess.run(  # nosec B607
            ["docker-compose", "up", "-d", "qdrant"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Wait for the service to be healthy
        time.sleep(5)

    try:
        yield
    finally:
        # Only stop the service if we started it
        if not was_running:
            subprocess.run(  # nosec B607
                ["docker-compose", "down"],
                check=True,
                capture_output=True,
                text=True,
            )


@pytest.mark.usefixtures("qdrant_service")
@pytest.fixture
def rag_instance() -> Generator[rag.RAG]:
    """Fixture to provide a RAG instance for testing.

    Yields:
        An instance of the RAG class.
    """
    client = qdrant_client.QdrantClient(
        url=settings.CHATBOT_SETTINGS.qdrant.url, api_key=settings.CHATBOT_SETTINGS.qdrant.api_key
    )
    collection_name = settings.CHATBOT_SETTINGS.qdrant.collection_name
    settings.CHATBOT_SETTINGS.qdrant.collection_name = "test"
    with contextlib.suppress(Exception):
        client.delete_collection(settings.CHATBOT_SETTINGS.qdrant.collection_name)

    rag_processor = rag.RAG()
    yield rag_processor

    settings.CHATBOT_SETTINGS.qdrant.collection_name = collection_name


@pytest.fixture(scope="module")
def openai_client() -> openai.OpenAI:
    """Fixture for OpenAI client.

    Returns:
        An OpenAI client instance.
    """
    return openai.OpenAI(base_url=TEST_SERVER_BASE_URL, api_key=TEST_API_KEY)
