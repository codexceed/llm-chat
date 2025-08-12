"""Integration tests for the RAG class."""

from unittest.mock import MagicMock

import httpx
import pytest
import pytest_httpx
import qdrant_client

from chatbot.rag import RAG
from chatbot.settings import CHATBOT_SETTINGS


def test_rag_initialization(rag_instance: RAG) -> None:
    """Test if the RAG class initializes correctly and creates the Qdrant collection."""
    assert rag_instance.client is not None
    assert rag_instance.embedding_model is not None
    assert rag_instance.index is not None
    assert rag_instance.retriever is not None

    # Verify that the collection was created in Qdrant
    client = qdrant_client.QdrantClient(url=CHATBOT_SETTINGS.qdrant.url, api_key=CHATBOT_SETTINGS.qdrant.api_key)
    collection_info = client.get_collection(CHATBOT_SETTINGS.qdrant.collection_name)
    assert collection_info is not None
    assert (
        collection_info.config.params.vectors.size  # type: ignore
        == CHATBOT_SETTINGS.qdrant.vector_size
    )


def test_process_uploaded_files(rag_instance: RAG) -> None:
    """Test processing of uploaded files."""
    # Create mock uploaded files
    mock_file_content = b"This is a test document for RAG processing about subject xyz with attributes 123."
    mock_file = MagicMock()
    mock_file.name = "test_file.txt"
    mock_file.getvalue.return_value = mock_file_content

    rag_instance.process_uploaded_files([mock_file])

    # Retrieve the document to verify it was indexed
    retrieved_chunks = rag_instance.retrieve("test document xyz 123")
    assert any("test document for RAG processing" in chunk for chunk in retrieved_chunks)


def is_example_url(request: httpx.Request) -> bool:
    """Check if the request is for example.com domain.

    Args:
        request: The HTTP request object to check.

    Returns:
        True if the request is for example.com domain, False otherwise.
    """
    return request.url.host == "example.com"


@pytest.mark.httpx_mock(should_mock=is_example_url)
@pytest.mark.asyncio
async def test_process_web_urls(rag_instance: RAG, httpx_mock: pytest_httpx.HTTPXMock) -> None:
    """Test processing of web URLs."""
    prompt = "Please check this website: http://example.com"
    key_text = "This is content xyz with features abc, gef and 123. Also, gubernatorial"
    mock_html_content = f"<html><body><h1>Test Website</h1><p>{key_text}</p></body></html>"

    # Mock the HTTP response using pytest_httpx
    httpx_mock.add_response(
        url="http://example.com",
        status_code=200,
        content=mock_html_content.encode(),
        headers={"content-type": "text/html"},
        is_reusable=True,
    )

    # Create an httpx client and pass it to the method
    async with httpx.AsyncClient() as client:
        await rag_instance.process_web_urls(prompt, client)

    # Retrieve the document to verify it was indexed
    retrieved_chunks = rag_instance.retrieve("Test Website content abc gef 123 gubernatorial")
    assert any(key_text in chunk for chunk in retrieved_chunks)


def test_retrieve_functionality(rag_instance: RAG) -> None:
    """Test the retrieve functionality with known content."""
    # Index a known document first
    mock_file_content = b"The quick brown fox jumps over the lazy dog."
    mock_file = MagicMock()
    mock_file.name = "test_retrieve.txt"
    mock_file.getvalue.return_value = mock_file_content
    rag_instance.process_uploaded_files([mock_file])

    # Test retrieval
    retrieved_chunks = rag_instance.retrieve("brown fox")
    assert len(retrieved_chunks) > 0
    assert "quick brown fox" in retrieved_chunks[0]

    # Test retrieval with no match
    retrieved_chunks = rag_instance.retrieve("non_existent_phrase")
    assert len(retrieved_chunks) == 0
