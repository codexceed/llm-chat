import unittest.mock

import hypothesis
import hypothesis.strategies
import numpy as np
import pytest

from chatbot import constants


@pytest.fixture
def rag_instance() -> unittest.mock.MagicMock:
    """Fixture for a mock rag.RAG instance.

    Returns:
        A mock rag.RAG instance.
    """
    rag = unittest.mock.MagicMock()
    rag._get_file_type = rag.RAG._get_file_type.__get__(rag, rag.RAG)
    rag._detect_code_language = rag.RAG._detect_code_language.__get__(rag, rag.RAG)
    rag._cosine_similarity = rag.RAG._vectorized_cosine_similarity.__get__(rag, rag.RAG)
    rag._vectorized_cosine_similarity = rag.RAG._vectorized_cosine_similarity.__get__(rag, rag.RAG)
    rag._deduplicate_chunks = rag.RAG._deduplicate_chunks.__get__(rag, rag.RAG)
    rag.process_web_urls = rag.RAG.process_web_urls.__get__(rag, rag.RAG)
    rag.retrieve = rag.RAG.retrieve.__get__(rag, rag.RAG)
    return rag


def test_get_file_type_python(rag_instance: unittest.mock.MagicMock) -> None:
    """Test file type detection for Python files."""
    assert rag_instance._get_file_type("test.py") == constants.FileTypes.CODE
    assert rag_instance._get_file_type("/path/to/script.py") == constants.FileTypes.CODE


def test_get_file_type_javascript(rag_instance: unittest.mock.MagicMock) -> None:
    """Test file type detection for JavaScript files."""
    assert rag_instance._get_file_type("app.js") == constants.FileTypes.CODE
    assert rag_instance._get_file_type("component.jsx") == constants.FileTypes.CODE
    assert rag_instance._get_file_type("types.ts") == constants.FileTypes.CODE
    assert rag_instance._get_file_type("Component.tsx") == constants.FileTypes.CODE


def test_get_file_type_markdown(rag_instance: unittest.mock.MagicMock) -> None:
    """Test file type detection for Markdown files."""
    assert rag_instance._get_file_type("README.md") == constants.FileTypes.MARKDOWN
    assert rag_instance._get_file_type("/docs/guide.md") == constants.FileTypes.MARKDOWN


def test_get_file_type_html(rag_instance: unittest.mock.MagicMock) -> None:
    """Test file type detection for HTML files."""
    assert rag_instance._get_file_type("index.html") == constants.FileTypes.HTML
    assert rag_instance._get_file_type("page.htm") == constants.FileTypes.HTML


def test_get_file_type_text(rag_instance: unittest.mock.MagicMock) -> None:
    """Test file type detection for text files."""
    assert rag_instance._get_file_type("notes.txt") == constants.FileTypes.TEXT


def test_get_file_type_unknown(rag_instance: unittest.mock.MagicMock) -> None:
    """Test file type detection for unknown files."""
    assert rag_instance._get_file_type("file.xyz") == constants.FileTypes.UNKNOWN
    assert rag_instance._get_file_type("no_extension") == constants.FileTypes.UNKNOWN
    assert rag_instance._get_file_type("") == constants.FileTypes.UNKNOWN


@hypothesis.given(file_path=hypothesis.strategies.text())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_get_file_type_arbitrary_paths(rag_instance: unittest.mock.MagicMock, file_path: str) -> None:
    """Test file type detection with arbitrary file paths."""
    result = rag_instance._get_file_type(file_path)
    assert isinstance(result, constants.FileTypes)


def test_detect_code_language_python(rag_instance: unittest.mock.MagicMock) -> None:
    """Test language detection for Python files."""
    assert rag_instance._detect_code_language("script.py") == "python"
    assert rag_instance._detect_code_language("/path/to/module.py") == "python"


def test_detect_code_language_javascript(rag_instance: unittest.mock.MagicMock) -> None:
    """Test language detection for JavaScript/TypeScript files."""
    assert rag_instance._detect_code_language("app.js") == "javascript"
    assert rag_instance._detect_code_language("component.jsx") == "javascript"
    assert rag_instance._detect_code_language("types.ts") == "typescript"
    assert rag_instance._detect_code_language("Component.tsx") == "typescript"


def test_detect_code_language_other_languages(rag_instance: unittest.mock.MagicMock) -> None:
    """Test language detection for other programming languages."""
    assert rag_instance._detect_code_language("Main.java") == "java"
    assert rag_instance._detect_code_language("program.cpp") == "cpp"
    assert rag_instance._detect_code_language("code.c") == "c"
    assert rag_instance._detect_code_language("header.h") == "c"
    assert rag_instance._detect_code_language("script.sh") == "bash"
    assert rag_instance._detect_code_language("query.sql") == "sql"


def test_detect_code_language_unknown(rag_instance: unittest.mock.MagicMock) -> None:
    """Test language detection for unknown file types."""
    assert rag_instance._detect_code_language("file.xyz") == "unknown"
    assert rag_instance._detect_code_language("no_extension") == "unknown"
    assert rag_instance._detect_code_language("") == "unknown"


@hypothesis.given(file_path=hypothesis.strategies.text())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_detect_code_language_arbitrary_paths(rag_instance: unittest.mock.MagicMock, file_path: str) -> None:
    """Test language detection with arbitrary file paths."""
    result = rag_instance._detect_code_language(file_path)
    assert isinstance(result, str)


def test_cosine_similarity_identical_vectors(rag_instance: unittest.mock.MagicMock) -> None:
    """Test cosine similarity with identical vectors."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]

    similarity = rag_instance._cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-10


def test_cosine_similarity_orthogonal_vectors(rag_instance: unittest.mock.MagicMock) -> None:
    """Test cosine similarity with orthogonal vectors."""
    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]

    similarity = rag_instance._cosine_similarity(vec1, vec2)
    assert abs(similarity - 0.0) < 1e-10


def test_cosine_similarity_opposite_vectors(rag_instance: unittest.mock.MagicMock) -> None:
    """Test cosine similarity with opposite vectors."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [-1.0, -2.0, -3.0]

    similarity = rag_instance._cosine_similarity(vec1, vec2)
    assert abs(similarity - (-1.0)) < 1e-10


def test_cosine_similarity_zero_vectors(rag_instance: unittest.mock.MagicMock) -> None:
    """Test cosine similarity with zero vectors."""
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 2.0, 3.0]

    similarity = rag_instance._cosine_similarity(vec1, vec2)
    assert similarity == 0.0


@hypothesis.given(
    vec1=hypothesis.strategies.lists(hypothesis.strategies.floats(min_value=-10, max_value=10), min_size=1, max_size=10)
)
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_cosine_similarity_arbitrary_vectors(rag_instance: unittest.mock.MagicMock, vec1: list[float]) -> None:
    """Test cosine similarity with arbitrary vectors."""
    # Create a second vector of same length
    vec2 = [x + 1.0 for x in vec1]

    similarity = rag_instance._vectorized_cosine_similarity(vec1, vec2)
    assert isinstance(similarity, float)
    # Allow for floating point precision errors
    assert -1.01 <= similarity <= 1.01


def test_vectorized_cosine_similarity_basic(rag_instance: unittest.mock.MagicMock) -> None:
    """Test vectorized cosine similarity with basic input."""
    embeddings1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    embeddings2 = np.array([[1.0, 0.0], [0.0, 1.0]])

    similarities = rag_instance._vectorized_cosine_similarity(embeddings1, embeddings2)

    # Should return 2x2 matrix
    assert similarities.shape == (2, 2)

    # Diagonal should be 1.0 (identical vectors)
    assert abs(similarities[0, 0] - 1.0) < 1e-10
    assert abs(similarities[1, 1] - 1.0) < 1e-10

    # Off-diagonal should be 0.0 (orthogonal vectors)
    assert abs(similarities[0, 1] - 0.0) < 1e-10
    assert abs(similarities[1, 0] - 0.0) < 1e-10


def test_vectorized_cosine_similarity_different_sizes(rag_instance: unittest.mock.MagicMock) -> None:
    """Test vectorized cosine similarity with different matrix sizes."""
    embeddings1 = np.array([[1.0, 0.0, 0.0]])  # 1x3
    embeddings2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2x3

    similarities = rag_instance._vectorized_cosine_similarity(embeddings1, embeddings2)

    # Should return 1x2 matrix
    assert similarities.shape == (1, 2)
    assert abs(similarities[0, 0] - 1.0) < 1e-10  # Same vector
    assert abs(similarities[0, 1] - 0.0) < 1e-10  # Orthogonal vectors


def test_vectorized_cosine_similarity_zero_norm_handling(rag_instance: unittest.mock.MagicMock) -> None:
    """Test vectorized cosine similarity handles zero norm vectors."""
    embeddings1 = np.array([[0.0, 0.0, 0.0]])  # Zero vector
    embeddings2 = np.array([[1.0, 2.0, 3.0]])  # Non-zero vector

    similarities = rag_instance._vectorized_cosine_similarity(embeddings1, embeddings2)

    # Should handle zero norm gracefully
    assert similarities.shape == (1, 1)
    assert isinstance(similarities[0, 0], float | np.floating)


@unittest.mock.patch("rag.RAG._vectorized_cosine_similarity")
def test_deduplicate_chunks_empty_list(
    mock_similarity: unittest.mock.MagicMock, rag_instance: unittest.mock.MagicMock
) -> None:
    """Test deduplication with empty chunk list."""
    result = rag_instance._deduplicate_chunks([])
    assert result == []
    mock_similarity.assert_not_called()


@unittest.mock.patch("rag.RAG._vectorized_cosine_similarity")
def test_deduplicate_chunks_single_chunk(
    mock_similarity: unittest.mock.MagicMock, rag_instance: unittest.mock.MagicMock
) -> None:
    """Test deduplication with single chunk."""
    rag_instance.embedding_model = unittest.mock.MagicMock()
    rag_instance.embedding_model.get_text_embedding_batch.return_value = [[1.0, 0.0]]

    chunks = ["Single chunk"]
    result = rag_instance._deduplicate_chunks(chunks)

    assert result == ["Single chunk"]
    mock_similarity.assert_not_called()


@unittest.mock.patch("CHATBOT_SETTINGS")
def test_deduplicate_chunks_no_duplicates(
    mock_settings: unittest.mock.MagicMock, rag_instance: unittest.mock.MagicMock
) -> None:
    """Test deduplication with no duplicate chunks."""
    mock_settings.rag.deduplication_similarity_threshold = 0.9

    rag_instance.embedding_model = unittest.mock.MagicMock()

    # Mock embeddings for different chunks (low similarity)
    rag_instance.embedding_model.get_text_embedding_batch.return_value = [
        [1.0, 0.0, 0.0],  # First chunk
        [0.0, 1.0, 0.0],  # Second chunk (orthogonal)
    ]

    chunks = ["First chunk", "Second chunk"]
    result = rag_instance._deduplicate_chunks(chunks)

    # Should keep both chunks as they're not similar
    assert len(result) == 2
    assert "First chunk" in result
    assert "Second chunk" in result


@unittest.mock.patch("CHATBOT_SETTINGS")
def test_deduplicate_chunks_with_duplicates(
    mock_settings: unittest.mock.MagicMock, rag_instance: unittest.mock.MagicMock
) -> None:
    """Test deduplication with duplicate chunks."""
    mock_settings.rag.deduplication_similarity_threshold = 0.9

    rag_instance.embedding_model = unittest.mock.MagicMock()

    # Mock embeddings for very similar chunks
    rag_instance.embedding_model.get_text_embedding_batch.return_value = [
        [1.0, 0.0, 0.0],  # First chunk
        [0.95, 0.31, 0.0],  # Second chunk (very similar)
    ]

    chunks = ["Original chunk", "Very similar chunk"]
    result = rag_instance._deduplicate_chunks(chunks)

    # Should keep only the first chunk
    assert len(result) == 1
    assert result[0] == "Original chunk"


@hypothesis.given(chunks=hypothesis.strategies.lists(hypothesis.strategies.text(min_size=1), min_size=1, max_size=5))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_deduplicate_chunks_arbitrary_input(rag_instance: unittest.mock.MagicMock, chunks: list[str]) -> None:
    """Test deduplication with arbitrary chunk lists."""
    rag_instance.embedding_model = unittest.mock.MagicMock()

    # Mock embeddings - make them orthogonal to avoid deduplication
    embeddings = []
    for i in range(len(chunks)):
        embedding = [0.0] * max(3, len(chunks))
        embedding[i % len(embedding)] = 1.0
        embeddings.append(embedding)

    rag_instance.embedding_model.get_text_embedding_batch.return_value = embeddings

    with unittest.mock.patch("CHATBOT_SETTINGS") as mock_settings:
        mock_settings.rag.deduplication_similarity_threshold = 0.9

        result = rag_instance._deduplicate_chunks(chunks)

        # Should return a list of strings
        assert isinstance(result, list)
        assert all(isinstance(chunk, str) for chunk in result)
        # Should not return more chunks than input
        assert len(result) <= len(chunks)


@pytest.mark.asyncio
async def test_process_web_urls_no_urls(rag_instance: unittest.mock.MagicMock) -> None:
    """Test processing web URLs when no URLs are found."""
    mock_client = unittest.mock.AsyncMock()

    with unittest.mock.patch("chatbot.utils.web.lookup_http_urls_in_prompt") as mock_lookup:
        mock_lookup.return_value = ([], [])

        await rag_instance.process_web_urls("No URLs here", mock_client)

        # Should not call index.insert_nodes when no URLs found
        rag_instance.index.insert_nodes.assert_not_called()


@pytest.mark.asyncio
async def test_process_web_urls_with_content(rag_instance: unittest.mock.MagicMock) -> None:
    """Test processing web URLs with actual content."""
    mock_client = unittest.mock.AsyncMock()

    with unittest.mock.patch("chatbot.utils.web.lookup_http_urls_in_prompt") as mock_lookup:
        mock_lookup.return_value = (["http://example.com"], ["<html><body>Test content</body></html>"])

        rag_instance._html_pipeline = unittest.mock.MagicMock()
        rag_instance._html_pipeline.run.return_value = ["processed_node"]

        with unittest.mock.patch("CHATBOT_SETTINGS") as mock_settings:
            mock_settings.rag.use_adaptive_parsing = True

            await rag_instance.process_web_urls("Visit http://example.com", mock_client)

            # Should process content and insert nodes
            rag_instance._html_pipeline.run.assert_called_once()
            rag_instance.index.insert_nodes.assert_called_once_with(["processed_node"])


@pytest.mark.asyncio
async def test_process_web_urls_empty_content(rag_instance: unittest.mock.MagicMock) -> None:
    """Test processing web URLs with empty content."""
    mock_client = unittest.mock.AsyncMock()

    with unittest.mock.patch("chatbot.utils.web.lookup_http_urls_in_prompt") as mock_lookup:
        mock_lookup.return_value = (
            ["http://example.com"],
            [""],  # Empty content
        )

        await rag_instance.process_web_urls("Visit http://example.com", mock_client)

        # Should not call index.insert_nodes when content is empty
        rag_instance.index.insert_nodes.assert_not_called()


@hypothesis.given(prompt=hypothesis.strategies.text())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
@pytest.mark.asyncio
async def test_process_web_urls_arbitrary_prompts(rag_instance: unittest.mock.MagicMock, prompt: str) -> None:
    """Test processing web URLs with arbitrary prompts."""
    mock_client = unittest.mock.AsyncMock()

    with unittest.mock.patch("chatbot.utils.web.lookup_http_urls_in_prompt") as mock_lookup:
        mock_lookup.return_value = ([], [])  # No URLs found

        # Should not raise an exception
        await rag_instance.process_web_urls(prompt, mock_client)

        rag_instance.index.insert_nodes.assert_not_called()


def test_retrieve_empty_query(rag_instance: unittest.mock.MagicMock) -> None:
    """Test retrieve with empty query."""
    rag_instance._deduplicate_chunks = unittest.mock.MagicMock(return_value=[])
    rag_instance.retriever = unittest.mock.MagicMock()
    rag_instance.retriever.retrieve.return_value = []

    with unittest.mock.patch("CHATBOT_SETTINGS") as mock_settings:
        mock_settings.rag.enable_relevance_filtering = False
        mock_settings.rag.top_k = 5

        result = rag_instance.retrieve("")
        assert result == []


def test_retrieve_with_relevance_filtering(rag_instance: unittest.mock.MagicMock) -> None:
    """Test retrieve with relevance filtering enabled."""
    rag_instance._deduplicate_chunks = unittest.mock.MagicMock(return_value=["filtered chunk"])

    # Mock nodes with different scores
    high_score_node = unittest.mock.MagicMock()
    high_score_node.text = "relevant content"
    high_score_node.score = 0.8

    low_score_node = unittest.mock.MagicMock()
    low_score_node.text = "irrelevant content"
    low_score_node.score = 0.6

    rag_instance.retriever = unittest.mock.MagicMock()
    rag_instance.retriever.retrieve.return_value = [high_score_node, low_score_node]

    with unittest.mock.patch("CHATBOT_SETTINGS") as mock_settings:
        mock_settings.rag.enable_relevance_filtering = True
        mock_settings.rag.relevance_threshold = 0.75
        mock_settings.rag.top_k = 5

        result = rag_instance.retrieve("test query")

        # Should only process high-scoring nodes
        rag_instance._deduplicate_chunks.assert_called_once_with(["relevant content"])
        assert result == ["filtered chunk"]


def test_retrieve_without_relevance_filtering(rag_instance: unittest.mock.MagicMock) -> None:
    """Test retrieve without relevance filtering."""
    rag_instance._deduplicate_chunks = unittest.mock.MagicMock(return_value=["all chunks"])

    # Mock nodes
    node1 = unittest.mock.MagicMock()
    node1.text = "content 1"
    node1.score = 0.8

    node2 = unittest.mock.MagicMock()
    node2.text = "content 2"
    node2.score = 0.6

    rag_instance.retriever = unittest.mock.MagicMock()
    rag_instance.retriever.retrieve.return_value = [node1, node2]

    with unittest.mock.patch("CHATBOT_SETTINGS") as mock_settings:
        mock_settings.rag.enable_relevance_filtering = False
        mock_settings.rag.top_k = 5

        result = rag_instance.retrieve("test query")

        # Should process all nodes
        rag_instance._deduplicate_chunks.assert_called_once_with(["content 1", "content 2"])
        assert result == ["all chunks"]


@hypothesis.given(query=hypothesis.strategies.text(min_size=1))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_retrieve_arbitrary_queries(rag_instance: unittest.mock.MagicMock, query: str) -> None:
    """Test retrieve with arbitrary query strings."""
    rag_instance._deduplicate_chunks = unittest.mock.MagicMock(return_value=[])
    rag_instance.retriever = unittest.mock.MagicMock()
    rag_instance.retriever.retrieve.return_value = []

    with unittest.mock.patch("CHATBOT_SETTINGS") as mock_settings:
        mock_settings.rag.enable_relevance_filtering = False
        mock_settings.rag.top_k = 5

        result = rag_instance.retrieve(query)
        assert isinstance(result, list)


def test_retrieve_handles_exceptions(rag_instance: unittest.mock.MagicMock) -> None:
    """Test that retrieve handles exceptions gracefully."""
    rag_instance.retriever = unittest.mock.MagicMock()
    rag_instance.retriever.retrieve.side_effect = Exception("Retrieval failed")

    result = rag_instance.retrieve("test query")
    assert result == []
