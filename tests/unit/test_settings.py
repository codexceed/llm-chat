import os

from hypothesis import given, strategies as st

from chatbot.settings import QdrantSettings, RAGSettings, Settings


def test_rag_settings_default_values() -> None:
    """Test that RAGSettings has expected default values."""
    settings = RAGSettings()

    assert settings.enabled is True
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.chunk_size == 1024
    assert settings.chunk_overlap == 100
    assert settings.top_k == 5
    assert settings.deduplication_similarity_threshold == 0.9
    assert settings.use_adaptive_parsing is True
    assert settings.code_chunk_lines == 40
    assert settings.code_chunk_overlap_lines == 15
    assert settings.semantic_breakpoint_threshold == 95
    assert settings.device == "cpu"
    assert settings.use_hybrid_retrieval is True
    assert settings.sparse_model == "Qdrant/bm25"
    assert settings.hybrid_top_k == 100
    assert settings.enable_relevance_filtering is True
    assert settings.relevance_threshold == 0.75


@given(enabled=st.booleans())
def test_rag_settings_enabled_property(enabled: bool) -> None:
    """Test RAGSettings enabled property with different boolean values."""
    settings = RAGSettings(enabled=enabled)
    assert settings.enabled == enabled


@given(model_name=st.text(min_size=1, max_size=100))
def test_rag_settings_embedding_model_property(model_name: str) -> None:
    """Test RAGSettings embedding_model property with different string values."""
    settings = RAGSettings(embedding_model=model_name)
    assert settings.embedding_model == model_name


@given(chunk_size=st.integers(min_value=1, max_value=10000))
def test_rag_settings_chunk_size_property(chunk_size: int) -> None:
    """Test RAGSettings chunk_size property with different integer values."""
    settings = RAGSettings(chunk_size=chunk_size)
    assert settings.chunk_size == chunk_size


@given(chunk_overlap=st.integers(min_value=0, max_value=1000))
def test_rag_settings_chunk_overlap_property(chunk_overlap: int) -> None:
    """Test RAGSettings chunk_overlap property with different integer values."""
    settings = RAGSettings(chunk_overlap=chunk_overlap)
    assert settings.chunk_overlap == chunk_overlap


@given(top_k=st.integers(min_value=1, max_value=100))
def test_rag_settings_top_k_property(top_k: int) -> None:
    """Test RAGSettings top_k property with different integer values."""
    settings = RAGSettings(top_k=top_k)
    assert settings.top_k == top_k


@given(threshold=st.floats(min_value=0.0, max_value=1.0))
def test_rag_settings_deduplication_similarity_threshold_property(threshold: float) -> None:
    """Test RAGSettings deduplication_similarity_threshold property."""
    settings = RAGSettings(deduplication_similarity_threshold=threshold)
    assert settings.deduplication_similarity_threshold == threshold


@given(threshold=st.floats(min_value=0.0, max_value=1.0))
def test_rag_settings_relevance_threshold_property(threshold: float) -> None:
    """Test RAGSettings relevance_threshold property."""
    settings = RAGSettings(relevance_threshold=threshold)
    assert settings.relevance_threshold == threshold


def test_qdrant_settings_default_values() -> None:
    """Test that QdrantSettings has expected default values."""
    settings = QdrantSettings()

    assert settings.url == "http://localhost:6333"
    assert settings.api_key is None
    assert settings.collection_name == "chatbot"
    assert settings.vector_size == 384


@given(url=st.text(min_size=1, max_size=100))
def test_qdrant_settings_url_property(url: str) -> None:
    """Test QdrantSettings url property with different string values."""
    settings = QdrantSettings(url=url)
    assert settings.url == url


@given(collection_name=st.text(min_size=1, max_size=100))
def test_qdrant_settings_collection_name_property(collection_name: str) -> None:
    """Test QdrantSettings collection_name property with different string values."""
    settings = QdrantSettings(collection_name=collection_name)
    assert settings.collection_name == collection_name


@given(vector_size=st.integers(min_value=1, max_value=10000))
def test_qdrant_settings_vector_size_property(vector_size: int) -> None:
    """Test QdrantSettings vector_size property with different integer values."""
    settings = QdrantSettings(vector_size=vector_size)
    assert settings.vector_size == vector_size


def test_settings_default_values() -> None:
    """Test that Settings has expected default values."""
    settings = Settings()

    assert settings.openai_api_base == "http://localhost:8000/v1"
    assert settings.openai_api_key == "not-needed"
    assert settings.llm_model_name == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    assert settings.temperature == 0.7
    assert settings.max_tokens == 2000
    assert settings.repetition_penalty == 1.1
    assert settings.seed == 1234
    assert settings.host == "127.0.0.1"
    assert settings.port == 8080
    assert settings.debug is True
    assert isinstance(settings.qdrant, QdrantSettings)
    assert isinstance(settings.rag, RAGSettings)
    assert settings.context_view_size == 1000


@given(api_base=st.text(min_size=1, max_size=100))
def test_settings_openai_api_base_property(api_base: str) -> None:
    """Test Settings openai_api_base property with different string values."""
    settings = Settings(openai_api_base=api_base)
    assert settings.openai_api_base == api_base


@given(model_name=st.text(min_size=1, max_size=100))
def test_settings_llm_model_name_property(model_name: str) -> None:
    """Test Settings llm_model_name property with different string values."""
    settings = Settings(llm_model_name=model_name)
    assert settings.llm_model_name == model_name


@given(temperature=st.floats(min_value=0.0, max_value=2.0))
def test_settings_temperature_property(temperature: float) -> None:
    """Test Settings temperature property with different float values."""
    settings = Settings(temperature=temperature)
    assert settings.temperature == temperature


@given(max_tokens=st.integers(min_value=1, max_value=10000))
def test_settings_max_tokens_property(max_tokens: int) -> None:
    """Test Settings max_tokens property with different integer values."""
    settings = Settings(max_tokens=max_tokens)
    assert settings.max_tokens == max_tokens


@given(penalty=st.floats(min_value=1.0, max_value=2.0))
def test_settings_repetition_penalty_property(penalty: float) -> None:
    """Test Settings repetition_penalty property with different float values."""
    settings = Settings(repetition_penalty=penalty)
    assert settings.repetition_penalty == penalty


@given(port=st.integers(min_value=1, max_value=65535))
def test_settings_port_property(port: int) -> None:
    """Test Settings port property with different integer values."""
    settings = Settings(port=port)
    assert settings.port == port


def test_env_file_loading() -> None:
    """Test that Settings can load configuration from environment variables."""
    # Store original values
    original_values = {
        "CHATBOT_TEMPERATURE": os.environ.get("CHATBOT_TEMPERATURE"),
        "CHATBOT_MAX_TOKENS": os.environ.get("CHATBOT_MAX_TOKENS"),
        "CHATBOT_QDRANT__URL": os.environ.get("CHATBOT_QDRANT__URL"),
        "CHATBOT_RAG__TOP_K": os.environ.get("CHATBOT_RAG__TOP_K"),
    }

    try:
        # Set environment variables
        os.environ["CHATBOT_TEMPERATURE"] = "0.5"
        os.environ["CHATBOT_MAX_TOKENS"] = "1500"
        os.environ["CHATBOT_QDRANT__URL"] = "http://test:6333"
        os.environ["CHATBOT_RAG__TOP_K"] = "10"

        settings = Settings()

        assert settings.temperature == 0.5
        assert settings.max_tokens == 1500
        assert settings.qdrant.url == "http://test:6333"
        assert settings.rag.top_k == 10

    finally:
        # Restore original values
        for key, value in original_values.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


def test_nested_settings() -> None:
    """Test that nested settings are properly initialized."""
    settings = Settings()

    # Test that nested settings are of correct type
    assert isinstance(settings.qdrant, QdrantSettings)
    assert isinstance(settings.rag, RAGSettings)

    # Test that nested settings have correct default values
    assert settings.qdrant.collection_name == "chatbot"
    assert settings.rag.enabled is True
