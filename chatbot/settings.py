import pydantic
import pydantic_settings
from qdrant_client.http import models


class RAGSettings(pydantic.BaseModel):
    """Settings for the RAG (Retrieval-Augmented Generation) processor."""

    enabled: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1024
    chunk_overlap: int = 100
    top_k: int = 5
    deduplication_similarity_threshold: float = 0.9

    # Adaptive parsing settings
    use_adaptive_parsing: bool = True
    code_chunk_lines: int = 40
    code_chunk_overlap_lines: int = 15
    semantic_breakpoint_threshold: int = 95
    device: str = "cpu"

    # Hybrid retrieval settings
    use_hybrid_retrieval: bool = True
    sparse_model: str = "Qdrant/bm25"  # BM42 sparse embedding model
    hybrid_top_k: int = 100

    # Relevance filtering settings
    enable_relevance_filtering: bool = True
    relevance_threshold: float = 0.6  # Minimum semantic similarity to query (0.0-1.0)


class QdrantSettings(pydantic.BaseModel):
    """Settings for the Qdrant vector database."""

    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "chatbot"
    vector_size: int = 384  # BGE-Small-EN-v1.5 outputs 384-dimensional embeddings
    distance_type: str = models.Distance.COSINE


class Settings(pydantic_settings.BaseSettings):
    """Configuration settings for the chatbot application."""

    openai_api_base: str = "http://localhost:8000/v1"
    openai_api_key: str = "not-needed"
    llm_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    temperature: float = 0.7
    max_tokens: int = 2000
    repetition_penalty: float = 1.1
    seed: int = 1234
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = True
    qdrant: QdrantSettings = pydantic.Field(default_factory=QdrantSettings)
    rag: RAGSettings = pydantic.Field(default_factory=RAGSettings)
    context_view_size: int = 1000

    class Config:
        """Configuration for the settings model."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
        env_nested_delimiter = "__"
        env_prefix = "CHATBOT_"


CHATBOT_SETTINGS = Settings()
