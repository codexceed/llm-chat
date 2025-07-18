"""RAG (Retrieval-Augmented Generation) functionality using LlamaIndex and Qdrant."""

import logging
from pathlib import Path
import tempfile

from llama_index import core
from llama_index.core import ingestion, node_parser
from llama_index.embeddings import huggingface
from llama_index.vector_stores import qdrant
import qdrant_client
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http import models
from streamlit.runtime import uploaded_file_manager

from chatbot.config import settings

LOGGER = logging.getLogger(__name__)


class RAG:
    """Handles document processing and retrieval for RAG functionality."""

    def __init__(self) -> None:
        """Initialize the RAG processor with Qdrant vector store."""
        self.client = qdrant_client.QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)
        self.vector_store = qdrant.QdrantVectorStore(
            client=self.client,
            collection_name=settings.qdrant.collection_name,
        )
        self.embedding_model = huggingface.HuggingFaceEmbedding(
            model_name=settings.rag.embedding_model,
        )

        # Configure LlamaIndex settings
        core.Settings.embed_model = self.embedding_model
        core.Settings.chunk_size = settings.rag.chunk_size
        core.Settings.chunk_overlap = settings.rag.chunk_overlap

        self.index = core.VectorStoreIndex.from_vector_store(  # type: ignore
            vector_store=self.vector_store
        )

        self._ensure_collection_exists()

        self.query_engine = self.index.as_query_engine(  # type: ignore
            similarity_top_k=settings.rag.top_k,
            response_mode="compact",
        )

    def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists."""
        try:
            self.client.get_collection(settings.qdrant.collection_name)
        except qdrant_exceptions.UnexpectedResponse:
            LOGGER.info(
                "Creating new Qdrant collection by name: %s",
                settings.qdrant.collection_name,
            )
            self.client.create_collection(
                collection_name=settings.qdrant.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.qdrant.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def process_uploaded_files(self, uploaded_files: list[uploaded_file_manager.UploadedFile]) -> None:
        """Process and index uploaded files.

        Args:
            uploaded_files: List of uploaded files from Streamlit.
        """
        if not uploaded_files:
            return

        # Read documents from uploaded files.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for uploaded_file in uploaded_files:
                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

            documents = core.SimpleDirectoryReader(temp_dir).load_data(show_progress=True)

        # Parse documents into chunks.
        document_parser = node_parser.SentenceSplitter(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
        )
        ingestion_pipeline = ingestion.IngestionPipeline(transformations=[document_parser, self.embedding_model])
        nodes = ingestion_pipeline.run(documents=documents)

        # Insert new nodes into the existing index.
        self.index.insert_nodes(nodes)

    def retrieve(self, query_text: str) -> list[str]:
        """Retrieve relevant context chunks without LLM generation.

        Args:
            query_text: The user's query.

        Returns:
            List of relevant text chunks.
        """
        try:
            retriever = self.index.as_retriever(similarity_top_k=settings.rag.top_k)
            nodes = retriever.retrieve(query_text)
            return [node.text for node in nodes]
        except Exception:
            # Return empty list if retrieval fails (e.g., no documents indexed)
            return []

    def query(self, query_text: str) -> str:
        """Query the RAG system.

        Args:
            query_text: The user's query.

        Returns:
            Retrieved context or empty string if no relevant documents found.
        """
        try:
            response = self.query_engine.query(query_text)
            return str(response)
        except Exception:
            # Return empty string if query fails (e.g., no documents indexed)
            return ""
