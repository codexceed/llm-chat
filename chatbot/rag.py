"""RAG (Retrieval-Augmented Generation) functionality using LlamaIndex and Qdrant."""

import tempfile
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np
import qdrant_client
from llama_index import core
from llama_index.core import ingestion, node_parser, schema
from llama_index.embeddings import huggingface
from llama_index.vector_stores import qdrant
from qdrant_client.http import exceptions as qdrant_exceptions, models
from streamlit import logger
from streamlit.runtime import uploaded_file_manager

from chatbot import constants
from chatbot.settings import settings
from chatbot.utils import web

LOGGER = logger.get_logger(__name__)


class RAG:
    """Handles document processing and retrieval for RAG functionality."""

    def __init__(self) -> None:
        """Initialize the RAG processor with Qdrant vector store."""
        LOGGER.info("Initializing RAG processor with Qdrant vector store.")
        self.client = qdrant_client.QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)
        self.embedding_model = huggingface.HuggingFaceEmbedding(model_name=settings.rag.embedding_model, device=settings.rag.device)

        # Configure LlamaIndex settings
        core.Settings.embed_model = self.embedding_model
        core.Settings.chunk_size = settings.rag.chunk_size
        core.Settings.chunk_overlap = settings.rag.chunk_overlap

        self._ensure_collection_exists()

        self.vector_store = qdrant.QdrantVectorStore(
            client=self.client,
            collection_name=settings.qdrant.collection_name,
        )
        self.index = core.VectorStoreIndex.from_vector_store(  # type: ignore
            vector_store=self.vector_store
        )
        self.retriever = self.index.as_retriever(similarity_top_k=settings.rag.top_k * 2)  # Get more candidates for deduplication

        # Initialize parsers and pipelines for adaptive parsing
        self._init_parsers()

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

    def _init_parsers(self) -> None:
        """Initialize cached parsers and pipelines for adaptive parsing."""
        self._sentence_parser = node_parser.SentenceSplitter(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
        )
        self._sentence_pipeline = ingestion.IngestionPipeline(transformations=[self._sentence_parser, self.embedding_model])

        if settings.rag.use_adaptive_parsing:
            self._markdown_parser = node_parser.MarkdownNodeParser()
            self._html_parser = node_parser.HTMLNodeParser()

            self._semantic_parser = node_parser.SemanticSplitterNodeParser(
                embed_model=self.embedding_model,
                buffer_size=1,
                breakpoint_percentile_threshold=settings.rag.semantic_breakpoint_threshold,
            )

            # Initialize pipelines (excluding code pipeline - created dynamically)
            self._markdown_pipeline = ingestion.IngestionPipeline(transformations=[self._markdown_parser, self.embedding_model])
            self._html_pipeline = ingestion.IngestionPipeline(transformations=[self._html_parser, self.embedding_model])
            self._semantic_pipeline = ingestion.IngestionPipeline(transformations=[self._semantic_parser, self.embedding_model])

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

        # Parse documents into chunks using adaptive parsing
        nodes: Sequence[schema.BaseNode]
        if settings.rag.use_adaptive_parsing:
            nodes = self._process_documents_adaptively(documents)
        else:
            nodes = self._sentence_pipeline.run(documents=documents)

        # Insert new nodes into the existing index.
        self.index.insert_nodes(nodes)

    async def process_web_urls(self, prompt: str, client: httpx.AsyncClient) -> None:
        """Extract web URLs from prompt and index their content for RAG.

        Args:
            prompt: User input containing potential web URLs.
            client: HTTP client for making requests to fetch URL content.
        """
        urls, web_docs_content = await web.lookup_http_urls_in_prompt(prompt, client)
        if not web_docs_content:
            return

        # Create documents with URL metadata for better tracking
        documents: list[core.Document] = []
        for url, content in zip(urls, web_docs_content, strict=False):
            if content:
                LOGGER.debug("Processing web URL: %s", url)
                LOGGER.debug("Content: %s", content[:100])
                doc = core.Document(text=content, metadata={"source": "web_url", "url": url, "content_type": "html"})
                documents.append(doc)

        if not documents:
            return

        # Process web documents using HTML pipeline for consistent parsing
        if settings.rag.use_adaptive_parsing and hasattr(self, "_html_pipeline"):
            nodes = self._html_pipeline.run(documents=documents)
        else:
            # Fallback to sentence-based parsing if adaptive parsing is disabled
            nodes = self._sentence_pipeline.run(documents=documents)

        self.index.insert_nodes(nodes)

    def retrieve(self, query_text: str) -> list[str]:
        """Retrieve relevant context chunks.

        Args:
            query_text: The user's query.

        Returns:
            List of relevant text chunks with duplicates removed.
        """
        try:
            nodes = self.retriever.retrieve(query_text)
            chunks = [node.text for node in nodes]
            return self._deduplicate_chunks(chunks)[: settings.rag.top_k]
        except Exception as e:
            LOGGER.error(f"Error during retrieval: {e}")
            return []

    def _deduplicate_chunks(self, chunks: list[str]) -> list[str]:
        """Remove duplicate and highly similar chunks.

        Args:
            chunks: List of text chunks to deduplicate.

        Returns:
            Deduplicated list of chunks.
        """
        if not chunks:
            return chunks

        deduplicated: list[str] = []
        similarity_threshold = 0.85  # Cosine similarity threshold

        for chunk in chunks:
            is_duplicate = False
            chunk_embedding = self.embedding_model.get_text_embedding(chunk)

            for existing_chunk in deduplicated:
                existing_embedding = self.embedding_model.get_text_embedding(existing_chunk)
                similarity = self._cosine_similarity(chunk_embedding, existing_embedding)

                if similarity > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(chunk)

        return deduplicated

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score.
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _get_file_type(self, file_path: str) -> constants.FileTypes:
        """Determine file type for parser selection.

        Args:
            file_path: Path to the file.

        Returns:
            File type category for parser selection.
        """
        suffix = Path(file_path).suffix.lower()

        for file_type, extensions in constants.FILE_EXTENSION_TYPE_MAPPING.items():
            if suffix in extensions:
                return file_type

        return constants.FileTypes.UNKNOWN

    def _process_documents_adaptively(self, documents: list[core.Document]) -> list[schema.BaseNode]:
        """Process documents with appropriate parsers per document type.

        Args:
            documents: List of documents to process.

        Returns:
            List of processed nodes.
        """
        all_nodes: list[schema.BaseNode] = []

        # Group documents by file type
        documents_by_type: dict[constants.FileTypes, list[core.Document]] = {
            constants.FileTypes.CODE: [],
            constants.FileTypes.MARKDOWN: [],
            constants.FileTypes.HTML: [],
            constants.FileTypes.TEXT: [],
            constants.FileTypes.UNKNOWN: [],
        }

        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            if file_path:
                file_type = self._get_file_type(file_path)
                documents_by_type[file_type].append(doc)
            else:
                documents_by_type[constants.FileTypes.UNKNOWN].append(doc)

        # Process code documents with dynamic language-specific parsing
        if code_docs := documents_by_type[constants.FileTypes.CODE]:
            code_nodes = self._process_code_documents(code_docs)
            all_nodes.extend(code_nodes)

        if markdown_docs := documents_by_type[constants.FileTypes.MARKDOWN]:
            md_nodes = self._markdown_pipeline.run(documents=markdown_docs)
            all_nodes.extend(md_nodes)

        if html_docs := documents_by_type[constants.FileTypes.HTML]:
            html_nodes = self._html_pipeline.run(documents=html_docs)
            all_nodes.extend(html_nodes)

        if text_docs := documents_by_type[constants.FileTypes.TEXT]:
            text_nodes = self._sentence_pipeline.run(documents=text_docs)
            all_nodes.extend(text_nodes)

        # Process unknown documents with semantic splitter (for safety)
        if unknown_docs := documents_by_type[constants.FileTypes.UNKNOWN]:
            unknown_nodes = self._semantic_pipeline.run(documents=unknown_docs)
            all_nodes.extend(unknown_nodes)

        return all_nodes

    def _process_code_documents(self, code_docs: list[core.Document]) -> list[schema.BaseNode]:
        """Process code documents with language-specific parsers.

        Args:
            code_docs: List of code documents to process.

        Returns:
            List of processed nodes with language-appropriate chunking.
        """
        all_nodes: list[schema.BaseNode] = []

        # Group documents by programming language
        docs_by_language: dict[str, list[core.Document]] = {}
        for doc in code_docs:
            file_path = doc.metadata.get("file_path", "")
            language = self._detect_code_language(file_path)

            if language not in docs_by_language:
                docs_by_language[language] = []
            docs_by_language[language].append(doc)

        # Process each language group
        for language, docs in docs_by_language.items():
            try:
                if language == "unknown":
                    # Fallback to semantic splitter for unknown code languages
                    LOGGER.warning(f"Unknown code language, falling back to semantic splitter for {len(docs)} documents")
                    nodes = self._semantic_pipeline.run(documents=docs)
                else:
                    # Create language-specific code parser
                    code_parser = node_parser.CodeSplitter(
                        language=language,
                        chunk_lines=settings.rag.code_chunk_lines,
                        chunk_lines_overlap=settings.rag.code_chunk_overlap_lines,
                        max_chars=settings.rag.chunk_size,
                    )
                    code_pipeline = ingestion.IngestionPipeline(transformations=[code_parser, self.embedding_model])
                    nodes = code_pipeline.run(documents=docs)

                all_nodes.extend(nodes)

            except Exception as e:  # noqa: PERF203
                LOGGER.error(f"CodeSplitter failed for language '{language}': {e}. Falling back to semantic splitter.")
                # Fallback to semantic splitter on any error
                fallback_nodes = self._semantic_pipeline.run(documents=docs)
                all_nodes.extend(fallback_nodes)

        return all_nodes

    def _detect_code_language(self, file_path: str) -> str:
        """Detect programming language from file path.

        Args:
            file_path: Path to the code file.

        Returns:
            Language identifier for CodeSplitter, or "unknown" if not detected.
        """
        if not file_path:
            return "unknown"

        suffix = Path(file_path).suffix.lower()
        return constants.EXTENSION_TO_LANGUAGE_MAPPING.get(suffix, "unknown")
