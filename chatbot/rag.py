"""RAG (Retrieval-Augmented Generation) functionality using LlamaIndex and Qdrant."""

import pathlib
import tempfile
from collections.abc import Sequence
from typing import Any, TypeAlias

import httpx
import numpy as np
import qdrant_client
from llama_index import core
from llama_index.core import ingestion, node_parser, postprocessor, schema
from llama_index.embeddings import huggingface
from llama_index.vector_stores import qdrant
from qdrant_client.http import exceptions as qdrant_exceptions, models
from streamlit import logger
from streamlit.runtime import uploaded_file_manager

from chatbot import constants, settings
from chatbot.web import http

LOGGER = logger.get_logger(__name__)

EmbeddingVectorType: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]


class RAG:
    """Handles document processing and retrieval for RAG functionality."""

    def __init__(self) -> None:
        """Initialize the RAG processor with Qdrant vector store and hybrid search."""
        LOGGER.info("Initializing RAG processor with Qdrant hybrid vector store.")
        self.client = qdrant_client.QdrantClient(
            url=settings.CHATBOT_SETTINGS.qdrant.url, api_key=settings.CHATBOT_SETTINGS.qdrant.api_key
        )
        self.embedding_model = huggingface.HuggingFaceEmbedding(
            model_name=settings.CHATBOT_SETTINGS.rag.embedding_model, device=settings.CHATBOT_SETTINGS.rag.device
        )

        # Configure LlamaIndex settings
        core.Settings.embed_model = self.embedding_model
        core.Settings.chunk_size = settings.CHATBOT_SETTINGS.rag.chunk_size
        core.Settings.chunk_overlap = settings.CHATBOT_SETTINGS.rag.chunk_overlap

        self._ensure_collection_exists()

        # Initialize vector store with hybrid search capability
        if settings.CHATBOT_SETTINGS.rag.use_hybrid_retrieval:
            LOGGER.info("Enabling hybrid search with sparse model: %s", settings.CHATBOT_SETTINGS.rag.sparse_model)
            self.vector_store = qdrant.QdrantVectorStore(
                client=self.client,
                collection_name=settings.CHATBOT_SETTINGS.qdrant.collection_name,
                fastembed_sparse_model=settings.CHATBOT_SETTINGS.rag.sparse_model,
            )
        else:
            self.vector_store = qdrant.QdrantVectorStore(
                client=self.client,
                collection_name=settings.CHATBOT_SETTINGS.qdrant.collection_name,
            )
        self.index = core.VectorStoreIndex.from_vector_store(  # pyright: ignore[reportUnknownMemberType]
            vector_store=self.vector_store, embed_model=self.embedding_model
        )  # type: ignore

        # Configure retriever with hybrid search parameters
        retriever_kwargs: dict[str, Any] = {"similarity_top_k": settings.CHATBOT_SETTINGS.rag.hybrid_top_k}
        if settings.CHATBOT_SETTINGS.rag.use_hybrid_retrieval:
            retriever_kwargs["sparse_top_k"] = settings.CHATBOT_SETTINGS.rag.hybrid_top_k

        # Add similarity postprocessor to filter low-relevance nodes
        retriever_kwargs["node_postprocessors"] = [postprocessor.SimilarityPostprocessor(similarity_cutoff=0.8)]

        self.retriever = self.index.as_retriever(**retriever_kwargs)

        # Initialize parsers and pipelines for adaptive parsing
        self._init_parsers()

    def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists."""
        try:
            self.client.get_collection(settings.CHATBOT_SETTINGS.qdrant.collection_name)
        except qdrant_exceptions.UnexpectedResponse:
            LOGGER.info(
                "Creating new Qdrant collection by name: %s",
                settings.CHATBOT_SETTINGS.qdrant.collection_name,
            )
            self.client.create_collection(
                collection_name=settings.CHATBOT_SETTINGS.qdrant.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.CHATBOT_SETTINGS.qdrant.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def _init_parsers(self) -> None:
        """Initialize cached parsers and pipelines for adaptive parsing."""
        self._sentence_parser = node_parser.SentenceSplitter(
            chunk_size=settings.CHATBOT_SETTINGS.rag.chunk_size,
            chunk_overlap=settings.CHATBOT_SETTINGS.rag.chunk_overlap,
        )
        self._sentence_pipeline = ingestion.IngestionPipeline(
            transformations=[self._sentence_parser, self.embedding_model]
        )

        if settings.CHATBOT_SETTINGS.rag.use_adaptive_parsing:
            self._markdown_parser = node_parser.MarkdownNodeParser()
            self._html_parser = node_parser.HTMLNodeParser()

            self._semantic_parser = node_parser.SemanticSplitterNodeParser(
                embed_model=self.embedding_model,
                buffer_size=1,
                breakpoint_percentile_threshold=settings.CHATBOT_SETTINGS.rag.semantic_breakpoint_threshold,
            )

            # Initialize pipelines (excluding code pipeline - created dynamically)
            self._markdown_pipeline = ingestion.IngestionPipeline(
                transformations=[self._markdown_parser, self.embedding_model]
            )
            self._html_pipeline = ingestion.IngestionPipeline(transformations=[self._html_parser, self.embedding_model])
            self._semantic_pipeline = ingestion.IngestionPipeline(
                transformations=[self._semantic_parser, self.embedding_model]
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
            temp_path = pathlib.Path(temp_dir)

            for uploaded_file in uploaded_files:
                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

            documents = core.SimpleDirectoryReader(temp_dir).load_data(show_progress=True)

        # Parse documents into chunks using adaptive parsing
        nodes: Sequence[schema.BaseNode]
        if settings.CHATBOT_SETTINGS.rag.use_adaptive_parsing:
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
        urls, web_docs_content = await http.fetch_from_http_urls_in_prompt(prompt, client)
        if not web_docs_content:
            return

        # Create documents with URL metadata for better tracking
        documents: list[core.Document] = []
        for url, content in zip(urls, web_docs_content, strict=True):
            if content:
                LOGGER.debug("Processing web URL: %s", url)
                LOGGER.debug("Content: %s", content[:100])
                doc = core.Document(text=content, metadata={"source": "web_url", "url": url, "content_type": "html"})
                documents.append(doc)

        if not documents:
            return

        # Process web documents using HTML pipeline for consistent parsing
        if settings.CHATBOT_SETTINGS.rag.use_adaptive_parsing and hasattr(self, "_html_pipeline"):
            nodes = self._html_pipeline.run(documents=documents)
        else:
            # Fallback to sentence-based parsing if adaptive parsing is disabled
            nodes = self._sentence_pipeline.run(documents=documents)

        self.index.insert_nodes(nodes)

    def retrieve(self, query_text: str) -> list[str]:
        """Retrieve relevant context chunks using Qdrant's hybrid retrieval.

        Args:
            query_text: The user's query.

        Returns:
            List of relevant text chunks with duplicates removed.
        """
        LOGGER.info("Retrieving context for query: %s", query_text)
        try:
            # Use Qdrant's built-in hybrid retrieval (dense + sparse)
            nodes = self.retriever.retrieve(query_text)

            # Apply relevance filtering if enabled using node scores
            if settings.CHATBOT_SETTINGS.rag.enable_relevance_filtering:
                filtered_nodes = [
                    node
                    for node in nodes
                    if node.score is not None and node.score >= settings.CHATBOT_SETTINGS.rag.relevance_threshold
                ]
                LOGGER.info(
                    "Filtered %d -> %d nodes by relevance (threshold: %s)",
                    len(nodes),
                    len(filtered_nodes),
                    settings.CHATBOT_SETTINGS.rag.relevance_threshold,
                )
                chunks = [node.text for node in filtered_nodes]
            else:
                chunks = [node.text for node in nodes]

            # Apply deduplication and return top_k results
            return self._deduplicate_chunks(chunks)[: settings.CHATBOT_SETTINGS.rag.top_k]

        except (qdrant_exceptions.UnexpectedResponse, ValueError, RuntimeError) as e:
            LOGGER.error("Error during retrieval: %s", e)
            return []

    def _deduplicate_chunks(self, chunks: list[str]) -> list[str]:
        """Remove duplicate and highly similar chunks using vectorized operations.

        Args:
            chunks: List of text chunks to deduplicate.

        Returns:
            Deduplicated list of chunks.
        """
        LOGGER.info("Deduplicating chunks.")
        if not chunks:
            return chunks

        # Batch compute all embeddings at once for efficiency
        chunk_embeddings = self.embedding_model.get_text_embedding_batch(chunks)
        embeddings_array = np.array(chunk_embeddings)

        deduplicated_chunks: list[str] = []
        deduplicated_indices: list[int] = []

        for i, chunk in enumerate(chunks):
            if not deduplicated_indices:
                # First chunk is always included
                deduplicated_chunks.append(chunk)
                deduplicated_indices.append(i)
                continue

            # Vectorized cosine similarity with all existing embeddings
            current_embedding = embeddings_array[i : i + 1]  # Keep 2D shape
            existing_embeddings = embeddings_array[deduplicated_indices]

            similarities = self._vectorized_cosine_similarity(current_embedding, existing_embeddings)

            # Check if any similarity exceeds threshold
            if np.all(similarities < settings.CHATBOT_SETTINGS.rag.deduplication_similarity_threshold):
                deduplicated_chunks.append(chunk)
                deduplicated_indices.append(i)

        return deduplicated_chunks

    def _vectorized_cosine_similarity(
        self, embeddings1: EmbeddingVectorType, embeddings2: EmbeddingVectorType
    ) -> EmbeddingVectorType:
        """Calculate vectorized cosine similarity between embedding matrices.

        Args:
            embeddings1: First embedding matrix (n1 x d).
            embeddings2: Second embedding matrix (n2 x d).

        Returns:
            Cosine similarity matrix (n1 x n2).
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        # Handle zero norms
        norm1 = np.where(norm1 == 0, 1, norm1)
        norm2 = np.where(norm2 == 0, 1, norm2)

        normalized1: EmbeddingVectorType = embeddings1 / norm1
        normalized2: EmbeddingVectorType = embeddings2 / norm2

        # Compute cosine similarity via dot product of normalized vectors
        return np.dot(normalized1, normalized2.T)  # type: ignore

    def _get_file_type(self, file_path: str) -> constants.FileTypes:
        """Determine file type for parser selection.

        Args:
            file_path: pathlib.Path to the file.

        Returns:
            File type category for parser selection.
        """
        suffix = pathlib.Path(file_path).suffix.lower()

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
                    LOGGER.warning(
                        "Unknown code language, falling back to semantic splitter for %d documents",
                        len(docs),
                    )
                    nodes = self._semantic_pipeline.run(documents=docs)
                else:
                    # Create language-specific code parser
                    code_parser = node_parser.CodeSplitter(
                        language=language,
                        chunk_lines=settings.CHATBOT_SETTINGS.rag.code_chunk_lines,
                        chunk_lines_overlap=settings.CHATBOT_SETTINGS.rag.code_chunk_overlap_lines,
                        max_chars=settings.CHATBOT_SETTINGS.rag.chunk_size,
                    )
                    code_pipeline = ingestion.IngestionPipeline(transformations=[code_parser, self.embedding_model])
                    nodes = code_pipeline.run(documents=docs)

                all_nodes.extend(nodes)

            except (ValueError, ImportError, RuntimeError) as e:  # noqa: PERF203
                LOGGER.error(
                    "CodeSplitter failed for language '%s': %s. Falling back to semantic splitter.", language, e
                )
                # Fallback to semantic splitter on any error
                fallback_nodes = self._semantic_pipeline.run(documents=docs)
                all_nodes.extend(fallback_nodes)

        return all_nodes

    def _detect_code_language(self, file_path: str) -> str:
        """Detect programming language from file path.

        Args:
            file_path: pathlib.Path to the code file.

        Returns:
            Language identifier for CodeSplitter, or "unknown" if not detected.
        """
        if not file_path:
            return "unknown"

        suffix = pathlib.Path(file_path).suffix.lower()
        return constants.EXTENSION_TO_LANGUAGE_MAPPING.get(suffix, "unknown")
