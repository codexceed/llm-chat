# ADR-002: RAG System Architecture - LlamaIndex + Qdrant Vector Store

## Status
Accepted

## Context
The chatbot application required Retrieval-Augmented Generation (RAG) capabilities to provide contextual responses based on uploaded documents and web content. The system needed to:

1. **Document Processing**: Handle multiple file types (code, markdown, HTML, text)
2. **Vector Storage**: Efficient similarity search and retrieval
3. **Embeddings**: Convert text chunks to vector representations
4. **Integration**: Seamless connection with the Streamlit chat interface
5. **Performance**: Fast retrieval for real-time chat responses
6. **Scalability**: Handle growing document collections
7. **Flexibility**: Support different parsing strategies per file type

## Decision
We chose **LlamaIndex** as the RAG framework and **Qdrant** as the vector database.

### RAG Framework: LlamaIndex
- Comprehensive document processing pipeline with multiple parsers
- Built-in support for various file formats and content types
- Flexible node/chunk management with metadata preservation
- Easy integration with multiple vector stores and LLM providers
- Rich ecosystem of tools for document ingestion and retrieval

### Vector Database: Qdrant
- High-performance vector similarity search
- Native support for metadata filtering and hybrid search
- Persistent storage with collection-based organization
- RESTful API with Python client library
- Docker deployment for easy local development
- Horizontal scaling capabilities for production use

### Embedding Model: BAAI/bge-small-en-v1.5
- Optimized for English text with good performance/size balance
- HuggingFace integration through LlamaIndex
- Suitable for general-purpose document retrieval tasks

## Alternatives Considered

### RAG Framework Alternatives
1. **LangChain**: More complex abstractions, less focused on document processing
2. **Custom Implementation**: Full control but significant development overhead
3. **Haystack**: Comprehensive but heavier framework with steeper learning curve
4. **txtai**: Simpler but less flexible for diverse document types

### Vector Database Alternatives
1. **Chroma**: Simpler setup but less performant for larger collections
2. **Pinecone**: Managed service but vendor lock-in and cost concerns
3. **Weaviate**: Feature-rich but more complex deployment and configuration
4. **FAISS**: High performance but requires custom persistence layer
5. **Elasticsearch**: Mature but overkill for pure vector search use case

### Embedding Model Alternatives
1. **OpenAI Embeddings**: High quality but API costs and external dependency
2. **sentence-transformers/all-MiniLM-L6-v2**: Smaller but lower quality
3. **text-embedding-ada-002**: Excellent quality but expensive for large volumes
4. **Local transformers models**: Custom fine-tuning potential but complexity

## Consequences

### Positive
- **Rapid Development**: LlamaIndex provides pre-built parsers and integrations
- **File Type Support**: Built-in handling for code, markdown, HTML, and text files
- **Performance**: Qdrant offers fast similarity search with metadata filtering
- **Local Development**: Docker setup enables offline development and testing
- **Flexibility**: Easy to experiment with different embedding models and chunk strategies
- **Metadata Preservation**: Rich document metadata support for enhanced retrieval
- **Persistence**: Qdrant provides reliable storage for document collections

### Negative
- **Learning Curve**: LlamaIndex abstractions require understanding of concepts
- **Memory Usage**: Embedding models and vector storage consume significant memory
- **Dependency Weight**: Large dependency tree compared to simpler alternatives
- **Version Compatibility**: Frequent updates may introduce breaking changes
- **Deployment Complexity**: Additional infrastructure component (Qdrant) to manage

### Risks and Mitigations
- **Performance Bottlenecks**: Can optimize chunk sizes and retrieval parameters
- **Storage Growth**: Implement collection cleanup and document lifecycle management
- **Embedding Quality**: Can switch to different models through LlamaIndex interface
- **Infrastructure Dependencies**: Docker Compose provides consistent local environment

## Implementation Details

### Document Processing Pipeline
1. **File Upload Detection**: Streamlit file uploader triggers processing
2. **Parser Selection**: File extension determines parsing strategy
3. **Chunk Generation**: Text splitting based on content type and size limits
4. **Embedding Creation**: Vector generation using HuggingFace model
5. **Storage**: Vectors and metadata stored in Qdrant collection
6. **Indexing**: Document chunks available for similarity search

### Retrieval Strategy
1. **Query Processing**: User message converted to embedding vector
2. **Similarity Search**: Qdrant finds most relevant document chunks
3. **Context Assembly**: Retrieved chunks formatted as RAG context
4. **LLM Integration**: Context appended to user query for enhanced responses

### Configuration
- Chunk size limits configurable per file type
- Embedding model selectable through environment variables
- Qdrant connection settings externalized
- Similarity thresholds tunable for retrieval quality

## Technical Architecture

```
User Upload → LlamaIndex Parser → Text Chunks → Embeddings → Qdrant Storage
                     ↓
User Query → Embedding → Similarity Search → Context Retrieval → LLM Response
```

## Performance Characteristics
- **Indexing**: ~1-2 seconds per MB of text content
- **Retrieval**: <100ms for similarity search queries
- **Memory**: ~500MB for embedding model + vector storage
- **Storage**: ~4KB per text chunk (embedding + metadata)

## Related Decisions
- [ADR-001: Initial Architecture Decision](001-initial-architecture-decision.md) - Streamlit integration
- [ADR-003: Adaptive Document Parsing Strategy](003-adaptive-document-parsing-strategy.md) - File type handling
- [ADR-004: URL Processing Architecture](004-url-processing-architecture.md) - Web content integration

## References
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [BAAI/bge-small-en-v1.5 Model](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [RAG Best Practices](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)