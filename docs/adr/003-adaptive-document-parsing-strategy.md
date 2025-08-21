# ADR-003: Adaptive Document Parsing Strategy

## Status
Accepted (Implemented in PR #7)

## Date
2025-07-20

## Context

The chatbot application initially operated as a simple LLM interface without the ability to process and understand external documents or context. Users could not upload files, reference web content, or maintain contextual awareness across conversations. This limitation significantly reduced the application's utility for document-based queries and knowledge synthesis tasks.

### Problem Statement
- No mechanism to process uploaded files or web content
- Limited conversational context beyond the immediate prompt
- Inability to answer questions about external documents
- No persistent knowledge base for information retrieval

### Technical Requirements
- Support multiple file types (code, markdown, HTML, text)
- Efficient vector-based similarity search
- Configurable chunk processing strategies
- Deduplication to avoid repetitive context
- Scalable storage for document embeddings

## Decision

We decided to implement an adaptive RAG (Retrieval-Augmented Generation) system using **LlamaIndex** and **Qdrant** vector database, with file-type specific parsing strategies.

### Key Architectural Decisions

**1. Vector Database Choice: Qdrant**
- Chosen for its performance with dense vector operations
- Built-in hybrid search capabilities (dense + sparse vectors)
- Easy local deployment via Docker
- Strong Python integration

**2. Framework Choice: LlamaIndex**
- Comprehensive document processing pipeline
- Built-in node parsers for different content types
- Seamless integration with vector stores
- Established patterns for RAG implementations

**3. Adaptive Parsing Strategy**
Instead of a one-size-fits-all approach, we implemented file-type specific parsing:
- **Code files**: Tree-sitter based language-specific parsing (`chatbot/rag.py:391-398`)
- **Markdown**: Structure-aware parsing preserving headers (`chatbot/rag.py:340-341`)  
- **HTML**: DOM-aware parsing for web content (`chatbot/rag.py:343-344`)
- **Text files**: Sentence-based splitting (`chatbot/rag.py:347-348`)
- **Unknown types**: Semantic splitting using embeddings (`chatbot/rag.py:352-354`)

**4. Deduplication Algorithm**
Implemented vectorized cosine similarity for chunk deduplication (`chatbot/rag.py:223-261`):
- Batch embedding computation for efficiency
- Configurable similarity threshold (default: 0.85)
- Prevents redundant context in retrieval results

## Implementation Details

### Core Components

**RAG Class (`chatbot/rag.py:27-76`)**
- Centralized document processing and retrieval
- Hybrid search configuration with similarity postprocessing
- Cached parser initialization for performance

**Key Methods:**
- `process_uploaded_files()` - File upload handling with temporary directory processing
- `process_web_urls()` - Asynchronous web content extraction and indexing  
- `retrieve()` - Context retrieval with relevance filtering and deduplication
- `_process_documents_adaptively()` - File-type routing and processing

**Configuration Management**
Environment-based settings with nested structure:
```python
# Qdrant connection settings
qdrant.url = "http://localhost:6333"
qdrant.collection_name = "chatbot_documents"

# RAG processing settings  
rag.embedding_model = "BAAI/bge-small-en-v1.5"
rag.chunk_size = 512
rag.use_adaptive_parsing = True
```

### File Processing Pipeline

1. **File Upload** → Temporary directory extraction
2. **Type Detection** → Extension-based classification (`chatbot/rag.py:289-304`)
3. **Adaptive Parsing** → Type-specific node parser selection
4. **Embedding Generation** → HuggingFace model processing
5. **Vector Storage** → Qdrant collection insertion
6. **Retrieval** → Hybrid search with similarity filtering

## Implementation History

Based on git commit analysis from PR #7:

**`6941447`** - Basic file-attachment based RAG pipeline
- Initial LlamaIndex + Qdrant integration
- Simple document processing workflow

**`966d9ee`** - Added `.env.example` configuration
- Environment-based configuration system
- Qdrant connection parameters

**`93fbb69`** - Improved retrieval precision with deduplication logic  
- Vectorized cosine similarity implementation
- Batch embedding processing for efficiency

**`82479c2`** - Improved RAG via adaptive parsing based on file types
- File extension to type mapping (`chatbot/constants.py`)
- Language-specific code parsing with Tree-sitter
- Markdown, HTML, and semantic parsing strategies

**`4398c51`** - Fixed sentence parser initialization bug
- Ensured fallback parser availability when adaptive parsing disabled
- Resolved runtime initialization errors

## Consequences

### Positive
- **Enhanced Capability**: Users can now upload documents and get contextual responses
- **Type-Aware Processing**: Different file types processed with appropriate strategies
- **Performance**: Efficient deduplication and hybrid search reduce irrelevant results
- **Scalability**: Vector database supports large document collections
- **Flexibility**: Configurable parsing strategies and retrieval parameters

### Negative  
- **Complexity**: Increased system complexity with multiple parsing strategies
- **Dependencies**: Additional external dependencies (Qdrant, Tree-sitter, HuggingFace)
- **Resource Usage**: Vector embeddings require significant memory and storage
- **Configuration**: More environment variables and setup complexity

### Risks Mitigated
- **Fallback Strategies**: Unknown file types handled via semantic parsing
- **Error Handling**: Parser failures gracefully degrade to alternative methods
- **Resource Management**: Temporary file cleanup and connection pooling

## Alternatives Considered

**1. Simple Text Concatenation**
- Rejected: No semantic understanding, poor relevance
- Would lose document structure and context

**2. Database Storage with Full-Text Search**  
- Rejected: Limited semantic similarity capabilities
- Traditional keyword search insufficient for contextual retrieval

**3. OpenAI Embeddings**
- Rejected: Cost concerns and external API dependency
- HuggingFace models provide local processing control

## Future Considerations

- **Query Engine Integration**: Adopt LlamaIndex QueryEngine for advanced response synthesis (referenced in `docs/proposals/context_retrieval_pipelines.md`)
- **Multi-Modal Support**: Extend parsing to handle images and PDFs
- **Semantic Chunking**: Improve chunking strategies based on document semantics
- **Performance Optimization**: Implement async embedding generation and parallel processing

---

**References:**
- Git commits: `6941447`, `93fbb69`, `82479c2`, `4398c51` 
- Pull Request #7: "Adaptive RAG for multiple file-types and use-cases"
- Implementation: `chatbot/rag.py`, `chatbot/constants.py`
- Proposal: `docs/proposals/context_retrieval_pipelines.md`