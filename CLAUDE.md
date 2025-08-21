# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a learning project meant to help the developer understand the state-of-the-art in ML engineering concepts such as inference optimization, LLMs, agentic systems, distributed systems, etc.

## Development Commands

### Setup and Installation
```bash
make install-all              # Install all dependencies in development mode
make install-dev              # Install with development dependencies (uses uv)
make install-based            # Install base dependencies necessary to run the chatbot
make install-uv               # Install uv package manager (if needed)
```

### Code Quality and Linting
```bash
make lint-all                 # Run all linters and formatting (recommended)
make lint-fast                # Run fast running linters and formatters
make format                   # Format code with ruff
make lint                     # Check with ruff linter
make lint-fix                 # Auto-fix linting issues
make type-check               # Run mypy type checker
make security                 # Run bandit security scanner
make check                    # Quick validation (no formatting)
```

### Testing
```bash
make test                     # Run all tests
make test-unit                # Run unit tests only
make test-coverage            # Run tests with coverage report
python -m pytest tests/ -v   # Direct pytest command
```

### Running the Application
```bash
chatbot                       # Start the Streamlit chatbot application
```

### Docker Services
```bash
# Start local LLM servers
docker-compose up vllm-qwen-2.5-coder-7b-instruct-awq
docker-compose up vllm-qwen-2.5-coder-14b-instruct-awq

# Start vector database
docker-compose up qdrant
```

## Architecture Overview

This is a modern chatbot application built with Streamlit that integrates Large Language Models via OpenAI-compatible APIs and includes RAG (Retrieval-Augmented Generation) capabilities.

### Core Components

- **`chatbot/app.py`** - Main Streamlit application with integrated web context pipeline
- **`chatbot/utils/chat.py`** - Core chat logic and LLM interaction with streaming responses
- **`chatbot/rag.py`** - RAG system with Qdrant vector store, hybrid retrieval, and adaptive parsing
- **`chatbot/utils/ui.py`** - Streamlit UI components and interface rendering
- **`chatbot/settings.py`** - Centralized configuration management using Pydantic settings
- **`chatbot/constants.py`** - Type definitions and file type mappings for RAG processing
- **`chatbot/cli.py`** - Command-line interface entry point
- **`chatbot/web/`** - Web processing module with search integration and URL handling
  - **`http.py`** - HTTP client utilities with retry logic and content sanitization
  - **`context.py`** - Web context pipeline for unified web content processing
  - **`search/`** - Search API integrations (Google Custom Search, Brave Search)
- **`chatbot/resources.py`** - Resource management and initialization

### Key Architecture Patterns

**RAG System Architecture:**
- LlamaIndex framework for document processing and indexing
- Qdrant vector database with hybrid retrieval (dense + Qdrant/bm25 sparse embeddings)
- Adaptive parsing strategy based on file types (code, markdown, HTML, text)
- Intelligent chunk deduplication using vectorized cosine similarity
- Language-specific code parsing with Tree-sitter integration

**Web Context Pipeline:**
- Independent web content processing separate from RAG system
- Search API integration (Google Custom Search, Brave Search) with automatic triggering
- Concurrent URL fetching with retry logic and content sanitization
- Force search capability with manual overrides (`search:` prefix)
- Unified context merging from multiple sources (search, URLs, RAG)

**Configuration Management:**
- Environment-based configuration with `.env` file support
- Nested settings structure (qdrant, rag, search sub-configurations)
- Prefix-based environment variables (`CHATBOT_*`)

**Enhanced Chat Flow:**
1. User input processed through Streamlit interface
2. File uploads automatically indexed into vector store
3. Web context pipeline gathers content from URLs and search APIs concurrently
4. RAG context retrieved from vector database with hybrid search
5. All context sources merged and appended to user queries
6. Streaming responses from OpenAI-compatible LLM API

### File Processing Strategy

The RAG system uses adaptive parsing based on file extensions:
- **Code files** (`.py`, `.js`, `.ts`, etc.) → Language-specific Tree-sitter parsing
- **Markdown files** (`.md`) → Markdown-aware node parsing
- **HTML files** (`.html`, `.htm`) → HTML structure parsing
- **Text files** (`.txt`) → Sentence-based splitting
- **Unknown types** → Semantic splitting with embeddings

### Dependencies and Integration

- **Streamlit** for web interface
- **OpenAI client** for LLM API communication  
- **LlamaIndex** for RAG document processing
- **Qdrant** for vector storage with hybrid search (dense + Qdrant/bm25 sparse)
- **HuggingFace** embeddings (default: BAAI/bge-small-en-v1.5)
- **Tree-sitter** for code parsing
- **Pydantic** for configuration management
- **httpx** for async HTTP requests with retry logic
- **trafilatura** for web content extraction and sanitization
- **Google Custom Search / Brave Search** APIs for web search

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and customize settings.

Key configuration areas:
- **LLM settings**: API endpoint, model name, temperature, token limits
- **RAG settings**: Embedding model, chunk size, hybrid retrieval, relevance filtering
- **Qdrant settings**: Database URL, collection name, vector dimensions
- **Search settings**: Provider (Google/Brave), API keys, trigger words, result limits
- **Adaptive parsing**: Code chunk sizes, semantic parsing thresholds

## Storage Structure

- **`storage/`** - Qdrant vector database persistent storage
- **`storage/collections/`** - Vector collections organized by name
- Local vector store maintains document chunks and embeddings for retrieval

## Development Notes

- Always check code quality using `code-quality` agent after performing code changes
- Think critically and deeply before you respond to:
  - Queries about major feature updates (consider design implications for scale, maintainability, effort)
  - Discussions about performance optimization
  - Discussions about code refactoring
- When writing tests:
  - Look for any functions or classes in the test scope that are likely to have a large number of test parameters due to their inflated functionality and flag them to the user
  - Use `pytest` and its test function format
  - Prefer using `hypothesis` library to generate property-based test cases over hard-coded test cases
  - Avoid mocking dependencies as much as possible. Consider looking up pre-existing libraries that can effectively mock dependencies.