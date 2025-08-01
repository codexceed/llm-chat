# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a learning project meant to help the developer understand the state-of-the-art in ML engineering concepts such as inference optimization, LLMs, agentic systems, distributed systems, etc.

## Development Commands

### Setup and Installation
```bash
pip install -e .              # Install package in development mode
make install-dev              # Install with development dependencies (uses uv)
make dev                      # Complete development environment setup
make install-uv               # Install uv package manager (if needed)
```

### Code Quality and Linting
```bash
make lint-all                 # Run all linters and formatting (recommended)
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

### Profiling
```bash
make profile-install          # Install profiling dependencies (with uv)
make profile-interactive      # Run interactive profiling with Streamlit
make profile-batch            # Run batch profiling tests
make profile-compare          # Compare profiling sessions
make profile-clean            # Clean profiling output
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

- **`chatbot/app.py`** - Main Streamlit application entry point
- **`chatbot/utils/chat.py`** - Core chat logic and LLM interaction with streaming responses
- **`chatbot/rag.py`** - RAG system with Qdrant vector store and adaptive document parsing
- **`chatbot/utils/ui.py`** - Streamlit UI components and interface rendering
- **`chatbot/settings.py`** - Centralized configuration management using Pydantic settings
- **`chatbot/constants.py`** - Type definitions and file type mappings for RAG processing
- **`chatbot/cli.py`** - Command-line interface entry point
- **`chatbot/utils/web.py`** - Web content processing and URL handling
- **`chatbot/resources.py`** - Resource management and initialization

### Key Architecture Patterns

**RAG System Architecture:**
- Uses LlamaIndex framework for document processing and indexing
- Qdrant vector database for similarity search
- Adaptive parsing strategy based on file types (code, markdown, HTML, text)
- Supports URL extraction and web content processing
- Intelligent chunk deduplication using cosine similarity
- Language-specific code parsing with Tree-sitter integration

**Configuration Management:**
- Environment-based configuration with `.env` file support
- Nested settings structure (qdrant, rag sub-configurations)
- Prefix-based environment variables (`CHATBOT_*`)

**Chat Flow:**
1. User input processed through Streamlit interface
2. File uploads automatically indexed into vector store
3. URLs extracted from messages and web content fetched/indexed asynchronously in background
4. RAG context retrieved and appended to user queries
5. Streaming responses from OpenAI-compatible LLM API

**Non-blocking URL Processing:**
- URLs in user messages are extracted and processed in background threads
- User receives immediate response while URL content is indexed asynchronously
- Prevents slow/failed URL fetches from blocking other user sessions

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
- **Qdrant** for vector storage and similarity search
- **HuggingFace** embeddings (default: BAAI/bge-small-en-v1.5)
- **Tree-sitter** for code parsing
- **Pydantic** for configuration management

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and customize settings.

Key configuration areas:
- **LLM settings**: API endpoint, model name, temperature, token limits
- **RAG settings**: Embedding model, chunk size, retrieval parameters
- **Qdrant settings**: Database URL, collection name, vector dimensions
- **Adaptive parsing**: Code chunk sizes, semantic parsing thresholds

## Storage Structure

- **`storage/`** - Qdrant vector database persistent storage
- **`storage/collections/`** - Vector collections organized by name
- Local vector store maintains document chunks and embeddings for retrieval

## Development Notes

- Use `make lint-all` before committing changes to ensure code quality
- The application supports both local LLM servers (via vLLM) and remote OpenAI-compatible APIs
- RAG indexing happens automatically on file upload and URL detection
- Vector embeddings are cached locally in the Qdrant storage directory