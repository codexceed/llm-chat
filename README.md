# LLM Chat

A modern chatbot application built with Streamlit that leverages Large Language Models via OpenAI-compatible APIs. Features include document upload, RAG (Retrieval-Augmented Generation) capabilities, and a clean, interactive UI.

## Features

- 🤖 **OpenAI-Compatible API Integration** - Works with local LLM servers like vLLM
- 📄 **Document Upload & Processing** - Support for various file types with intelligent parsing
- 🔍 **RAG (Retrieval-Augmented Generation)** - Vector search with Qdrant for enhanced responses
- 🎨 **Clean Streamlit UI** - Modern, responsive chat interface
- ⚙️ **Configurable** - Extensive configuration options via environment variables
- 🐳 **Docker Support** - Ready-to-use Docker Compose setup for LLM servers

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for local LLM servers)
- NVIDIA GPU (optional, for local LLM inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llm-chat
   ```

2. **Install the package:**
   ```bash
   # Basic installation
   pip install -e .
   
   # Or install all dependencies (recommended for development)
   pip install -e ".[all]"
   
   # Using make targets (see Requirements section below)
   make install-all
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

4. **Run the chatbot:**
   ```bash
   chatbot
   ```

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

### Core Settings
```bash
CHATBOT_OPENAI_API_BASE=http://localhost:8000/v1  # LLM API endpoint
CHATBOT_LLM_MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
CHATBOT_TEMPERATURE=0.7
CHATBOT_MAX_TOKENS=2000
CHATBOT_HOST=127.0.0.1
CHATBOT_PORT=8080
```

### RAG Configuration
```bash
CHATBOT_QDRANT__URL=http://localhost:6333         # Vector database URL
CHATBOT_QDRANT__COLLECTION_NAME=chatbot          # Collection name
CHATBOT_RAG__EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHATBOT_RAG__CHUNK_SIZE=1024
CHATBOT_RAG__TOP_K=5
```

## Local LLM Setup

### Using Docker Compose

Start a local LLM server with vLLM:

```bash
# For 7B model (recommended for most users)
docker-compose up vllm-qwen-2.5-coder-7b-instruct-awq

# For 14B model (requires more VRAM)
docker-compose up vllm-qwen-2.5-coder-14b-instruct-awq

# Start vector database
docker-compose up qdrant
```

**Note:** Update the model paths in `docker-compose.yaml` to match your local model storage.

## Requirements Management

The project provides flexible dependency installation through make targets:

```bash
# Install all dependencies (default, recommended)
make install-all

# Install only base/production dependencies
make install-base

# Install development dependencies only
make install-dev

# Install profiling dependencies only
make install-profiling

# Install uv package manager for faster installs
make install-uv

# Show all available requirements targets
make -f .makefiles/requirements.mk help
```

### Available Dependency Groups

- **Base**: Core application dependencies (streamlit, openai, llama-index, etc.)
- **Dev**: Testing and linting tools (pytest, ruff, mypy, bandit)
- **Profiling**: Performance analysis tools (matplotlib, plotly, py-spy)
- **All**: Combines all dependency groups above

## Development

### Setup Development Environment

```bash
# Install all dependencies (includes dev and profiling tools)
make install-all

# Set up pre-commit hooks and complete dev environment
make dev

# Install uv package manager for faster dependency management
make install-uv
```

### Code Quality

```bash
# Run all linting and formatting
make lint-all

# Individual tools
make format      # Format with ruff
make lint        # Check with ruff  
make type-check  # Type check with mypy
make security    # Security scan with bandit

# Quick validation (no formatting changes)
make check
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Unit tests only
make test-unit
```

## Architecture

The application follows a modular design with clear separation of concerns:

- **UI Layer** (`chatbot/utils/ui.py`) - Streamlit interface components
- **Chat Logic** (`chatbot/utils/chat.py`) - LLM interaction and conversation management  
- **RAG System** (`chatbot/rag.py`) - Document processing and vector search
- **Configuration** (`chatbot/settings.py`) - Centralized settings management
- **Web Processing** (`chatbot/utils/web.py`) - URL handling and content fetching
- **Resources** (`chatbot/resources.py`) - Resource management and initialization

## Supported Models

The application works with any OpenAI-compatible API. Tested models include:

- **Qwen2.5-Coder series** (7B, 14B) - Optimized for code generation
- **Custom models** via vLLM server
- **OpenAI GPT models** (with API key)

