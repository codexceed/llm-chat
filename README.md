# Streamlit Chatbot

A chatbot application that leverages LLMs served via OpenAI spec APIs.

## Installation

1. **Install the package:**

   ```bash
   pip install -e .
   ```

## Usage

1. **Update the `.env` file with your API keys and other necessary settings.**

   ```.env
   OPENAI_API_BASE=http://localhost:8000/v1
   MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
   TEMPERATURE=0.7
   MAX_TOKENS=2000
   UPLOAD_DIR=uploads
   PAGE_TITLE="My Custom Chatbot"
   OPENAI_API_KEY=not-needed
   HOST=0.0.0.0
   PORT=8080
   DEBUG=True
   ```

2.**Run the chatbot:**

   ```bash
   chatbot
   ```
