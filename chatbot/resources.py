"""Shared or re-usable resources for the chatbot application."""

import streamlit as st
import transformers
from llama_index.embeddings import huggingface

from chatbot import rag, settings


@st.cache_resource
def get_rag_processor() -> rag.RAG:
    """Returns the RAG processor instance.

    Returns:
        RAG processor instance for handling retrieval-augmented generation.
    """
    return rag.RAG()


@st.cache_resource
def get_tokenizer() -> transformers.PreTrainedTokenizerFast:
    """Get cached tokenizer for token counting.

    Returns:
        Tokenizer instance for the configured model
    """
    return transformers.AutoTokenizer.from_pretrained(  # nosec B615: revision pinned via settings
        settings.CHATBOT_SETTINGS.llm_model_name,
        revision=settings.CHATBOT_SETTINGS.llm_model_revision,
        trust_remote_code=False,
    )


@st.cache_resource
def get_embedding_model() -> huggingface.HuggingFaceEmbedding:
    """Get cached embedding model for similarity computation.

    Returns:
        Embedding model instance
    """
    return huggingface.HuggingFaceEmbedding(
        model_name=settings.CHATBOT_SETTINGS.rag.embedding_model,
        device=settings.CHATBOT_SETTINGS.rag.device,
    )
