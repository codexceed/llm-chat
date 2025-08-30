"""Shared or re-usable resources for the chatbot application."""

import sentence_transformers
import streamlit as st
import transformers

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
def get_embedding_model() -> sentence_transformers.SentenceTransformer:
    """Get cached embedding model for similarity computation.

    Returns:
        Embedding model instance
    """
    model = sentence_transformers.SentenceTransformer(settings.CHATBOT_SETTINGS.rag.embedding_model)
    model.eval()
    return model
