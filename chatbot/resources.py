import streamlit as st

from chatbot import rag


@st.cache_resource
def get_rag_processor() -> rag.RAG:
    """Returns the RAG processor instance.'

    Returns:
        RAG processor instance for handling retrieval-augmented generation.
    """
    return rag.RAG()
