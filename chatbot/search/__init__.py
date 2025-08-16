"""Search module for web search integration with RAG system."""

from chatbot.search import base, brave, google, manager

SearchAPIError = base.SearchAPIError
SearchClient = base.SearchClient
SearchResult = base.SearchResult
BraveSearchClient = brave.BraveSearchClient
GoogleSearchClient = google.GoogleSearchClient
SearchManager = manager.SearchManager
SearchProvider = manager.SearchEngineProvider

__all__ = ["BraveSearchClient", "GoogleSearchClient", "SearchAPIError", "SearchClient", "SearchManager", "SearchResult"]
