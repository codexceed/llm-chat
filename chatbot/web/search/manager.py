"""Search manager for coordinating web search operations."""

import enum
from typing import Any, Final

from streamlit import logger

from chatbot.web.search import base, brave, google

LOGGER = logger.get_logger(__name__)


class SearchEngineProvider(enum.Enum):
    """Search engine provider names."""

    GOOGLE = "google"
    BRAVE = "brave"


SEARCH_CLIENT_NAME_MAP: Final[dict[SearchEngineProvider, type[base.SearchClient]]] = {
    SearchEngineProvider.GOOGLE: google.GoogleSearchClient,
    SearchEngineProvider.BRAVE: brave.BraveSearchClient,
}


class SearchManager:
    """Manages web search operations across different providers."""

    def __init__(self, provider: SearchEngineProvider, api_key: str, trigger_words: list[str], **kwargs: Any) -> None:
        """Initialize search manager with settings.

        Args:
            provider: Search engine provider to use.
            api_key: API key for the search engine provider
            trigger_words: Set of words that trigger search
            **kwargs: Additional keyword arguments for the search client
        """
        self._provider = provider
        self._client = SEARCH_CLIENT_NAME_MAP[provider](api_key=api_key, **kwargs)
        self._trigger_words = trigger_words

    async def search(self, query: str, num_results: int) -> list[base.SearchResult]:
        """Perform web search and return results with metadata.

        Args:
            query: Search query string
            num_results: Number of results to return (uses setting default if None)

        Returns:
            List of SearchResult objects with metadata

        Raises:
            SearchAPIError: When search operation fails
        """
        try:
            results = await self._client.search(query, num_results)

            # Log search metrics
            LOGGER.info("Search completed: query='%s', provider='%s', results=%d", query, self._provider, len(results))

            return results

        except base.SearchAPIError:
            LOGGER.error("Search API error for query: %s", query)
            raise
        except Exception as e:
            error_msg = f"Unexpected error during search: {e}"
            LOGGER.error(error_msg)
            raise base.SearchAPIError(error_msg) from e

    def should_trigger_search(self, query: str) -> bool:
        """Determine if a query should trigger web search.

        Args:
            query: User query string

        Returns:
            True if search should be triggered
        """
        query_lower = query.lower()
        for keyword in self._trigger_words:
            if keyword in query_lower:
                LOGGER.info("Search triggered by keyword '%s' in query: %s", keyword, query)
                return True

        return False
