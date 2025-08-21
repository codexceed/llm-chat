"""Base classes for search API integration."""

import abc
import urllib.parse
from typing import Any

import pydantic


class SearchResult(pydantic.BaseModel):
    """Represents a single search result with metadata."""

    title: str
    url: str
    snippet: str
    rank: int
    domain: str = ""
    published_date: str | None = None
    search_query: str = ""

    @pydantic.model_validator(mode="after")
    def extract_domain_from_url(self) -> "SearchResult":
        """Extract domain from URL if not provided.

        Returns:
            The SearchResult instance with domain extracted if needed.
        """
        if not self.domain and self.url:
            parsed = urllib.parse.urlparse(self.url)
            self.domain = parsed.netloc
        return self


class SearchClient(abc.ABC):
    """Abstract base class for search API clients."""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initialize search client with API credentials.

        Args:
            api_key: API key for the search service
            kwargs: Additional keyword arguments for initialization
        """
        self.api_key = api_key
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abc.abstractmethod
    async def search(self, query: str, num_results: int = 10, **kwargs: Any) -> list[SearchResult]:
        """Perform a search query and return results.

        Args:
            query: Search query string
            num_results: Maximum number of results to return
            **kwargs: Additional search parameters specific to the API

        Returns:
            List of SearchResult objects

        Raises:
            SearchAPIError: When search API request fails
        """


class SearchAPIError(Exception):
    """Exception raised when search API requests fail."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize search API error.

        Args:
            message: Error message
            status_code: HTTP status code if applicable
        """
        super().__init__(message)
        self.status_code = status_code
