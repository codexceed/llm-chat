"""Brave Search API client implementation."""

import logging
from typing import Any

import httpx
import tenacity
from streamlit import logger

from chatbot.search import base

LOGGER = logger.get_logger(__name__)


class BraveSearchClient(base.SearchClient):
    """Brave Search API client."""

    def __init__(self, api_key: str, **kwargs: dict[str, Any]) -> None:
        """Initialize Brave Search client.

        Args:
            api_key: Brave Search API key
            kwargs: Additional keyword arguments for initialization
        """
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=4),
        before=tenacity.before_log(LOGGER, logging.WARNING),
        after=tenacity.after_log(LOGGER, logging.WARNING),
    )
    async def search(self, query: str, num_results: int = 10, **kwargs: dict[str, Any]) -> list[base.SearchResult]:
        """Perform Brave Search and return results.

        Args:
            query: Search query string
            num_results: Maximum number of results to return (max 20 per request)
            **kwargs: Additional search parameters

        Returns:
            List of SearchResult objects

        Raises:
            base.SearchAPIError: When Brave API request fails
        """
        LOGGER.info("Performing Brave search for query: %s", query)

        # Brave Search API limits to 20 results per request
        num_results = min(num_results, 20)

        params = {
            "q": query,
            "count": num_results,
            "offset": 0,
            "mkt": "en-US",
            "safesearch": "moderate",
            "textDecorations": False,
            "textFormat": "Raw",
            **kwargs,
        }

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(self.base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

                results = []
                web_results = data.get("web", {}).get("results", [])

                for i, item in enumerate(web_results):
                    # Extract published date if available
                    published_date = None
                    if item.get("age"):
                        published_date = item["age"]
                    elif item.get("published"):
                        published_date = item["published"]

                    search_result = base.SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", ""),
                        rank=i + 1,
                        domain="",  # Will be extracted in __post_init__
                        published_date=published_date,
                        search_query=query,
                    )
                    results.append(search_result)

                LOGGER.info("Brave search returned %d results", len(results))
                return results

            except httpx.HTTPStatusError as e:
                error_msg = f"Brave Search API HTTP error: {e.response.status_code}"
                if e.response.status_code == 429:
                    error_msg += " (Rate limit exceeded)"
                elif e.response.status_code == 401:
                    error_msg += " (Invalid API key)"
                elif e.response.status_code == 403:
                    error_msg += " (Quota exceeded)"
                LOGGER.error(error_msg)
                raise base.SearchAPIError(error_msg, e.response.status_code) from e

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                error_msg = f"Brave Search API connection error: {e}"
                LOGGER.error(error_msg)
                raise base.SearchAPIError(error_msg) from e

            except Exception as e:
                error_msg = f"Unexpected error in Brave Search: {e}"
                LOGGER.error(error_msg)
                raise base.SearchAPIError(error_msg) from e
