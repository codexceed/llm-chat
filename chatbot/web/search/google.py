"""Google Custom Search API client implementation."""

import asyncio
import logging
from typing import Any

import tenacity
from googleapiclient import discovery, errors as google_errors
from streamlit import logger

from chatbot.web.search import base

LOGGER = logger.get_logger(__name__)


class GoogleSearchClient(base.SearchClient):
    """Google Custom Search API client."""

    def __init__(self, api_key: str, search_engine_id: str, **kwargs: dict[str, Any]) -> None:
        """Initialize Google Search client.

        Args:
            api_key: Google API key
            search_engine_id: Custom Search Engine ID
            kwargs: Additional key-word arguments
        """
        super().__init__(api_key)
        self.search_engine_id = search_engine_id
        self.service = discovery.build("customsearch", "v1", developerKey=api_key)

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((google_errors.HttpError, ConnectionError, TimeoutError)),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=4),
        before=tenacity.before_log(LOGGER, logging.WARNING),
        after=tenacity.after_log(LOGGER, logging.WARNING),
    )
    async def search(self, query: str, num_results: int = 10, **kwargs: Any) -> list[base.SearchResult]:
        """Perform Google Custom Search and return results.

        Args:
            query: Search query string
            num_results: Maximum number of results to return (max 10 per request)
            **kwargs: Additional search parameters

        Returns:
            List of SearchResult objects

        Raises:
            base.SearchAPIError: When Google API request fails
        """
        LOGGER.info("Performing Google search for query: %s", query)

        # Google Custom Search API limits to 10 results per request
        num_results = min(num_results, 10)

        # Build search parameters
        search_params = {
            "q": query,
            "cx": self.search_engine_id,
            "num": num_results,
            "fields": "items(title,link,snippet,displayLink,pagemap)",
        }

        # Add optional parameters from kwargs
        search_params.update({k: v for k, v in kwargs.items() if v is not None})

        try:
            # Execute search in a thread pool since google-api-python-client is synchronous
            data = await asyncio.to_thread(lambda: self.service.cse().list(**search_params).execute())

            results = []
            items = data.get("items", [])

            for i, item in enumerate(items):
                # Extract published date from pagemap if available
                published_date = None
                pagemap = item.get("pagemap", {})
                if pagemap.get("metatags"):
                    metatag = pagemap["metatags"][0]
                    # Look for common date fields
                    for date_field in ("article:published_time", "article:modified_time", "og:updated_time"):
                        if date_field in metatag:
                            published_date = metatag[date_field]
                            break

                search_result = base.SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    rank=i + 1,
                    domain=item.get("displayLink", ""),
                    published_date=published_date,
                    search_query=query,
                )
                results.append(search_result)

            LOGGER.info("Google search returned %d results", len(results))
            return results

        except google_errors.HttpError as e:
            error_msg = f"Google Search API HTTP error: {e.resp.status}"
            if e.resp.status == 429:
                error_msg += " (Rate limit exceeded)"
            elif e.resp.status == 403:
                error_msg += " (Quota exceeded or invalid API key)"
            elif e.resp.status == 400:
                error_msg += " (Bad request - check search parameters)"
            LOGGER.error(error_msg)
            raise base.SearchAPIError(error_msg, e.resp.status) from e

        except (ConnectionError, TimeoutError) as e:
            error_msg = f"Google Search API connection error: {e}"
            LOGGER.error(error_msg)
            raise base.SearchAPIError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error in Google Search: {e}"
            LOGGER.error(error_msg)
            raise base.SearchAPIError(error_msg) from e
