import asyncio
import re
from collections.abc import Sequence

import httpx
from streamlit import logger

LOGGER = logger.get_logger(__name__)
URL_REGEX = re.compile(r"https?:\/\/(?:[-\w.])+(?:\:[0-9]+)?(?:\/(?:[\w\/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?", re.IGNORECASE)

# HTTP headers to appear as a regular browser
DEFAULT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "<set_this_to_parent_url_or_homepage>",
    "Connection": "keep-alive",
    "DNT": "1",  # Do Not Track header
    "Upgrade-Insecure-Requests": "1",
}

# Retry configuration
MAX_RETRIES = 3
INITIAL_DELAY = 1.0
MAX_DELAY = 8.0


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from a text block.

    Args:
        text: Input text to search for URLs

    Returns:
        List of unique URLs found in the text
    """
    urls = URL_REGEX.findall(text)
    unique_urls = list(set(urls))
    LOGGER.info("Extracted URLs from prompt:\n-%s", "\n-".join(unique_urls))
    return unique_urls


async def _fetch_url_with_retry(url: str, client: httpx.AsyncClient) -> str:
    """Fetch content from a single URL with retry logic.

    Args:
        url: URL to fetch
        client: HTTP client for making requests

    Returns:
        Response text from the URL
    """
    delay = INITIAL_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url, headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:  # noqa: PERF203
            if e.response.status_code == 403:
                LOGGER.warning("Access forbidden for %s (attempt %d/%d): %s", url, attempt + 1, MAX_RETRIES, e)
            elif e.response.status_code in (429, 503, 502, 504):
                LOGGER.warning("Server error for %s (attempt %d/%d): %s", url, attempt + 1, MAX_RETRIES, e)
            else:
                LOGGER.warning("HTTP error for %s: %s", url, e)
                return ""

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)

        except httpx.TimeoutException:
            LOGGER.warning("Timeout fetching %s (attempt %d/%d)", url, attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)

        except Exception as e:
            LOGGER.warning("Unexpected error fetching %s: %s", url, e)
            return ""

    LOGGER.error("Failed to fetch %s after %d attempts", url, MAX_RETRIES)
    return ""


async def fetch_content_from_urls(urls: Sequence[str], client: httpx.AsyncClient) -> list[str]:
    """Fetch content from a list of URLs asynchronously.

    Args:
        urls: List of URLs to fetch
        client: HTTP client for making requests

    Returns:
        List of response text from the URLs
    """
    LOGGER.info("Fetching content from URLs")
    tasks = [_fetch_url_with_retry(url, client) for url in urls]
    responses = await asyncio.gather(*tasks)

    return [response for response in responses if response]


async def lookup_http_urls_in_prompt(prompt: str, client: httpx.AsyncClient) -> tuple[list[str], list[str]]:
    """Look up HTTP URLs in a prompt and fetch their content.

    Args:
        prompt: User input containing potential web URLs
        client: HTTP client for making requests

    Returns:
        Tuple containing a list of URLs and their corresponding content
    """
    urls = extract_urls_from_text(prompt)
    if not urls:
        return [], []

    LOGGER.debug("Looking up URLs in prompt:\n-%s", "- ".join(urls))

    docs = await fetch_content_from_urls(urls, client)

    return urls, docs
