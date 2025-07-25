import asyncio
from collections.abc import Sequence
import logging
import re

import httpx

LOGGER = logging.getLogger(__name__)
URL_REGEX = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from a text block.

    Args:
        text: Input text to search for URLs

    Returns:
        List of unique URLs found in the text
    """
    urls = URL_REGEX.findall(text, re.IGNORECASE)
    unique_urls = list(set(urls))
    LOGGER.info("Extracted URLs from prompt:\n-%s", "\n-".join(unique_urls))
    return unique_urls


async def _fetch_url(url: str, client: httpx.AsyncClient) -> str:
    """Fetch content from a single URL.

    Args:
        url: URL to fetch
        client: HTTP client for making requests

    Returns:
        Response text from the URL
    """
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
    except httpx.HTTPStatusError as e:
        LOGGER.warning("Failed to fetch %s: %s", url, e)
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
    tasks = [_fetch_url(url, client) for url in urls]
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
    docs = await fetch_content_from_urls(urls, client)

    return urls, docs
