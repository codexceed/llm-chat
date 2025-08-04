import asyncio
import logging
import re
from collections.abc import Sequence

import httpx
import tenacity
from streamlit import logger

LOGGER = logger.get_logger(__name__)
URL_REGEX = re.compile(
    r"https?:\/\/(?:[-\w.])+(?:\:[0-9]+)?(?:\/(?:[\w\/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?", re.IGNORECASE
)


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


@tenacity.retry(
    retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)),
    stop=tenacity.stop_after_attempt(4),  # 3 retries + 1 initial attempt
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=8),  # 1s, 2s, 4s, 8s
    before=tenacity.before_log(LOGGER, logging.WARNING),
    after=tenacity.after_log(LOGGER, logging.WARNING),
)
async def _fetch_url(url: str, client: httpx.AsyncClient) -> str:
    """Fetch content from a single URL with retry logic using tenacity decorator.

    Args:
        url: URL to fetch
        client: HTTP client for making requests

    Returns:
        Response text from the URL

    Raises:
        httpx.ConnectError: Error when establishing connection
        httpx.ConnectTimeout: Connection timed out
        httpx.ReadTimeout: Read timeout
    """
    # Headers to appear more like a legitimate browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

    try:
        response = await client.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
        LOGGER.warning("Error fetching %s: %s", url, e)
        raise
    except httpx.HTTPStatusError as e:
        LOGGER.warning("HTTP error for %s: %s", url, e)
        return ""
    except Exception as e:
        LOGGER.warning("Unexpected error fetching %s: %s", url, e)
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

    LOGGER.debug("Looking up URLs in prompt:\n-%s", "- ".join(urls))

    docs = await fetch_content_from_urls(urls, client)

    return urls, docs
