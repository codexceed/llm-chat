import asyncio
import logging
import re
from collections.abc import Sequence

import httpx
import tenacity
import trafilatura
from streamlit import logger

LOGGER = logger.get_logger(__name__)
URL_REGEX = re.compile(
    r"https?\:\/\/(?:[\w\d\.\:\-\@]+)(?:\/[\w\d\-\%\/\.]+)?(?:\?(?:[\w\d]+\=[\w\d\:\/\.\@\;]+)(?:\&[\w\d]+\=[\w\d\:\/\.\@\;]+)*)?(?:\#(?:[\w.])*)?",
    re.IGNORECASE,
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
    LOGGER.info("Extracted URLs from prompt:\n- %s", "\n- ".join(unique_urls))
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
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.5",
        "priority": "u=0, i",
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Brave";v="138"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "sec-gpc": "1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
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
    except (httpx.RequestError, ValueError, UnicodeDecodeError) as e:
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
    LOGGER.info("Fetching content from URLs:\n- %s", "\n- ".join(urls))
    tasks = [_fetch_url(url, client) for url in urls]
    return await asyncio.gather(*tasks)


async def fetch_from_http_urls_in_prompt(prompt: str, client: httpx.AsyncClient) -> tuple[list[str], list[str]]:
    """Fetch content from HTTP URLs in a prompt and fetch their content.

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


def sanitize_web_content(raw_contents: list[str]) -> list[str | None]:
    """Extract web content as clean text.

    Args:
       raw_contents: List of raw web page contents.

    Returns:
        Sanitized web content.
    """
    sanitized_contents = []
    for raw_content in raw_contents:
        if raw_content and (extracted := trafilatura.extract(raw_content)):
            sanitized_contents.append(extracted)
        else:
            sanitized_contents.append(None)

    return sanitized_contents


async def fetch_sanitized_web_content_from_urls(urls: Sequence[str], client: httpx.AsyncClient) -> list[str | None]:
    """Fetch and sanitize content from URLs.

    Args:
        urls: List of URLs to fetch
        client: HTTP client for making requests

    Returns:
        List of sanitized text content from URLs
    """
    if not urls:
        return []

    LOGGER.info("Fetching and sanitizing content from %d URLs", len(urls))

    # Fetch raw content first
    raw_contents = await fetch_content_from_urls(urls, client)

    return sanitize_web_content(raw_contents)


async def fetch_sanitized_web_content_from_http_urls_in_prompt(prompt: str, client: httpx.AsyncClient) -> str:
    """Extract URLs from prompt and fetch their sanitized content.

    Args:
        prompt: User input containing potential URLs
        client: HTTP client for making requests

    Returns:
        Concatenated sanitized content from all URLs found in prompt
    """
    urls, raw_web_pages = await fetch_from_http_urls_in_prompt(prompt, client)

    sanitized_contents = sanitize_web_content(raw_web_pages)

    if not sanitized_contents:
        return ""

    # Combine all content with URL headers
    combined_content = []
    for url, content in zip(urls, sanitized_contents, strict=True):
        if content:
            combined_content.append(f"Content from {url}:\n{content}")

    return "\n\n".join(combined_content)
