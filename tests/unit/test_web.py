from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from hypothesis import given, strategies as st

from chatbot.utils.web import URL_REGEX, extract_urls_from_text, fetch_content_from_urls, lookup_http_urls_in_prompt


def test_extract_single_http_url() -> None:
    """Test extracting a single HTTP URL from text."""
    text = "Check out this website: http://example.com"
    urls = extract_urls_from_text(text)
    assert urls == ["http://example.com"]


def test_extract_single_https_url() -> None:
    """Test extracting a single HTTPS URL from text."""
    text = "Visit https://secure.example.com for more info"
    urls = extract_urls_from_text(text)
    assert urls == ["https://secure.example.com"]


def test_extract_multiple_urls() -> None:
    """Test extracting multiple URLs from text."""
    text = "Visit http://example.com and https://test.org for details"
    urls = extract_urls_from_text(text)
    assert set(urls) == {"http://example.com", "https://test.org"}


def test_extract_urls_with_paths() -> None:
    """Test extracting URLs with paths and query parameters."""
    text = "API endpoint: https://api.example.com/v1/users?page=1&limit=10"
    urls = extract_urls_from_text(text)
    assert urls == ["https://api.example.com/v1/users?page=1&limit=10"]


def test_extract_urls_with_fragments() -> None:
    """Test extracting URLs with fragments."""
    text = "Read more at https://example.com/docs#section1"
    urls = extract_urls_from_text(text)
    assert urls == ["https://example.com/docs#section1"]


def test_extract_no_urls() -> None:
    """Test text with no URLs returns empty list."""
    text = "This text has no URLs in it at all"
    urls = extract_urls_from_text(text)
    assert urls == []


def test_extract_duplicate_urls() -> None:
    """Test that duplicate URLs are deduplicated."""
    text = "Visit http://example.com and also http://example.com again"
    urls = extract_urls_from_text(text)
    assert urls == ["http://example.com"]


def test_extract_urls_case_insensitive() -> None:
    """Test URL extraction is case insensitive."""
    text = "HTTP://EXAMPLE.COM and https://Example.Com"
    urls = extract_urls_from_text(text)
    # URLs should be preserved as-is, but regex should match
    assert len(urls) == 2
    assert "HTTP://EXAMPLE.COM" in urls
    assert "https://Example.Com" in urls


@given(text=st.text())
def test_extract_urls_no_crashes_on_arbitrary_text(text: str) -> None:
    """Test that URL extraction doesn't crash on arbitrary text."""
    urls = extract_urls_from_text(text)
    assert isinstance(urls, list)
    # All extracted items should be strings
    assert all(isinstance(url, str) for url in urls)


def test_url_regex_pattern() -> None:
    """Test the URL regex pattern directly."""
    # Valid URLs and their expected captures
    test_cases = [
        ("http://example.com", "http://example.com"),
        ("https://example.com", "https://example.com"),
        ("http://sub.example.com", "http://sub.example.com"),
        ("https://example.com:8080", "https://example.com:8080"),
        ("http://example.com/path", "http://example.com/path"),
        ("https://example.com/path/to/resource", "https://example.com/path/to/resource"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("https://127.0.0.1:8080/api/v1", "https://127.0.0.1:8080/api/v1"),
    ]

    for url, expected in test_cases:
        match = URL_REGEX.search(url)
        assert match is not None, f"URL should match: {url}"
        assert match.group() == expected, f"Expected {expected}, got {match.group()}"

    # Test that URLs with query params and fragments are at least partially matched
    complex_urls = [
        "http://example.com?query=value",
        "https://example.com/path?query=value&other=123",
        "http://example.com#fragment",
        "https://example.com/path#fragment",
    ]

    for url in complex_urls:
        match = URL_REGEX.search(url)
        assert match is not None, f"URL should match: {url}"
        # Just verify that some part of the URL is captured
        captured = match.group()
        assert captured.startswith(("http://", "https://")), f"Should capture URL prefix: {captured}"


def test_extract_urls_with_special_characters() -> None:
    """Test URL extraction with special characters in paths."""
    text = "Download from https://example.com/files/document_v2.pdf"
    urls = extract_urls_from_text(text)
    assert urls == ["https://example.com/files/document_v2.pdf"]


@pytest.mark.asyncio
async def test_fetch_single_url_success() -> None:
    """Test successful fetch of content from a single URL."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.text = "Test content"
    mock_response.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_response

    content = await fetch_content_from_urls(["http://example.com"], mock_client)

    assert content == ["Test content"]
    mock_client.get.assert_called_once_with("http://example.com", follow_redirects=True)


@pytest.mark.asyncio
async def test_fetch_multiple_urls_success() -> None:
    """Test successful fetch of content from multiple URLs."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # Mock responses for different URLs
    responses = []
    for _i, content_text in enumerate(["Content 1", "Content 2"]):
        mock_response = MagicMock()
        mock_response.text = content_text
        mock_response.raise_for_status = MagicMock()
        responses.append(mock_response)

    mock_client.get.side_effect = responses

    urls = ["http://example1.com", "http://example2.com"]
    content = await fetch_content_from_urls(urls, mock_client)

    assert content == ["Content 1", "Content 2"]
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_fetch_url_http_error() -> None:
    """Test handling of HTTP errors during fetch."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found", request=MagicMock(), response=MagicMock()
    )
    mock_client.get.return_value = mock_response

    content = await fetch_content_from_urls(["http://notfound.com"], mock_client)

    # Should return empty list when all URLs fail
    assert content == []


@pytest.mark.asyncio
async def test_fetch_mixed_success_failure() -> None:
    """Test fetch with mix of successful and failed URLs."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # First URL succeeds
    success_response = MagicMock()
    success_response.text = "Success content"
    success_response.raise_for_status = MagicMock()

    # Second URL fails
    error_response = MagicMock()
    error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Server Error", request=MagicMock(), response=MagicMock()
    )

    mock_client.get.side_effect = [success_response, error_response]

    urls = ["http://success.com", "http://error.com"]
    content = await fetch_content_from_urls(urls, mock_client)

    # Should only return content from successful URLs
    assert content == ["Success content"]


@pytest.mark.asyncio
async def test_fetch_empty_urls_list() -> None:
    """Test fetch with empty URLs list."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    content = await fetch_content_from_urls([], mock_client)

    assert content == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_single_url() -> None:
    """Test lookup of a single URL in prompt."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.text = "Web content"
    mock_response.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_response

    prompt = "Please analyze http://example.com"
    urls, content = await lookup_http_urls_in_prompt(prompt, mock_client)

    assert urls == ["http://example.com"]
    assert content == ["Web content"]


@pytest.mark.asyncio
async def test_lookup_multiple_urls() -> None:
    """Test lookup of multiple URLs in prompt."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    responses = []
    for _i, content_text in enumerate(["Content 1", "Content 2"]):
        mock_response = MagicMock()
        mock_response.text = content_text
        mock_response.raise_for_status = MagicMock()
        responses.append(mock_response)

    mock_client.get.side_effect = responses

    prompt = "Compare http://site1.com and https://site2.com"
    urls, content = await lookup_http_urls_in_prompt(prompt, mock_client)

    assert set(urls) == {"http://site1.com", "https://site2.com"}
    assert set(content) == {"Content 1", "Content 2"}


@pytest.mark.asyncio
async def test_lookup_no_urls() -> None:
    """Test lookup when prompt contains no URLs."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    prompt = "This prompt has no URLs"
    urls, content = await lookup_http_urls_in_prompt(prompt, mock_client)

    assert urls == []
    assert content == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_with_failed_fetches() -> None:
    """Test lookup when some URL fetches fail."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found", request=MagicMock(), response=MagicMock()
    )
    mock_client.get.return_value = mock_response

    prompt = "Check http://broken.com"
    urls, content = await lookup_http_urls_in_prompt(prompt, mock_client)

    assert urls == ["http://broken.com"]
    assert content == []  # Failed fetches result in empty content


@given(prompt=st.text())
@pytest.mark.asyncio
async def test_lookup_no_crashes_on_arbitrary_prompts(prompt: str) -> None:
    """Test that URL lookup doesn't crash on arbitrary prompts."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    urls, content = await lookup_http_urls_in_prompt(prompt, mock_client)

    assert isinstance(urls, list)
    assert isinstance(content, list)
    assert len(content) <= len(urls)  # Content can be less due to failed fetches
