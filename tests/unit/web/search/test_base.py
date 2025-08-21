"""Tests for base search classes and data structures."""

import hypothesis
import hypothesis.strategies as st

import chatbot.web.search.base

URL_REGEX = r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^\s]*)?"
DOMAIN_REGEX = r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"


@hypothesis.given(
    url=st.from_regex(URL_REGEX),
    title=st.text(min_size=1, max_size=50),
    snippet=st.text(min_size=0, max_size=100),
    rank=st.integers(min_value=1, max_value=100),
    domain=st.one_of(
        st.just(""),  # Empty domain - should extract from URL
        st.from_regex(DOMAIN_REGEX),  # Custom domain - should preserve
    ),
)
def test_initialize_search_result_success(url: str, title: str, snippet: str, rank: int, domain: str) -> None:
    """Test domain extraction: extracts from URL when empty, preserves when provided."""
    result = chatbot.web.search.base.SearchResult(
        title=title,
        url=url,
        snippet=snippet,
        rank=rank,
        domain=domain,
    )

    if domain == "":
        # When domain is empty, should extract from URL
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        expected_domain = parsed.netloc
        assert result.domain == expected_domain
    else:
        # When domain is provided, should preserve it
        assert result.domain == domain
