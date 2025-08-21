"""Shared fixtures for web search tests."""

import json
import pathlib
import typing
import unittest.mock

import pytest

import chatbot.web.search.brave
import chatbot.web.search.google
import chatbot.web.search.manager


@pytest.fixture(scope="session")
def fixture_data() -> dict[str, typing.Any]:
    """Load test response data from JSON files.

    Returns:
        dict[str, Any]: Loaded test data with 'brave' and 'google' keys.
    """
    fixtures_dir = pathlib.Path(__file__).parent.parent.parent.parent / "fixtures" / "web" / "search"

    with open(fixtures_dir / "brave_responses.json", encoding="utf-8") as f:
        brave_data = json.load(f)

    with open(fixtures_dir / "google_responses.json", encoding="utf-8") as f:
        google_data = json.load(f)

    return {"brave": brave_data, "google": google_data}


@pytest.fixture
def brave_client() -> chatbot.web.search.brave.BraveSearchClient:
    """Create a BraveSearchClient for testing.

    Returns:
        BraveSearchClient: A configured Brave search client for testing.
    """
    return chatbot.web.search.brave.BraveSearchClient(api_key="test-brave-key")


@pytest.fixture
def mock_google_service() -> tuple[unittest.mock.MagicMock, unittest.mock.MagicMock]:
    """Mock the Google API service discovery.

    Returns:
        tuple[MagicMock, MagicMock]: Mock service and CSE objects.
    """
    mock_service = unittest.mock.MagicMock()
    mock_cse = unittest.mock.MagicMock()
    mock_service.cse.return_value = mock_cse
    return mock_service, mock_cse


@pytest.fixture
def google_client(
    mock_google_service: tuple[unittest.mock.MagicMock, unittest.mock.MagicMock],
) -> chatbot.web.search.google.GoogleSearchClient:
    """Create a GoogleSearchClient for testing with mocked service.

    Returns:
        GoogleSearchClient: A configured Google search client for testing.
    """
    client = chatbot.web.search.google.GoogleSearchClient(api_key="test-google-key", search_engine_id="test-engine-id")
    # Replace the service with our mock
    client.service = mock_google_service[0]
    return client


@pytest.fixture
def brave_manager() -> chatbot.web.search.manager.SearchManager:
    """Create a SearchManager with Brave provider for testing.

    Returns:
        SearchManager: A configured search manager with Brave provider.
    """
    return chatbot.web.search.manager.SearchManager(
        provider=chatbot.web.search.manager.SearchEngineProvider.BRAVE,
        api_key="test-brave-key",
        trigger_words=["latest", "recent", "news", "breaking"],
    )


@pytest.fixture
def google_manager() -> chatbot.web.search.manager.SearchManager:
    """Create a SearchManager with Google provider for testing.

    Returns:
        SearchManager: A configured search manager with Google provider.
    """
    return chatbot.web.search.manager.SearchManager(
        provider=chatbot.web.search.manager.SearchEngineProvider.GOOGLE,
        api_key="test-google-key",
        search_engine_id="test-engine-id",
        trigger_words=["search", "find", "lookup"],
    )


# Global pytest-httpx configuration for consistent behavior
pytestmark = pytest.mark.httpx_mock(assert_all_requests_were_expected=False)
