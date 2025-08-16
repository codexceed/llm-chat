"""Web Context Pipeline for independent web content processing."""

import asyncio

import httpx
from streamlit import logger

from chatbot import search, settings
from chatbot.utils import web

LOGGER = logger.get_logger(__name__)


class WebContextPipeline:
    """Manages web context processing independently from RAG system."""

    def __init__(self, search_manager: search.SearchManager | None = None) -> None:
        """Initialize web context pipeline.

        Args:
            search_manager: Optional search manager for web search capabilities
        """
        self._search_manager = search_manager

    async def web_search_and_fetch(self, query: str, client: httpx.AsyncClient, num_results: int = 5) -> str:
        """Perform web search and fetch content from search results.

        Args:
            query: Search query string
            client: HTTP client for making requests
            num_results: Number of search results to fetch content from

        Returns:
            Combined and formatted content from search results and their URLs
        """
        if not self._search_manager:
            LOGGER.warning("No search manager configured, skipping web search")
            return ""

        LOGGER.info("Performing web search for query: %s", query)
        search_results = await self._search_manager.search(query, num_results)

        if not search_results:
            LOGGER.info("No search results found for query: %s", query)
            return ""

        result_urls = [result.url for result in search_results]
        LOGGER.info("Found %d search result URLs to fetch", len(result_urls))

        # Compile search result string context
        url_contents = await web.fetch_sanitized_web_content_from_urls(result_urls, client)
        search_context_parts = []
        search_summary = "Search Results:\n"
        for i, result in enumerate(search_results, 1):
            search_summary += f"{i}. {result.title}\n   {result.snippet}\n   Source: {result.url}\n\n"
        search_context_parts.append(search_summary)

        # Fetch URL content from search results
        if url_contents:
            content_section = "Detailed Content from Search Results:\n\n"
            for url, content in zip(result_urls, url_contents, strict=True):
                if content:
                    # Truncate very long content based on settings
                    max_len = settings.CHATBOT_SETTINGS.search.result_text_max_len
                    truncated_content = content[:max_len] + "..." if len(content) > max_len else content
                    content_section += f"Content from {url}:\n{truncated_content}\n\n"
            search_context_parts.append(content_section)

        return "\n".join(search_context_parts)

    async def gather_web_context(
        self, prompt: str, client: httpx.AsyncClient, enable_search: bool = True, search_num_results: int = 5
    ) -> dict[str, str]:
        """Gather all web context asynchronously.

        Args:
            prompt: User query/prompt
            client: HTTP client for making requests
            enable_search: Whether to perform web search
            search_num_results: Number of search results to fetch

        Returns:
            Dictionary containing different types of web context
        """
        LOGGER.info("Gathering web context for prompt")

        # Prepare async tasks
        tasks = []
        task_names = []

        # Always fetch URLs from prompt
        tasks.append(web.fetch_sanitized_web_content_from_http_urls_in_prompt(prompt, client))
        task_names.append("prompt_urls")

        # Optionally perform web search
        if enable_search and self._search_manager:
            # Check if search should be triggered based on query
            if self._search_manager.should_trigger_search(prompt):
                tasks.append(self.web_search_and_fetch(prompt, client, search_num_results))
                task_names.append("web_search")
            else:
                LOGGER.info("Search not triggered for this query")

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Process results
        context = {}
        for name, result in zip(task_names, results, strict=True):
            if isinstance(result, str):
                context[name] = result
            else:
                LOGGER.error("Error in %s: %s", name, result)
                context[name] = ""

        return context

    def merge_context(self, web_context: dict[str, str], rag_context: str = "") -> str:
        """Merge different types of context.

        Args:
            web_context: Dictionary of web context from different sources
            rag_context: RAG context from vector database

        Returns:
            Merged context string
        """
        LOGGER.info("Merging context")

        context_parts = []

        # Add web search context
        if web_context.get("web_search"):
            context_parts.append("=== Web Search Context ===\n" + web_context["web_search"])

        # Add URL context from prompt
        if web_context.get("prompt_urls"):
            context_parts.append("=== URLs from Prompt ===\n" + web_context["prompt_urls"])

        # Add RAG context
        if rag_context.strip():
            context_parts.append("=== Document Context ===\n" + rag_context)

        if not context_parts:
            return ""

        # Combine all context sections
        merged_context = "\n\n".join(context_parts)

        LOGGER.info("Merged context length: %d characters", len(merged_context))
        return merged_context
