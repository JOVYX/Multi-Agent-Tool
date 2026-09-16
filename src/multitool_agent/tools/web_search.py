"""Web search tool supporting configurable search providers (Tavily, Brave, DuckDuckGo)."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from multitool_agent.tools.base import BaseTool
from multitool_agent.utils.logging import get_logger

logger = get_logger("multitool_agent.tools.web_search")

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RESULTS = 5


class WebSearchTool(BaseTool):
    """Tool for querying external web search APIs with structured result formatting."""

    def __init__(
        self,
        api_key: str = "",
        provider: str = "tavily",
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.provider = provider.strip().lower()
        self.max_results = max_results
        self.timeout = timeout
        self._custom_client = http_client

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the public web for real-time, up-to-date information, news, documentation, "
            "and facts. Returns top relevant search results with title, URL, and snippet."
        )

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or keywords to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute web search query against configured provider."""
        if not isinstance(arguments, dict) or "query" not in arguments:
            return "Error: Missing required argument 'query'."

        query = str(arguments["query"]).strip()
        if not query:
            return "Error: Search query cannot be empty."

        logger.debug(f"Web search requested: '{query}' via provider '{self.provider}'")

        if self.provider == "tavily":
            return self._search_tavily(query)
        elif self.provider == "brave":
            return self._search_brave(query)
        elif self.provider == "duckduckgo":
            return self._search_duckduckgo(query)
        else:
            return (
                f"Error: Unsupported search provider '{self.provider}'. "
                "Supported providers are 'tavily', 'brave', and 'duckduckgo'."
            )

    def _get_client(self) -> httpx.Client:
        """Return injected HTTP client or a new standard client."""
        if self._custom_client is not None:
            return self._custom_client
        return httpx.Client(timeout=self.timeout)

    def _search_tavily(self, query: str) -> str:
        """Search via Tavily Search API."""
        if not self.api_key:
            return (
                "Error: Tavily search API key is not configured. "
                "Please set SEARCH_API_KEY in your .env file."
            )

        endpoint = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": "basic",
        }

        try:
            client = self._get_client()
            # If client is injected, don't close it; otherwise close
            should_close = client is not self._custom_client
            try:
                response = client.post(endpoint, json=payload)
            finally:
                if should_close:
                    client.close()

            if response.status_code == 401:
                return "Error: Invalid Tavily API key provided (HTTP 401 Unauthorized)."
            elif response.status_code == 429:
                return "Error: Tavily API rate limit exceeded (HTTP 429 Too Many Requests)."
            elif response.status_code != 200:
                return f"Error: Tavily search request failed with status code {response.status_code}: {response.text}"

            data = response.json()
            raw_results = data.get("results", [])
            if not raw_results:
                return f"No search results found for query: '{query}'."

            items = []
            for res in raw_results[: self.max_results]:
                items.append({
                    "title": res.get("title", "No title"),
                    "url": res.get("url", ""),
                    "snippet": res.get("content", "").strip(),
                })

            return self._format_results(query, items)

        except httpx.TimeoutException:
            return f"Error: Web search request timed out after {self.timeout} seconds."
        except httpx.RequestError as e:
            return f"Error: Network error occurred while searching: {type(e).__name__}: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected Tavily search error: {e}")
            return f"Error executing search: {type(e).__name__}: {str(e)}"

    def _search_brave(self, query: str) -> str:
        """Search via Brave Search API."""
        if not self.api_key:
            return (
                "Error: Brave search API key is not configured. "
                "Please set SEARCH_API_KEY in your .env file."
            )

        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": self.max_results,
        }

        try:
            client = self._get_client()
            should_close = client is not self._custom_client
            try:
                response = client.get(endpoint, headers=headers, params=params)
            finally:
                if should_close:
                    client.close()

            if response.status_code == 401:
                return "Error: Invalid Brave API key provided (HTTP 401 Unauthorized)."
            elif response.status_code == 429:
                return "Error: Brave API rate limit exceeded (HTTP 429 Too Many Requests)."
            elif response.status_code != 200:
                return f"Error: Brave search failed with status code {response.status_code}: {response.text}"

            data = response.json()
            web_results = data.get("web", {}).get("results", [])
            if not web_results:
                return f"No search results found for query: '{query}'."

            items = []
            for res in web_results[: self.max_results]:
                items.append({
                    "title": res.get("title", "No title"),
                    "url": res.get("url", ""),
                    "snippet": res.get("description", "").strip(),
                })

            return self._format_results(query, items)

        except httpx.TimeoutException:
            return f"Error: Web search request timed out after {self.timeout} seconds."
        except httpx.RequestError as e:
            return f"Error: Network error occurred while searching: {type(e).__name__}: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected Brave search error: {e}")
            return f"Error executing search: {type(e).__name__}: {str(e)}"

    def _search_duckduckgo(self, query: str) -> str:
        """Search via DuckDuckGo Instant Answer API (keyless fallback)."""
        endpoint = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }

        try:
            client = self._get_client()
            should_close = client is not self._custom_client
            try:
                response = client.get(endpoint, params=params)
            finally:
                if should_close:
                    client.close()

            if response.status_code != 200:
                return f"Error: DuckDuckGo search failed with status code {response.status_code}."

            data = response.json()
            items = []

            # Check Abstract
            abstract = data.get("AbstractText", "").strip()
            abstract_url = data.get("AbstractURL", "").strip()
            heading = data.get("Heading", query)
            if abstract:
                items.append({"title": heading, "url": abstract_url, "snippet": abstract})

            # Check Related Topics
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict) and "Text" in topic:
                    items.append({
                        "title": topic.get("Text", "")[:60] + "...",
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    })
                if len(items) >= self.max_results:
                    break

            if not items:
                return f"No search results found for query: '{query}'."

            return self._format_results(query, items)

        except httpx.TimeoutException:
            return f"Error: Web search request timed out after {self.timeout} seconds."
        except httpx.RequestError as e:
            return f"Error: Network error occurred while searching: {type(e).__name__}: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected DuckDuckGo search error: {e}")
            return f"Error executing search: {type(e).__name__}: {str(e)}"

    def _format_results(self, query: str, items: list[dict[str, str]]) -> str:
        """Format search results into clean markdown text for the LLM."""
        lines = [f"Search results for '{query}':\n"]
        for idx, item in enumerate(items, 1):
            title = item.get("title", "Untitled")
            url = item.get("url", "N/A")
            snippet = item.get("snippet", "")
            lines.append(f"{idx}. {title}")
            lines.append(f"   URL: {url}")
            lines.append(f"   Snippet: {snippet}")
            lines.append("")

        return "\n".join(lines).strip()
