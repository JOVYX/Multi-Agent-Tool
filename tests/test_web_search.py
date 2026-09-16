"""Unit tests for the web search tool using mocked HTTP requests."""

import httpx
import pytest

from multitool_agent.tools.web_search import WebSearchTool


def test_schema() -> None:
    tool = WebSearchTool(api_key="test-key", provider="tavily")
    schema = tool.schema()
    assert schema["name"] == "web_search"
    assert "query" in schema["input_schema"]["properties"]
    assert "query" in schema["input_schema"]["required"]


def test_missing_api_key_tavily() -> None:
    tool = WebSearchTool(api_key="", provider="tavily")
    res = tool.execute({"query": "Python news"})
    assert "Error: Tavily search API key is not configured" in res


def test_missing_api_key_brave() -> None:
    tool = WebSearchTool(api_key="", provider="brave")
    res = tool.execute({"query": "Python news"})
    assert "Error: Brave search API key is not configured" in res


def test_empty_query() -> None:
    tool = WebSearchTool(api_key="test-key", provider="tavily")
    assert "Missing required argument" in tool.execute({})
    assert "Search query cannot be empty" in tool.execute({"query": "   "})


def test_tavily_successful_search() -> None:
    mock_response = httpx.Response(
        status_code=200,
        json={
            "results": [
                {
                    "title": "Python 3.14 Documentation",
                    "url": "https://docs.python.org/3.14",
                    "content": "Official Python 3.14 documentation and release highlights.",
                },
                {
                    "title": "Python Release Schedule",
                    "url": "https://peps.python.org/pep-0745",
                    "content": "Schedule for the Python 3.14 release cycle.",
                },
            ]
        },
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )

    client = httpx.Client(transport=httpx.MockTransport(lambda req: mock_response))
    tool = WebSearchTool(api_key="tvly-mock-key", provider="tavily", http_client=client)

    result = tool.execute({"query": "Python 3.14 release"})
    assert "Python 3.14 Documentation" in result
    assert "https://docs.python.org/3.14" in result
    assert "Python Release Schedule" in result


def test_brave_successful_search() -> None:
    mock_response = httpx.Response(
        status_code=200,
        json={
            "web": {
                "results": [
                    {
                        "title": "Brave Python Result",
                        "url": "https://python.org",
                        "description": "Welcome to Python.org official website.",
                    }
                ]
            }
        },
        request=httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )

    client = httpx.Client(transport=httpx.MockTransport(lambda req: mock_response))
    tool = WebSearchTool(api_key="brave-mock-key", provider="brave", http_client=client)

    result = tool.execute({"query": "python"})
    assert "Brave Python Result" in result
    assert "https://python.org" in result
    assert "Welcome to Python.org" in result


def test_empty_search_results() -> None:
    mock_response = httpx.Response(
        status_code=200,
        json={"results": []},
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )

    client = httpx.Client(transport=httpx.MockTransport(lambda req: mock_response))
    tool = WebSearchTool(api_key="tvly-mock-key", provider="tavily", http_client=client)

    result = tool.execute({"query": "nonexistentquery123xyz"})
    assert "No search results found" in result


def test_invalid_api_key_401() -> None:
    mock_response = httpx.Response(
        status_code=401,
        text="Unauthorized",
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )

    client = httpx.Client(transport=httpx.MockTransport(lambda req: mock_response))
    tool = WebSearchTool(api_key="invalid-key", provider="tavily", http_client=client)

    result = tool.execute({"query": "test"})
    assert "401 Unauthorized" in result


def test_rate_limit_429() -> None:
    mock_response = httpx.Response(
        status_code=429,
        text="Too Many Requests",
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )

    client = httpx.Client(transport=httpx.MockTransport(lambda req: mock_response))
    tool = WebSearchTool(api_key="tvly-mock-key", provider="tavily", http_client=client)

    result = tool.execute({"query": "test"})
    assert "rate limit exceeded" in result


def test_api_server_error_500() -> None:
    mock_response = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )

    client = httpx.Client(transport=httpx.MockTransport(lambda req: mock_response))
    tool = WebSearchTool(api_key="tvly-mock-key", provider="tavily", http_client=client)

    result = tool.execute({"query": "test"})
    assert "status code 500" in result


def test_timeout_handling() -> None:
    def raise_timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Connection timed out.")

    client = httpx.Client(transport=httpx.MockTransport(raise_timeout))
    tool = WebSearchTool(api_key="tvly-mock-key", provider="tavily", http_client=client)

    result = tool.execute({"query": "test"})
    assert "timed out" in result
