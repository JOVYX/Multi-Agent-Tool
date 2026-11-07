"""Tests for the Multi-Tool AI Agent FastAPI service."""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from multitool_agent.api import app
from multitool_agent.models import AgentResponse, AgentStep, ToolCall, ToolResult

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test health check returns healthy status and active provider."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "provider" in data
    assert "model" in data
    assert data["tools_count"] == 4


def test_list_tools_endpoint() -> None:
    """Test that all registered tools and their schemas are enumerated."""
    response = client.get("/api/v1/tools")
    assert response.status_code == 200
    tools = response.json()
    assert len(tools) == 4
    tool_names = {t["name"] for t in tools}
    assert "calculator" in tool_names
    assert "web_search" in tool_names
    assert "file_read" in tool_names
    assert "file_write" in tool_names


@patch("multitool_agent.agent.Agent.run")
def test_chat_endpoint(mock_run: MagicMock) -> None:
    """Test non-streaming chat execution returns structured reasoning steps."""
    mock_run.return_value = AgentResponse(
        text="The calculation result is 42.",
        iterations=1,
        steps=[
            AgentStep(
                iteration=1,
                tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"expression": "6 * 7"})],
                tool_results=[ToolResult(tool_use_id="call_1", content="42", is_error=False)],
                assistant_text="Calculating 6 * 7...",
            )
        ],
        finish_reason="end_turn",
    )

    payload = {"message": "What is 6 * 7?", "session_id": "test-session-123"}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-123"
    assert data["response"] == "The calculation result is 42."
    assert data["iterations"] == 1
    assert len(data["steps"]) == 1
    assert data["steps"][0]["tool_calls"][0]["name"] == "calculator"


def test_root_serves_html() -> None:
    """Test that root endpoint serves the interactive web client HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Multi-Tool AI Agent" in response.text


def test_clear_session_not_found() -> None:
    """Test clearing a non-existent session returns 404."""
    response = client.delete("/api/v1/sessions/non-existent-session-id")
    assert response.status_code == 404

