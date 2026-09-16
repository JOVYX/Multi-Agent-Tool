"""Optional integration tests connecting to live LLM APIs.

These tests are excluded by default from standard test runs.
To execute live integration tests:
    pytest -m integration
Ensure GROQ_API_KEY (or ANTHROPIC_API_KEY) is configured in your environment or .env.
"""

from pathlib import Path
import pytest

from multitool_agent.agent import Agent
from multitool_agent.config import load_settings


@pytest.mark.integration
def test_live_groq_calculator(tmp_path: Path) -> None:
    """Live test verifying Groq model calls calculator tool."""
    settings = load_settings(reload=True)
    if not settings.has_groq_key or "your_" in settings.groq_api_key:
        pytest.skip("GROQ_API_KEY is not set or contains placeholder value.")

    agent = Agent(settings=settings)
    response = agent.run("Calculate 17 * 23 using your calculator tool.")

    # 17 * 23 = 391
    assert "391" in response.text
    assert any(
        any(tc.name == "calculator" for tc in step.tool_calls)
        for step in response.steps
    )


@pytest.mark.integration
def test_live_anthropic_calculator(tmp_path: Path) -> None:
    """Live test verifying Anthropic model calls calculator tool."""
    settings = load_settings(reload=True)
    if not settings.has_anthropic_key or "your_" in settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set or contains placeholder value.")

    from multitool_agent.llm import AnthropicClient
    llm = AnthropicClient(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    agent = Agent(settings=settings, llm_client=llm)

    response = agent.run("Calculate 17 * 23 using your calculator tool.")

    assert "391" in response.text
    assert any(
        any(tc.name == "calculator" for tc in step.tool_calls)
        for step in response.steps
    )
