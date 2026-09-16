"""Unit tests for the Agent loop with mocked LLM client."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multitool_agent.agent import Agent
from multitool_agent.config import Settings
from multitool_agent.llm import BaseLLMClient
from multitool_agent.models import ToolCall


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=BaseLLMClient)
    llm.is_configured = True
    llm.provider_name = "groq"
    return llm


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_key",
        groq_model="openai/gpt-oss-120b",
        groq_temperature=1.0,
        groq_max_completion_tokens=2048,
        groq_top_p=1.0,
        groq_reasoning_effort="medium",
        anthropic_api_key="sk-ant-test-key",
        anthropic_model="claude-test-model",
        search_api_key="tvly-test",
        search_provider="tavily",
        max_tool_iterations=5,
        max_file_size=1024 * 1024,
        max_output_chars=8000,
        workspace_dir=tmp_path,
        log_level="DEBUG",
    )


def test_agent_normal_response_no_tools(mock_llm: MagicMock, test_settings: Settings) -> None:
    """Scenario 1: Model immediately responds with text without requesting any tools."""
    mock_llm.create_completion.return_value = ("Hello! How can I help you today?", [])

    agent = Agent(settings=test_settings, llm_client=mock_llm)
    response = agent.run("Hello")

    assert response.text == "Hello! How can I help you today?"
    assert response.iterations == 1
    assert response.finish_reason == "end_turn"
    assert len(response.steps) == 1
    assert len(response.steps[0].tool_calls) == 0


def test_agent_single_tool_call(mock_llm: MagicMock, test_settings: Settings) -> None:
    """Scenario 2: Model requests one tool call (calculator) and then produces final answer."""
    # Turn 1: Model requests calculator
    turn1 = (
        "Let me calculate that.",
        [ToolCall(id="call_calc_1", name="calculator", arguments={"expression": "123 * 456"})],
    )
    # Turn 2: Model provides final answer after receiving result
    turn2 = ("123 multiplied by 456 is 56088.", [])
    mock_llm.create_completion.side_effect = [turn1, turn2]

    agent = Agent(settings=test_settings, llm_client=mock_llm)
    response = agent.run("What is 123 * 456?")

    assert response.iterations == 2
    assert "56088" in response.text
    assert len(response.steps) == 2

    # Step 1 verified tool call and result
    step1 = response.steps[0]
    assert len(step1.tool_calls) == 1
    assert step1.tool_calls[0].name == "calculator"
    assert step1.tool_calls[0].arguments == {"expression": "123 * 456"}
    assert step1.tool_results[0].content == "56088"

    # Step 2 verified final answer
    step2 = response.steps[1]
    assert len(step2.tool_calls) == 0


def test_agent_multiple_sequential_tool_calls(mock_llm: MagicMock, test_settings: Settings) -> None:
    """Scenario 3: Model calls calculator, then saves result with file_write in sequence."""
    turn1 = (
        "",
        [ToolCall(id="call_calc_1", name="calculator", arguments={"expression": "15% of 2000"})],
    )
    turn2 = (
        "",
        [ToolCall(id="call_write_1", name="file_write", arguments={"path": "result.txt", "content": "300"})],
    )
    turn3 = ("I calculated 300 and saved it to result.txt.", [])
    mock_llm.create_completion.side_effect = [turn1, turn2, turn3]

    agent = Agent(settings=test_settings, llm_client=mock_llm)
    response = agent.run("Calculate 15% of 2000 and save to result.txt")

    assert response.iterations == 3
    assert "saved it to result.txt" in response.text
    assert (test_settings.workspace_dir / "result.txt").read_text() == "300"


def test_agent_multiple_tools_in_single_response(mock_llm: MagicMock, test_settings: Settings) -> None:
    """Scenario 4: Model returns multiple tool_use blocks in a single turn."""
    turn1 = (
        "",
        [
            ToolCall(id="call_1", name="calculator", arguments={"expression": "10 * 10"}),
            ToolCall(id="call_2", name="calculator", arguments={"expression": "20 * 20"}),
        ],
    )
    turn2 = ("Results are 100 and 400.", [])
    mock_llm.create_completion.side_effect = [turn1, turn2]

    agent = Agent(settings=test_settings, llm_client=mock_llm)
    response = agent.run("Compute 10*10 and 20*20")

    assert response.iterations == 2
    step1 = response.steps[0]
    assert len(step1.tool_calls) == 2
    assert step1.tool_results[0].content == "100"
    assert step1.tool_results[1].content == "400"


def test_agent_unknown_tool_handling(mock_llm: MagicMock, test_settings: Settings) -> None:
    """Scenario 5: Model requests an unknown tool, agent reports error in tool_result."""
    turn1 = (
        "",
        [ToolCall(id="call_unknown", name="nonexistent_tool", arguments={"foo": "bar"})],
    )
    turn2 = ("Sorry, nonexistent_tool is not available.", [])
    mock_llm.create_completion.side_effect = [turn1, turn2]

    agent = Agent(settings=test_settings, llm_client=mock_llm)
    response = agent.run("Do something nonexistent")

    assert response.iterations == 2
    step1 = response.steps[0]
    assert "is not recognized" in step1.tool_results[0].content
    assert step1.tool_results[0].is_error is True


def test_agent_tool_failure_handling(mock_llm: MagicMock, test_settings: Settings) -> None:
    """Scenario 6: Tool fails (e.g. division by zero), agent reports error cleanly to model."""
    turn1 = (
        "",
        [ToolCall(id="call_div_zero", name="calculator", arguments={"expression": "10 / 0"})],
    )
    turn2 = ("The calculation failed because division by zero is undefined.", [])
    mock_llm.create_completion.side_effect = [turn1, turn2]

    agent = Agent(settings=test_settings, llm_client=mock_llm)
    response = agent.run("What is 10 / 0?")

    assert response.iterations == 2
    step1 = response.steps[0]
    assert "Division by zero" in step1.tool_results[0].content
    assert step1.tool_results[0].is_error is True


def test_agent_max_tool_iterations(mock_llm: MagicMock, test_settings: Settings) -> None:
    """Scenario 7: Agent terminates gracefully when MAX_TOOL_ITERATIONS is exceeded."""
    repeating_call = (
        "",
        [ToolCall(id="infinite_call", name="calculator", arguments={"expression": "1 + 1"})],
    )
    mock_llm.create_completion.return_value = repeating_call

    # max_tool_iterations is 5 in test_settings
    agent = Agent(settings=test_settings, llm_client=mock_llm)
    response = agent.run("Loop forever")

    assert response.iterations == 5
    assert response.finish_reason == "max_iterations"
    assert "maximum allowed tool iterations" in response.text


def test_conversation_history_retention(mock_llm: MagicMock, test_settings: Settings) -> None:
    """Test that conversation history persists across sequential user queries."""
    msg1_call = (
        "",
        [ToolCall(id="c1", name="calculator", arguments={"expression": "25 * 20"})],
    )
    msg1_final = ("500", [])

    msg2_call = (
        "",
        [ToolCall(id="c2", name="calculator", arguments={"expression": "10% of 500"})],
    )
    msg2_final = ("50", [])

    mock_llm.create_completion.side_effect = [msg1_call, msg1_final, msg2_call, msg2_final]

    agent = Agent(settings=test_settings, llm_client=mock_llm)

    res1 = agent.run("What is 25 * 20?")
    assert res1.text == "500"

    res2 = agent.run("Now calculate 10% of that.")
    assert res2.text == "50"

    assert len(agent.conversation_history) > 4
    agent.clear_history()
    assert len(agent.conversation_history) == 0
