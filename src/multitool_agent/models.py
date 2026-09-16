"""Data models and type definitions for the multi-tool agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """Represents a tool call emitted by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Represents the output from executing a tool."""

    tool_use_id: str
    content: str
    is_error: bool = False

    def to_content_block(self) -> dict[str, Any]:
        """Convert to the Anthropic tool_result block format."""
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            block["is_error"] = True
        return block


@dataclass
class AgentStep:
    """Record of a single reasoning step in the agent loop."""

    iteration: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    assistant_text: Optional[str] = None


@dataclass
class AgentResponse:
    """Final output from the agent."""

    text: str
    iterations: int
    steps: list[AgentStep] = field(default_factory=list)
    finish_reason: str = "end_turn"
