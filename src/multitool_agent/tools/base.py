"""Base tool interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from multitool_agent.utils.logging import get_logger

logger = get_logger("multitool_agent.tools")


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A detailed description explaining what the tool does and when to use it."""
        pass

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Return the JSON schema conforming to Anthropic's tool definition standard."""
        pass

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with validated arguments and return a string result.

        Args:
            arguments: Dictionary of input parameters parsed from the model.

        Returns:
            String output to be fed back into the conversation context as tool_result.
        """
        pass


class ToolRegistry:
    """Registry that holds available tools and handles schema export and dispatch."""

    def __init__(self, max_output_chars: int = 8000) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.max_output_chars = max_output_chars

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting already registered tool: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return a list of all registered tool instances."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return the list of tool schemas formatted for the Anthropic Messages API."""
        return [tool.schema() for tool in self._tools.values()]

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        """Return the list of tool schemas formatted for Groq / OpenAI function calling."""
        schemas = []
        for tool in self._tools.values():
            base_schema = tool.schema()
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": base_schema.get(
                        "input_schema", {"type": "object", "properties": {}}
                    ),
                },
            })
        return schemas

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch execution to the named tool, safely capturing and formatting any errors.

        Args:
            name: The name of the tool to execute.
            arguments: Dictionary of arguments passed by the model.

        Returns:
            The string result or a descriptive error message.
        """
        tool = self.get(name)
        if tool is None:
            available = ", ".join(self.list_names()) or "none"
            return f"Error: Tool '{name}' is not recognized. Available tools: {available}."

        try:
            result = tool.execute(arguments)
        except Exception as e:
            logger.exception(f"Error while executing tool '{name}'")
            return f"Error executing tool '{name}': {type(e).__name__}: {str(e)}"

        # Output character truncation guard to protect context window
        if len(result) > self.max_output_chars:
            truncated_len = len(result) - self.max_output_chars
            result = (
                result[: self.max_output_chars]
                + f"\n\n[Output truncated: {truncated_len} additional characters omitted to stay within context limit.]"
            )

        return result
