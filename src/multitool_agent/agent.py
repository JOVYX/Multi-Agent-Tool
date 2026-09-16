"""Core Agent implementation featuring the iterative reasoning and tool execution loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from multitool_agent.config import Settings, get_settings
from multitool_agent.llm import BaseLLMClient, create_llm_client
from multitool_agent.models import AgentResponse, AgentStep, ToolCall, ToolResult
from multitool_agent.tools.base import ToolRegistry
from multitool_agent.tools.calculator import CalculatorTool
from multitool_agent.tools.file_tools import FileReadTool, FileWriteTool
from multitool_agent.tools.web_search import WebSearchTool
from multitool_agent.utils.logging import get_logger, sanitize_dict_for_logging

logger = get_logger("multitool_agent.agent")

DEFAULT_SYSTEM_PROMPT = """You are a capable, honest, and tool-using AI assistant.

Follow these strict operational guidelines:
1. Tool Selection: Use tools whenever they provide necessary information or execute actions needed to fulfill the user's request.
2. Arithmetic & Calculations: ALWAYS delegate mathematical calculations to the `calculator` tool rather than doing arithmetic mentally.
3. Current Facts & Information: ALWAYS use the `web_search` tool for real-time information, recent events, release dates, or current facts.
4. Filesystem Boundaries: Use `file_read` and `file_write` for file operations. NEVER attempt to access files outside the designated workspace sandbox.
5. Honesty & Integrity: NEVER pretend a tool was executed. NEVER fabricate, hallucinate, or assume tool results.
6. Error Handling: If a tool returns an error, explain the issue honestly and decide if an alternative query/expression or a clarifying response is appropriate.
7. Multi-step Reasoning: After receiving tool results, evaluate whether additional tool calls are required to complete the task.
8. Termination: Stop calling tools once you have sufficient information to answer the user's question with a concise, helpful final answer.
"""


class Agent:
    """Multi-tool AI Agent orchestrating LLM tool calling, execution, and conversation state."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_client: Optional[BaseLLMClient] = None,
        registry: Optional[ToolRegistry] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        on_tool_start: Optional[Callable[[ToolCall], None]] = None,
        on_tool_end: Optional[Callable[[ToolResult], None]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm_client or create_llm_client(self.settings)
        self.system_prompt = system_prompt
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end

        # Initialize registry and register standard tools
        if registry is not None:
            self.registry = registry
        else:
            self.registry = ToolRegistry(max_output_chars=self.settings.max_output_chars)
            self._register_default_tools()

        # Conversation history holds API message dictionaries
        self.conversation_history: list[dict[str, Any]] = []

    def _register_default_tools(self) -> None:
        """Register the 4 standard tools with configured settings."""
        self.registry.register(CalculatorTool())
        self.registry.register(
            WebSearchTool(
                api_key=self.settings.search_api_key,
                provider=self.settings.search_provider,
            )
        )
        self.registry.register(
            FileReadTool(
                workspace_dir=self.settings.workspace_dir,
                max_file_size=self.settings.max_file_size,
            )
        )
        self.registry.register(
            FileWriteTool(
                workspace_dir=self.settings.workspace_dir,
            )
        )

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self.conversation_history.clear()
        logger.debug("Conversation history cleared.")

    def run(
        self,
        user_prompt: str,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> AgentResponse:
        """Process a user prompt through the multi-step agent reasoning loop.

        Args:
            user_prompt: User's textual input.
            stream: Whether to stream tokens during final generation.
            stream_callback: Optional callback for streamed text chunks.

        Returns:
            AgentResponse containing the final text, iteration count, and execution steps.
        """
        clean_input = user_prompt.strip()
        if not clean_input:
            return AgentResponse(
                text="Please provide a non-empty question or instruction.",
                iterations=0,
            )

        logger.info(f"User request: {clean_input}")

        # Append new user message to session history
        self.conversation_history.append({"role": "user", "content": clean_input})

        steps: list[AgentStep] = []
        iteration = 0

        # Choose schema format based on provider
        if self.llm.provider_name == "anthropic":
            tool_schemas = self.registry.get_schemas()
        else:
            tool_schemas = self.registry.get_openai_schemas()

        while iteration < self.settings.max_tool_iterations:
            iteration += 1
            logger.debug(f"Starting agent iteration {iteration}/{self.settings.max_tool_iterations}")

            assistant_text, tool_calls = self.llm.create_completion(
                messages=self.conversation_history,
                tools=tool_schemas,
                system=self.system_prompt,
                stream=stream,
                stream_callback=stream_callback,
            )

            # Record assistant turn in history
            if self.llm.provider_name == "anthropic":
                # For Anthropic native format
                content_blocks: list[dict[str, Any]] = []
                if assistant_text:
                    content_blocks.append({"type": "text", "text": assistant_text})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
            else:
                # For Groq / OpenAI format
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_text or "",
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                self.conversation_history.append(assistant_msg)

            # If no tool calls were requested, the model has delivered its final answer
            if not tool_calls:
                logger.info(f"Final answer reached after {iteration} iteration(s).")
                step = AgentStep(
                    iteration=iteration,
                    tool_calls=[],
                    tool_results=[],
                    assistant_text=assistant_text,
                )
                steps.append(step)
                return AgentResponse(
                    text=assistant_text or "(No response text generated.)",
                    iterations=iteration,
                    steps=steps,
                    finish_reason="end_turn",
                )

            # Execute tool calls
            tool_results: list[ToolResult] = []
            for tc in tool_calls:
                logger.debug(f"Model requested tool: {tc.name} with ID {tc.id}")
                logger.debug(f"Arguments: {sanitize_dict_for_logging(tc.arguments)}")

                if self.on_tool_start:
                    try:
                        self.on_tool_start(tc)
                    except Exception as err:
                        logger.warning(f"Error in on_tool_start callback: {err}")

                result_str = self.registry.execute(tc.name, tc.arguments)
                is_error = result_str.startswith("Error")

                logger.debug(
                    f"Tool result for {tc.name} ({'error' if is_error else 'success'}): {result_str[:200]}"
                )

                tool_result = ToolResult(
                    tool_use_id=tc.id,
                    content=result_str,
                    is_error=is_error,
                )
                tool_results.append(tool_result)

                if self.on_tool_end:
                    try:
                        self.on_tool_end(tool_result)
                    except Exception as err:
                        logger.warning(f"Error in on_tool_end callback: {err}")

            step = AgentStep(
                iteration=iteration,
                tool_calls=tool_calls,
                tool_results=tool_results,
                assistant_text=assistant_text,
            )
            steps.append(step)

            # Append tool results back to conversation history
            if self.llm.provider_name == "anthropic":
                user_tool_results = [tr.to_content_block() for tr in tool_results]
                self.conversation_history.append({
                    "role": "user",
                    "content": user_tool_results,
                })
            else:
                for tc, tr in zip(tool_calls, tool_results):
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": tr.content,
                    })

        # If loop exited because maximum iterations were exhausted:
        limit_msg = (
            f"Notice: The agent reached the maximum allowed tool iterations ({self.settings.max_tool_iterations}) "
            "without finalizing an answer."
        )
        logger.warning(limit_msg)
        return AgentResponse(
            text=limit_msg,
            iterations=iteration,
            steps=steps,
            finish_reason="max_iterations",
        )
