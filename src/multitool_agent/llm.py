"""LLM client implementations for Groq (console.groq.com) and Anthropic."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from multitool_agent.models import ToolCall
from multitool_agent.utils.logging import get_logger

logger = get_logger("multitool_agent.llm")


class LLMClientError(Exception):
    """Base exception for LLM client failures."""


class MissingAPIKeyError(LLMClientError):
    """Raised when an API key is not configured."""


class BaseLLMClient(ABC):
    """Abstract interface for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'groq' or 'anthropic')."""
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if the client has an API key configured."""
        pass

    @abstractmethod
    def create_completion(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[Optional[str], list[ToolCall]]:
        """Send chat messages and tools to the LLM.

        Returns:
            A tuple of (assistant_text_or_none, list_of_tool_calls).
        """
        pass


class GroqClient(BaseLLMClient):
    """Client for Groq's high-speed inference API (console.groq.com)."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "openai/gpt-oss-120b",
        temperature: float = 1.0,
        max_completion_tokens: int = 2048,
        top_p: float = 1.0,
        reasoning_effort: str = "medium",
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort

        if client is not None:
            self._client = client
        elif self.api_key:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        else:
            try:
                from groq import Groq
                # Attempt to initialize using GROQ_API_KEY from environment
                self._client = Groq()
                self.api_key = getattr(self._client, "api_key", "")
            except Exception:
                self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def is_configured(self) -> bool:
        return self._client is not None and bool(self.api_key or getattr(self._client, "api_key", None))

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[Optional[str], list[ToolCall]]:
        """Call Groq chat completions with tool support and optional streaming."""
        if not self.is_configured:
            raise MissingAPIKeyError(
                "Groq API key is not set. Please set GROQ_API_KEY in your environment or .env file "
                "(get a free key at https://console.groq.com)."
            )

        import groq

        # Prepare messages payload, prepending system prompt if not present
        full_messages: list[dict[str, Any]] = []
        if system:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                full_messages.append({"role": "system", "content": system})

        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "top_p": self.top_p,
            "stop": None,
        }

        # reasoning_effort is supported on reasoning models like openai/gpt-oss-120b
        if self.reasoning_effort and "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = self.reasoning_effort

        if tools:
            kwargs["tools"] = tools

        logger.debug(
            f"Calling Groq model '{self.model}' with {len(full_messages)} messages, "
            f"stream={stream}, reasoning_effort='{self.reasoning_effort}'."
        )

        try:
            if stream:
                kwargs["stream"] = True
                completion = self._client.chat.completions.create(**kwargs)
                return self._handle_streaming_response(completion, stream_callback)
            else:
                kwargs["stream"] = False
                response = self._client.chat.completions.create(**kwargs)
                return self._handle_non_streaming_response(response, stream_callback)

        except groq.AuthenticationError as e:
            raise LLMClientError(
                f"Groq Authentication failed (HTTP 401). Check GROQ_API_KEY. Details: {e.message}"
            ) from e
        except groq.RateLimitError as e:
            raise LLMClientError(
                f"Groq API rate limit exceeded (HTTP 429). Details: {e.message}"
            ) from e
        except groq.APIConnectionError as e:
            raise LLMClientError(
                f"Failed to connect to Groq API. Check your network connection. Details: {str(e)}"
            ) from e
        except groq.APIStatusError as e:
            raise LLMClientError(
                f"Groq API returned status {e.status_code}: {e.message}"
            ) from e
        except Exception as e:
            raise LLMClientError(
                f"Unexpected error communicating with Groq API: {type(e).__name__}: {str(e)}"
            ) from e

    def _handle_non_streaming_response(
        self,
        response: Any,
        stream_callback: Optional[Callable[[str], None]],
    ) -> tuple[Optional[str], list[ToolCall]]:
        choice = response.choices[0]
        msg = choice.message
        content_text = getattr(msg, "content", None) or ""

        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in raw_tool_calls:
            tc_id = getattr(tc, "id", "")
            func = getattr(tc, "function", None)
            tc_name = getattr(func, "name", "") if func else ""
            raw_args = getattr(func, "arguments", "{}") if func else "{}"
            parsed_args = self._parse_json_args(raw_args)
            tool_calls.append(ToolCall(id=tc_id, name=tc_name, arguments=parsed_args))

        if stream_callback and content_text and not tool_calls:
            stream_callback(content_text)

        return (content_text.strip() or None, tool_calls)

    def _handle_streaming_response(
        self,
        completion: Any,
        stream_callback: Optional[Callable[[str], None]],
    ) -> tuple[Optional[str], list[ToolCall]]:
        content_parts: list[str] = []
        tool_calls_dict: dict[int, dict[str, str]] = {}

        for chunk in completion:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            delta_content = getattr(delta, "content", None) or ""
            if delta_content:
                content_parts.append(delta_content)
                if stream_callback:
                    stream_callback(delta_content)

            delta_tool_calls = getattr(delta, "tool_calls", None) or []
            for tc_delta in delta_tool_calls:
                idx = getattr(tc_delta, "index", 0)
                if idx not in tool_calls_dict:
                    tool_calls_dict[idx] = {"id": "", "name": "", "arguments": ""}

                tc_id = getattr(tc_delta, "id", None)
                if tc_id:
                    tool_calls_dict[idx]["id"] = tc_id

                func = getattr(tc_delta, "function", None)
                if func:
                    func_name = getattr(func, "name", None)
                    if func_name:
                        tool_calls_dict[idx]["name"] += func_name
                    func_args = getattr(func, "arguments", None)
                    if func_args:
                        tool_calls_dict[idx]["arguments"] += func_args

        final_text = "".join(content_parts).strip() or None
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_calls_dict.keys()):
            tc_info = tool_calls_dict[idx]
            parsed_args = self._parse_json_args(tc_info["arguments"])
            tool_calls.append(
                ToolCall(
                    id=tc_info["id"] or f"call_{idx}",
                    name=tc_info["name"],
                    arguments=parsed_args,
                )
            )

        return (final_text, tool_calls)

    @staticmethod
    def _parse_json_args(raw_str: str) -> dict[str, Any]:
        """Safely parse tool argument JSON string."""
        if not raw_str or not raw_str.strip():
            return {}
        try:
            return json.loads(raw_str)
        except Exception:
            return {"raw_input": raw_str}


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Messages API (Claude)."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-3-5-sonnet-20241022",
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()

        if client is not None:
            self._client = client
        elif self.api_key:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        else:
            self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def is_configured(self) -> bool:
        return self._client is not None and bool(self.api_key)

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[Optional[str], list[ToolCall]]:
        """Call Anthropic messages.create with tool schemas."""
        if not self.is_configured:
            raise MissingAPIKeyError(
                "Anthropic API key is not set. Please set ANTHROPIC_API_KEY in your environment or .env file."
            )

        import anthropic

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.messages.create(**kwargs)
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            for block in response.content:
                block_type = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if block_type == "text":
                    text_parts.append(getattr(block, "text", "") or "")
                elif block_type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=getattr(block, "id", ""),
                            name=getattr(block, "name", ""),
                            arguments=getattr(block, "input", {}) or {},
                        )
                    )

            final_text = "".join(text_parts).strip() or None
            if stream_callback and final_text and not tool_calls:
                stream_callback(final_text)

            return (final_text, tool_calls)

        except anthropic.AuthenticationError as e:
            raise LLMClientError(f"Anthropic Authentication failed: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMClientError(f"Anthropic rate limit exceeded: {e.message}") from e
        except Exception as e:
            raise LLMClientError(f"Anthropic API error: {str(e)}") from e


def create_llm_client(settings: Any) -> BaseLLMClient:
    """Factory creating appropriate LLMClient based on settings."""
    if settings.llm_provider == "anthropic":
        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
    return GroqClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_completion_tokens=settings.groq_max_completion_tokens,
        top_p=settings.groq_top_p,
        reasoning_effort=settings.groq_reasoning_effort,
    )
