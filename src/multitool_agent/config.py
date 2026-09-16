"""Centralized configuration management for the multi-tool agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application configuration settings loaded from environment variables."""

    llm_provider: str
    groq_api_key: str
    groq_model: str
    groq_temperature: float
    groq_max_completion_tokens: int
    groq_top_p: float
    groq_reasoning_effort: str
    anthropic_api_key: str
    anthropic_model: str
    search_api_key: str
    search_provider: str
    max_tool_iterations: int
    max_file_size: int
    max_output_chars: int
    workspace_dir: Path
    log_level: str

    @property
    def has_groq_key(self) -> bool:
        """Return True if a Groq API key is provided and non-empty."""
        return bool(self.groq_api_key.strip())

    @property
    def has_anthropic_key(self) -> bool:
        """Return True if an Anthropic API key is provided and non-empty."""
        return bool(self.anthropic_api_key.strip())

    @property
    def has_active_llm_key(self) -> bool:
        """Return True if the key for the active LLM provider is present."""
        if self.llm_provider == "groq":
            return self.has_groq_key
        return self.has_anthropic_key

    @property
    def has_search_key(self) -> bool:
        """Return True if a Search API key is provided and non-empty."""
        return bool(self.search_api_key.strip())

    def masked_summary(self) -> dict[str, str]:
        """Return a dictionary of settings with secrets masked for safe logging/display."""
        return {
            "llm_provider": self.llm_provider,
            "groq_model": self.groq_model,
            "groq_api_key": _mask_secret(self.groq_api_key),
            "groq_reasoning_effort": self.groq_reasoning_effort,
            "anthropic_model": self.anthropic_model,
            "anthropic_api_key": _mask_secret(self.anthropic_api_key),
            "search_provider": self.search_provider,
            "search_api_key": _mask_secret(self.search_api_key),
            "max_tool_iterations": str(self.max_tool_iterations),
            "max_file_size_bytes": str(self.max_file_size),
            "max_output_chars": str(self.max_output_chars),
            "workspace_dir": str(self.workspace_dir),
            "log_level": self.log_level,
        }


def _mask_secret(val: str) -> str:
    """Mask sensitive string leaving only first 3 and last 3 characters visible."""
    if not val:
        return "<not set>"
    val = val.strip()
    if len(val) <= 8:
        return "********"
    return f"{val[:3]}...{val[-3:]}"


_CACHED_SETTINGS: Optional[Settings] = None


def load_settings(env_path: Optional[Path | str] = None, reload: bool = False) -> Settings:
    """Load configuration settings from environment variables and .env file."""
    global _CACHED_SETTINGS

    if _CACHED_SETTINGS is not None and not reload:
        return _CACHED_SETTINGS

    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=False)

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    try:
        groq_temperature = float(os.getenv("GROQ_TEMPERATURE", "1.0"))
    except ValueError:
        groq_temperature = 1.0

    try:
        groq_max_completion_tokens = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "2048"))
    except ValueError:
        groq_max_completion_tokens = 2048

    try:
        groq_top_p = float(os.getenv("GROQ_TOP_P", "1.0"))
    except ValueError:
        groq_top_p = 1.0

    groq_reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "medium").strip()

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()

    # Default to 'groq' if GROQ_API_KEY is present or if no LLM_PROVIDER is set
    llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not llm_provider:
        if anthropic_api_key and not groq_api_key:
            llm_provider = "anthropic"
        else:
            llm_provider = "groq"

    search_api_key = os.getenv("SEARCH_API_KEY", "").strip()
    search_provider = os.getenv("SEARCH_PROVIDER", "tavily").strip().lower()

    try:
        max_tool_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", "10"))
    except ValueError:
        max_tool_iterations = 10

    try:
        max_file_size = int(os.getenv("MAX_FILE_SIZE", str(1024 * 1024)))
    except ValueError:
        max_file_size = 1024 * 1024

    try:
        max_output_chars = int(os.getenv("MAX_OUTPUT_CHARS", "8000"))
    except ValueError:
        max_output_chars = 8000

    workspace_str = os.getenv("WORKSPACE_DIR", "workspace").strip()
    workspace_path = Path(workspace_str).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level = "INFO"

    settings = Settings(
        llm_provider=llm_provider,
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        groq_temperature=groq_temperature,
        groq_max_completion_tokens=groq_max_completion_tokens,
        groq_top_p=groq_top_p,
        groq_reasoning_effort=groq_reasoning_effort,
        anthropic_api_key=anthropic_api_key,
        anthropic_model=anthropic_model,
        search_api_key=search_api_key,
        search_provider=search_provider,
        max_tool_iterations=max_tool_iterations,
        max_file_size=max_file_size,
        max_output_chars=max_output_chars,
        workspace_dir=workspace_path,
        log_level=log_level,
    )

    _CACHED_SETTINGS = settings
    return settings


def get_settings() -> Settings:
    """Retrieve current settings, loading them if not yet initialized."""
    return load_settings()
