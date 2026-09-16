"""Unit tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

from multitool_agent.config import Settings, _mask_secret, load_settings


def test_settings_defaults(tmp_path: Path) -> None:
    """Test default values when no environment variables are set."""
    with patch.dict(os.environ, {}, clear=True):
        settings = load_settings(reload=True)
        assert settings.llm_provider == "groq"
        assert settings.groq_model == "openai/gpt-oss-120b"
        assert settings.groq_reasoning_effort == "medium"
        assert settings.groq_temperature == 1.0
        assert settings.groq_max_completion_tokens == 2048
        assert settings.groq_top_p == 1.0
        assert settings.search_provider == "tavily"
        assert settings.max_tool_iterations == 10
        assert settings.max_file_size == 1024 * 1024
        assert settings.max_output_chars == 8000
        assert settings.log_level == "INFO"
        assert not settings.has_groq_key
        assert not settings.has_anthropic_key
        assert not settings.has_active_llm_key
        assert not settings.has_search_key


def test_settings_environment_override(tmp_path: Path) -> None:
    """Test that environment variables properly override defaults."""
    env = {
        "GROQ_API_KEY": "gsk_test_groq_key_1234567890abcdef",
        "GROQ_MODEL": "openai/gpt-oss-120b",
        "GROQ_TEMPERATURE": "0.7",
        "GROQ_MAX_COMPLETION_TOKENS": "4096",
        "SEARCH_API_KEY": "tvly-test-search-key-1234",
        "SEARCH_PROVIDER": "brave",
        "MAX_TOOL_ITERATIONS": "5",
        "MAX_FILE_SIZE": "2048",
        "MAX_OUTPUT_CHARS": "4000",
        "WORKSPACE_DIR": str(tmp_path / "custom_workspace"),
        "LOG_LEVEL": "DEBUG",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = load_settings(reload=True)
        assert settings.llm_provider == "groq"
        assert settings.groq_api_key == "gsk_test_groq_key_1234567890abcdef"
        assert settings.groq_model == "openai/gpt-oss-120b"
        assert settings.groq_temperature == 0.7
        assert settings.groq_max_completion_tokens == 4096
        assert settings.search_api_key == "tvly-test-search-key-1234"
        assert settings.search_provider == "brave"
        assert settings.max_tool_iterations == 5
        assert settings.max_file_size == 2048
        assert settings.max_output_chars == 4000
        assert settings.workspace_dir == (tmp_path / "custom_workspace").resolve()
        assert settings.log_level == "DEBUG"
        assert settings.has_groq_key
        assert settings.has_active_llm_key
        assert settings.has_search_key


def test_mask_secret() -> None:
    """Test that secret keys are safely masked."""
    assert _mask_secret("") == "<not set>"
    assert _mask_secret("short") == "********"
    masked = _mask_secret("gsk_1234567890abcdef123456")
    assert masked.startswith("gsk")
    assert masked.endswith("456")
    assert "1234567890" not in masked


def test_masked_summary(tmp_path: Path) -> None:
    """Test that summary dictionary obscures sensitive keys."""
    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_super_secret_groq_key_value",
        groq_model="openai/gpt-oss-120b",
        groq_temperature=1.0,
        groq_max_completion_tokens=2048,
        groq_top_p=1.0,
        groq_reasoning_effort="medium",
        anthropic_api_key="sk-ant-super-secret-anthropic-key",
        anthropic_model="claude-3-5-sonnet",
        search_api_key="tvly-super-secret-tavily-key",
        search_provider="tavily",
        max_tool_iterations=10,
        max_file_size=1024,
        max_output_chars=5000,
        workspace_dir=tmp_path,
        log_level="INFO",
    )
    summary = settings.masked_summary()
    assert "super_secret" not in summary["groq_api_key"]
    assert "super-secret" not in summary["anthropic_api_key"]
    assert "super-secret" not in summary["search_api_key"]
    assert summary["groq_model"] == "openai/gpt-oss-120b"
