"""Multi-tool AI Agent package.

A production-style, modular Python AI agent built with Anthropic function calling.
"""

from multitool_agent.agent import Agent
from multitool_agent.config import Settings, get_settings
from multitool_agent.tools.base import BaseTool, ToolRegistry

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "BaseTool",
    "ToolRegistry",
    "Settings",
    "get_settings",
]
