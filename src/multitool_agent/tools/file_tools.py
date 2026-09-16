"""Sandboxed file reading and writing tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from multitool_agent.tools.base import BaseTool
from multitool_agent.utils.logging import get_logger
from multitool_agent.utils.validation import SecurityError, validate_workspace_path

logger = get_logger("multitool_agent.tools.file_tools")


class FileReadTool(BaseTool):
    """Tool for reading text files safely from the workspace directory."""

    def __init__(self, workspace_dir: Path, max_file_size: int = 1024 * 1024) -> None:
        self.workspace_dir = workspace_dir
        self.max_file_size = max_file_size

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read the text content of a file located within the workspace sandbox directory. "
            "Access outside the workspace is strictly forbidden."
        )

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path to the file inside the workspace (e.g., 'notes.txt', 'data/report.md')."
                        ),
                    }
                },
                "required": ["path"],
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        """Read file inside workspace sandbox with size and traversal checks."""
        if not isinstance(arguments, dict) or "path" not in arguments:
            return "Error: Missing required argument 'path'."

        raw_path = str(arguments["path"]).strip()
        if not raw_path:
            return "Error: File path cannot be empty."

        try:
            target_path = validate_workspace_path(raw_path, self.workspace_dir)
        except (SecurityError, ValueError) as err:
            return f"Error: {str(err)}"

        if not target_path.exists():
            return f"Error: File '{raw_path}' does not exist in workspace."

        if target_path.is_dir():
            return f"Error: Path '{raw_path}' is a directory, not a readable file."

        try:
            file_size = target_path.stat().st_size
            if file_size > self.max_file_size:
                size_mb = file_size / (1024 * 1024)
                limit_mb = self.max_file_size / (1024 * 1024)
                return (
                    f"Error: File '{raw_path}' size ({size_mb:.2f} MB) exceeds maximum allowed limit "
                    f"of {limit_mb:.2f} MB."
                )

            # Read text with UTF-8, gracefully replacing non-decodable bytes if any
            with target_path.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            logger.debug(f"Read {len(content)} characters from {target_path}")
            return content

        except PermissionError:
            return f"Error: Permission denied while attempting to read '{raw_path}'."
        except Exception as e:
            logger.exception(f"Unexpected error reading '{raw_path}'")
            return f"Error reading file '{raw_path}': {type(e).__name__}: {str(e)}"


class FileWriteTool(BaseTool):
    """Tool for writing text files safely inside the workspace directory."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write text content to a file inside the workspace sandbox directory. "
            "Creates parent directories inside workspace automatically if needed. "
            "Writing outside the workspace is strictly forbidden."
        )

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path to the destination file inside the workspace "
                            "(e.g., 'result.txt', 'output/summary.txt')."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        """Write content to file inside workspace sandbox with security checks."""
        if not isinstance(arguments, dict):
            return "Error: Invalid arguments format."

        if "path" not in arguments:
            return "Error: Missing required argument 'path'."

        if "content" not in arguments:
            return "Error: Missing required argument 'content'."

        raw_path = str(arguments["path"]).strip()
        content = str(arguments["content"])

        if not raw_path:
            return "Error: File path cannot be empty."

        try:
            target_path = validate_workspace_path(raw_path, self.workspace_dir)
        except (SecurityError, ValueError) as err:
            return f"Error: {str(err)}"

        if target_path.is_dir():
            return f"Error: Target path '{raw_path}' is an existing directory. Cannot overwrite directory."

        try:
            # Create parent directories safely inside workspace
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with target_path.open("w", encoding="utf-8") as f:
                f.write(content)

            try:
                rel_display = target_path.relative_to(self.workspace_dir.resolve())
            except ValueError:
                rel_display = target_path.name

            char_count = len(content)
            logger.debug(f"Wrote {char_count} characters to {target_path}")
            return f"Successfully wrote {char_count} characters to workspace/{rel_display}"

        except PermissionError:
            return f"Error: Permission denied while attempting to write '{raw_path}'."
        except Exception as e:
            logger.exception(f"Unexpected error writing to '{raw_path}'")
            return f"Error writing file '{raw_path}': {type(e).__name__}: {str(e)}"
