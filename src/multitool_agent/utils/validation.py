"""Validation utilities for security boundaries and arguments."""

from __future__ import annotations

from pathlib import Path


class SecurityError(Exception):
    """Raised when a security boundary violation occurs."""


def validate_workspace_path(path_str: str, workspace_dir: Path) -> Path:
    """Validate that a requested path strictly resides within the workspace directory.

    Protects against:
    - Path traversal attacks (e.g. '../../secret.txt')
    - Absolute paths targeting outside workspace (e.g. '/etc/passwd')
    - Symlink escapes targeting outside workspace

    Args:
        path_str: Target file or directory path string.
        workspace_dir: Absolute or relative Path of the allowed workspace sandbox.

    Returns:
        The safely resolved Path object inside workspace_dir.

    Raises:
        SecurityError: If the target path resolves outside the workspace sandbox.
        ValueError: If path_str is empty.
    """
    clean_str = path_str.strip()
    if not clean_str:
        raise ValueError("Path argument cannot be empty.")

    resolved_workspace = workspace_dir.resolve()
    target_path = Path(clean_str)

    # If relative, anchor to workspace
    if not target_path.is_absolute():
        candidate = (resolved_workspace / target_path).resolve()
    else:
        candidate = target_path.resolve()

    # Check that candidate is inside or equal to resolved_workspace
    try:
        candidate.relative_to(resolved_workspace)
    except ValueError as err:
        raise SecurityError(
            f"Security violation: Path '{path_str}' resolves outside the allowed workspace "
            f"sandbox '{resolved_workspace}'."
        ) from err

    return candidate
