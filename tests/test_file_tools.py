"""Unit tests for sandboxed file tools."""

from pathlib import Path
import pytest

from multitool_agent.tools.file_tools import FileReadTool, FileWriteTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "test_workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def file_reader(workspace: Path) -> FileReadTool:
    return FileReadTool(workspace_dir=workspace, max_file_size=1024)  # 1 KB limit for tests


@pytest.fixture
def file_writer(workspace: Path) -> FileWriteTool:
    return FileWriteTool(workspace_dir=workspace)


def test_file_schemas(file_reader: FileReadTool, file_writer: FileWriteTool) -> None:
    read_schema = file_reader.schema()
    assert read_schema["name"] == "file_read"
    assert "path" in read_schema["input_schema"]["properties"]

    write_schema = file_writer.schema()
    assert write_schema["name"] == "file_write"
    assert "path" in write_schema["input_schema"]["properties"]
    assert "content" in write_schema["input_schema"]["properties"]


def test_file_write_and_read(file_reader: FileReadTool, file_writer: FileWriteTool) -> None:
    # Write normal file
    write_res = file_writer.execute({"path": "notes.txt", "content": "Hello World!\nLine 2"})
    assert "Successfully wrote" in write_res
    assert "workspace/notes.txt" in write_res

    # Read back file
    read_res = file_reader.execute({"path": "notes.txt"})
    assert read_res == "Hello World!\nLine 2"


def test_file_overwrite(file_reader: FileReadTool, file_writer: FileWriteTool) -> None:
    file_writer.execute({"path": "data.txt", "content": "Initial content"})
    file_writer.execute({"path": "data.txt", "content": "Updated content"})
    read_res = file_reader.execute({"path": "data.txt"})
    assert read_res == "Updated content"


def test_file_write_nested_directories(file_reader: FileReadTool, file_writer: FileWriteTool) -> None:
    write_res = file_writer.execute({
        "path": "sub/deep/folder/result.md",
        "content": "# Nested Result",
    })
    assert "Successfully wrote" in write_res

    read_res = file_reader.execute({"path": "sub/deep/folder/result.md"})
    assert read_res == "# Nested Result"


def test_file_read_missing(file_reader: FileReadTool) -> None:
    res = file_reader.execute({"path": "nonexistent.txt"})
    assert "does not exist in workspace" in res


def test_file_read_directory(workspace: Path, file_reader: FileReadTool) -> None:
    sub = workspace / "test_dir"
    sub.mkdir()
    res = file_reader.execute({"path": "test_dir"})
    assert "is a directory" in res


def test_file_read_oversized(workspace: Path, file_reader: FileReadTool) -> None:
    large_file = workspace / "large.txt"
    # max_file_size is set to 1024 bytes in fixture
    large_file.write_text("x" * 2048)
    res = file_reader.execute({"path": "large.txt"})
    assert "exceeds maximum allowed limit" in res


def test_path_traversal_read_rejection(file_reader: FileReadTool) -> None:
    # Relative escape
    res = file_reader.execute({"path": "../../etc/passwd"})
    assert "Security violation" in res or "resolves outside" in res

    # Absolute escape
    res = file_reader.execute({"path": "/etc/passwd"})
    assert "Security violation" in res or "resolves outside" in res


def test_path_traversal_write_rejection(file_writer: FileWriteTool) -> None:
    res = file_writer.execute({"path": "../escape.txt", "content": "malicious"})
    assert "Security violation" in res or "resolves outside" in res

    res = file_writer.execute({"path": "/tmp/root_escape.txt", "content": "malicious"})
    assert "Security violation" in res or "resolves outside" in res


def test_missing_parameters(file_reader: FileReadTool, file_writer: FileWriteTool) -> None:
    assert "Missing required argument" in file_reader.execute({})
    assert "File path cannot be empty" in file_reader.execute({"path": "  "})
    assert "Missing required argument" in file_writer.execute({"path": "test.txt"})
    assert "Missing required argument" in file_writer.execute({"content": "hello"})
