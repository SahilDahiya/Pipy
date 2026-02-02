import base64
import re
from pathlib import Path

import pytest

from pi_ai.types import ImageContent, TextContent
from pi_tools import create_bash_tool, create_edit_tool, create_read_tool, create_write_tool
from pi_tools.bash import BashToolOptions
import pi_tools.shell as shell_module


def get_text_output(result) -> str:
    return "\n".join(
        block.text for block in result.content if isinstance(block, TextContent)
    )


@pytest.mark.asyncio
async def test_read_file_contents_within_limits(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "test.txt"
    content = "Hello, world!\nLine 2\nLine 3"
    test_file.write_text(content, encoding="utf-8")

    result = await read.execute("test-call-1", {"path": str(test_file)})

    assert get_text_output(result) == content
    assert "Use offset=" not in get_text_output(result)
    assert result.details is None


@pytest.mark.asyncio
async def test_read_handles_missing_files(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "missing.txt"

    with pytest.raises(Exception) as excinfo:
        await read.execute("test-call-2", {"path": str(test_file)})
    assert "not found" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_read_truncates_by_lines(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "large.txt"
    lines = [f"Line {idx}" for idx in range(1, 2501)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    result = await read.execute("test-call-3", {"path": str(test_file)})
    output = get_text_output(result)

    assert "Line 1" in output
    assert "Line 2000" in output
    assert "Line 2001" not in output
    assert "[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in output


@pytest.mark.asyncio
async def test_read_truncates_by_bytes(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "large-bytes.txt"
    lines = [f"Line {idx}: {'x' * 200}" for idx in range(1, 501)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    result = await read.execute("test-call-4", {"path": str(test_file)})
    output = get_text_output(result)

    assert "Line 1:" in output
    assert re.search(
        r"\[Showing lines 1-\d+ of 500 \(.* limit\)\. Use offset=\d+ to continue\.\]",
        output,
    )


@pytest.mark.asyncio
async def test_read_offset_parameter(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "offset.txt"
    lines = [f"Line {idx}" for idx in range(1, 101)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    result = await read.execute("test-call-5", {"path": str(test_file), "offset": 51})
    output = get_text_output(result)

    assert "Line 50" not in output
    assert "Line 51" in output
    assert "Line 100" in output
    assert "Use offset=" not in output


@pytest.mark.asyncio
async def test_read_limit_parameter(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "limit.txt"
    lines = [f"Line {idx}" for idx in range(1, 101)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    result = await read.execute("test-call-6", {"path": str(test_file), "limit": 10})
    output = get_text_output(result)

    assert "Line 1" in output
    assert "Line 10" in output
    assert "Line 11" not in output
    assert "[90 more lines in file. Use offset=11 to continue.]" in output


@pytest.mark.asyncio
async def test_read_offset_limit_together(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "offset-limit.txt"
    lines = [f"Line {idx}" for idx in range(1, 101)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    result = await read.execute(
        "test-call-7",
        {"path": str(test_file), "offset": 41, "limit": 20},
    )
    output = get_text_output(result)

    assert "Line 40" not in output
    assert "Line 41" in output
    assert "Line 60" in output
    assert "Line 61" not in output
    assert "[40 more lines in file. Use offset=61 to continue.]" in output


@pytest.mark.asyncio
async def test_read_offset_out_of_bounds(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "short.txt"
    test_file.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        await read.execute("test-call-8", {"path": str(test_file), "offset": 100})
    assert "Offset 100 is beyond end of file (3 lines total)" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_includes_truncation_details(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "large-file.txt"
    lines = [f"Line {idx}" for idx in range(1, 2501)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    result = await read.execute("test-call-9", {"path": str(test_file)})

    assert result.details is not None
    truncation = result.details["truncation"]
    assert truncation["truncated"] is True
    assert truncation["truncated_by"] == "lines"
    assert truncation["total_lines"] == 2500
    assert truncation["output_lines"] == 2000


@pytest.mark.asyncio
async def test_read_detects_image_magic(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+X2Z0AAAAASUVORK5CYII="
    )
    test_file = tmp_path / "image.txt"
    test_file.write_bytes(base64.b64decode(png_base64))

    result = await read.execute("test-call-img-1", {"path": str(test_file)})

    assert result.content[0].type == "text"
    assert "Read image file [image/png]" in get_text_output(result)
    image_block = next(
        (block for block in result.content if isinstance(block, ImageContent)),
        None,
    )
    assert image_block is not None
    assert image_block.mime_type == "image/png"
    assert image_block.data


@pytest.mark.asyncio
async def test_read_treats_non_image_content_as_text(tmp_path: Path) -> None:
    read = create_read_tool(str(tmp_path))
    test_file = tmp_path / "not-an-image.png"
    test_file.write_text("definitely not a png", encoding="utf-8")

    result = await read.execute("test-call-img-2", {"path": str(test_file)})

    assert "definitely not a png" in get_text_output(result)
    assert not any(isinstance(block, ImageContent) for block in result.content)


@pytest.mark.asyncio
async def test_write_tool_writes_content(tmp_path: Path) -> None:
    write = create_write_tool(str(tmp_path))
    test_file = tmp_path / "write-test.txt"
    content = "Test content"

    result = await write.execute("test-call-10", {"path": str(test_file), "content": content})

    assert "Successfully wrote" in get_text_output(result)
    assert str(test_file) in get_text_output(result)


@pytest.mark.asyncio
async def test_write_tool_creates_parent_dirs(tmp_path: Path) -> None:
    write = create_write_tool(str(tmp_path))
    test_file = tmp_path / "nested" / "dir" / "test.txt"
    content = "Nested content"

    result = await write.execute("test-call-11", {"path": str(test_file), "content": content})

    assert "Successfully wrote" in get_text_output(result)
    assert test_file.exists()


@pytest.mark.asyncio
async def test_edit_tool_replaces_text(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-test.txt"
    test_file.write_text("Hello, world!", encoding="utf-8")

    result = await edit.execute(
        "test-call-12",
        {"path": str(test_file), "oldText": "world", "newText": "testing"},
    )

    assert "Successfully replaced" in get_text_output(result)
    assert "diff" in result.details
    assert "testing" in result.details["diff"]


@pytest.mark.asyncio
async def test_edit_tool_fails_when_text_not_found(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-test.txt"
    test_file.write_text("Hello, world!", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        await edit.execute(
            "test-call-13",
            {"path": str(test_file), "oldText": "nonexistent", "newText": "testing"},
        )
    assert "Could not find the exact text" in str(excinfo.value)


@pytest.mark.asyncio
async def test_edit_tool_fails_when_multiple_matches(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-test.txt"
    test_file.write_text("foo foo foo", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        await edit.execute(
            "test-call-14",
            {"path": str(test_file), "oldText": "foo", "newText": "bar"},
        )
    assert "Found 3 occurrences" in str(excinfo.value)


@pytest.mark.asyncio
async def test_bash_executes_commands(tmp_path: Path) -> None:
    bash = create_bash_tool(str(tmp_path))
    result = await bash.execute("test-call-15", {"command": "echo 'test output'"})
    assert "test output" in get_text_output(result)


@pytest.mark.asyncio
async def test_bash_handles_command_errors(tmp_path: Path) -> None:
    bash = create_bash_tool(str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        await bash.execute("test-call-16", {"command": "exit 1"})
    assert "code 1" in str(excinfo.value)


@pytest.mark.asyncio
async def test_bash_respects_timeout(tmp_path: Path) -> None:
    bash = create_bash_tool(str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        await bash.execute("test-call-17", {"command": "sleep 5", "timeout": 1})
    assert "timed out" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_bash_cwd_must_exist(tmp_path: Path) -> None:
    nonexistent = tmp_path / "missing-dir"
    bash = create_bash_tool(str(nonexistent))

    with pytest.raises(RuntimeError) as excinfo:
        await bash.execute("test-call-18", {"command": "echo test"})
    assert "Working directory does not exist" in str(excinfo.value)


@pytest.mark.asyncio
async def test_bash_spawn_errors(tmp_path: Path, monkeypatch) -> None:
    def _bad_shell_config():
        return {"shell": "/nonexistent-shell-path-xyz123", "args": ["-c"]}

    monkeypatch.setattr(shell_module, "get_shell_config", _bad_shell_config)
    bash = create_bash_tool(str(tmp_path))

    with pytest.raises(Exception) as excinfo:
        await bash.execute("test-call-19", {"command": "echo test"})
    assert "no such file" in str(excinfo.value).lower() or "enoent" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_bash_command_prefix(tmp_path: Path) -> None:
    bash = create_bash_tool(str(tmp_path), BashToolOptions(command_prefix="export TEST_VAR=hello"))
    result = await bash.execute("test-prefix-1", {"command": "echo $TEST_VAR"})
    assert get_text_output(result).strip() == "hello"


@pytest.mark.asyncio
async def test_bash_prefix_and_command_output(tmp_path: Path) -> None:
    bash = create_bash_tool(str(tmp_path), BashToolOptions(command_prefix="echo prefix-output"))
    result = await bash.execute("test-prefix-2", {"command": "echo command-output"})
    assert get_text_output(result).strip() == "prefix-output\ncommand-output"


@pytest.mark.asyncio
async def test_bash_without_prefix(tmp_path: Path) -> None:
    bash = create_bash_tool(str(tmp_path), BashToolOptions())
    result = await bash.execute("test-prefix-3", {"command": "echo no-prefix"})
    assert get_text_output(result).strip() == "no-prefix"
