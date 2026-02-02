import asyncio
import base64

import pytest

from pi_tools import create_bash_tool, create_edit_tool, create_read_tool, create_write_tool
from pi_ai.types import ImageContent, TextContent


@pytest.mark.asyncio
async def test_write_read_edit(tmp_path):
    cwd = str(tmp_path)
    write = create_write_tool(cwd)
    read = create_read_tool(cwd)
    edit = create_edit_tool(cwd)

    await write.execute("1", {"path": "note.txt", "content": "hello"})
    result = await read.execute("2", {"path": "note.txt"})
    assert any(
        isinstance(block, TextContent) and "hello" in block.text for block in result.content
    )

    await edit.execute("3", {"path": "note.txt", "oldText": "hello", "newText": "hi"})
    updated = await read.execute("4", {"path": "note.txt"})
    assert any(
        isinstance(block, TextContent) and "hi" in block.text for block in updated.content
    )


@pytest.mark.asyncio
async def test_read_offset_limit(tmp_path):
    cwd = str(tmp_path)
    path = tmp_path / "lines.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    read = create_read_tool(cwd)

    result = await read.execute("1", {"path": "lines.txt", "offset": 2, "limit": 2})
    assert any(isinstance(block, TextContent) and "b\nc" in block.text for block in result.content)


@pytest.mark.asyncio
async def test_read_image(tmp_path):
    cwd = str(tmp_path)
    read = create_read_tool(cwd)

    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO7m2NQAAAAASUVORK5CYII="
    )
    img_path = tmp_path / "pixel.png"
    img_path.write_bytes(base64.b64decode(png_base64))

    result = await read.execute("1", {"path": "pixel.png"})
    assert any(isinstance(block, ImageContent) for block in result.content)


@pytest.mark.asyncio
async def test_bash_tool(tmp_path):
    cwd = str(tmp_path)
    bash = create_bash_tool(cwd)

    result = await bash.execute("1", {"command": "printf \"hello\""})
    assert any(
        isinstance(block, TextContent) and "hello" in block.text for block in result.content
    )
