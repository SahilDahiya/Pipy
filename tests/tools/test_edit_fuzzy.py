from pathlib import Path

import pytest

from pi_ai.types import TextContent
from pi_tools import create_edit_tool


def get_text_output(result) -> str:
    return "\n".join(
        block.text for block in result.content if isinstance(block, TextContent)
    )


@pytest.mark.asyncio
async def test_edit_matches_trailing_whitespace(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "trailing-ws.txt"
    test_file.write_text("line one   \nline two  \nline three\n", encoding="utf-8")

    result = await edit.execute(
        "test-fuzzy-1",
        {"path": str(test_file), "old_text": "line one\nline two\n", "new_text": "replaced\n"},
    )

    assert "Successfully replaced" in get_text_output(result)
    assert test_file.read_text(encoding="utf-8") == "replaced\nline three\n"


@pytest.mark.asyncio
async def test_edit_matches_smart_single_quotes(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "smart-quotes.txt"
    test_file.write_text("console.log(\u2018hello\u2019);\n", encoding="utf-8")

    result = await edit.execute(
        "test-fuzzy-2",
        {"path": str(test_file), "old_text": "console.log('hello');", "new_text": "console.log('world');"},
    )

    assert "Successfully replaced" in get_text_output(result)
    assert "world" in test_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_edit_matches_smart_double_quotes(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "smart-double-quotes.txt"
    test_file.write_text("const msg = \u201CHello World\u201D;\n", encoding="utf-8")

    result = await edit.execute(
        "test-fuzzy-3",
        {"path": str(test_file), "old_text": 'const msg = "Hello World";', "new_text": 'const msg = "Goodbye";'},
    )

    assert "Successfully replaced" in get_text_output(result)
    assert "Goodbye" in test_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_edit_matches_unicode_dashes(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "unicode-dashes.txt"
    test_file.write_text("range: 1\u20135\nbreak\u2014here\n", encoding="utf-8")

    result = await edit.execute(
        "test-fuzzy-4",
        {
            "path": str(test_file),
            "old_text": "range: 1-5\nbreak-here",
            "new_text": "range: 10-50\nbreak--here",
        },
    )

    assert "Successfully replaced" in get_text_output(result)
    content = test_file.read_text(encoding="utf-8")
    assert "10-50" in content


@pytest.mark.asyncio
async def test_edit_matches_nbsp(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "nbsp.txt"
    test_file.write_text("hello\u00A0world\n", encoding="utf-8")

    result = await edit.execute(
        "test-fuzzy-5",
        {"path": str(test_file), "old_text": "hello world", "new_text": "hello universe"},
    )

    assert "Successfully replaced" in get_text_output(result)
    assert "universe" in test_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_edit_prefers_exact_match(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "exact-preferred.txt"
    test_file.write_text("const x = 'exact';\nconst y = 'other';\n", encoding="utf-8")

    result = await edit.execute(
        "test-fuzzy-6",
        {"path": str(test_file), "old_text": "const x = 'exact';", "new_text": "const x = 'changed';"},
    )

    assert "Successfully replaced" in get_text_output(result)
    assert test_file.read_text(encoding="utf-8") == "const x = 'changed';\nconst y = 'other';\n"


@pytest.mark.asyncio
async def test_edit_fails_when_no_match(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "no-match.txt"
    test_file.write_text("completely different content\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        await edit.execute(
            "test-fuzzy-7",
            {"path": str(test_file), "old_text": "this does not exist", "new_text": "replacement"},
        )
    assert "Could not find the exact text" in str(excinfo.value)


@pytest.mark.asyncio
async def test_edit_detects_fuzzy_duplicates(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "fuzzy-dups.txt"
    test_file.write_text("hello world   \nhello world\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        await edit.execute(
            "test-fuzzy-8",
            {"path": str(test_file), "old_text": "hello world", "new_text": "replaced"},
        )
    assert "Found 2 occurrences" in str(excinfo.value)


@pytest.mark.asyncio
async def test_edit_matches_lf_text_against_crlf(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "crlf-test.txt"
    test_file.write_text("line one\r\nline two\r\nline three\r\n", encoding="utf-8")

    result = await edit.execute(
        "test-crlf-1",
        {"path": str(test_file), "old_text": "line two\n", "new_text": "replaced line\n"},
    )

    assert "Successfully replaced" in get_text_output(result)


@pytest.mark.asyncio
async def test_edit_preserves_crlf_line_endings(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "crlf-preserve.txt"
    test_file.write_text("first\r\nsecond\r\nthird\r\n", encoding="utf-8")

    await edit.execute(
        "test-crlf-2",
        {"path": str(test_file), "old_text": "second\n", "new_text": "REPLACED\n"},
    )

    assert test_file.read_bytes().decode("utf-8") == "first\r\nREPLACED\r\nthird\r\n"


@pytest.mark.asyncio
async def test_edit_preserves_lf_line_endings(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "lf-preserve.txt"
    test_file.write_text("first\nsecond\nthird\n", encoding="utf-8")

    await edit.execute(
        "test-lf-1",
        {"path": str(test_file), "old_text": "second\n", "new_text": "REPLACED\n"},
    )

    assert test_file.read_text(encoding="utf-8") == "first\nREPLACED\nthird\n"


@pytest.mark.asyncio
async def test_edit_detects_duplicates_across_crlf_variants(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "mixed-endings.txt"
    test_file.write_text("hello\r\nworld\r\n---\r\nhello\nworld\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        await edit.execute(
            "test-crlf-dup",
            {"path": str(test_file), "old_text": "hello\nworld\n", "new_text": "replaced\n"},
        )
    assert "Found 2 occurrences" in str(excinfo.value)


@pytest.mark.asyncio
async def test_edit_preserves_bom(tmp_path: Path) -> None:
    edit = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "bom-test.txt"
    test_file.write_text("\ufefffirst\r\nsecond\r\nthird\r\n", encoding="utf-8")

    await edit.execute(
        "test-bom",
        {"path": str(test_file), "old_text": "second\n", "new_text": "REPLACED\n"},
    )

    assert test_file.read_bytes().decode("utf-8") == "\ufefffirst\r\nREPLACED\r\nthird\r\n"
