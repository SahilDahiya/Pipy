import os
import tempfile

from pi_tools.path_utils import expand_path, resolve_read_path, resolve_to_cwd


def test_expand_path():
    assert "~" not in expand_path("~")
    assert "~" not in expand_path("~/Documents/file.txt")

    nbsp = "file\u00A0name.txt"
    assert expand_path(nbsp) == "file name.txt"


def test_resolve_to_cwd():
    assert resolve_to_cwd("/absolute/path/file.txt", "/some/cwd") == "/absolute/path/file.txt"
    result = resolve_to_cwd("relative/file.txt", "/some/cwd")
    assert result.endswith("/some/cwd/relative/file.txt")


def test_resolve_read_path_variants():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = "test-file.txt"
        path = os.path.join(temp_dir, file_name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("content")
        assert resolve_read_path(file_name, temp_dir) == path

    with tempfile.TemporaryDirectory() as temp_dir:
        nfd_name = "file\u0065\u0301.txt"
        nfc_name = "file\u00e9.txt"
        with open(os.path.join(temp_dir, nfd_name), "w", encoding="utf-8") as handle:
            handle.write("content")
        resolved = resolve_read_path(nfc_name, temp_dir)
        assert resolved.endswith(".txt")

    with tempfile.TemporaryDirectory() as temp_dir:
        curly = "Capture d\u2019cran.txt"
        straight = "Capture d'cran.txt"
        with open(os.path.join(temp_dir, curly), "w", encoding="utf-8") as handle:
            handle.write("content")
        assert resolve_read_path(straight, temp_dir) == os.path.join(temp_dir, curly)

    with tempfile.TemporaryDirectory() as temp_dir:
        nfc_curly = "Capture d\u2019\u00e9cran.txt"
        nfc_straight = "Capture d'\u00e9cran.txt"
        with open(os.path.join(temp_dir, nfc_curly), "w", encoding="utf-8") as handle:
            handle.write("content")
        assert resolve_read_path(nfc_straight, temp_dir) == os.path.join(temp_dir, nfc_curly)

    with tempfile.TemporaryDirectory() as temp_dir:
        macos_name = "Screenshot 2024-01-01 at 10.00.00\u202FAM.png"
        user_name = "Screenshot 2024-01-01 at 10.00.00 AM.png"
        with open(os.path.join(temp_dir, macos_name), "w", encoding="utf-8") as handle:
            handle.write("content")
        assert resolve_read_path(user_name, temp_dir) == os.path.join(temp_dir, macos_name)
