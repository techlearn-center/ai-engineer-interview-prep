"""Tests for File Operations exercises."""
import pytest
import json
import csv
import os
import tempfile
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from p4_file_operations import (
    read_file_content, read_file_lines, read_file_with_line_numbers,
    count_words_in_file, write_text_file, append_to_file,
    write_lines_to_file, create_numbered_file, read_json_file,
    write_json_file, update_json_file, merge_json_files,
    read_csv_as_dicts, read_csv_as_lists, write_csv_from_dicts,
    filter_csv_rows, get_file_info, list_files_in_directory,
    find_files_recursive, ensure_directory_exists, safe_read_file,
    safe_write_file, backup_file, atomic_write_file,
    parse_log_file, filter_log_by_level, write_log_entry,
    read_config_file, write_config_file, update_config_value,
    copy_file, get_file_hash, compare_files, process_data_files
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


# =============================================================================
# Exercise 1: Reading Text Files
# =============================================================================

class TestReadTextFiles:
    def test_read_file_content(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")
        with open(filepath, "w") as f:
            f.write("Hello, World!")

        content = read_file_content(filepath)
        assert content == "Hello, World!"

    def test_read_file_content_multiline(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")
        with open(filepath, "w") as f:
            f.write("Line 1\nLine 2\nLine 3")

        content = read_file_content(filepath)
        assert "Line 1" in content
        assert "Line 2" in content

    def test_read_file_content_not_found(self, temp_dir):
        filepath = os.path.join(temp_dir, "nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            read_file_content(filepath)

    def test_read_file_lines(self, temp_dir):
        filepath = os.path.join(temp_dir, "names.txt")
        with open(filepath, "w") as f:
            f.write("Alice\nBob\nCharlie\n")

        lines = read_file_lines(filepath)
        assert lines == ["Alice", "Bob", "Charlie"]

    def test_read_file_with_line_numbers(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")
        with open(filepath, "w") as f:
            f.write("First\nSecond\nThird")

        result = read_file_with_line_numbers(filepath)
        assert result[1] == "First"
        assert result[2] == "Second"
        assert result[3] == "Third"

    def test_count_words_in_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "story.txt")
        with open(filepath, "w") as f:
            f.write("The quick brown fox\njumps over the lazy dog")

        count = count_words_in_file(filepath)
        assert count == 9


# =============================================================================
# Exercise 2: Writing Text Files
# =============================================================================

class TestWriteTextFiles:
    def test_write_text_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "output.txt")
        write_text_file(filepath, "Hello, World!")

        with open(filepath) as f:
            assert f.read() == "Hello, World!"

    def test_write_text_file_overwrites(self, temp_dir):
        filepath = os.path.join(temp_dir, "output.txt")
        write_text_file(filepath, "First")
        write_text_file(filepath, "Second")

        with open(filepath) as f:
            assert f.read() == "Second"

    def test_append_to_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "log.txt")
        write_text_file(filepath, "Line 1\n")
        append_to_file(filepath, "Line 2\n")

        with open(filepath) as f:
            content = f.read()
        assert "Line 1" in content
        assert "Line 2" in content

    def test_write_lines_to_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "names.txt")
        write_lines_to_file(filepath, ["Alice", "Bob", "Charlie"])

        with open(filepath) as f:
            lines = f.read().strip().split("\n")
        assert lines == ["Alice", "Bob", "Charlie"]

    def test_create_numbered_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "todo.txt")
        create_numbered_file(filepath, ["Buy milk", "Call mom"])

        with open(filepath) as f:
            content = f.read()
        assert "1. Buy milk" in content
        assert "2. Call mom" in content


# =============================================================================
# Exercise 3: JSON File Operations
# =============================================================================

class TestJsonFiles:
    def test_read_json_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "config.json")
        with open(filepath, "w") as f:
            json.dump({"name": "app", "version": "1.0"}, f)

        data = read_json_file(filepath)
        assert data["name"] == "app"
        assert data["version"] == "1.0"

    def test_write_json_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "output.json")
        write_json_file(filepath, {"key": "value"})

        with open(filepath) as f:
            data = json.load(f)
        assert data["key"] == "value"

    def test_update_json_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "config.json")
        with open(filepath, "w") as f:
            json.dump({"name": "app"}, f)

        result = update_json_file(filepath, {"version": "2.0"})
        assert result["name"] == "app"
        assert result["version"] == "2.0"

        # Verify file was updated
        with open(filepath) as f:
            data = json.load(f)
        assert data["version"] == "2.0"

    def test_merge_json_files(self, temp_dir):
        file1 = os.path.join(temp_dir, "file1.json")
        file2 = os.path.join(temp_dir, "file2.json")
        output = os.path.join(temp_dir, "merged.json")

        with open(file1, "w") as f:
            json.dump({"a": 1, "b": 2}, f)
        with open(file2, "w") as f:
            json.dump({"c": 3, "b": 99}, f)

        result = merge_json_files([file1, file2], output)
        assert result["a"] == 1
        assert result["b"] == 99  # Overridden by file2
        assert result["c"] == 3


# =============================================================================
# Exercise 4: CSV File Operations
# =============================================================================

class TestCsvFiles:
    def test_read_csv_as_dicts(self, temp_dir):
        filepath = os.path.join(temp_dir, "users.csv")
        with open(filepath, "w", newline="") as f:
            f.write("name,age\nAlice,30\nBob,25\n")

        data = read_csv_as_dicts(filepath)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"
        assert data[0]["age"] == "30"

    def test_read_csv_as_lists(self, temp_dir):
        filepath = os.path.join(temp_dir, "data.csv")
        with open(filepath, "w", newline="") as f:
            f.write("name,age\nAlice,30\nBob,25\n")

        data = read_csv_as_lists(filepath, skip_header=True)
        assert data == [["Alice", "30"], ["Bob", "25"]]

    def test_write_csv_from_dicts(self, temp_dir):
        filepath = os.path.join(temp_dir, "output.csv")
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        write_csv_from_dicts(filepath, data)

        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    def test_filter_csv_rows(self, temp_dir):
        input_path = os.path.join(temp_dir, "users.csv")
        output_path = os.path.join(temp_dir, "filtered.csv")

        with open(input_path, "w", newline="") as f:
            f.write("name,status\nAlice,active\nBob,inactive\nCharlie,active\n")

        count = filter_csv_rows(input_path, output_path, "status", "active")
        assert count == 2

        with open(output_path) as f:
            content = f.read()
        assert "Alice" in content
        assert "Charlie" in content
        assert "Bob" not in content


# =============================================================================
# Exercise 5: File Path Operations
# =============================================================================

class TestFilePaths:
    def test_get_file_info_exists(self, temp_dir):
        filepath = os.path.join(temp_dir, "document.pdf")
        with open(filepath, "w") as f:
            f.write("test content")

        info = get_file_info(filepath)
        assert info["exists"] is True
        assert info["is_file"] is True
        assert info["is_directory"] is False
        assert info["name"] == "document.pdf"
        assert info["stem"] == "document"
        assert info["extension"] == ".pdf"
        assert info["size"] > 0

    def test_get_file_info_not_exists(self, temp_dir):
        filepath = os.path.join(temp_dir, "missing.txt")
        info = get_file_info(filepath)
        assert info["exists"] is False

    def test_list_files_in_directory(self, temp_dir):
        # Create test files
        for name in ["file1.py", "file2.py", "file3.txt"]:
            with open(os.path.join(temp_dir, name), "w") as f:
                f.write("test")

        all_files = list_files_in_directory(temp_dir)
        assert len(all_files) == 3

        py_files = list_files_in_directory(temp_dir, ".py")
        assert len(py_files) == 2
        assert all(f.endswith(".py") for f in py_files)

    def test_find_files_recursive(self, temp_dir):
        # Create subdirectory with files
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)

        with open(os.path.join(temp_dir, "main.py"), "w") as f:
            f.write("test")
        with open(os.path.join(subdir, "utils.py"), "w") as f:
            f.write("test")

        files = find_files_recursive(temp_dir, "*.py")
        assert len(files) == 2

    def test_ensure_directory_exists(self, temp_dir):
        new_dir = os.path.join(temp_dir, "new", "nested", "dir")

        result = ensure_directory_exists(new_dir)
        assert result is True
        assert os.path.isdir(new_dir)

        result = ensure_directory_exists(new_dir)
        assert result is False  # Already exists


# =============================================================================
# Exercise 6: Safe File Operations
# =============================================================================

class TestSafeFileOps:
    def test_safe_read_file_exists(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")
        with open(filepath, "w") as f:
            f.write("content")

        result = safe_read_file(filepath)
        assert result == "content"

    def test_safe_read_file_missing(self, temp_dir):
        filepath = os.path.join(temp_dir, "missing.txt")
        result = safe_read_file(filepath, "default")
        assert result == "default"

    def test_safe_write_file_creates_dirs(self, temp_dir):
        filepath = os.path.join(temp_dir, "new", "path", "file.txt")

        result = safe_write_file(filepath, "hello", create_dirs=True)
        assert result is True

        with open(filepath) as f:
            assert f.read() == "hello"

    def test_backup_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "important.txt")
        with open(filepath, "w") as f:
            f.write("important data")

        backup_path = backup_file(filepath)
        assert backup_path.endswith(".bak")
        assert os.path.exists(backup_path)

    def test_atomic_write_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "config.json")
        atomic_write_file(filepath, '{"version": 2}')

        with open(filepath) as f:
            content = f.read()
        assert content == '{"version": 2}'


# =============================================================================
# Exercise 7: Log File Operations
# =============================================================================

class TestLogFiles:
    def test_parse_log_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "app.log")
        with open(filepath, "w") as f:
            f.write("2024-01-15 10:30:00 INFO User logged in\n")
            f.write("2024-01-15 10:31:00 ERROR Connection failed\n")

        entries = parse_log_file(filepath)
        assert len(entries) == 2
        assert entries[0]["level"] == "INFO"
        assert entries[0]["message"] == "User logged in"
        assert entries[1]["level"] == "ERROR"

    def test_filter_log_by_level(self, temp_dir):
        filepath = os.path.join(temp_dir, "app.log")
        with open(filepath, "w") as f:
            f.write("2024-01-15 10:30:00 INFO Message 1\n")
            f.write("2024-01-15 10:31:00 ERROR Error 1\n")
            f.write("2024-01-15 10:32:00 INFO Message 2\n")

        errors = filter_log_by_level(filepath, "ERROR")
        assert len(errors) == 1
        assert errors[0]["level"] == "ERROR"

    def test_write_log_entry(self, temp_dir):
        filepath = os.path.join(temp_dir, "app.log")
        write_log_entry(filepath, "INFO", "Application started")

        with open(filepath) as f:
            content = f.read()
        assert "INFO" in content
        assert "Application started" in content


# =============================================================================
# Exercise 8: Configuration File Operations
# =============================================================================

class TestConfigFiles:
    def test_read_config_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "app.config")
        with open(filepath, "w") as f:
            f.write("# Comment\n")
            f.write("HOST=localhost\n")
            f.write("PORT=8080\n")
            f.write("\n")  # Empty line

        config = read_config_file(filepath)
        assert config["HOST"] == "localhost"
        assert config["PORT"] == "8080"
        assert len(config) == 2

    def test_write_config_file(self, temp_dir):
        filepath = os.path.join(temp_dir, "app.config")
        write_config_file(filepath, {"HOST": "localhost"}, "Server config")

        with open(filepath) as f:
            content = f.read()
        assert "# Server config" in content
        assert "HOST=localhost" in content

    def test_update_config_value(self, temp_dir):
        filepath = os.path.join(temp_dir, "app.config")
        with open(filepath, "w") as f:
            f.write("HOST=localhost\nPORT=8080\n")

        result = update_config_value(filepath, "PORT", "9090")
        assert result is True

        with open(filepath) as f:
            content = f.read()
        assert "PORT=9090" in content
        assert "PORT=8080" not in content


# =============================================================================
# Exercise 9: Binary File Operations
# =============================================================================

class TestBinaryFiles:
    def test_copy_file(self, temp_dir):
        source = os.path.join(temp_dir, "source.txt")
        dest = os.path.join(temp_dir, "dest.txt")

        with open(source, "w") as f:
            f.write("test content")

        bytes_copied = copy_file(source, dest)
        assert bytes_copied > 0
        assert os.path.exists(dest)

        with open(dest) as f:
            assert f.read() == "test content"

    def test_get_file_hash(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.txt")
        with open(filepath, "w") as f:
            f.write("test")

        hash_md5 = get_file_hash(filepath, "md5")
        assert len(hash_md5) == 32  # MD5 produces 32 hex chars

        hash_sha256 = get_file_hash(filepath, "sha256")
        assert len(hash_sha256) == 64  # SHA256 produces 64 hex chars

    def test_compare_files_identical(self, temp_dir):
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")

        with open(file1, "w") as f:
            f.write("same content")
        with open(file2, "w") as f:
            f.write("same content")

        assert compare_files(file1, file2) is True

    def test_compare_files_different(self, temp_dir):
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")

        with open(file1, "w") as f:
            f.write("content 1")
        with open(file2, "w") as f:
            f.write("content 2")

        assert compare_files(file1, file2) is False


# =============================================================================
# Exercise 10: Data Pipeline
# =============================================================================

class TestDataPipeline:
    def test_process_data_files(self, temp_dir):
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)

        # Create test CSV
        with open(os.path.join(input_dir, "users.csv"), "w", newline="") as f:
            f.write("name,email,status\n")
            f.write("Alice,alice@test.com,active\n")
            f.write("Bob,bob@test.com,inactive\n")
            f.write("Charlie,charlie@test.com,active\n")

        config = {
            "columns": ["name", "email"],
            "filter_column": "status",
            "filter_value": "active",
            "add_timestamp": False
        }

        stats = process_data_files(input_dir, output_dir, config)
        assert stats["files_processed"] == 1
        assert stats["total_rows_in"] == 3
        assert stats["total_rows_out"] == 2

        # Check output file
        output_file = os.path.join(output_dir, "users.csv")
        assert os.path.exists(output_file)
