"""
File Operations in Python

This module covers reading and writing files - essential for any Python developer.
You'll practice working with text, JSON, CSV files, and handling file paths.

Run tests with:
    pytest 01_python_fundamentals/tests/test_p4_file_operations.py -v
"""
import os
import json
import csv
from pathlib import Path
from typing import Any


# =============================================================================
# EXERCISE 1: Reading Text Files
# =============================================================================

def read_file_content(filepath: str) -> str:
    """
    Read the entire content of a text file and return it as a string.

    Args:
        filepath: Path to the file (can be absolute or relative)

    Returns:
        The file contents as a string

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> content = read_file_content("data/sample.txt")
        >>> print(content)
        'Hello, World!'
    """
    # YOUR CODE HERE
    pass


def read_file_lines(filepath: str) -> list:
    """
    Read a file and return a list of lines (without newline characters).

    Args:
        filepath: Path to the file

    Returns:
        List of strings, one per line

    Example:
        >>> lines = read_file_lines("data/names.txt")
        >>> lines
        ['Alice', 'Bob', 'Charlie']
    """
    # YOUR CODE HERE
    pass


def read_file_with_line_numbers(filepath: str) -> dict:
    """
    Read a file and return a dict mapping line numbers to content.
    Line numbers start at 1.

    Args:
        filepath: Path to the file

    Returns:
        Dict like {1: "first line", 2: "second line", ...}

    Example:
        >>> read_file_with_line_numbers("data/sample.txt")
        {1: 'Hello', 2: 'World'}
    """
    # YOUR CODE HERE
    pass


def count_words_in_file(filepath: str) -> int:
    """
    Count the total number of words in a file.
    Words are separated by whitespace.

    Args:
        filepath: Path to the file

    Returns:
        Total word count

    Example:
        >>> count_words_in_file("data/story.txt")
        42
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Writing Text Files
# =============================================================================

def write_text_file(filepath: str, content: str) -> None:
    """
    Write content to a text file. Creates the file if it doesn't exist,
    overwrites if it does.

    Args:
        filepath: Path where file should be written
        content: Text content to write

    Example:
        >>> write_text_file("output/hello.txt", "Hello, World!")
        # Creates file with content "Hello, World!"
    """
    # YOUR CODE HERE
    pass


def append_to_file(filepath: str, content: str) -> None:
    """
    Append content to the end of a file.
    Creates the file if it doesn't exist.

    Args:
        filepath: Path to the file
        content: Text to append

    Example:
        >>> append_to_file("log.txt", "New log entry\n")
    """
    # YOUR CODE HERE
    pass


def write_lines_to_file(filepath: str, lines: list) -> None:
    """
    Write a list of strings to a file, one per line.

    Args:
        filepath: Path to the file
        lines: List of strings to write

    Example:
        >>> write_lines_to_file("names.txt", ["Alice", "Bob", "Charlie"])
        # Creates file with:
        # Alice
        # Bob
        # Charlie
    """
    # YOUR CODE HERE
    pass


def create_numbered_file(filepath: str, lines: list) -> None:
    """
    Write lines to a file with line numbers prefixed.
    Format: "1. First line"

    Args:
        filepath: Path to the file
        lines: List of strings to write

    Example:
        >>> create_numbered_file("todo.txt", ["Buy milk", "Call mom"])
        # Creates:
        # 1. Buy milk
        # 2. Call mom
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: JSON File Operations
# =============================================================================

def read_json_file(filepath: str) -> Any:
    """
    Read a JSON file and return the parsed data.

    Args:
        filepath: Path to the JSON file

    Returns:
        Parsed JSON data (dict, list, etc.)

    Example:
        >>> data = read_json_file("config.json")
        >>> data
        {'name': 'app', 'version': '1.0'}
    """
    # YOUR CODE HERE
    pass


def write_json_file(filepath: str, data: Any, indent: int = 2) -> None:
    """
    Write data to a JSON file with pretty formatting.

    Args:
        filepath: Path to the JSON file
        data: Data to serialize to JSON
        indent: Indentation level for formatting

    Example:
        >>> write_json_file("output.json", {"name": "test"})
    """
    # YOUR CODE HERE
    pass


def update_json_file(filepath: str, updates: dict) -> dict:
    """
    Read a JSON file, update it with new key-value pairs, and save it.
    Returns the updated data.

    Args:
        filepath: Path to the JSON file
        updates: Dict of key-value pairs to add/update

    Returns:
        The updated data

    Example:
        >>> # config.json contains {"name": "app"}
        >>> update_json_file("config.json", {"version": "2.0"})
        {'name': 'app', 'version': '2.0'}
    """
    # YOUR CODE HERE
    pass


def merge_json_files(filepaths: list, output_path: str) -> dict:
    """
    Merge multiple JSON files (all must contain dicts) into one.
    Later files override earlier ones for duplicate keys.

    Args:
        filepaths: List of JSON file paths to merge
        output_path: Where to save the merged result

    Returns:
        The merged dictionary

    Example:
        >>> # file1.json: {"a": 1}, file2.json: {"b": 2}
        >>> merge_json_files(["file1.json", "file2.json"], "merged.json")
        {'a': 1, 'b': 2}
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: CSV File Operations
# =============================================================================

def read_csv_as_dicts(filepath: str) -> list:
    """
    Read a CSV file and return a list of dictionaries.
    Uses the first row as headers.

    Args:
        filepath: Path to the CSV file

    Returns:
        List of dicts, one per row

    Example:
        >>> # users.csv:
        >>> # name,age
        >>> # Alice,30
        >>> # Bob,25
        >>> read_csv_as_dicts("users.csv")
        [{'name': 'Alice', 'age': '30'}, {'name': 'Bob', 'age': '25'}]
    """
    # YOUR CODE HERE
    pass


def read_csv_as_lists(filepath: str, skip_header: bool = True) -> list:
    """
    Read a CSV file and return a list of lists (rows).

    Args:
        filepath: Path to the CSV file
        skip_header: If True, skip the first row

    Returns:
        List of lists

    Example:
        >>> read_csv_as_lists("data.csv", skip_header=True)
        [['Alice', '30'], ['Bob', '25']]
    """
    # YOUR CODE HERE
    pass


def write_csv_from_dicts(filepath: str, data: list, fieldnames: list = None) -> None:
    """
    Write a list of dictionaries to a CSV file.

    Args:
        filepath: Path to the CSV file
        data: List of dicts to write
        fieldnames: Column headers (if None, use keys from first dict)

    Example:
        >>> data = [{'name': 'Alice', 'age': 30}]
        >>> write_csv_from_dicts("output.csv", data)
    """
    # YOUR CODE HERE
    pass


def filter_csv_rows(input_path: str, output_path: str,
                    column: str, value: str) -> int:
    """
    Filter a CSV file to only include rows where column equals value.
    Write filtered rows to output file.

    Args:
        input_path: Source CSV file
        output_path: Destination CSV file
        column: Column name to filter on
        value: Value to match

    Returns:
        Number of rows written (excluding header)

    Example:
        >>> filter_csv_rows("users.csv", "adults.csv", "status", "active")
        5  # 5 rows had status='active'
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 5: File Path Operations
# =============================================================================

def get_file_info(filepath: str) -> dict:
    """
    Get information about a file.

    Args:
        filepath: Path to the file

    Returns:
        Dict with keys:
        - exists: bool
        - is_file: bool
        - is_directory: bool
        - name: filename with extension
        - stem: filename without extension
        - extension: file extension (with dot)
        - parent: parent directory path
        - size: file size in bytes (0 if doesn't exist)

    Example:
        >>> get_file_info("/path/to/document.pdf")
        {
            'exists': True,
            'is_file': True,
            'is_directory': False,
            'name': 'document.pdf',
            'stem': 'document',
            'extension': '.pdf',
            'parent': '/path/to',
            'size': 1024
        }
    """
    # YOUR CODE HERE
    pass


def list_files_in_directory(directory: str, extension: str = None) -> list:
    """
    List all files in a directory.

    Args:
        directory: Path to directory
        extension: If provided, only return files with this extension
                   (include the dot, e.g., ".py")

    Returns:
        List of filenames (not full paths)

    Example:
        >>> list_files_in_directory("/code", ".py")
        ['main.py', 'utils.py', 'test.py']
    """
    # YOUR CODE HERE
    pass


def find_files_recursive(directory: str, pattern: str) -> list:
    """
    Recursively find all files matching a pattern.

    Args:
        directory: Root directory to search
        pattern: Glob pattern (e.g., "*.txt", "**/*.py")

    Returns:
        List of full file paths

    Example:
        >>> find_files_recursive("/project", "*.py")
        ['/project/main.py', '/project/src/utils.py']
    """
    # YOUR CODE HERE
    pass


def ensure_directory_exists(directory: str) -> bool:
    """
    Create a directory if it doesn't exist (including parent directories).

    Args:
        directory: Path to directory

    Returns:
        True if directory was created, False if it already existed

    Example:
        >>> ensure_directory_exists("/path/to/new/folder")
        True  # Created
        >>> ensure_directory_exists("/path/to/new/folder")
        False  # Already exists
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 6: Safe File Operations
# =============================================================================

def safe_read_file(filepath: str, default: str = "") -> str:
    """
    Read a file, returning a default value if the file doesn't exist.

    Args:
        filepath: Path to the file
        default: Value to return if file doesn't exist

    Returns:
        File content or default value

    Example:
        >>> safe_read_file("exists.txt")
        'file content'
        >>> safe_read_file("missing.txt", "N/A")
        'N/A'
    """
    # YOUR CODE HERE
    pass


def safe_write_file(filepath: str, content: str, create_dirs: bool = True) -> bool:
    """
    Safely write to a file, optionally creating parent directories.

    Args:
        filepath: Path to the file
        content: Content to write
        create_dirs: If True, create parent directories if they don't exist

    Returns:
        True if successful, False otherwise

    Example:
        >>> safe_write_file("/new/path/file.txt", "hello", create_dirs=True)
        True
    """
    # YOUR CODE HERE
    pass


def backup_file(filepath: str) -> str:
    """
    Create a backup of a file by copying it with .bak extension.
    If backup already exists, add a number (file.bak.1, file.bak.2, etc.)

    Args:
        filepath: Path to the file to backup

    Returns:
        Path to the backup file

    Raises:
        FileNotFoundError: If source file doesn't exist

    Example:
        >>> backup_file("important.txt")
        'important.txt.bak'
        >>> backup_file("important.txt")  # called again
        'important.txt.bak.1'
    """
    # YOUR CODE HERE
    pass


def atomic_write_file(filepath: str, content: str) -> None:
    """
    Write to a file atomically (write to temp file, then rename).
    This prevents partial writes if the program crashes.

    Args:
        filepath: Path to the file
        content: Content to write

    Example:
        >>> atomic_write_file("config.json", '{"version": 2}')
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 7: Log File Operations
# =============================================================================

def parse_log_file(filepath: str) -> list:
    """
    Parse a log file with format: "TIMESTAMP LEVEL MESSAGE"
    Example line: "2024-01-15 10:30:00 INFO User logged in"

    Args:
        filepath: Path to log file

    Returns:
        List of dicts with keys: timestamp, level, message

    Example:
        >>> parse_log_file("app.log")
        [
            {'timestamp': '2024-01-15 10:30:00', 'level': 'INFO', 'message': 'User logged in'},
            {'timestamp': '2024-01-15 10:31:00', 'level': 'ERROR', 'message': 'Connection failed'}
        ]
    """
    # YOUR CODE HERE
    pass


def filter_log_by_level(filepath: str, level: str) -> list:
    """
    Filter log entries by level.

    Args:
        filepath: Path to log file
        level: Log level to filter for ("INFO", "ERROR", "WARNING", etc.)

    Returns:
        List of matching log entries (as dicts)

    Example:
        >>> filter_log_by_level("app.log", "ERROR")
        [{'timestamp': '...', 'level': 'ERROR', 'message': '...'}]
    """
    # YOUR CODE HERE
    pass


def write_log_entry(filepath: str, level: str, message: str) -> None:
    """
    Append a log entry to a log file with current timestamp.
    Format: "YYYY-MM-DD HH:MM:SS LEVEL MESSAGE"

    Args:
        filepath: Path to log file
        level: Log level (INFO, ERROR, WARNING, DEBUG)
        message: Log message

    Example:
        >>> write_log_entry("app.log", "INFO", "Application started")
        # Appends: "2024-01-15 10:30:00 INFO Application started"
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 8: Configuration File Operations
# =============================================================================

def read_config_file(filepath: str) -> dict:
    """
    Read a simple config file with KEY=VALUE format.
    Lines starting with # are comments.
    Empty lines are ignored.

    Args:
        filepath: Path to config file

    Returns:
        Dict of key-value pairs

    Example:
        >>> # config.txt:
        >>> # # This is a comment
        >>> # HOST=localhost
        >>> # PORT=8080
        >>> read_config_file("config.txt")
        {'HOST': 'localhost', 'PORT': '8080'}
    """
    # YOUR CODE HERE
    pass


def write_config_file(filepath: str, config: dict, comment: str = None) -> None:
    """
    Write a config file in KEY=VALUE format.

    Args:
        filepath: Path to config file
        config: Dict of key-value pairs
        comment: Optional comment to add at the top

    Example:
        >>> write_config_file("config.txt", {"HOST": "localhost"}, "Server config")
        # Creates:
        # # Server config
        # HOST=localhost
    """
    # YOUR CODE HERE
    pass


def update_config_value(filepath: str, key: str, value: str) -> bool:
    """
    Update a single value in a config file, preserving other entries and comments.

    Args:
        filepath: Path to config file
        key: Key to update
        value: New value

    Returns:
        True if key was found and updated, False if key wasn't found

    Example:
        >>> update_config_value("config.txt", "PORT", "9090")
        True
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 9: Binary File Operations
# =============================================================================

def copy_file(source: str, destination: str) -> int:
    """
    Copy a file (works for both text and binary files).

    Args:
        source: Path to source file
        destination: Path to destination file

    Returns:
        Number of bytes copied

    Example:
        >>> copy_file("image.png", "backup/image.png")
        1024  # bytes copied
    """
    # YOUR CODE HERE
    pass


def get_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """
    Calculate the hash of a file.

    Args:
        filepath: Path to the file
        algorithm: Hash algorithm ("md5", "sha1", "sha256")

    Returns:
        Hex string of the hash

    Example:
        >>> get_file_hash("document.pdf", "sha256")
        'a1b2c3d4e5...'
    """
    # YOUR CODE HERE
    pass


def compare_files(file1: str, file2: str) -> bool:
    """
    Check if two files have identical content.

    Args:
        file1: Path to first file
        file2: Path to second file

    Returns:
        True if files are identical, False otherwise

    Example:
        >>> compare_files("original.txt", "copy.txt")
        True
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 10: Real-World Project - Data Pipeline
# =============================================================================

def process_data_files(input_dir: str, output_dir: str, config: dict) -> dict:
    """
    Process all CSV files in input_dir according to config and save to output_dir.

    Config options:
    - columns: List of columns to keep (if None, keep all)
    - filter_column: Column name to filter on
    - filter_value: Value to filter for
    - add_timestamp: If True, add a 'processed_at' column

    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory to save processed files
        config: Processing configuration dict

    Returns:
        Dict with stats:
        - files_processed: number of files processed
        - total_rows_in: total input rows
        - total_rows_out: total output rows

    Example:
        >>> config = {
        ...     "columns": ["name", "email"],
        ...     "filter_column": "status",
        ...     "filter_value": "active",
        ...     "add_timestamp": True
        ... }
        >>> process_data_files("input/", "output/", config)
        {'files_processed': 3, 'total_rows_in': 100, 'total_rows_out': 45}
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TEST YOUR UNDERSTANDING
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil

    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    print(f"Testing in: {test_dir}")

    try:
        # Test 1: Write and read a text file
        print("\n=== Test 1: Text Files ===")
        test_file = os.path.join(test_dir, "test.txt")
        write_text_file(test_file, "Hello, World!")
        content = read_file_content(test_file)
        print(f"Wrote and read: '{content}'")

        # Test 2: Write and read lines
        print("\n=== Test 2: Lines ===")
        names_file = os.path.join(test_dir, "names.txt")
        write_lines_to_file(names_file, ["Alice", "Bob", "Charlie"])
        lines = read_file_lines(names_file)
        print(f"Lines: {lines}")

        # Test 3: JSON operations
        print("\n=== Test 3: JSON ===")
        json_file = os.path.join(test_dir, "config.json")
        write_json_file(json_file, {"name": "app", "version": "1.0"})
        data = read_json_file(json_file)
        print(f"JSON data: {data}")

        updated = update_json_file(json_file, {"debug": True})
        print(f"Updated: {updated}")

        # Test 4: CSV operations
        print("\n=== Test 4: CSV ===")
        csv_file = os.path.join(test_dir, "users.csv")
        users = [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ]
        write_csv_from_dicts(csv_file, users)
        read_users = read_csv_as_dicts(csv_file)
        print(f"CSV users: {read_users}")

        # Test 5: File info
        print("\n=== Test 5: File Info ===")
        info = get_file_info(test_file)
        print(f"File info: {info}")

        # Test 6: Config file
        print("\n=== Test 6: Config File ===")
        config_file = os.path.join(test_dir, "app.config")
        write_config_file(config_file, {"HOST": "localhost", "PORT": "8080"}, "App Config")
        config = read_config_file(config_file)
        print(f"Config: {config}")

        print("\n=== All tests passed! ===")

    finally:
        # Clean up
        shutil.rmtree(test_dir)
        print(f"\nCleaned up: {test_dir}")
