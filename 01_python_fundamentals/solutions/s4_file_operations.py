"""
Solutions for File Operations exercises.
"""
import os
import json
import csv
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any


# =============================================================================
# EXERCISE 1: Reading Text Files - SOLUTIONS
# =============================================================================

def read_file_content(filepath: str) -> str:
    """Read the entire content of a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def read_file_lines(filepath: str) -> list:
    """Read a file and return a list of lines (without newlines)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def read_file_with_line_numbers(filepath: str) -> dict:
    """Read a file and return a dict mapping line numbers to content."""
    result = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            result[i] = line.rstrip('\n')
    return result


def count_words_in_file(filepath: str) -> int:
    """Count the total number of words in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return len(content.split())


# =============================================================================
# EXERCISE 2: Writing Text Files - SOLUTIONS
# =============================================================================

def write_text_file(filepath: str, content: str) -> None:
    """Write content to a text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def append_to_file(filepath: str, content: str) -> None:
    """Append content to the end of a file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(content)


def write_lines_to_file(filepath: str, lines: list) -> None:
    """Write a list of strings to a file, one per line."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def create_numbered_file(filepath: str, lines: list) -> None:
    """Write lines to a file with line numbers prefixed."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines, 1):
            f.write(f"{i}. {line}\n")


# =============================================================================
# EXERCISE 3: JSON File Operations - SOLUTIONS
# =============================================================================

def read_json_file(filepath: str) -> Any:
    """Read a JSON file and return the parsed data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(filepath: str, data: Any, indent: int = 2) -> None:
    """Write data to a JSON file with pretty formatting."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def update_json_file(filepath: str, updates: dict) -> dict:
    """Read a JSON file, update it with new key-value pairs, and save it."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data.update(updates)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return data


def merge_json_files(filepaths: list, output_path: str) -> dict:
    """Merge multiple JSON files into one."""
    merged = {}
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merged.update(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    return merged


# =============================================================================
# EXERCISE 4: CSV File Operations - SOLUTIONS
# =============================================================================

def read_csv_as_dicts(filepath: str) -> list:
    """Read a CSV file and return a list of dictionaries."""
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_csv_as_lists(filepath: str, skip_header: bool = True) -> list:
    """Read a CSV file and return a list of lists (rows)."""
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        return list(reader)


def write_csv_from_dicts(filepath: str, data: list, fieldnames: list = None) -> None:
    """Write a list of dictionaries to a CSV file."""
    if not data:
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def filter_csv_rows(input_path: str, output_path: str,
                    column: str, value: str) -> int:
    """Filter a CSV file to only include rows where column equals value."""
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get(column) == value]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# =============================================================================
# EXERCISE 5: File Path Operations - SOLUTIONS
# =============================================================================

def get_file_info(filepath: str) -> dict:
    """Get information about a file."""
    path = Path(filepath)
    exists = path.exists()

    return {
        'exists': exists,
        'is_file': path.is_file() if exists else False,
        'is_directory': path.is_dir() if exists else False,
        'name': path.name,
        'stem': path.stem,
        'extension': path.suffix,
        'parent': str(path.parent),
        'size': path.stat().st_size if exists and path.is_file() else 0
    }


def list_files_in_directory(directory: str, extension: str = None) -> list:
    """List all files in a directory."""
    path = Path(directory)
    files = []

    for item in path.iterdir():
        if item.is_file():
            if extension is None or item.suffix == extension:
                files.append(item.name)

    return files


def find_files_recursive(directory: str, pattern: str) -> list:
    """Recursively find all files matching a pattern."""
    path = Path(directory)
    return [str(p) for p in path.rglob(pattern)]


def ensure_directory_exists(directory: str) -> bool:
    """Create a directory if it doesn't exist."""
    path = Path(directory)
    if path.exists():
        return False

    path.mkdir(parents=True, exist_ok=True)
    return True


# =============================================================================
# EXERCISE 6: Safe File Operations - SOLUTIONS
# =============================================================================

def safe_read_file(filepath: str, default: str = "") -> str:
    """Read a file, returning a default value if the file doesn't exist."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return default


def safe_write_file(filepath: str, content: str, create_dirs: bool = True) -> bool:
    """Safely write to a file, optionally creating parent directories."""
    try:
        if create_dirs:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception:
        return False


def backup_file(filepath: str) -> str:
    """Create a backup of a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    backup_path = filepath + ".bak"

    # If backup exists, find a numbered alternative
    if os.path.exists(backup_path):
        counter = 1
        while os.path.exists(f"{backup_path}.{counter}"):
            counter += 1
        backup_path = f"{backup_path}.{counter}"

    shutil.copy2(filepath, backup_path)
    return backup_path


def atomic_write_file(filepath: str, content: str) -> None:
    """Write to a file atomically."""
    temp_path = filepath + ".tmp"

    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Atomic rename (on most systems)
    os.replace(temp_path, filepath)


# =============================================================================
# EXERCISE 7: Log File Operations - SOLUTIONS
# =============================================================================

def parse_log_file(filepath: str) -> list:
    """Parse a log file with format: TIMESTAMP LEVEL MESSAGE"""
    entries = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: "2024-01-15 10:30:00 INFO User logged in"
            parts = line.split(' ', 3)
            if len(parts) >= 4:
                timestamp = f"{parts[0]} {parts[1]}"
                level = parts[2]
                message = parts[3]
                entries.append({
                    'timestamp': timestamp,
                    'level': level,
                    'message': message
                })

    return entries


def filter_log_by_level(filepath: str, level: str) -> list:
    """Filter log entries by level."""
    entries = parse_log_file(filepath)
    return [e for e in entries if e['level'] == level]


def write_log_entry(filepath: str, level: str, message: str) -> None:
    """Append a log entry to a log file with current timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} {level} {message}\n"

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(entry)


# =============================================================================
# EXERCISE 8: Configuration File Operations - SOLUTIONS
# =============================================================================

def read_config_file(filepath: str) -> dict:
    """Read a simple config file with KEY=VALUE format."""
    config = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    return config


def write_config_file(filepath: str, config: dict, comment: str = None) -> None:
    """Write a config file in KEY=VALUE format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        if comment:
            f.write(f"# {comment}\n")

        for key, value in config.items():
            f.write(f"{key}={value}\n")


def update_config_value(filepath: str, key: str, value: str) -> bool:
    """Update a single value in a config file, preserving other entries."""
    lines = []
    found = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(f"{key}="):
                lines.append(f"{key}={value}\n")
                found = True
            else:
                lines.append(line)

    if found:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    return found


# =============================================================================
# EXERCISE 9: Binary File Operations - SOLUTIONS
# =============================================================================

def copy_file(source: str, destination: str) -> int:
    """Copy a file (works for both text and binary files)."""
    # Ensure parent directory exists
    Path(destination).parent.mkdir(parents=True, exist_ok=True)

    bytes_copied = 0
    with open(source, 'rb') as src:
        with open(destination, 'wb') as dst:
            while True:
                chunk = src.read(8192)
                if not chunk:
                    break
                dst.write(chunk)
                bytes_copied += len(chunk)

    return bytes_copied


def get_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """Calculate the hash of a file."""
    hash_func = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hash_func.update(chunk)

    return hash_func.hexdigest()


def compare_files(file1: str, file2: str) -> bool:
    """Check if two files have identical content."""
    # Quick check: different sizes means different content
    if os.path.getsize(file1) != os.path.getsize(file2):
        return False

    # Compare content
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            chunk1 = f1.read(8192)
            chunk2 = f2.read(8192)
            if chunk1 != chunk2:
                return False
            if not chunk1:
                return True


# =============================================================================
# EXERCISE 10: Real-World Project - Data Pipeline - SOLUTION
# =============================================================================

def process_data_files(input_dir: str, output_dir: str, config: dict) -> dict:
    """Process all CSV files in input_dir according to config."""
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats = {
        'files_processed': 0,
        'total_rows_in': 0,
        'total_rows_out': 0
    }

    # Find all CSV files
    input_path = Path(input_dir)
    csv_files = list(input_path.glob("*.csv"))

    for csv_file in csv_files:
        # Read input file
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        stats['total_rows_in'] += len(rows)

        # Filter rows if specified
        filter_column = config.get('filter_column')
        filter_value = config.get('filter_value')

        if filter_column and filter_value:
            rows = [r for r in rows if r.get(filter_column) == filter_value]

        # Select columns if specified
        columns = config.get('columns')
        if columns:
            rows = [{k: r[k] for k in columns if k in r} for r in rows]

        # Add timestamp if requested
        if config.get('add_timestamp'):
            timestamp = datetime.now().isoformat()
            for row in rows:
                row['processed_at'] = timestamp

        # Write output file
        output_file = Path(output_dir) / csv_file.name
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        stats['total_rows_out'] += len(rows)
        stats['files_processed'] += 1

    return stats
