# Hints - Python Fundamentals

## P1: Data Structures

### two_sum
- Think about what data structure lets you look up values in O(1) time
- As you iterate, store each number's index in a dictionary
- For each number, check if `target - number` exists in your dictionary

### group_anagrams
- What property do all anagrams share? (Think about sorted characters)
- Use `collections.defaultdict(list)` for grouping
- The key for grouping should be the sorted version of each word

### flatten_nested_dict
- This is a classic recursion problem
- Base case: value is NOT a dict -> add to result
- Recursive case: value IS a dict -> recurse with updated key prefix
- Build the new key with `f"{parent_key}{sep}{k}"`

### lru_cache_dict
- Python 3.7+ dicts maintain insertion order!
- "Most recently used" = move to END of dict (delete + re-insert)
- "Least recently used" = FIRST item in dict (`next(iter(dict))`)
- When capacity full: pop the first key before inserting new one

## P2: Comprehensions & Generators

### matrix_transpose
- `zip(*matrix)` is the magic trick - it unpacks rows and zips columns
- Remember to convert each tuple back to a list

### filter_and_transform
- Sort FIRST (by score descending), then filter, then transform
- `sorted(data, key=lambda x: x["score"], reverse=True)`
- List comprehension with `if` clause for filtering

### fibonacci_generator
- Use `yield` instead of `return`
- Keep two variables: current and next
- `a, b = b, a + b` is the Fibonacci update step

### sliding_window_avg
- Iterate from `i=0` to `len(data) - window_size`
- Slice `data[i:i+window_size]` for each window
- `yield sum(window) / len(window)`

## P3: Decorators & Closures

### timer_decorator
- `import functools` and use `@functools.wraps(func)`
- Record `time.time()` before and after calling `func`
- Store result as `wrapper.last_elapsed = elapsed`

### retry_decorator
- This is a decorator FACTORY: function -> decorator -> wrapper (3 levels of nesting)
- Use a for loop with `range(max_retries)`
- Catch the exception, save it, and if all retries fail, re-raise the last one

### memoize
- Use a dictionary `wrapper.cache = {}`
- Key = `args` tuple (it's hashable)
- Check if args already in cache before calling the function

### validate_types
- Use `inspect.signature(func)` to get parameter names
- Match positional args with parameter names by index
- Use `isinstance(arg, expected_type)` to check

## P4: File Operations

### Reading Text Files

**read_file_content:**
```python
with open(filepath, 'r', encoding='utf-8') as f:
    return f.read()
```

**read_file_lines:**
- Use `f.readlines()` or iterate over `f`
- Strip newlines: `line.rstrip('\n')`
- Filter empty lines if needed

**count_words_in_file:**
- Read content, then `len(content.split())`

### Writing Text Files

**write_text_file:**
- Use `'w'` mode: `open(filepath, 'w', encoding='utf-8')`
- The `with` statement handles closing automatically

**append_to_file:**
- Use `'a'` mode instead of `'w'`

**write_lines_to_file:**
```python
for line in lines:
    f.write(line + '\n')
```

### JSON Files

**read_json_file:**
```python
import json
with open(filepath) as f:
    return json.load(f)  # Not loads!
```

**write_json_file:**
```python
json.dump(data, f, indent=indent)  # Not dumps!
```

**update_json_file:**
1. Read existing data
2. `data.update(updates)`
3. Write back to file

### CSV Files

**read_csv_as_dicts:**
```python
import csv
with open(filepath, newline='') as f:
    reader = csv.DictReader(f)
    return list(reader)
```

**write_csv_from_dicts:**
```python
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(data)
```

### Path Operations (use pathlib!)

**get_file_info:**
```python
from pathlib import Path
path = Path(filepath)
path.exists()    # True/False
path.is_file()   # True/False
path.name        # "file.txt"
path.stem        # "file"
path.suffix      # ".txt"
path.parent      # Path object of parent
path.stat().st_size  # file size in bytes
```

**find_files_recursive:**
```python
path = Path(directory)
return list(path.rglob(pattern))  # rglob = recursive glob
```

**ensure_directory_exists:**
```python
path.mkdir(parents=True, exist_ok=True)
```

### Safe File Operations

**safe_read_file:**
- Use try/except to catch `FileNotFoundError`
- Return default on exception

**backup_file:**
- Use `shutil.copy2()` to preserve metadata
- Check if `.bak` exists, if so try `.bak.1`, `.bak.2`, etc.

**atomic_write_file:**
1. Write to a temp file (same directory)
2. Use `os.replace(temp_path, filepath)` - this is atomic

### Log File Operations

**parse_log_file:**
- Format: `"TIMESTAMP LEVEL MESSAGE"`
- Split on space, first 2 parts = timestamp, part 3 = level, rest = message
- `parts = line.split(' ', 3)` - split into max 4 parts

**write_log_entry:**
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
entry = f"{timestamp} {level} {message}\n"
```

### Config Files

**read_config_file:**
```python
if line.startswith('#'):  # Skip comments
    continue
if '=' in line:
    key, value = line.split('=', 1)  # Split on first = only
```

**update_config_value:**
- Read all lines, modify the matching one, write all back
- Preserve comments and other lines

### Binary Files

**copy_file:**
- Use `'rb'` and `'wb'` modes
- Read/write in chunks: `while chunk := f.read(8192):`

**get_file_hash:**
```python
import hashlib
hash_func = hashlib.new(algorithm)  # "md5", "sha256"
hash_func.update(chunk)
return hash_func.hexdigest()
```

**compare_files:**
- Quick check: compare file sizes first
- Then compare content chunk by chunk
