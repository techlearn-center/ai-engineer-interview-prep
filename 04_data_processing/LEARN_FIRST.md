# Learn: Data Processing for AI Engineers

Data processing is what you do BEFORE training a model.
"Garbage in, garbage out" - the quality of your data determines your model's quality.

---

## 1. Why Data Processing Matters

In a real ML project, you spend 80% of your time on data, 20% on the model.
Interviewers know this, which is why they test it.

```
Raw Data (messy, inconsistent, missing values)
        |
        v
[Text Cleaning / Feature Engineering]
        |
        v
Clean Features (ready for model training)
```

---

## 2. Text Tokenization

Tokenization = splitting text into individual units (tokens/words).

```python
# Simplest approach: split on spaces
"Hello World".split()  # ["Hello", "World"]

# But real text is messy:
"Hello, World! How's it going?"
# You want: ["hello", "world", "hows", "it", "going"]
# Not: ["Hello,", "World!", "How's", "it", "going?"]

# Solution: clean first, then split
import re

def tokenize(text):
    text = text.lower()                        # Normalize case
    text = re.sub(r'[^a-z0-9\s]', '', text)   # Remove punctuation
    return text.split()                         # Split on whitespace

tokenize("Hello, World! It's 2024.")
# ["hello", "world", "its", "2024"]
```

**Why tokenize?** ML models don't understand raw text. They need numbers.
Tokenization is the first step toward converting text to numbers.

---

## 3. TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF measures how important a word is to a document within a collection.

**The intuition:**
- A word that appears OFTEN in a document is probably important to that document (TF)
- But if it appears in EVERY document (like "the", "is"), it's not useful (IDF penalizes this)

**Term Frequency (TF):**
```python
# How often does a word appear in THIS document?
# TF(word) = count(word) / total_words_in_document

document = ["the", "cat", "sat", "on", "the", "mat"]
# TF("the") = 2/6 = 0.333
# TF("cat") = 1/6 = 0.167
```

**Inverse Document Frequency (IDF):**
```python
import math

# How rare is this word ACROSS ALL documents?
# IDF(word) = log(total_documents / documents_containing_word)

docs = [["the", "cat"], ["the", "dog"], ["a", "bird"]]
# IDF("the") = log(3/2) = 0.405  (appears in 2 of 3 docs - common)
# IDF("cat") = log(3/1) = 1.099  (appears in 1 of 3 docs - rare, more important)
```

**TF-IDF = TF * IDF**
```
TF-IDF("the", doc1) = 0.333 * 0.405 = 0.135  (common word, low score)
TF-IDF("cat", doc1) = 0.167 * 1.099 = 0.183  (rarer word, higher score)
```

**Why this matters:** TF-IDF is still used for search, document similarity, and as
features for classification. It's also the conceptual foundation for embeddings.

---

## 4. Text Cleaning for ML

Real-world text is messy. Before processing, you need to clean it.

```python
import re

text = """
    Visit https://example.com or email info@test.com
    for more info!!!

    Call us at 555-1234.   Extra   spaces   everywhere.
"""

# Step 1: Remove URLs
text = re.sub(r'https?://\S+', '', text)

# Step 2: Remove email addresses
text = re.sub(r'\S+@\S+\.\S+', '', text)

# Step 3: Remove non-ASCII characters (emojis, special chars)
text = text.encode('ascii', errors='ignore').decode('ascii')

# Step 4: Collapse multiple spaces/newlines into single space
text = re.sub(r'\s+', ' ', text)

# Step 5: Strip leading/trailing whitespace
text = text.strip()

# Result: "Visit or email for more info!!! Call us at 555-1234. Extra spaces everywhere."
```

**Common regex patterns to know:**
```python
r'https?://\S+'      # URLs
r'\S+@\S+\.\S+'      # Email addresses
r'[^a-zA-Z0-9\s]'    # Non-alphanumeric (except spaces)
r'\s+'                # One or more whitespace characters
r'\d+'                # One or more digits
```

---

## 5. Text Chunking for RAG

When documents are too long for embedding models, you split them into chunks.

```
Long document (5000 words)
    |
    v
Chunk 1: words 1-200
Chunk 2: words 151-350     <-- overlap!
Chunk 3: words 301-500
...
```

**Why overlap?** If an important sentence is split between two chunks,
the overlap ensures it's fully captured in at least one.

```python
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    step = chunk_size - overlap  # How far to advance each step

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks
```

---

## 6. Feature Engineering

Feature engineering = creating new useful columns from raw data.

### One-Hot Encoding
Convert categories to numbers:
```python
# Raw: ["cat", "dog", "cat", "bird"]
# Encoded:
#        bird  cat  dog
#   0:    0     1    0
#   1:    0     0    1
#   2:    0     1    0
#   3:    1     0    0

import numpy as np

def one_hot_encode(labels):
    unique = sorted(set(labels))
    mapping = {label: i for i, label in enumerate(unique)}
    encoded = np.zeros((len(labels), len(unique)))
    for i, label in enumerate(labels):
        encoded[i, mapping[label]] = 1
    return encoded, unique
```

### Min-Max Scaling
Scale features to [0, 1] range:
```python
# Why? Features on different scales confuse models.
# Age: 20-80, Income: 20000-200000
# After scaling: both are 0-1

def min_max_scale(X):
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    return (X - col_min) / (col_max - col_min)
```

### Polynomial Features
Create new features from existing ones:
```python
# If X = [2], degree=3:
# Result = [2, 4, 8]  (X, X^2, X^3)
# This lets linear models learn non-linear patterns

def polynomial_features(X, degree=2):
    return np.hstack([X**d for d in range(1, degree+1)])
```

### Handling Missing Values
```python
# Strategy 1: Fill with mean
col_mean = np.nanmean(X[:, col])  # nanmean ignores NaN
X[np.isnan(X[:, col]), col] = col_mean

# Strategy 2: Fill with median (better for skewed data)
col_median = np.nanmedian(X[:, col])

# Strategy 3: Fill with zero
X[np.isnan(X[:, col]), col] = 0
```

---

## 7. Regular Expressions Quick Reference

Regex is essential for text processing. Know these patterns:

```python
import re

# re.sub(pattern, replacement, string) - find and replace
re.sub(r'\d+', 'NUM', "I have 3 cats and 2 dogs")
# "I have NUM cats and NUM dogs"

# re.findall(pattern, string) - find all matches
re.findall(r'\d+', "ages: 25, 30, 35")
# ['25', '30', '35']

# re.search(pattern, string) - find first match
match = re.search(r'(\d{4})-(\d{2})-(\d{2})', "Date: 2024-01-15")
match.group(0)  # "2024-01-15"
match.group(1)  # "2024"

# Common patterns
r'\d'        # Any digit
r'\w'        # Any word character (letter, digit, underscore)
r'\s'        # Any whitespace
r'\S'        # Any NON-whitespace
r'.'         # Any character
r'+'         # One or more
r'*'         # Zero or more
r'?'         # Zero or one
r'^'         # Start of string
r'$'         # End of string
r'[abc]'     # Any of a, b, or c
r'[^abc]'    # Any character EXCEPT a, b, c
```

---

## Now Try the Problems

1. `p1_text_preprocessing.py` - tokenize, TF-IDF, text cleaning, chunking
2. `p2_feature_engineering.py` - one-hot encode, scaling, polynomial features, missing values

```bash
pytest 04_data_processing/ -v
```
