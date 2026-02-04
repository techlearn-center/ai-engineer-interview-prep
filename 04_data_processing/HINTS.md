# Hints - Data Processing

## P1: Text Preprocessing

### tokenize
- `text.lower()` for case normalization
- `re.sub(r'[^a-z0-9\s]', '', text)` removes special characters
- `text.split()` splits on whitespace

### compute_tf
- `from collections import Counter` to count words
- TF = count / total_words
- Return a dict: `{word: count/total for word, count in counter.items()}`

### compute_idf
- For each document, convert to `set()` to get unique words
- Count how many documents each word appears in
- IDF = `math.log(total_docs / doc_freq)`

### compute_tfidf
- Combine: tokenize -> compute TF per doc -> compute IDF -> multiply
- TF-IDF(word, doc) = TF(word, doc) * IDF(word)

### clean_for_embedding
- URL regex: `r'https?://\S+'`
- Email regex: `r'\S+@\S+\.\S+'`
- Remove non-ASCII: `text.encode('ascii', errors='ignore').decode('ascii')`
- Collapse whitespace: `re.sub(r'\s+', ' ', text).strip()`

### chunk_text
- Split into words: `words = text.split()`
- Step size = `chunk_size - overlap`
- Loop: `for i in range(0, len(words), step)`
- Each chunk: `" ".join(words[i:i+chunk_size])`

## P2: Feature Engineering

### one_hot_encode
- Get sorted unique labels: `sorted(set(labels))`
- Create mapping: `{label: index for index, label in enumerate(unique)}`
- Create zero matrix, set 1 at the right positions

### min_max_scale
- Per column: `(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))`
- Handle constant columns (max == min): set range to 1 to avoid division by zero

### add_polynomial_features
- For each degree d from 1 to degree: compute `X ** d`
- Stack them horizontally: `np.hstack([X**1, X**2, X**3, ...])`

### handle_missing_values
- Find NaN: `np.isnan(X[:, col])`
- Compute stats ignoring NaN: `np.nanmean()`, `np.nanmedian()`
- Replace: `X[mask, col] = fill_value`
