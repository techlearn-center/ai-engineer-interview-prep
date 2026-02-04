# AI Engineer Interview Prep - Coding Practice

Practice problems for AI/ML engineer coding interviews. Each problem has **tests** so you know immediately if your solution is correct.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Practice

1. Open a problem file (e.g., `01_python_fundamentals/p1_data_structures.py`)
2. Read the docstrings - they explain the problem and give examples
3. Write your solution where it says `# YOUR CODE HERE`
4. Run the tests to check:

```bash
# Run tests for a specific problem
pytest 01_python_fundamentals/tests/test_p1_data_structures.py -v

# Run ALL tests for a topic
pytest 01_python_fundamentals/ -v

# Run ALL tests
pytest -v
```

## Topics & Problems

### 01 - Python Fundamentals
| Problem | File | Topics |
|---------|------|--------|
| Data Structures | `p1_data_structures.py` | dict/hashmap, two-sum, anagrams, LRU cache |
| Comprehensions & Generators | `p2_comprehensions_generators.py` | list comp, generators, sliding window |
| Decorators & Closures | `p3_decorators_closures.py` | timer, retry, memoize, type validation |

### 02 - NumPy & Pandas
| Problem | File | Topics |
|---------|------|--------|
| NumPy Operations | `p1_numpy_operations.py` | normalize, dot product, cosine sim, softmax, broadcasting |
| Pandas Wrangling | `p2_pandas_wrangling.py` | groupby, rolling avg, pivot tables, data cleaning |

### 03 - ML from Scratch
| Problem | File | Topics |
|---------|------|--------|
| Linear Regression | `p1_linear_regression.py` | gradient descent, MSE, prediction |
| Logistic Regression | `p2_logistic_regression.py` | sigmoid, binary cross-entropy, accuracy |
| K-Means Clustering | `p3_kmeans.py` | centroid init, assignment, update, inertia |

### 04 - Data Processing
| Problem | File | Topics |
|---------|------|--------|
| Text Preprocessing | `p1_text_preprocessing.py` | tokenize, TF-IDF, text cleaning, chunking |
| Feature Engineering | `p2_feature_engineering.py` | one-hot encode, scaling, polynomial features, missing values |

### 05 - API / Model Serving
| Problem | File | Topics |
|---------|------|--------|
| FastAPI Inference | `p1_fastapi_inference.py` | Pydantic models, REST endpoints, model registry |

### 06 - LLM & RAG Patterns
| Problem | File | Topics |
|---------|------|--------|
| RAG Pipeline | `p1_rag_pipeline.py` | vector store, similarity search, prompt building, chat messages |

## Recommended Order (2-Day Plan)

### Day 1: Foundations
1. `01_python_fundamentals/p1_data_structures.py`
2. `01_python_fundamentals/p2_comprehensions_generators.py`
3. `01_python_fundamentals/p3_decorators_closures.py`
4. `02_numpy_pandas/p1_numpy_operations.py`
5. `02_numpy_pandas/p2_pandas_wrangling.py`

### Day 2: ML & AI-Specific
1. `03_ml_from_scratch/p1_linear_regression.py`
2. `03_ml_from_scratch/p2_logistic_regression.py`
3. `03_ml_from_scratch/p3_kmeans.py`
4. `04_data_processing/p1_text_preprocessing.py`
5. `05_api_serving/p1_fastapi_inference.py`
6. `06_llm_rag_patterns/p1_rag_pipeline.py`

## Tips for the Interview

- **Talk out loud** as you code - explain your thought process
- **Start with pseudocode** before writing the real solution
- **Handle edge cases** - interviewers watch for this
- **Use meaningful variable names** - not `x`, `y`, `temp`
- **Know your complexity** - be ready to discuss Big-O
- **Test as you go** - run pytest frequently

## Solutions

Try to solve each problem yourself first! If you get stuck, the test files show the expected behavior. You can also ask Claude Code for hints.
