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
| File Operations | `p4_file_operations.py` | read/write text, JSON, CSV, paths, logging, config files |

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

### 07 - GCP AI Platform
| Problem | File | Topics |
|---------|------|--------|
| GCS Operations | `p1_gcs_operations.py` | upload/download JSON & CSV, list blobs, organize training data |
| BigQuery & BQML | `p2_bigquery_ml.py` | SQL queries, feature stats, BigQuery ML CREATE MODEL |
| Vertex AI Pipeline | `p3_vertex_ai_pipeline.py` | training config, model registry, endpoints, A/B testing, pipeline design |

### 08 - LangChain (Zero to Hero)
| Problem | File | Topics |
|---------|------|--------|
| LangChain Basics | `p1_langchain_basics.py` | PromptTemplate, ChatPromptTemplate, Chains, OutputParsers, Memory |
| RAG Application | `p2_rag_application.py` | TextSplitter, VectorStore, Retriever, RAGChain, DocumentLoaders |

### 09 - MCP (Model Context Protocol)
| Problem | File | Topics |
|---------|------|--------|
| MCP Basics | `p1_mcp_basics.py` | ToolDefinition, ResourceDefinition, JSON-RPC, ToolRegistry, ResourceStore |
| MCP Server | `p2_mcp_server.py` | Complete MCP server: tools, resources, prompts, routing |

## Learning Guides (LEARN_FIRST.md)

Every topic has a LEARN_FIRST.md that teaches the concepts from scratch before you code:

| Topic | Guide | What you'll learn |
|-------|-------|-------------------|
| Python Fundamentals | `01_python_fundamentals/LEARN_FIRST.md` | Dicts, two-sum pattern, comprehensions, generators, decorators |
| NumPy & Pandas | `02_numpy_pandas/LEARN_FIRST.md` | Arrays, vectorization, broadcasting, DataFrames, groupby |
| ML from Scratch | `03_ml_from_scratch/LEARN_FIRST.md` | Gradient descent, loss functions, how training works |
| Data Processing | `04_data_processing/LEARN_FIRST.md` | Tokenization, TF-IDF, regex, feature engineering |
| APIs & FastAPI | `05_api_serving/LEARN_FIRST.md` | HTTP basics, REST, Pydantic, building endpoints |
| LLM & RAG | `06_llm_rag_patterns/LEARN_FIRST.md` | Embeddings, vector stores, cosine similarity, RAG pipeline |
| GCP for AI | `07_gcp_ai_platform/LEARN_FIRST.md` | GCS, BigQuery, Vertex AI, Cloud Run, IAM, architecture |
| LangChain | `08_langchain/LEARN_FIRST.md` | LangChain, LangGraph, LangSmith - complete zero-to-hero guide |
| MCP | `09_mcp/LEARN_FIRST.md` | Model Context Protocol - connecting AI to tools and data |

## Recommended Order (2-Day Plan)

### Day 1: Foundations + GCP Concepts
1. `01_python_fundamentals/p1_data_structures.py`
2. `01_python_fundamentals/p2_comprehensions_generators.py`
3. `01_python_fundamentals/p3_decorators_closures.py`
4. `02_numpy_pandas/p1_numpy_operations.py`
5. `02_numpy_pandas/p2_pandas_wrangling.py`
6. Read `07_gcp_ai_platform/LEARN_FIRST.md` (evening study)

### Day 2: ML, AI & Cloud
1. `03_ml_from_scratch/p1_linear_regression.py`
2. `03_ml_from_scratch/p2_logistic_regression.py`
3. `04_data_processing/p1_text_preprocessing.py`
4. `05_api_serving/p1_fastapi_inference.py` (read LEARN_FIRST.md first)
5. `06_llm_rag_patterns/p1_rag_pipeline.py` (read LEARN_FIRST.md first)
6. `07_gcp_ai_platform/p1_gcs_operations.py`
7. `07_gcp_ai_platform/p2_bigquery_ml.py`
8. `07_gcp_ai_platform/p3_vertex_ai_pipeline.py`

## Tips for the Interview

- **Talk out loud** as you code - explain your thought process
- **Start with pseudocode** before writing the real solution
- **Handle edge cases** - interviewers watch for this
- **Use meaningful variable names** - not `x`, `y`, `temp`
- **Know your complexity** - be ready to discuss Big-O
- **Test as you go** - run pytest frequently

## Each Topic Folder Contains

```
topic_folder/
  LEARN_FIRST.md       <-- Teaches the concept from scratch (read first if new to topic)
  HINTS.md             <-- Step-by-step hints (read before looking at solutions)
  p1_xxx.py            <-- Problem file (you write code here)
  tests/test_p1_xxx.py <-- Tests (run pytest to check your answers)
  solutions/s1_xxx.py  <-- Full working solutions (last resort)
```

## Cheatsheets (Quick Reference)

Review these the night before or right before your interview:

| Cheatsheet | What's in it |
|-----------|-------------|
| `cheatsheets/python-syntax.md` | All Python syntax: data structures, comprehensions, classes, regex, NumPy, Pandas |
| `cheatsheets/ml-ai-glossary.md` | 150+ terms: ML concepts, model types, metrics, LLM/RAG, MLOps, math |
| `cheatsheets/gcp-cheatsheet.md` | GCP services, CLI commands, Python SDK patterns, architecture diagrams |

## Solutions

Try to solve each problem yourself first! If you get stuck:
1. Read the HINTS.md file for the topic
2. Check the test files to understand expected behavior
3. Look at solutions/ only after giving it a real attempt
