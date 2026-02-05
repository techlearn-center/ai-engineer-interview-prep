# ML & AI Glossary

Quick reference for terms that come up in AI engineer interviews.
Organized by topic so you can review the section relevant to your conversation.

---

## Core ML Concepts

| Term | Definition | Example |
|------|-----------|---------|
| **Supervised Learning** | Learning from labeled data (input → known output) | Predicting house prices from features |
| **Unsupervised Learning** | Finding patterns in unlabeled data | Grouping customers by behavior (clustering) |
| **Feature** | An input variable to the model | Age, income, number of clicks |
| **Label / Target** | The value you're trying to predict | Price, churn (yes/no), sentiment |
| **Training Set** | Data used to train the model | 80% of your data |
| **Validation Set** | Data used to tune hyperparameters | 10% of your data |
| **Test Set** | Data used for final evaluation (never seen during training) | 10% of your data |
| **Epoch** | One complete pass through the training data | Training for 100 epochs |
| **Batch Size** | Number of samples processed before updating weights | batch_size=32 |
| **Learning Rate** | Step size for weight updates | lr=0.001 |
| **Loss Function** | Measures how wrong the model is | MSE, cross-entropy |
| **Gradient** | Direction of steepest increase of the loss | Used to update weights |
| **Gradient Descent** | Algorithm that minimizes loss by following gradients | weights -= lr * gradient |
| **Overfitting** | Model memorizes training data, fails on new data | 99% train accuracy, 60% test accuracy |
| **Underfitting** | Model is too simple, misses patterns | 50% accuracy on both train and test |
| **Regularization** | Technique to prevent overfitting | L1/L2 penalty, dropout |
| **Hyperparameter** | Setting you choose (not learned) | Learning rate, number of layers |
| **Cross-Validation** | Splitting data multiple ways to get robust evaluation | 5-fold CV |

---

## Model Types

| Model | Type | Use Case | Key Idea |
|-------|------|----------|----------|
| **Linear Regression** | Supervised | Predict continuous values | y = wX + b |
| **Logistic Regression** | Supervised | Binary classification | y = sigmoid(wX + b) |
| **Decision Tree** | Supervised | Classification/regression | If-else rules learned from data |
| **Random Forest** | Supervised | Classification/regression | Many decision trees voting together |
| **XGBoost / Gradient Boosting** | Supervised | Tabular data (Kaggle winner) | Trees that correct each other's errors |
| **K-Means** | Unsupervised | Clustering | Assign points to nearest centroid |
| **KNN** | Supervised | Classification | Predict based on K nearest neighbors |
| **SVM** | Supervised | Classification | Find the best separating boundary |
| **Neural Network** | Supervised | Any (images, text, tabular) | Layers of learned transformations |
| **CNN** | Supervised | Images | Learns spatial patterns with filters |
| **RNN / LSTM** | Supervised | Sequences (time series, text) | Remembers previous inputs |
| **Transformer** | Supervised | NLP, vision, everything | Attention mechanism, parallelizable |

---

## Metrics

| Metric | Formula / Meaning | When to Use |
|--------|-------------------|-------------|
| **Accuracy** | correct / total | Balanced classes |
| **Precision** | TP / (TP + FP) | When false positives are costly (spam filter) |
| **Recall** | TP / (TP + FN) | When false negatives are costly (disease detection) |
| **F1 Score** | 2 * (P * R) / (P + R) | Balance precision and recall |
| **MSE** | mean((y - y_pred)^2) | Regression |
| **RMSE** | sqrt(MSE) | Regression (same units as target) |
| **MAE** | mean(abs(y - y_pred)) | Regression (robust to outliers) |
| **R^2** | 1 - (SS_res / SS_tot) | Regression (0-1, how much variance explained) |
| **AUC-ROC** | Area under ROC curve | Binary classification (threshold-independent) |
| **Log Loss** | -mean(y*log(p) + (1-y)*log(1-p)) | Probabilistic classification |

**Confusion Matrix:**
```
                 Predicted Positive    Predicted Negative
Actual Positive       TP                    FN
Actual Negative       FP                    TN
```

---

## Deep Learning & LLM Terms

| Term | Definition |
|------|-----------|
| **Embedding** | Dense vector representation of text/items that captures meaning |
| **Attention** | Mechanism that lets model focus on relevant parts of input |
| **Self-Attention** | Each token attends to all other tokens in the sequence |
| **Transformer** | Architecture using self-attention (basis for GPT, BERT, etc.) |
| **Fine-Tuning** | Taking a pre-trained model and training further on your data |
| **Transfer Learning** | Using knowledge from one task to help with another |
| **Pre-training** | Training on large general data before specializing |
| **Tokenizer** | Converts text to token IDs for the model (BPE, WordPiece) |
| **Token** | A sub-word unit (not always a full word) |
| **Context Window** | Maximum number of tokens a model can process at once |
| **Temperature** | Controls randomness in generation (0=deterministic, 1=creative) |
| **Top-k / Top-p** | Sampling strategies for text generation |
| **Prompt Engineering** | Designing inputs to get better outputs from LLMs |
| **Few-Shot Learning** | Providing examples in the prompt to guide the model |
| **Zero-Shot** | Asking the model to do a task with no examples |
| **Hallucination** | Model generates false or fabricated information |
| **Grounding** | Providing factual context to reduce hallucination |
| **RLHF** | Reinforcement Learning from Human Feedback (how ChatGPT was trained) |
| **LoRA / QLoRA** | Efficient fine-tuning methods (low-rank adaptation) |
| **Quantization** | Reducing model precision (FP32 → INT8) to save memory/speed |

---

## RAG (Retrieval-Augmented Generation)

| Term | Definition |
|------|-----------|
| **RAG** | Retrieve relevant docs, stuff into prompt, then generate answer |
| **Vector Store / Vector DB** | Database optimized for similarity search on embeddings |
| **Cosine Similarity** | Measures angle between two vectors (-1 to 1). 1 = identical |
| **Semantic Search** | Search by meaning (using embeddings) vs. keyword matching |
| **Chunking** | Splitting long documents into smaller pieces for embedding |
| **Overlap** | Shared words between consecutive chunks (prevents lost context) |
| **Top-k Retrieval** | Return the k most similar documents |
| **Re-ranking** | Second pass to improve retrieval quality with a more powerful model |
| **Hybrid Search** | Combining vector search + keyword search for better results |
| **HyDE** | Hypothetical Document Embeddings - generate fake answer, embed it, search |
| **Context Window Stuffing** | Filling the prompt with as much relevant context as possible |
| **Citation / Sourcing** | Returning which documents the answer came from |

**Popular Tools:**
| Category | Options |
|----------|---------|
| Embedding Models | OpenAI text-embedding-3, Cohere embed-v3, sentence-transformers |
| Vector DBs | Pinecone, ChromaDB, Weaviate, Qdrant, pgvector, FAISS |
| Frameworks | LangChain, LlamaIndex, Haystack |

---

## MLOps & Production

| Term | Definition |
|------|-----------|
| **MLOps** | DevOps practices applied to ML (CI/CD for models) |
| **Model Registry** | Versioned storage for trained models |
| **Model Serving** | Deploying a model to handle prediction requests |
| **Online Prediction** | Real-time, single request inference (API) |
| **Batch Prediction** | Process a large dataset of predictions at once |
| **A/B Testing** | Route traffic between model versions to compare performance |
| **Canary Deployment** | New model gets small % of traffic, increase if good |
| **Blue/Green Deployment** | Two identical environments, switch traffic between them |
| **Data Drift** | Input data distribution changes over time |
| **Model Drift / Decay** | Model performance degrades as data changes |
| **Feature Store** | Centralized storage for computed features (reuse across models) |
| **Pipeline** | Automated sequence: data → preprocess → train → evaluate → deploy |
| **CI/CD** | Continuous Integration / Continuous Deployment |
| **Containerization** | Packaging app + dependencies in Docker for consistent deployment |
| **Orchestration** | Managing complex workflows (Airflow, Kubeflow, Vertex AI Pipelines) |

---

## Data Processing

| Term | Definition |
|------|-----------|
| **ETL** | Extract, Transform, Load - moving and cleaning data |
| **Feature Engineering** | Creating useful input variables from raw data |
| **One-Hot Encoding** | Converting categories to binary vectors |
| **Normalization** | Scaling data to [0, 1] range |
| **Standardization** | Scaling to mean=0, std=1 (Z-score) |
| **Imputation** | Filling in missing values (mean, median, mode, model-based) |
| **Tokenization** | Splitting text into words or sub-word units |
| **TF-IDF** | Term Frequency * Inverse Document Frequency (word importance) |
| **Stop Words** | Common words removed before processing ("the", "is", "a") |
| **Stemming** | Reducing words to root form ("running" → "run") |
| **Lemmatization** | Like stemming but linguistically correct ("better" → "good") |
| **Bag of Words** | Representing text as word count vectors |
| **N-gram** | Sequence of n consecutive words ("New York" = bigram) |

---

## Math You Should Know

| Concept | Formula | When it comes up |
|---------|---------|-----------------|
| **Dot Product** | a . b = Σ(ai * bi) | Predictions, similarity |
| **Matrix Multiply** | C = A @ B | Neural networks, linear models |
| **Sigmoid** | 1 / (1 + e^(-x)) | Logistic regression, neural networks |
| **Softmax** | e^xi / Σ(e^xj) | Multi-class classification, attention |
| **MSE** | (1/n) Σ(y - y_pred)^2 | Regression loss |
| **Cross-Entropy** | -Σ(y * log(p)) | Classification loss |
| **Cosine Similarity** | (a . b) / (\|\|a\|\| * \|\|b\|\|) | Embedding similarity |
| **L2 Norm** | sqrt(Σ(xi^2)) | Vector magnitude, regularization |
| **Derivative / Gradient** | Rate of change of a function | Gradient descent |
| **Chain Rule** | d(f(g(x)))/dx = f'(g(x)) * g'(x) | Backpropagation |

---

## API / Web Terms

| Term | Definition |
|------|-----------|
| **REST API** | Web API using HTTP methods (GET, POST, PUT, DELETE) |
| **Endpoint** | A URL path that handles a specific request |
| **JSON** | Data format used by APIs (like a Python dict) |
| **HTTP Status Codes** | 200=OK, 400=Bad Request, 404=Not Found, 500=Server Error |
| **Request Body** | Data sent with POST/PUT (usually JSON) |
| **Query Parameter** | Data in the URL: `/search?q=python&limit=10` |
| **Path Parameter** | Data in the URL path: `/models/123` |
| **Authentication** | Verifying identity (API keys, OAuth, JWT) |
| **Rate Limiting** | Restricting number of requests per time period |
| **Middleware** | Code that runs before/after every request |
| **Serialization** | Converting objects to JSON (and back) |
| **Pydantic** | Python library for data validation and serialization |
| **FastAPI** | Modern Python web framework for building APIs |
| **Uvicorn** | ASGI server that runs FastAPI apps |
