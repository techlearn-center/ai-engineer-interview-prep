# Learn: LangChain from Zero to Hero

This guide assumes you know NOTHING about LangChain. We'll build up piece by piece
until you can build a complete RAG chatbot.

---

## 1. What is LangChain?

LangChain is a Python library that makes it easier to build applications with LLMs.

**Without LangChain:**
```python
import openai

# You manually handle:
# - Formatting prompts
# - Managing chat history
# - Connecting to vector databases
# - Chaining multiple LLM calls
# - Error handling and retries
# - Switching between different LLM providers
# ... lots of repetitive code
```

**With LangChain:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Cleaner, modular, reusable components
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = prompt | llm
response = chain.invoke({"topic": "Python"})
```

**Key idea:** LangChain provides building blocks (prompts, models, chains, memory, tools)
that you can snap together like Lego.

---

## 2. Installation & Setup

```bash
pip install langchain langchain-openai langchain-community chromadb
```

**API Key Setup:**
```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY="sk-..."

# Option 2: In code (not recommended for production)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

For our practice problems, we'll use **mock LLMs** so you don't need an API key.

---

## 3. Core Concept #1: Models (LLMs and Chat Models)

A **Model** is your connection to an LLM.

```python
from langchain_openai import ChatOpenAI

# Create a model
llm = ChatOpenAI(
    model="gpt-4",           # or "gpt-3.5-turbo"
    temperature=0.7,          # 0 = deterministic, 1 = creative
)

# Simple invocation
response = llm.invoke("What is Python?")
print(response.content)  # The text response
```

**Types of models:**
- **LLM:** Old-style, takes a string, returns a string
- **Chat Model:** Takes messages (system/user/assistant), returns a message

Most modern code uses Chat Models.

---

## 4. Core Concept #2: Prompt Templates

Instead of hardcoding prompts, use **templates** with variables:

```python
from langchain_core.prompts import ChatPromptTemplate

# Create a template with a variable
prompt = ChatPromptTemplate.from_template(
    "You are an expert in {topic}. Explain it to a beginner in 3 sentences."
)

# Format with actual values
formatted = prompt.format(topic="machine learning")
# "You are an expert in machine learning. Explain it to a beginner in 3 sentences."

# Or invoke directly to get a message object
messages = prompt.invoke({"topic": "machine learning"})
```

**Chat prompts with roles:**
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}."),
    ("user", "{question}"),
])

messages = prompt.invoke({"role": "teacher", "question": "What is an API?"})
# [SystemMessage(...), HumanMessage(...)]
```

---

## 5. Core Concept #3: Chains (The Pipe Operator)

A **Chain** connects components together. The output of one becomes the input of the next.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Components
prompt = ChatPromptTemplate.from_template("Explain {topic} simply.")
llm = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()   # Extracts the string content from the response

# Chain them with the pipe operator |
chain = prompt | llm | parser

# Invoke the whole chain
result = chain.invoke({"topic": "recursion"})
print(result)  # A string explanation of recursion
```

**What happens:**
```
{"topic": "recursion"}
    → prompt (formats the template)
    → llm (sends to GPT-4)
    → parser (extracts text from response)
    → "Recursion is when a function calls itself..."
```

---

## 6. Core Concept #4: Output Parsers

Convert LLM output (unstructured text) into structured data:

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the structure you want
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="One sentence summary")

parser = JsonOutputParser(pydantic_object=MovieReview)

# Include parsing instructions in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie review details.\n{format_instructions}"),
    ("user", "{review}"),
])

# Add format instructions to the prompt
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

result = chain.invoke({"review": "Inception was mind-blowing! 9/10, must watch!"})
# {"title": "Inception", "rating": 9, "summary": "A mind-blowing must-watch film."}
```

---

## 7. Core Concept #5: Document Loaders

Load data from various sources:

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
)

# Plain text
loader = TextLoader("notes.txt")
docs = loader.load()  # Returns list of Document objects

# PDF
loader = PyPDFLoader("report.pdf")
docs = loader.load()

# CSV (each row becomes a document)
loader = CSVLoader("data.csv")
docs = loader.load()

# Web page
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# Each Document has:
doc = docs[0]
doc.page_content   # The text content
doc.metadata       # {"source": "notes.txt", "page": 0, ...}
```

---

## 8. Core Concept #6: Text Splitters

LLMs have limited context windows. Split long documents into chunks:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Max characters per chunk
    chunk_overlap=50,     # Overlap between chunks
    separators=["\n\n", "\n", " ", ""],  # Try to split on these
)

# Split a document
text = "Your very long document text here..."
chunks = splitter.split_text(text)

# Or split Document objects
docs = loader.load()
split_docs = splitter.split_documents(docs)
```

**Why overlap?** If important info spans two chunks, overlap ensures it's captured.

---

## 9. Core Concept #7: Embeddings

Convert text to vectors (numbers) that capture meaning:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single text
vector = embeddings.embed_query("What is machine learning?")
# Returns: [0.012, -0.045, 0.098, ...] (1536 numbers)

# Embed multiple texts
vectors = embeddings.embed_documents([
    "Python is great",
    "JavaScript is popular",
    "Machine learning is powerful",
])
# Returns: list of 3 vectors
```

**Key insight:** Similar texts have similar vectors. This enables semantic search.

---

## 10. Core Concept #8: Vector Stores

Store and search embeddings:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create a vector store from documents
docs = [...]  # Your Document objects
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./my_vectorstore",  # Optional: save to disk
)

# Search for similar documents
results = vectorstore.similarity_search(
    "How do I train a model?",
    k=3,  # Return top 3 most similar
)

# Results is a list of Document objects, sorted by similarity
for doc in results:
    print(doc.page_content)
```

**Popular vector stores:**
- **Chroma:** Great for local development
- **Pinecone:** Cloud-hosted, scalable
- **Weaviate:** Open source, feature-rich
- **FAISS:** Fast, by Facebook, local only

---

## 11. Core Concept #9: Retrievers

A **Retriever** finds relevant documents for a query:

```python
# Convert vector store to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 5},    # Return top 5
)

# Use the retriever
docs = retriever.invoke("What is gradient descent?")
```

The retriever is what you plug into a RAG chain.

---

## 12. Building a RAG Chain

Now let's put it all together - a complete RAG (Retrieval-Augmented Generation) pipeline:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Set up components
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4")

# 2. Create the prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you don't know, say "I don't know."

Context: {context}

Question: {question}

Answer:
""")

# 3. Helper to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Build the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Use it!
answer = rag_chain.invoke("What is the refund policy?")
print(answer)
```

**What happens:**
```
"What is the refund policy?"
    → retriever finds 3 relevant docs
    → format_docs joins them into a string
    → prompt combines context + question
    → llm generates answer based on context
    → parser extracts the text
    → "According to the policy, refunds are available within 30 days..."
```

---

## 13. Memory (Chat History)

For multi-turn conversations:

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create a simple in-memory history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Your base chain
chain = prompt | llm | parser

# Wrap with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# Use with a session ID
config = {"configurable": {"session_id": "user_123"}}
response1 = chain_with_history.invoke({"question": "My name is Alice"}, config=config)
response2 = chain_with_history.invoke({"question": "What's my name?"}, config=config)
# response2 will know your name is Alice!
```

---

## 14. Common Patterns

### Pattern 1: Simple Q&A
```python
chain = ChatPromptTemplate.from_template("Answer: {question}") | llm | StrOutputParser()
```

### Pattern 2: Summarization
```python
chain = ChatPromptTemplate.from_template(
    "Summarize this in 3 bullet points:\n\n{text}"
) | llm | StrOutputParser()
```

### Pattern 3: Multi-step (Sequential Chains)
```python
# Step 1: Translate
translate = ChatPromptTemplate.from_template(
    "Translate to English: {text}"
) | llm | StrOutputParser()

# Step 2: Summarize
summarize = ChatPromptTemplate.from_template(
    "Summarize: {text}"
) | llm | StrOutputParser()

# Chain them
full_chain = {"text": translate} | summarize
result = full_chain.invoke({"text": "Bonjour, comment ça va?"})
```

### Pattern 4: RAG with Sources
```python
def rag_with_sources(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = chain.invoke({"context": context, "question": question})
    sources = [doc.metadata["source"] for doc in docs]
    return {"answer": answer, "sources": sources}
```

---

## 15. Debugging Tips

```python
# See what's happening in the chain
from langchain.globals import set_debug
set_debug(True)  # Prints all intermediate steps

# Or use verbose mode on specific components
llm = ChatOpenAI(verbose=True)
```

---

## 16. Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                     YOUR RAG APP                         │
│                                                          │
│  ┌──────────┐   ┌───────────┐   ┌───────────────────┐   │
│  │ Document │ → │ Text      │ → │ Embeddings        │   │
│  │ Loaders  │   │ Splitters │   │ (OpenAI, Cohere)  │   │
│  └──────────┘   └───────────┘   └─────────┬─────────┘   │
│                                            │             │
│                                            ▼             │
│                                   ┌────────────────┐     │
│                                   │ Vector Store   │     │
│                                   │ (Chroma, etc.) │     │
│                                   └───────┬────────┘     │
│                                           │              │
│  ┌──────────┐                             │              │
│  │ User     │ ──────────────────────────────────┐        │
│  │ Query    │                             │     │        │
│  └──────────┘                             ▼     │        │
│                                   ┌────────────┐│        │
│                                   │ Retriever  ││        │
│                                   └─────┬──────┘│        │
│                                         │       │        │
│                                         ▼       ▼        │
│                                   ┌────────────────┐     │
│                                   │ Prompt         │     │
│                                   │ Template       │     │
│                                   └───────┬────────┘     │
│                                           │              │
│                                           ▼              │
│                                   ┌────────────────┐     │
│                                   │ LLM (GPT-4)    │     │
│                                   └───────┬────────┘     │
│                                           │              │
│                                           ▼              │
│                                   ┌────────────────┐     │
│                                   │ Output Parser  │     │
│                                   └───────┬────────┘     │
│                                           │              │
│                                           ▼              │
│                                      [Answer]            │
└─────────────────────────────────────────────────────────┘
```

---

---

# Part 2: LangGraph

LangGraph is for building **stateful, multi-actor applications** with LLMs.
Think: AI agents that can loop, branch, and use tools.

## 17. What is LangGraph?

LangChain chains are linear: A → B → C → done.

LangGraph adds:
- **Loops:** Keep running until a condition is met
- **Branching:** Different paths based on decisions
- **State:** Remember information across steps
- **Human-in-the-loop:** Pause for human approval

```
LangChain:   Input → Step1 → Step2 → Step3 → Output  (linear)

LangGraph:   Input → Step1 → Decision?
                       ↓         ↓
                     Yes        No
                       ↓         ↓
                    Step2 ←──┐ Step3
                       ↓      │
                    Check ────┘  (loops back if needed)
                       ↓
                    Output
```

---

## 18. LangGraph Core Concepts

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Define your state (what gets passed between nodes)
class AgentState(TypedDict):
    messages: list
    current_step: str
    iteration_count: int

# 2. Define nodes (functions that process state)
def analyze(state: AgentState) -> AgentState:
    # Do something with state
    return {"messages": state["messages"] + ["Analyzed!"], "current_step": "decide"}

def decide(state: AgentState) -> AgentState:
    return {"current_step": "act" if len(state["messages"]) < 5 else "end"}

def act(state: AgentState) -> AgentState:
    return {"messages": state["messages"] + ["Acted!"], "current_step": "analyze"}

# 3. Define routing (which node to go to next)
def router(state: AgentState) -> str:
    if state["current_step"] == "end":
        return END
    return state["current_step"]

# 4. Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze", analyze)
workflow.add_node("decide", decide)
workflow.add_node("act", act)

# Add edges
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "decide")
workflow.add_conditional_edges("decide", router)
workflow.add_edge("act", "analyze")  # Loop back

# 5. Compile and run
app = workflow.compile()
result = app.invoke({"messages": [], "current_step": "", "iteration_count": 0})
```

---

## 19. LangGraph Agent Example

Here's a simple agent that can use tools:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define tools the agent can use
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

# Create the agent
llm = ChatOpenAI(model="gpt-4")
tools = [search, calculate]
agent = create_react_agent(llm, tools)

# Run it
result = agent.invoke({"messages": [("user", "What is 25 * 4?")]})
```

---

## 20. When to Use LangGraph

| Use Case | Tool |
|----------|------|
| Simple Q&A, summarization | LangChain (chains) |
| Linear pipeline (load → process → respond) | LangChain (chains) |
| Agent that uses tools | LangGraph |
| Complex workflows with loops | LangGraph |
| Multi-step reasoning (ReAct, CoT) | LangGraph |
| Human-in-the-loop approval | LangGraph |
| Parallel processing | LangGraph |

---

# Part 3: LangSmith

LangSmith is for **observability, testing, and debugging** LLM applications.

## 21. What is LangSmith?

LangSmith lets you:
- **Trace:** See every step of your chain (inputs, outputs, latency, tokens)
- **Debug:** Find where things went wrong
- **Evaluate:** Run test datasets and measure quality
- **Monitor:** Track production performance

```
Your LangChain App
       ↓
   (sends traces)
       ↓
┌──────────────────┐
│   LangSmith      │
│  ┌────────────┐  │
│  │ Traces     │  │ ← See every LLM call
│  │ Datasets   │  │ ← Store test cases
│  │ Evaluators │  │ ← Measure quality
│  │ Dashboards │  │ ← Monitor production
│  └────────────┘  │
└──────────────────┘
```

---

## 22. Setting Up LangSmith

```bash
pip install langsmith
```

```bash
# Get your API key from smith.langchain.com
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls__..."
export LANGCHAIN_PROJECT="my-project"  # Optional: organize by project
```

That's it! Your LangChain code automatically sends traces to LangSmith.

---

## 23. Viewing Traces

After running your chain, go to **smith.langchain.com**.

You'll see:
```
Run: "What is Python?"
├── ChatPromptTemplate (0.5ms)
│   └── Input: {"topic": "Python"}
│   └── Output: [SystemMessage, HumanMessage]
├── ChatOpenAI (1.2s)
│   └── Input: [messages]
│   └── Output: AIMessage("Python is...")
│   └── Tokens: 150 in, 200 out
└── StrOutputParser (0.1ms)
    └── Output: "Python is..."
```

**Super useful for:**
- Finding slow steps
- Seeing exact prompts sent to the LLM
- Debugging unexpected outputs

---

## 24. Creating Datasets for Evaluation

```python
from langsmith import Client

client = Client()

# Create a dataset
dataset = client.create_dataset("my-qa-dataset")

# Add examples
client.create_examples(
    inputs=[
        {"question": "What is Python?"},
        {"question": "What is JavaScript?"},
    ],
    outputs=[
        {"answer": "A programming language known for readability"},
        {"answer": "A language for web development"},
    ],
    dataset_id=dataset.id,
)
```

---

## 25. Running Evaluations

```python
from langsmith.evaluation import evaluate

def my_chain(inputs: dict) -> dict:
    # Your LangChain code here
    result = chain.invoke(inputs)
    return {"answer": result}

# Define how to grade outputs
def correctness(run, example) -> dict:
    # Compare run output to expected output
    predicted = run.outputs["answer"]
    expected = example.outputs["answer"]
    score = 1.0 if expected.lower() in predicted.lower() else 0.0
    return {"key": "correctness", "score": score}

# Run evaluation
results = evaluate(
    my_chain,
    data="my-qa-dataset",
    evaluators=[correctness],
)
```

---

## 26. LangSmith in Production

```python
from langsmith import traceable

# Wrap any function to trace it
@traceable(name="my_rag_pipeline")
def answer_question(question: str) -> str:
    docs = retriever.invoke(question)
    response = chain.invoke({"question": question, "context": docs})
    return response

# Traces show up in LangSmith automatically
answer_question("What is the return policy?")
```

**Key features for production:**
- **Feedback:** Collect user 👍/👎 on responses
- **Monitoring:** Set alerts for errors, latency spikes
- **A/B testing:** Compare different prompts/models

---

## 27. LangChain Ecosystem Summary

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **LangChain** | Building blocks for LLM apps | Core framework, always |
| **LangGraph** | Complex, stateful workflows | Agents, loops, branching |
| **LangSmith** | Observability & evaluation | Debugging, testing, production |
| **LangServe** | Deploy chains as APIs | When you need a REST API |

---

# Part 4: Testing & Evaluating LangChain Applications

Building LLM apps is one thing — knowing they work correctly is another.

## 28. GenAI Testing Tools Overview

| Category | Tools | Purpose |
|----------|-------|---------|
| **Evaluation Frameworks** | Ragas, DeepEval, Promptfoo | Measure output quality (faithfulness, relevance, hallucination) |
| **Observability & Tracing** | LangSmith, LangFuse, Arize Phoenix | Trace every step, debug failures, monitor production |
| **Load & Performance** | Locust, LiteLLM | Benchmark latency, throughput, and cost across providers |
| **Guardrails & Safety** | Guardrails AI, NeMo Guardrails, Rebuff | Validate output structure, enforce safety rails, detect prompt injection |
| **General / E2E** | pytest + DeepEval, Giskard, TruLens | Unit/integration tests, bias scanning, groundedness checks |

---

## 29. Testing RAG with Ragas

Ragas is the go-to framework for evaluating RAG pipelines:

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Prepare evaluation data
eval_data = Dataset.from_dict({
    "question": ["What is the refund policy?"],
    "answer": ["Refunds are available within 30 days of purchase."],
    "contexts": [["Our refund policy allows returns within 30 days. Items must be unused."]],
    "ground_truth": ["Refunds are available within 30 days."],
})

# Run evaluation
results = evaluate(
    eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.92, 'context_precision': 0.88, ...}
```

**Key Ragas metrics:**
- **Faithfulness:** Is the answer grounded in the retrieved context? (no hallucination)
- **Answer Relevancy:** Does the answer actually address the question?
- **Context Precision:** Are the retrieved documents relevant to the question?
- **Context Recall:** Did the retriever find all the necessary information?

---

## 30. Unit Testing LLM Outputs with DeepEval

```bash
pip install deepeval
```

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric

def test_no_hallucination():
    test_case = LLMTestCase(
        input="What is our return policy?",
        actual_output="You can return items within 30 days for a full refund.",
        context=["Our return policy allows returns within 30 days. Full refund provided."],
    )
    metric = HallucinationMetric(threshold=0.5)
    assert_test(test_case, [metric])

def test_answer_relevancy():
    test_case = LLMTestCase(
        input="What is Python?",
        actual_output="Python is a programming language known for its readability.",
    )
    metric = AnswerRelevancyMetric(threshold=0.7)
    assert_test(test_case, [metric])
```

```bash
# Run with pytest
deepeval test run test_llm.py
```

---

## 31. Prompt Regression Testing with Promptfoo

Test prompts across models and catch regressions:

```bash
npx promptfoo@latest init
```

```yaml
# promptfooconfig.yaml
prompts:
  - "Answer this question: {{question}}"
  - "You are a helpful assistant. Answer: {{question}}"

providers:
  - openai:gpt-4
  - openai:gpt-3.5-turbo

tests:
  - vars:
      question: "What is Python?"
    assert:
      - type: contains
        value: "programming language"
      - type: llm-rubric
        value: "Answer should be concise and accurate"
  - vars:
      question: "What is 2+2?"
    assert:
      - type: contains
        value: "4"
```

```bash
npx promptfoo@latest eval
npx promptfoo@latest view  # Opens a comparison UI in the browser
```

---

## 32. Guardrails for LangChain

### Using Guardrails AI

Validate LLM outputs match expected structure and rules:

```bash
pip install guardrails-ai
```

```python
from guardrails import Guard
from guardrails.hub import ValidLength, ToxicLanguage

guard = Guard().use_many(
    ValidLength(min=10, max=500, on_fail="reask"),
    ToxicLanguage(on_fail="fix"),
)

# Wrap your LangChain call
raw_output = chain.invoke({"question": "Tell me about Python"})
validated = guard.validate(raw_output)
print(validated.validated_output)
```

### Using NeMo Guardrails (NVIDIA)

Add programmable safety rails to your LangChain app:

```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(
    messages=[{"role": "user", "content": "How do I hack a website?"}]
)
# Response is blocked or redirected by the rails
```

---

## 33. Testing Quick Reference

| What You Want to Test | Tool | How |
|----------------------|------|-----|
| RAG quality (faithfulness, relevancy) | Ragas | `evaluate()` with metrics |
| Hallucination detection | DeepEval | `HallucinationMetric` |
| Prompt regression across models | Promptfoo | YAML config + `eval` |
| Output structure validation | Guardrails AI | `Guard.validate()` |
| End-to-end tracing | LangSmith / LangFuse | Auto-tracing with env vars |
| Safety & toxicity | NeMo Guardrails | Colang rules |
| Load testing LLM APIs | Locust | Standard HTTP load test |
| Cost/latency comparison | LiteLLM | Proxy across providers |

---

# Part 5: OWASP Top 10 for LLM Applications

The **OWASP Top 10 for Large Language Model Applications** is a security framework
that identifies the most critical vulnerabilities in LLM-powered apps. This is
frequently asked about in AI engineer interviews.

## 34. The OWASP Top 10 for LLMs

| # | Vulnerability | Description |
|---|--------------|-------------|
| 1 | **Prompt Injection** | Manipulating the model via crafted inputs to bypass instructions. Can be **direct** (user types malicious prompt) or **indirect** (injected via external data sources like web pages or documents). |
| 2 | **Sensitive Information Disclosure** | Model leaking PII, API keys, system prompts, or training data in its responses. |
| 3 | **Supply Chain Vulnerabilities** | Compromised pre-trained models, poisoned training datasets, or vulnerable third-party plugins/packages. |
| 4 | **Data and Model Poisoning** | Corrupting training or fine-tuning data to introduce backdoors, biases, or incorrect behavior. |
| 5 | **Improper Output Handling** | Trusting LLM output without validation or sanitization, leading to XSS, SQL injection, or code execution. |
| 6 | **Excessive Agency** | Granting the LLM too many permissions, tools, or autonomy without proper guardrails or human oversight. |
| 7 | **System Prompt Leakage** | Attackers extracting system instructions through prompt manipulation techniques. |
| 8 | **Vector and Embedding Weaknesses** | Manipulating RAG retrieval by poisoning vector store embeddings or exploiting weak access controls. |
| 9 | **Misinformation** | Model generating hallucinated, false, or misleading content presented as fact. |
| 10 | **Unbounded Consumption** | No limits on token usage, API calls, or resource consumption — leading to denial-of-wallet or denial-of-service. |

---

## 35. OWASP Mitigations in LangChain Apps

Here's how each vulnerability maps to LangChain-specific defenses:

### Prompt Injection (#1)
```python
# Use strict prompt templates — never concatenate raw user input into system prompts
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Only answer questions about our products."),
    ("user", "{user_input}"),  # Kept separate from system instructions
])

# Add input validation
def sanitize_input(text: str) -> str:
    forbidden = ["ignore previous", "system prompt", "forget instructions"]
    for phrase in forbidden:
        if phrase.lower() in text.lower():
            return "I can only help with product questions."
    return text
```

### Sensitive Information Disclosure (#2)
```python
# Filter outputs before returning to user
from langchain_core.output_parsers import StrOutputParser

def filter_sensitive(output: str) -> str:
    import re
    # Remove potential API keys, emails, SSNs
    output = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[REDACTED]', output)
    output = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', output)
    return output

chain = prompt | llm | StrOutputParser() | filter_sensitive
```

### Improper Output Handling (#5)
```python
# Always sanitize LLM output before using in downstream systems
import html

def safe_output(text: str) -> str:
    return html.escape(text)  # Prevent XSS

# Never pass LLM output directly to eval(), exec(), or SQL queries
# BAD:  eval(llm_response)
# GOOD: use structured output parsers with validation
```

### Excessive Agency (#6)
```python
# Limit tool access and add human approval
from langgraph.prebuilt import create_react_agent

# Only give the agent READ-ONLY tools
agent = create_react_agent(llm, tools=[search_tool])  # No write/delete tools

# Add human-in-the-loop for sensitive actions
# Use LangGraph's interrupt_before for approval steps
```

### Unbounded Consumption (#10)
```python
# Set token limits and rate limiting
llm = ChatOpenAI(
    model="gpt-4",
    max_tokens=500,          # Cap output length
    request_timeout=30,      # Timeout long requests
)

# Track costs with LangSmith or callbacks
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"question": "Explain quantum physics"})
    print(f"Cost: ${cb.total_cost:.4f}, Tokens: {cb.total_tokens}")
```

---

## 36. Security Checklist for LangChain Apps

- [ ] Separate system and user messages in prompt templates
- [ ] Validate and sanitize all user inputs before passing to chains
- [ ] Never pass LLM output directly to `eval()`, `exec()`, or raw SQL
- [ ] Use output parsers with Pydantic validation for structured responses
- [ ] Set `max_tokens` and request timeouts on all LLM calls
- [ ] Limit tool permissions — principle of least privilege
- [ ] Add human-in-the-loop for destructive or high-impact actions
- [ ] Monitor costs and token usage with callbacks or LangSmith
- [ ] Implement rate limiting on user-facing endpoints
- [ ] Scan vector store data for injection attempts before indexing
- [ ] Never expose system prompts — treat them as secrets
- [ ] Use Guardrails or NeMo Guardrails for production safety rails

---

## 37. Interview Quick Reference

**"What are the OWASP Top 10 for LLMs?"**
> A security framework identifying the 10 most critical vulnerabilities in LLM applications — from prompt injection and data poisoning to excessive agency and unbounded consumption. It guides teams on building safer AI-powered systems.

**"How do you test a GenAI application?"**
> Use a combination of: (1) evaluation frameworks like Ragas for RAG quality, (2) DeepEval or Promptfoo for unit testing and prompt regression, (3) LangSmith or LangFuse for observability and tracing, and (4) Guardrails AI or NeMo Guardrails for output validation and safety.

**"How do you prevent prompt injection?"**
> Separate system and user messages in templates, validate inputs, use guardrails, and never concatenate raw user input into system prompts. For indirect injection via RAG, scan and sanitize documents before indexing.

---

## Now Try the Problems

The practice problems build up step by step:

1. `p1_langchain_basics.py` - Prompts, chains, output parsers
2. `p2_rag_application.py` - Document loading, splitting, embeddings, vector store
3. `p3_langgraph_agent.py` - Building a stateful agent with tools

```bash
pytest 08_langchain/ -v
```
