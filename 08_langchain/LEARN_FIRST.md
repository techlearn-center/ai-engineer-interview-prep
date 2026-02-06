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

## Now Try the Problems

The practice problems build up step by step:

1. `p1_langchain_basics.py` - Prompts, chains, output parsers
2. `p2_rag_application.py` - Document loading, splitting, embeddings, vector store
3. `p3_langgraph_agent.py` - Building a stateful agent with tools

```bash
pytest 08_langchain/ -v
```
