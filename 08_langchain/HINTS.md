# Hints - LangChain

## P1: LangChain Basics

### PromptTemplate
- Store the template string in `self.template`
- `format(**kwargs)`: just use Python's built-in `self.template.format(**kwargs)`
- `get_variables()`: use regex `re.findall(r'\{(\w+)\}', self.template)`

### ChatPromptTemplate
- Store the list of `(role, content_template)` tuples
- In `format()`, loop through each tuple, format the content, create dict
- Result: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`

### StrOutputParser
- Just return `message.content`

### Chain
- Store prompt, llm, parser in `__init__`
- In `invoke(inputs)`:
  1. `formatted = self.prompt.format(**inputs)`
  2. `message = self.llm.invoke(formatted)`
  3. `return self.parser.parse(message)`

### JsonOutputParser
- First try to find JSON in a code block: `re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)`
- If not found, try raw JSON: `re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)`
- Use `json.loads()` to parse
- Raise `ValueError` if parsing fails

### ConversationChain
- `self.history = []` in `__init__`
- In `invoke()`:
  1. `self.history.append({"role": "user", "content": user_message})`
  2. `message = self.llm.invoke(user_message)`
  3. `self.history.append({"role": "assistant", "content": message.content})`
  4. `return message.content`
- `get_history()`: return a copy of the list
- `clear_history()`: reset to empty list

## P2: RAG Application

### TextSplitter
- Split on sentence boundaries: `re.split(r'(?<=[.!?])\s+', text)`
- Accumulate sentences until you exceed `chunk_size`
- When saving a chunk, keep the last `chunk_overlap` characters for the next chunk
- For `split_documents`, loop through docs, split each, create new Document with same metadata

### VectorStore
- Store two parallel lists: `self.documents` and `self.vectors`
- `add_documents`: for each doc, embed its `page_content`, append both
- `add_texts`: convert to Documents first, then call `add_documents`
- `similarity_search_with_score`:
  1. Embed the query
  2. Compute cosine similarity with each stored vector
  3. Create list of `(doc, score)` tuples
  4. Sort by score descending
  5. Return top k
- Cosine similarity: `dot(a,b) / (norm(a) * norm(b))`

### Retriever
- Just wraps the vector store
- `invoke()`: call `self.vector_store.similarity_search(query, k=self.search_kwargs.get("k", 4))`
- `get_relevant_documents()`: alias for `invoke()`

### RAGChain
- `format_docs()`: `"\n\n".join(doc.page_content for doc in docs)`
- `invoke(question)`:
  1. `docs = self.retriever.invoke(question)`
  2. `context = self.format_docs(docs)`
  3. `prompt = self.prompt_template.format(context=context, question=question)`
  4. `response = self.llm.invoke(prompt)`
  5. `return {"answer": response.content, "source_documents": docs}`

### TextLoader
- Store content and source in `__init__`
- `load()`: return `[Document(page_content=self.content, metadata={"source": self.source})]`

### CSVLoader
- Use `csv.DictReader(io.StringIO(self.content))`
- For each row, format as `"key: value\n"` pairs joined together
- Include `row` number in metadata
