"""
Problem 1: LangChain Basics
============================
Difficulty: Easy -> Medium

Learn the fundamentals: prompts, chains, and output parsers.
Uses MOCK components so you don't need an API key.

Run tests:
    pytest 08_langchain/tests/test_p1_langchain_basics.py -v
"""
from typing import Any


# ============================================================
# MOCK COMPONENTS (simulates LangChain behavior)
# These work just like real LangChain but without API calls
# ============================================================

class MockLLM:
    """Simulates a language model."""

    def __init__(self, responses: dict[str, str] = None):
        # Map of input patterns to responses
        self.responses = responses or {
            "python": "Python is a high-level programming language known for readability.",
            "javascript": "JavaScript is a scripting language for web development.",
            "summarize": "Here is a brief summary of the key points.",
            "translate": "Translation: Hello, how are you?",
        }

    def invoke(self, prompt: str) -> "MockMessage":
        """Generate a response based on the prompt."""
        prompt_lower = prompt.lower() if isinstance(prompt, str) else str(prompt).lower()
        for keyword, response in self.responses.items():
            if keyword in prompt_lower:
                return MockMessage(response)
        return MockMessage("I can help you with that question.")


class MockMessage:
    """Simulates an AI message response."""

    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content


# ============================================================
# YOUR TASKS
# ============================================================


class PromptTemplate:
    """
    TASK 1: Implement a simple prompt template.

    A prompt template has a template string with {variable} placeholders.
    When you call .format(), it replaces placeholders with actual values.

    Example:
        template = PromptTemplate("Hello, {name}! You are {age} years old.")
        result = template.format(name="Alice", age=30)
        # "Hello, Alice! You are 30 years old."

    Implement:
        - __init__(self, template: str): Store the template
        - format(self, **kwargs) -> str: Replace placeholders with values
        - get_variables(self) -> list[str]: Return list of variable names in the template

    Hint for get_variables: Use regex to find all {word} patterns
    """

    def __init__(self, template: str):
        # YOUR CODE HERE
        pass

    def format(self, **kwargs) -> str:
        # YOUR CODE HERE
        pass

    def get_variables(self) -> list[str]:
        # YOUR CODE HERE
        pass


class ChatPromptTemplate:
    """
    TASK 2: Implement a chat prompt template with roles.

    Chat models expect messages with roles (system, user, assistant).
    This template builds a list of message dicts.

    Example:
        template = ChatPromptTemplate([
            ("system", "You are a helpful {role}."),
            ("user", "{question}"),
        ])
        messages = template.format(role="teacher", question="What is Python?")
        # [
        #     {"role": "system", "content": "You are a helpful teacher."},
        #     {"role": "user", "content": "What is Python?"},
        # ]

    Implement:
        - __init__(self, messages: list[tuple[str, str]]): Store message templates
        - format(self, **kwargs) -> list[dict]: Return formatted messages
    """

    def __init__(self, messages: list[tuple[str, str]]):
        # YOUR CODE HERE
        pass

    def format(self, **kwargs) -> list[dict]:
        # YOUR CODE HERE
        pass


class StrOutputParser:
    """
    TASK 3: Implement a string output parser.

    Takes a MockMessage and extracts just the string content.

    Example:
        parser = StrOutputParser()
        message = MockMessage("Hello world")
        result = parser.parse(message)
        # "Hello world"
    """

    def parse(self, message: MockMessage) -> str:
        # YOUR CODE HERE
        pass


class Chain:
    """
    TASK 4: Implement a simple chain that connects components.

    A chain connects: prompt → llm → parser

    Example:
        prompt = PromptTemplate("Explain {topic} simply.")
        llm = MockLLM()
        parser = StrOutputParser()

        chain = Chain(prompt, llm, parser)
        result = chain.invoke({"topic": "Python"})
        # "Python is a high-level programming language known for readability."

    Implement:
        - __init__(self, prompt, llm, parser): Store components
        - invoke(self, inputs: dict) -> str: Run the chain
            1. Format the prompt with inputs
            2. Pass to LLM
            3. Parse the output
            4. Return the string result
    """

    def __init__(self, prompt: PromptTemplate, llm: MockLLM, parser: StrOutputParser):
        # YOUR CODE HERE
        pass

    def invoke(self, inputs: dict) -> str:
        # YOUR CODE HERE
        pass


class JsonOutputParser:
    """
    TASK 5: Implement a JSON output parser.

    Extracts JSON from LLM responses. The LLM might return text with
    JSON embedded in it (between ```json and ``` or just raw JSON).

    Example:
        parser = JsonOutputParser()

        # Plain JSON
        result = parser.parse(MockMessage('{"name": "Alice", "age": 30}'))
        # {"name": "Alice", "age": 30}

        # JSON in markdown code block
        result = parser.parse(MockMessage('''
        Here is the data:
        ```json
        {"name": "Bob"}
        ```
        '''))
        # {"name": "Bob"}

    Implement:
        - parse(self, message: MockMessage) -> dict: Extract and parse JSON
        - Raise ValueError if no valid JSON found
    """

    def parse(self, message: MockMessage) -> dict:
        # YOUR CODE HERE
        pass


class ConversationChain:
    """
    TASK 6: Implement a chain with memory.

    This chain remembers previous messages in the conversation.

    Example:
        chain = ConversationChain(MockLLM())

        chain.invoke("My name is Alice")
        # Stores: user said "My name is Alice"

        chain.invoke("What is my name?")
        # Can access history to see previous messages

        history = chain.get_history()
        # [
        #     {"role": "user", "content": "My name is Alice"},
        #     {"role": "assistant", "content": "..."},
        #     {"role": "user", "content": "What is my name?"},
        #     {"role": "assistant", "content": "..."},
        # ]

    Implement:
        - __init__(self, llm: MockLLM): Store llm and initialize empty history
        - invoke(self, user_message: str) -> str:
            1. Add user message to history
            2. Get LLM response
            3. Add assistant response to history
            4. Return the response content
        - get_history(self) -> list[dict]: Return the conversation history
        - clear_history(self): Clear the history
    """

    def __init__(self, llm: MockLLM):
        # YOUR CODE HERE
        pass

    def invoke(self, user_message: str) -> str:
        # YOUR CODE HERE
        pass

    def get_history(self) -> list[dict]:
        # YOUR CODE HERE
        pass

    def clear_history(self):
        # YOUR CODE HERE
        pass
