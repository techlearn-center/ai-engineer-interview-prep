import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p1_langchain_basics import (
    MockLLM,
    MockMessage,
    PromptTemplate,
    ChatPromptTemplate,
    StrOutputParser,
    Chain,
    JsonOutputParser,
    ConversationChain,
)


class TestPromptTemplate:
    def test_format_single_variable(self):
        template = PromptTemplate("Hello, {name}!")
        result = template.format(name="Alice")
        assert result == "Hello, Alice!"

    def test_format_multiple_variables(self):
        template = PromptTemplate("{greeting}, {name}! You are {age} years old.")
        result = template.format(greeting="Hi", name="Bob", age=25)
        assert result == "Hi, Bob! You are 25 years old."

    def test_get_variables(self):
        template = PromptTemplate("Hello {name}, your score is {score}.")
        variables = template.get_variables()
        assert set(variables) == {"name", "score"}

    def test_get_variables_empty(self):
        template = PromptTemplate("No variables here.")
        variables = template.get_variables()
        assert variables == []


class TestChatPromptTemplate:
    def test_format_messages(self):
        template = ChatPromptTemplate([
            ("system", "You are a {role}."),
            ("user", "{question}"),
        ])
        messages = template.format(role="teacher", question="What is Python?")

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a teacher."}
        assert messages[1] == {"role": "user", "content": "What is Python?"}

    def test_format_no_variables(self):
        template = ChatPromptTemplate([
            ("system", "You are helpful."),
            ("user", "Hello"),
        ])
        messages = template.format()
        assert messages[0]["content"] == "You are helpful."


class TestStrOutputParser:
    def test_parse(self):
        parser = StrOutputParser()
        message = MockMessage("Hello world")
        result = parser.parse(message)
        assert result == "Hello world"

    def test_parse_empty(self):
        parser = StrOutputParser()
        message = MockMessage("")
        result = parser.parse(message)
        assert result == ""


class TestChain:
    def test_invoke(self):
        prompt = PromptTemplate("Tell me about {topic}.")
        llm = MockLLM({"python": "Python is great!"})
        parser = StrOutputParser()

        chain = Chain(prompt, llm, parser)
        result = chain.invoke({"topic": "Python"})
        assert "Python" in result

    def test_chain_flow(self):
        prompt = PromptTemplate("Summarize: {text}")
        llm = MockLLM({"summarize": "Here is the summary."})
        parser = StrOutputParser()

        chain = Chain(prompt, llm, parser)
        result = chain.invoke({"text": "Long text here..."})
        assert "summary" in result.lower()


class TestJsonOutputParser:
    def test_parse_plain_json(self):
        parser = JsonOutputParser()
        message = MockMessage('{"name": "Alice", "age": 30}')
        result = parser.parse(message)
        assert result == {"name": "Alice", "age": 30}

    def test_parse_json_in_code_block(self):
        parser = JsonOutputParser()
        message = MockMessage('''
Here is the data:
```json
{"name": "Bob", "score": 95}
```
Done.
''')
        result = parser.parse(message)
        assert result == {"name": "Bob", "score": 95}

    def test_parse_invalid_raises(self):
        parser = JsonOutputParser()
        message = MockMessage("This is not JSON at all")
        with pytest.raises(ValueError):
            parser.parse(message)

    def test_parse_nested_json(self):
        parser = JsonOutputParser()
        message = MockMessage('{"user": {"name": "Alice"}, "scores": [1, 2, 3]}')
        result = parser.parse(message)
        assert result["user"]["name"] == "Alice"
        assert result["scores"] == [1, 2, 3]


class TestConversationChain:
    def test_invoke_stores_history(self):
        llm = MockLLM()
        chain = ConversationChain(llm)

        chain.invoke("Hello")
        history = chain.get_history()

        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

    def test_multiple_turns(self):
        llm = MockLLM()
        chain = ConversationChain(llm)

        chain.invoke("First message")
        chain.invoke("Second message")
        history = chain.get_history()

        assert len(history) == 4  # 2 turns * 2 messages each

    def test_clear_history(self):
        llm = MockLLM()
        chain = ConversationChain(llm)

        chain.invoke("Hello")
        assert len(chain.get_history()) > 0

        chain.clear_history()
        assert len(chain.get_history()) == 0

    def test_returns_response(self):
        llm = MockLLM({"hello": "Hi there!"})
        chain = ConversationChain(llm)

        response = chain.invoke("Hello")
        assert isinstance(response, str)
        assert len(response) > 0
