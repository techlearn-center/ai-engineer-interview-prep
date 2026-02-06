"""
SOLUTIONS - LangChain Basics
==============================
Try to solve the problems yourself first!
"""
import re
import json
from p1_langchain_basics import MockLLM, MockMessage


class PromptTemplate:
    """Simple prompt template with variable substitution."""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    def get_variables(self) -> list[str]:
        # Find all {variable_name} patterns
        return re.findall(r'\{(\w+)\}', self.template)


class ChatPromptTemplate:
    """Chat prompt template with roles."""

    def __init__(self, messages: list[tuple[str, str]]):
        self.messages = messages

    def format(self, **kwargs) -> list[dict]:
        result = []
        for role, content_template in self.messages:
            content = content_template.format(**kwargs)
            result.append({"role": role, "content": content})
        return result


class StrOutputParser:
    """Extract string content from a message."""

    def parse(self, message: MockMessage) -> str:
        return message.content


class Chain:
    """Connect prompt -> llm -> parser."""

    def __init__(self, prompt: PromptTemplate, llm: MockLLM, parser: StrOutputParser):
        self.prompt = prompt
        self.llm = llm
        self.parser = parser

    def invoke(self, inputs: dict) -> str:
        # 1. Format the prompt
        formatted_prompt = self.prompt.format(**inputs)
        # 2. Get LLM response
        message = self.llm.invoke(formatted_prompt)
        # 3. Parse and return
        return self.parser.parse(message)


class JsonOutputParser:
    """Extract JSON from LLM response."""

    def parse(self, message: MockMessage) -> dict:
        content = message.content

        # Try to find JSON in code block first
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Try to find raw JSON (starts with { or [)
            json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("No valid JSON found in response")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")


class ConversationChain:
    """Chain with memory for multi-turn conversations."""

    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.history = []

    def invoke(self, user_message: str) -> str:
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Get LLM response
        message = self.llm.invoke(user_message)

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": message.content})

        return message.content

    def get_history(self) -> list[dict]:
        return self.history.copy()

    def clear_history(self):
        self.history = []
