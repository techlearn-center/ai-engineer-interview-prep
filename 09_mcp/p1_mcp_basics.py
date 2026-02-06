"""
MCP Basics - Understanding the Core Concepts

MCP (Model Context Protocol) is Anthropic's standard for connecting AI to external tools.
Before building a full server, let's understand the building blocks.

These exercises simulate MCP concepts without needing the full SDK.
"""
from dataclasses import dataclass, field
from typing import Any
import json
import re


# =============================================================================
# EXERCISE 1: Tool Definition
# =============================================================================
# MCP tools have a name, description, and input schema (JSON Schema format).
# Create a class that represents a tool definition.

@dataclass
class ToolDefinition:
    """
    Represents an MCP tool definition.

    Attributes:
        name: The tool's identifier (e.g., "search_files")
        description: What the tool does (shown to AI)
        parameters: Dict defining the input schema

    Example:
        >>> tool = ToolDefinition(
        ...     name="calculate",
        ...     description="Perform math calculations",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "expression": {"type": "string", "description": "Math expression"}
        ...         },
        ...         "required": ["expression"]
        ...     }
        ... )
        >>> tool.name
        'calculate'
        >>> tool.get_required_params()
        ['expression']
    """
    name: str
    description: str
    parameters: dict = field(default_factory=dict)

    def get_required_params(self) -> list:
        """Return list of required parameter names."""
        # YOUR CODE HERE
        pass

    def get_param_names(self) -> list:
        """Return list of all parameter names."""
        # YOUR CODE HERE
        pass

    def validate_arguments(self, arguments: dict) -> tuple:
        """
        Validate that arguments match the schema.
        Returns (is_valid: bool, error_message: str or None)

        Example:
            >>> tool.validate_arguments({"expression": "2+2"})
            (True, None)
            >>> tool.validate_arguments({})
            (False, "Missing required parameter: expression")
        """
        # YOUR CODE HERE
        pass

    def to_dict(self) -> dict:
        """Convert to MCP-style dict format."""
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 2: Resource Definition
# =============================================================================
# MCP resources are data that AI can read, identified by URIs.

@dataclass
class ResourceDefinition:
    """
    Represents an MCP resource.

    Attributes:
        uri: Unique identifier (e.g., "file:///path/to/file.txt")
        name: Human-readable name
        description: What this resource contains
        mime_type: Content type (e.g., "text/plain", "application/json")

    Example:
        >>> res = ResourceDefinition(
        ...     uri="config://app",
        ...     name="App Configuration",
        ...     description="Application settings",
        ...     mime_type="application/json"
        ... )
        >>> res.get_scheme()
        'config'
        >>> res.get_path()
        'app'
    """
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"

    def get_scheme(self) -> str:
        """
        Extract the scheme from the URI.
        "file:///path" -> "file"
        "db://users" -> "db"
        """
        # YOUR CODE HERE
        pass

    def get_path(self) -> str:
        """
        Extract the path from the URI.
        "file:///path/to/file" -> "/path/to/file"
        "db://users" -> "users"
        """
        # YOUR CODE HERE
        pass

    def to_dict(self) -> dict:
        """Convert to MCP-style dict format."""
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 3: JSON-RPC Message Handling
# =============================================================================
# MCP uses JSON-RPC 2.0 for communication. Let's parse and create messages.

class JsonRpcHandler:
    """
    Handle JSON-RPC 2.0 messages used by MCP.

    JSON-RPC format:
    Request:  {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {...}}
    Response: {"jsonrpc": "2.0", "id": 1, "result": {...}}
    Error:    {"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "..."}}

    Example:
        >>> handler = JsonRpcHandler()
        >>> handler.parse_request('{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
        {'id': 1, 'method': 'tools/list', 'params': {}}
    """

    def parse_request(self, message: str) -> dict:
        """
        Parse a JSON-RPC request string.
        Returns dict with 'id', 'method', and 'params'.
        Raises ValueError for invalid messages.

        Example:
            >>> handler.parse_request('{"jsonrpc":"2.0","id":1,"method":"test","params":{"a":1}}')
            {'id': 1, 'method': 'test', 'params': {'a': 1}}
        """
        # YOUR CODE HERE
        pass

    def create_response(self, request_id: int, result: Any) -> str:
        """
        Create a JSON-RPC success response.

        Example:
            >>> handler.create_response(1, {"tools": []})
            '{"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}'
        """
        # YOUR CODE HERE
        pass

    def create_error(self, request_id: int, code: int, message: str) -> str:
        """
        Create a JSON-RPC error response.

        Common codes:
        -32700: Parse error
        -32600: Invalid request
        -32601: Method not found
        -32602: Invalid params

        Example:
            >>> handler.create_error(1, -32601, "Method not found")
            '{"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}}'
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 4: Tool Registry
# =============================================================================
# A tool registry manages available tools and dispatches calls.

class ToolRegistry:
    """
    Registry for MCP tools. Stores tool definitions and handlers.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register_tool(
        ...     name="greet",
        ...     description="Say hello",
        ...     parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        ...     handler=lambda args: f"Hello, {args.get('name', 'World')}!"
        ... )
        >>> registry.list_tools()
        [{'name': 'greet', 'description': 'Say hello', 'inputSchema': {...}}]
        >>> registry.call_tool("greet", {"name": "Alice"})
        'Hello, Alice!'
    """

    def __init__(self):
        self._tools: dict = {}  # name -> (ToolDefinition, handler)

    def register_tool(self, name: str, description: str, parameters: dict, handler: callable):
        """Register a new tool with its handler function."""
        # YOUR CODE HERE
        pass

    def list_tools(self) -> list:
        """Return list of tool definitions in MCP format."""
        # YOUR CODE HERE
        pass

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        # YOUR CODE HERE
        pass

    def call_tool(self, name: str, arguments: dict) -> Any:
        """
        Call a tool by name with given arguments.
        Raises ValueError if tool not found.
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 5: Resource Store
# =============================================================================
# A resource store manages data that AI can access.

class ResourceStore:
    """
    Store for MCP resources.

    Example:
        >>> store = ResourceStore()
        >>> store.add_resource("notes://meeting", "Meeting Notes", "Discuss Q1 goals")
        >>> store.add_resource("notes://todo", "Todo List", "1. Review PRs\\n2. Deploy")
        >>> store.list_resources()
        [{'uri': 'notes://meeting', 'name': 'Meeting Notes', ...}, ...]
        >>> store.read_resource("notes://meeting")
        'Discuss Q1 goals'
    """

    def __init__(self):
        self._resources: dict = {}  # uri -> (ResourceDefinition, content)

    def add_resource(self, uri: str, name: str, content: str,
                     description: str = "", mime_type: str = "text/plain"):
        """Add a resource to the store."""
        # YOUR CODE HERE
        pass

    def list_resources(self) -> list:
        """Return list of resource definitions in MCP format."""
        # YOUR CODE HERE
        pass

    def has_resource(self, uri: str) -> bool:
        """Check if a resource exists."""
        # YOUR CODE HERE
        pass

    def read_resource(self, uri: str) -> str:
        """
        Read a resource's content by URI.
        Raises ValueError if not found.
        """
        # YOUR CODE HERE
        pass

    def update_resource(self, uri: str, content: str):
        """
        Update an existing resource's content.
        Raises ValueError if not found.
        """
        # YOUR CODE HERE
        pass

    def delete_resource(self, uri: str):
        """
        Delete a resource.
        Raises ValueError if not found.
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 6: Simple MCP Message Router
# =============================================================================
# Route incoming MCP messages to the right handler.

class MCPRouter:
    """
    Route MCP protocol messages to appropriate handlers.

    Handles these MCP methods:
    - initialize: Return server info
    - tools/list: Return available tools
    - tools/call: Execute a tool
    - resources/list: Return available resources
    - resources/read: Read a resource

    Example:
        >>> router = MCPRouter("my-server", "1.0.0")
        >>> router.tool_registry.register_tool("echo", "Echo input", {...}, lambda a: a["text"])
        >>> result = router.handle_message('{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
        >>> # Returns JSON-RPC response with tool list
    """

    def __init__(self, server_name: str, version: str = "1.0.0"):
        self.server_name = server_name
        self.version = version
        self.tool_registry = ToolRegistry()
        self.resource_store = ResourceStore()
        self._json_rpc = JsonRpcHandler()

    def handle_message(self, message: str) -> str:
        """
        Parse and route an MCP message, returning the response.

        Routes to:
        - "initialize" -> _handle_initialize
        - "tools/list" -> _handle_tools_list
        - "tools/call" -> _handle_tools_call
        - "resources/list" -> _handle_resources_list
        - "resources/read" -> _handle_resources_read
        """
        # YOUR CODE HERE
        pass

    def _handle_initialize(self, request_id: int, params: dict) -> str:
        """Handle initialize request. Return server info."""
        # YOUR CODE HERE
        pass

    def _handle_tools_list(self, request_id: int, params: dict) -> str:
        """Handle tools/list request. Return available tools."""
        # YOUR CODE HERE
        pass

    def _handle_tools_call(self, request_id: int, params: dict) -> str:
        """
        Handle tools/call request. Execute the tool.
        params contains: {"name": "tool_name", "arguments": {...}}
        """
        # YOUR CODE HERE
        pass

    def _handle_resources_list(self, request_id: int, params: dict) -> str:
        """Handle resources/list request. Return available resources."""
        # YOUR CODE HERE
        pass

    def _handle_resources_read(self, request_id: int, params: dict) -> str:
        """
        Handle resources/read request. Return resource content.
        params contains: {"uri": "resource://uri"}
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# TEST YOUR UNDERSTANDING
# =============================================================================

if __name__ == "__main__":
    # Test Exercise 1: Tool Definition
    print("=" * 50)
    print("Testing ToolDefinition...")

    calc_tool = ToolDefinition(
        name="calculate",
        description="Perform math calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
                "precision": {"type": "integer", "description": "Decimal places"}
            },
            "required": ["expression"]
        }
    )
    print(f"Tool name: {calc_tool.name}")
    print(f"Required params: {calc_tool.get_required_params()}")
    print(f"All params: {calc_tool.get_param_names()}")
    print(f"Valid args: {calc_tool.validate_arguments({'expression': '2+2'})}")
    print(f"Invalid args: {calc_tool.validate_arguments({})}")

    # Test Exercise 4: Tool Registry
    print("\n" + "=" * 50)
    print("Testing ToolRegistry...")

    registry = ToolRegistry()
    registry.register_tool(
        name="greet",
        description="Say hello to someone",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        },
        handler=lambda args: f"Hello, {args['name']}!"
    )

    print(f"Tools: {registry.list_tools()}")
    print(f"Call greet: {registry.call_tool('greet', {'name': 'World'})}")

    # Test Exercise 6: MCP Router
    print("\n" + "=" * 50)
    print("Testing MCPRouter...")

    router = MCPRouter("test-server", "1.0.0")
    router.tool_registry.register_tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        handler=lambda args: args["a"] + args["b"]
    )

    # Test initialize
    init_response = router.handle_message(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
    )
    print(f"Initialize: {init_response}")

    # Test tools/list
    list_response = router.handle_message(
        '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
    )
    print(f"Tools list: {list_response}")

    # Test tools/call
    call_response = router.handle_message(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":5,"b":3}}}'
    )
    print(f"Tool call: {call_response}")
