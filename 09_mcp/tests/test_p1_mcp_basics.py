"""Tests for MCP Basics exercises."""
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from p1_mcp_basics import (
    ToolDefinition, ResourceDefinition, JsonRpcHandler,
    ToolRegistry, ResourceStore, MCPRouter
)


# =============================================================================
# Exercise 1: ToolDefinition Tests
# =============================================================================

class TestToolDefinition:
    def test_get_required_params(self):
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"},
                    "optional_param": {"type": "integer"}
                },
                "required": ["required_param"]
            }
        )
        assert tool.get_required_params() == ["required_param"]

    def test_get_required_params_empty(self):
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={"type": "object", "properties": {}}
        )
        assert tool.get_required_params() == []

    def test_get_param_names(self):
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "integer"},
                    "c": {"type": "boolean"}
                }
            }
        )
        params = tool.get_param_names()
        assert set(params) == {"a", "b", "c"}

    def test_validate_arguments_valid(self):
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        )
        is_valid, error = tool.validate_arguments({"name": "test"})
        assert is_valid is True
        assert error is None

    def test_validate_arguments_missing_required(self):
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        )
        is_valid, error = tool.validate_arguments({})
        assert is_valid is False
        assert "name" in error.lower()

    def test_to_dict(self):
        tool = ToolDefinition(
            name="calculate",
            description="Math calculator",
            parameters={"type": "object", "properties": {}}
        )
        d = tool.to_dict()
        assert d["name"] == "calculate"
        assert d["description"] == "Math calculator"
        assert "inputSchema" in d


# =============================================================================
# Exercise 2: ResourceDefinition Tests
# =============================================================================

class TestResourceDefinition:
    def test_get_scheme_file(self):
        res = ResourceDefinition(uri="file:///path/to/file", name="File")
        assert res.get_scheme() == "file"

    def test_get_scheme_custom(self):
        res = ResourceDefinition(uri="db://users", name="Users")
        assert res.get_scheme() == "db"

    def test_get_path_file(self):
        res = ResourceDefinition(uri="file:///path/to/file", name="File")
        assert res.get_path() == "/path/to/file"

    def test_get_path_simple(self):
        res = ResourceDefinition(uri="notes://meeting", name="Meeting")
        assert res.get_path() == "meeting"

    def test_to_dict(self):
        res = ResourceDefinition(
            uri="test://resource",
            name="Test Resource",
            description="A test",
            mime_type="application/json"
        )
        d = res.to_dict()
        assert d["uri"] == "test://resource"
        assert d["name"] == "Test Resource"
        assert d["description"] == "A test"
        assert d["mimeType"] == "application/json"


# =============================================================================
# Exercise 3: JsonRpcHandler Tests
# =============================================================================

class TestJsonRpcHandler:
    def test_parse_request_basic(self):
        handler = JsonRpcHandler()
        result = handler.parse_request('{"jsonrpc":"2.0","id":1,"method":"test"}')
        assert result["id"] == 1
        assert result["method"] == "test"
        assert result["params"] == {}

    def test_parse_request_with_params(self):
        handler = JsonRpcHandler()
        result = handler.parse_request(
            '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"add"}}'
        )
        assert result["id"] == 5
        assert result["method"] == "tools/call"
        assert result["params"]["name"] == "add"

    def test_parse_request_invalid_json(self):
        handler = JsonRpcHandler()
        with pytest.raises(ValueError):
            handler.parse_request('not valid json')

    def test_parse_request_missing_method(self):
        handler = JsonRpcHandler()
        with pytest.raises(ValueError):
            handler.parse_request('{"jsonrpc":"2.0","id":1}')

    def test_create_response(self):
        handler = JsonRpcHandler()
        response = handler.create_response(1, {"tools": []})
        data = json.loads(response)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["result"] == {"tools": []}

    def test_create_error(self):
        handler = JsonRpcHandler()
        response = handler.create_error(1, -32601, "Method not found")
        data = json.loads(response)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["error"]["code"] == -32601
        assert data["error"]["message"] == "Method not found"


# =============================================================================
# Exercise 4: ToolRegistry Tests
# =============================================================================

class TestToolRegistry:
    def test_register_and_list_tools(self):
        registry = ToolRegistry()
        registry.register_tool(
            name="greet",
            description="Say hello",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: "Hello!"
        )
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "greet"

    def test_has_tool(self):
        registry = ToolRegistry()
        registry.register_tool("test", "Test", {}, lambda a: None)
        assert registry.has_tool("test") is True
        assert registry.has_tool("nonexistent") is False

    def test_call_tool(self):
        registry = ToolRegistry()
        registry.register_tool(
            name="add",
            description="Add numbers",
            parameters={},
            handler=lambda args: args["a"] + args["b"]
        )
        result = registry.call_tool("add", {"a": 5, "b": 3})
        assert result == 8

    def test_call_nonexistent_tool(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError):
            registry.call_tool("nonexistent", {})


# =============================================================================
# Exercise 5: ResourceStore Tests
# =============================================================================

class TestResourceStore:
    def test_add_and_list_resources(self):
        store = ResourceStore()
        store.add_resource("notes://meeting", "Meeting Notes", "Content here")
        resources = store.list_resources()
        assert len(resources) == 1
        assert resources[0]["uri"] == "notes://meeting"

    def test_has_resource(self):
        store = ResourceStore()
        store.add_resource("test://res", "Test", "Content")
        assert store.has_resource("test://res") is True
        assert store.has_resource("test://other") is False

    def test_read_resource(self):
        store = ResourceStore()
        store.add_resource("notes://todo", "Todo", "Buy groceries")
        content = store.read_resource("notes://todo")
        assert content == "Buy groceries"

    def test_read_nonexistent_resource(self):
        store = ResourceStore()
        with pytest.raises(ValueError):
            store.read_resource("notes://nonexistent")

    def test_update_resource(self):
        store = ResourceStore()
        store.add_resource("notes://test", "Test", "Original")
        store.update_resource("notes://test", "Updated content")
        assert store.read_resource("notes://test") == "Updated content"

    def test_delete_resource(self):
        store = ResourceStore()
        store.add_resource("notes://test", "Test", "Content")
        store.delete_resource("notes://test")
        assert store.has_resource("notes://test") is False


# =============================================================================
# Exercise 6: MCPRouter Tests
# =============================================================================

class TestMCPRouter:
    def test_handle_initialize(self):
        router = MCPRouter("test-server", "1.0.0")
        response = router.handle_message(
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
        )
        data = json.loads(response)
        assert "result" in data
        assert data["result"]["name"] == "test-server"

    def test_handle_tools_list(self):
        router = MCPRouter("test-server")
        router.tool_registry.register_tool("test", "Test tool", {}, lambda a: None)
        response = router.handle_message(
            '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
        )
        data = json.loads(response)
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) == 1

    def test_handle_tools_call(self):
        router = MCPRouter("test-server")
        router.tool_registry.register_tool(
            "multiply",
            "Multiply numbers",
            {},
            lambda args: args["a"] * args["b"]
        )
        response = router.handle_message(
            '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"multiply","arguments":{"a":6,"b":7}}}'
        )
        data = json.loads(response)
        assert data["result"]["content"][0]["text"] == "42"

    def test_handle_resources_list(self):
        router = MCPRouter("test-server")
        router.resource_store.add_resource("test://res", "Test", "Content")
        response = router.handle_message(
            '{"jsonrpc":"2.0","id":1,"method":"resources/list","params":{}}'
        )
        data = json.loads(response)
        assert len(data["result"]["resources"]) == 1

    def test_handle_resources_read(self):
        router = MCPRouter("test-server")
        router.resource_store.add_resource("test://data", "Data", "The content")
        response = router.handle_message(
            '{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"uri":"test://data"}}'
        )
        data = json.loads(response)
        assert "The content" in str(data["result"])

    def test_handle_unknown_method(self):
        router = MCPRouter("test-server")
        response = router.handle_message(
            '{"jsonrpc":"2.0","id":1,"method":"unknown/method","params":{}}'
        )
        data = json.loads(response)
        assert "error" in data
        assert data["error"]["code"] == -32601
