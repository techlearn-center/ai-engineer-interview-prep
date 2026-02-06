"""
Solutions for MCP Basics exercises.
"""
from dataclasses import dataclass, field
from typing import Any
import json
import re


# =============================================================================
# EXERCISE 1: Tool Definition - SOLUTION
# =============================================================================

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)

    def get_required_params(self) -> list:
        """Return list of required parameter names."""
        return self.parameters.get("required", [])

    def get_param_names(self) -> list:
        """Return list of all parameter names."""
        properties = self.parameters.get("properties", {})
        return list(properties.keys())

    def validate_arguments(self, arguments: dict) -> tuple:
        """Validate that arguments match the schema."""
        required = self.get_required_params()
        for param in required:
            if param not in arguments:
                return (False, f"Missing required parameter: {param}")
        return (True, None)

    def to_dict(self) -> dict:
        """Convert to MCP-style dict format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters
        }


# =============================================================================
# EXERCISE 2: Resource Definition - SOLUTION
# =============================================================================

@dataclass
class ResourceDefinition:
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"

    def get_scheme(self) -> str:
        """Extract the scheme from the URI."""
        # "file:///path" -> "file", "db://users" -> "db"
        if "://" in self.uri:
            return self.uri.split("://")[0]
        return ""

    def get_path(self) -> str:
        """Extract the path from the URI."""
        if "://" in self.uri:
            parts = self.uri.split("://", 1)
            path = parts[1] if len(parts) > 1 else ""
            # Handle file:/// (three slashes)
            if path.startswith("/"):
                return path
            return path
        return self.uri

    def to_dict(self) -> dict:
        """Convert to MCP-style dict format."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


# =============================================================================
# EXERCISE 3: JSON-RPC Message Handling - SOLUTION
# =============================================================================

class JsonRpcHandler:
    def parse_request(self, message: str) -> dict:
        """Parse a JSON-RPC request string."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON")

        if "method" not in data:
            raise ValueError("Missing 'method' field")

        return {
            "id": data.get("id"),
            "method": data["method"],
            "params": data.get("params", {})
        }

    def create_response(self, request_id: int, result: Any) -> str:
        """Create a JSON-RPC success response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
        return json.dumps(response)

    def create_error(self, request_id: int, code: int, message: str) -> str:
        """Create a JSON-RPC error response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        return json.dumps(response)


# =============================================================================
# EXERCISE 4: Tool Registry - SOLUTION
# =============================================================================

class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register_tool(self, name: str, description: str, parameters: dict, handler: callable):
        """Register a new tool with its handler function."""
        tool_def = ToolDefinition(name=name, description=description, parameters=parameters)
        self._tools[name] = (tool_def, handler)

    def list_tools(self) -> list:
        """Return list of tool definitions in MCP format."""
        return [tool_def.to_dict() for tool_def, _ in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools

    def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool by name with given arguments."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")

        _, handler = self._tools[name]
        return handler(arguments)


# =============================================================================
# EXERCISE 5: Resource Store - SOLUTION
# =============================================================================

class ResourceStore:
    def __init__(self):
        self._resources: dict = {}

    def add_resource(self, uri: str, name: str, content: str,
                     description: str = "", mime_type: str = "text/plain"):
        """Add a resource to the store."""
        resource_def = ResourceDefinition(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type
        )
        self._resources[uri] = (resource_def, content)

    def list_resources(self) -> list:
        """Return list of resource definitions in MCP format."""
        return [res_def.to_dict() for res_def, _ in self._resources.values()]

    def has_resource(self, uri: str) -> bool:
        """Check if a resource exists."""
        return uri in self._resources

    def read_resource(self, uri: str) -> str:
        """Read a resource's content by URI."""
        if uri not in self._resources:
            raise ValueError(f"Resource not found: {uri}")
        _, content = self._resources[uri]
        return content

    def update_resource(self, uri: str, content: str):
        """Update an existing resource's content."""
        if uri not in self._resources:
            raise ValueError(f"Resource not found: {uri}")
        res_def, _ = self._resources[uri]
        self._resources[uri] = (res_def, content)

    def delete_resource(self, uri: str):
        """Delete a resource."""
        if uri not in self._resources:
            raise ValueError(f"Resource not found: {uri}")
        del self._resources[uri]


# =============================================================================
# EXERCISE 6: Simple MCP Message Router - SOLUTION
# =============================================================================

class MCPRouter:
    def __init__(self, server_name: str, version: str = "1.0.0"):
        self.server_name = server_name
        self.version = version
        self.tool_registry = ToolRegistry()
        self.resource_store = ResourceStore()
        self._json_rpc = JsonRpcHandler()

    def handle_message(self, message: str) -> str:
        """Parse and route an MCP message, returning the response."""
        try:
            request = self._json_rpc.parse_request(message)
            request_id = request["id"]
            method = request["method"]
            params = request["params"]

            # Route to appropriate handler
            if method == "initialize":
                return self._handle_initialize(request_id, params)
            elif method == "tools/list":
                return self._handle_tools_list(request_id, params)
            elif method == "tools/call":
                return self._handle_tools_call(request_id, params)
            elif method == "resources/list":
                return self._handle_resources_list(request_id, params)
            elif method == "resources/read":
                return self._handle_resources_read(request_id, params)
            else:
                return self._json_rpc.create_error(
                    request_id, -32601, f"Method not found: {method}"
                )

        except ValueError as e:
            return self._json_rpc.create_error(0, -32600, str(e))
        except Exception as e:
            return self._json_rpc.create_error(0, -32603, str(e))

    def _handle_initialize(self, request_id: int, params: dict) -> str:
        """Handle initialize request."""
        result = {
            "name": self.server_name,
            "version": self.version,
            "capabilities": {
                "tools": True,
                "resources": True
            }
        }
        return self._json_rpc.create_response(request_id, result)

    def _handle_tools_list(self, request_id: int, params: dict) -> str:
        """Handle tools/list request."""
        result = {"tools": self.tool_registry.list_tools()}
        return self._json_rpc.create_response(request_id, result)

    def _handle_tools_call(self, request_id: int, params: dict) -> str:
        """Handle tools/call request."""
        try:
            name = params["name"]
            arguments = params.get("arguments", {})
            tool_result = self.tool_registry.call_tool(name, arguments)
            result = {
                "content": [
                    {"type": "text", "text": str(tool_result)}
                ]
            }
            return self._json_rpc.create_response(request_id, result)
        except Exception as e:
            return self._json_rpc.create_error(request_id, -32602, str(e))

    def _handle_resources_list(self, request_id: int, params: dict) -> str:
        """Handle resources/list request."""
        result = {"resources": self.resource_store.list_resources()}
        return self._json_rpc.create_response(request_id, result)

    def _handle_resources_read(self, request_id: int, params: dict) -> str:
        """Handle resources/read request."""
        try:
            uri = params["uri"]
            content = self.resource_store.read_resource(uri)
            result = {
                "content": [
                    {"type": "text", "text": content}
                ]
            }
            return self._json_rpc.create_response(request_id, result)
        except Exception as e:
            return self._json_rpc.create_error(request_id, -32602, str(e))
