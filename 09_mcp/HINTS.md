# Hints - MCP (Model Context Protocol)

## P1: MCP Basics

### ToolDefinition

**get_required_params:**
- Look in `self.parameters` for the "required" key
- Return `self.parameters.get("required", [])`

**get_param_names:**
- Properties are in `self.parameters.get("properties", {})`
- Return the keys: `list(properties.keys())`

**validate_arguments:**
- Loop through required params
- Check if each is in arguments
- Return `(True, None)` if valid, `(False, "Missing: X")` if not

**to_dict:**
```python
return {
    "name": self.name,
    "description": self.description,
    "inputSchema": self.parameters
}
```

### ResourceDefinition

**get_scheme:**
- Split on "://" and take the first part
- `self.uri.split("://")[0]`

**get_path:**
- Split on "://" and take the second part
- Handle edge cases (file:/// has three slashes)

**to_dict:**
```python
return {
    "uri": self.uri,
    "name": self.name,
    "description": self.description,
    "mimeType": self.mime_type  # Note: camelCase for MCP
}
```

### JsonRpcHandler

**parse_request:**
1. `json.loads(message)` to parse
2. Check for "method" key, raise ValueError if missing
3. Return dict with id, method, params (default to {})

**create_response:**
```python
return json.dumps({
    "jsonrpc": "2.0",
    "id": request_id,
    "result": result
})
```

**create_error:**
```python
return json.dumps({
    "jsonrpc": "2.0",
    "id": request_id,
    "error": {"code": code, "message": message}
})
```

### ToolRegistry

**register_tool:**
- Create a ToolDefinition
- Store tuple: `self._tools[name] = (tool_def, handler)`

**list_tools:**
- Loop through tools, call `.to_dict()` on each ToolDefinition
- `[tool_def.to_dict() for tool_def, _ in self._tools.values()]`

**call_tool:**
- Check if tool exists, raise ValueError if not
- Get the handler and call it: `handler(arguments)`

### ResourceStore

**add_resource:**
- Create ResourceDefinition
- Store: `self._resources[uri] = (resource_def, content)`

**read_resource:**
- Check if uri exists, raise ValueError if not
- Return the content (second element of tuple)

### MCPRouter

**handle_message:**
1. Parse with `self._json_rpc.parse_request(message)`
2. Extract id, method, params
3. Route based on method name
4. Call appropriate handler
5. Return JSON-RPC response

**Method routing:**
```python
if method == "initialize":
    return self._handle_initialize(request_id, params)
elif method == "tools/list":
    return self._handle_tools_list(request_id, params)
# ... etc
else:
    return self._json_rpc.create_error(request_id, -32601, "Method not found")
```

**_handle_tools_call:**
1. Extract name and arguments from params
2. Call `self.tool_registry.call_tool(name, arguments)`
3. Wrap result in MCP content format:
```python
result = {"content": [{"type": "text", "text": str(tool_result)}]}
return self._json_rpc.create_response(request_id, result)
```

---

## P2: MCP Server

### NotesStorage

**create:**
1. Generate ID: `f"note_{self._next_id}"` then increment
2. Create Note object with all fields
3. Store in `self._notes[note_id]`
4. Return the note

**update:**
1. Check if note exists, raise ValueError if not
2. Update only non-None fields
3. Always update `updated_at`
4. Return the note

**search:**
```python
query_lower = query.lower()
return [note for note in self._notes.values()
        if query_lower in note.title.lower()
        or query_lower in note.content.lower()]
```

### get_notes_tools

Return a list of dicts, each with:
- `name`: Tool name (e.g., "create_note")
- `description`: What it does
- `inputSchema`: JSON Schema for inputs

Example:
```python
{
    "name": "create_note",
    "description": "Create a new note",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Note title"},
            "content": {"type": "string", "description": "Note content"}
        },
        "required": ["title", "content"]
    }
}
```

### get_notes_resources

Loop through storage notes:
```python
for note in storage.list_all():
    resources.append({
        "uri": f"notes://{note.id}",
        "name": note.title,
        "description": note.content[:50] + "...",
        "mimeType": "text/plain"
    })
```

### render_prompt

Switch on prompt_name, build appropriate message:

```python
if prompt_name == "brainstorm":
    topic = arguments["topic"]
    count = arguments.get("count", 5)
    return [{
        "role": "user",
        "content": f"Generate {count} ideas about: {topic}"
    }]
```

### NotesMCPServer.handle

1. Parse JSON: `data = json.loads(message)`
2. Extract: `request_id`, `method`, `params`
3. Use a dict to route methods:
```python
handlers = {
    "initialize": self._handle_initialize,
    "tools/list": self._handle_tools_list,
    # ...
}
if method in handlers:
    result = handlers[method](params)
    return self._create_response(request_id, result)
```

### _handle_tools_call

```python
name = params["name"]
arguments = params.get("arguments", {})

if name == "create_note":
    note = self.storage.create(
        title=arguments["title"],
        content=arguments["content"],
        tags=arguments.get("tags", [])
    )
    result = f"Created: {note.id}"

return {"content": [{"type": "text", "text": result}]}
```

### _handle_resources_read

```python
uri = params["uri"]
note_id = uri.replace("notes://", "")
note = self.storage.get(note_id)

if note:
    return {"content": [{"type": "text", "text": note.content}]}
else:
    return {"content": [{"type": "text", "text": "Not found"}]}
```

---

## Common MCP Patterns

### Tool Result Format
```python
{
    "content": [
        {"type": "text", "text": "Result here"}
    ]
}
```

### Resource Format
```python
{
    "uri": "scheme://path",
    "name": "Display Name",
    "description": "What it is",
    "mimeType": "text/plain"
}
```

### Prompt Message Format
```python
[
    {"role": "user", "content": "The prompt text..."}
]
```

### Error Codes
- `-32700`: Parse error
- `-32600`: Invalid request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error
