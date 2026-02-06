"""
Solutions for MCP Server exercises.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import json


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Note:
    id: str
    title: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags
        }


# =============================================================================
# EXERCISE 1: Notes Storage - SOLUTION
# =============================================================================

class NotesStorage:
    def __init__(self):
        self._notes: dict = {}
        self._next_id = 1

    def create(self, title: str, content: str, tags: list = None) -> Note:
        """Create a new note and return it."""
        note_id = f"note_{self._next_id}"
        self._next_id += 1

        note = Note(
            id=note_id,
            title=title,
            content=content,
            tags=tags if tags else []
        )
        self._notes[note_id] = note
        return note

    def get(self, note_id: str) -> Optional[Note]:
        """Get a note by ID."""
        return self._notes.get(note_id)

    def update(self, note_id: str, title: str = None, content: str = None,
               tags: list = None) -> Note:
        """Update a note."""
        if note_id not in self._notes:
            raise ValueError(f"Note not found: {note_id}")

        note = self._notes[note_id]
        if title is not None:
            note.title = title
        if content is not None:
            note.content = content
        if tags is not None:
            note.tags = tags
        note.updated_at = datetime.now().isoformat()

        return note

    def delete(self, note_id: str) -> bool:
        """Delete a note."""
        if note_id in self._notes:
            del self._notes[note_id]
            return True
        return False

    def list_all(self) -> list:
        """Return all notes as a list."""
        return list(self._notes.values())

    def search(self, query: str) -> list:
        """Search notes by title or content."""
        query_lower = query.lower()
        results = []
        for note in self._notes.values():
            if query_lower in note.title.lower() or query_lower in note.content.lower():
                results.append(note)
        return results

    def get_by_tag(self, tag: str) -> list:
        """Return all notes with a specific tag."""
        return [note for note in self._notes.values() if tag in note.tags]


# =============================================================================
# EXERCISE 2: Tool Definitions - SOLUTION
# =============================================================================

def get_notes_tools() -> list:
    """Return the tool definitions for the notes server."""
    return [
        {
            "name": "create_note",
            "description": "Create a new note",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Note title"},
                    "content": {"type": "string", "description": "Note content"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the note"
                    }
                },
                "required": ["title", "content"]
            }
        },
        {
            "name": "get_note",
            "description": "Get a note by ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Note ID"}
                },
                "required": ["id"]
            }
        },
        {
            "name": "update_note",
            "description": "Update an existing note",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Note ID"},
                    "title": {"type": "string", "description": "New title"},
                    "content": {"type": "string", "description": "New content"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags"
                    }
                },
                "required": ["id"]
            }
        },
        {
            "name": "delete_note",
            "description": "Delete a note",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Note ID"}
                },
                "required": ["id"]
            }
        },
        {
            "name": "search_notes",
            "description": "Search notes by title or content",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "list_notes",
            "description": "List all notes",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ]


# =============================================================================
# EXERCISE 3: Resource Definitions - SOLUTION
# =============================================================================

def get_notes_resources(storage: NotesStorage) -> list:
    """Generate resource definitions for all notes in storage."""
    resources = []
    for note in storage.list_all():
        description = note.content[:50]
        if len(note.content) > 50:
            description += "..."

        resources.append({
            "uri": f"notes://{note.id}",
            "name": note.title,
            "description": description,
            "mimeType": "text/plain"
        })
    return resources


# =============================================================================
# EXERCISE 4: Prompt Definitions - SOLUTION
# =============================================================================

def get_notes_prompts() -> list:
    """Return prompt definitions for the notes server."""
    return [
        {
            "name": "summarize_note",
            "description": "Summarize a note's content",
            "arguments": [
                {"name": "note_id", "description": "ID of the note to summarize", "required": True}
            ]
        },
        {
            "name": "brainstorm",
            "description": "Generate ideas based on a topic",
            "arguments": [
                {"name": "topic", "description": "Topic to brainstorm about", "required": True},
                {"name": "count", "description": "Number of ideas (default 5)", "required": False}
            ]
        },
        {
            "name": "meeting_notes",
            "description": "Template for meeting notes",
            "arguments": [
                {"name": "meeting_title", "description": "Title of the meeting", "required": True},
                {"name": "attendees", "description": "List of attendees", "required": False}
            ]
        }
    ]


def render_prompt(prompt_name: str, arguments: dict, storage: NotesStorage) -> list:
    """Render a prompt into messages."""
    if prompt_name == "summarize_note":
        note_id = arguments["note_id"]
        note = storage.get(note_id)
        if note:
            content = f"Please summarize the following note:\n\nTitle: {note.title}\n\nContent:\n{note.content}"
        else:
            content = f"Note with ID '{note_id}' not found."
        return [{"role": "user", "content": content}]

    elif prompt_name == "brainstorm":
        topic = arguments["topic"]
        count = arguments.get("count", 5)
        content = f"""Generate {count} creative ideas about: {topic}

Please provide:
1. Each idea as a numbered item
2. A brief explanation for each idea
3. Consider both practical and innovative approaches"""
        return [{"role": "user", "content": content}]

    elif prompt_name == "meeting_notes":
        title = arguments["meeting_title"]
        attendees = arguments.get("attendees", "")

        content = f"""Create meeting notes for: {title}

Attendees: {attendees if attendees else 'TBD'}

Template:
## Meeting: {title}
**Date:** [Date]
**Attendees:** {attendees if attendees else '[List attendees]'}

### Agenda
-

### Discussion Points
-

### Action Items
- [ ]

### Next Steps
- """
        return [{"role": "user", "content": content}]

    raise ValueError(f"Unknown prompt: {prompt_name}")


# =============================================================================
# EXERCISE 5: Complete Notes MCP Server - SOLUTION
# =============================================================================

class NotesMCPServer:
    def __init__(self):
        self.name = "notes-server"
        self.version = "1.0.0"
        self.storage = NotesStorage()

        # Pre-populate with sample notes
        self.storage.create("Welcome", "Welcome to your notes!", ["intro"])
        self.storage.create("Todo", "1. Learn MCP\n2. Build something cool", ["tasks"])

    def handle(self, message: str) -> str:
        """Handle an incoming JSON-RPC message."""
        try:
            data = json.loads(message)
            request_id = data.get("id", 0)
            method = data.get("method", "")
            params = data.get("params", {})

            # Route to handler
            handlers = {
                "initialize": self._handle_initialize,
                "tools/list": self._handle_tools_list,
                "tools/call": self._handle_tools_call,
                "resources/list": self._handle_resources_list,
                "resources/read": self._handle_resources_read,
                "prompts/list": self._handle_prompts_list,
                "prompts/get": self._handle_prompts_get,
            }

            if method in handlers:
                result = handlers[method](params)
                return self._create_response(request_id, result)
            else:
                return self._create_error(request_id, -32601, f"Method not found: {method}")

        except json.JSONDecodeError:
            return self._create_error(0, -32700, "Parse error")
        except Exception as e:
            return self._create_error(0, -32603, str(e))

    def _create_response(self, request_id: int, result: Any) -> str:
        """Create a success response."""
        return json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        })

    def _create_error(self, request_id: int, code: int, message: str) -> str:
        """Create an error response."""
        return json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message}
        })

    def _handle_initialize(self, params: dict) -> dict:
        """Return server capabilities."""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": True
            }
        }

    def _handle_tools_list(self, params: dict) -> dict:
        """Return available tools."""
        return {"tools": get_notes_tools()}

    def _handle_tools_call(self, params: dict) -> dict:
        """Execute a tool and return result."""
        name = params["name"]
        arguments = params.get("arguments", {})

        try:
            if name == "create_note":
                note = self.storage.create(
                    title=arguments["title"],
                    content=arguments["content"],
                    tags=arguments.get("tags", [])
                )
                result = f"Created note: {note.id}"

            elif name == "get_note":
                note = self.storage.get(arguments["id"])
                if note:
                    result = json.dumps(note.to_dict())
                else:
                    result = f"Note not found: {arguments['id']}"

            elif name == "update_note":
                note = self.storage.update(
                    note_id=arguments["id"],
                    title=arguments.get("title"),
                    content=arguments.get("content"),
                    tags=arguments.get("tags")
                )
                result = f"Updated note: {note.id}"

            elif name == "delete_note":
                deleted = self.storage.delete(arguments["id"])
                result = "Deleted" if deleted else "Not found"

            elif name == "search_notes":
                notes = self.storage.search(arguments["query"])
                result = json.dumps([n.to_dict() for n in notes])

            elif name == "list_notes":
                notes = self.storage.list_all()
                result = json.dumps([n.to_dict() for n in notes])

            else:
                result = f"Unknown tool: {name}"

            return {"content": [{"type": "text", "text": result}]}

        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}]}

    def _handle_resources_list(self, params: dict) -> dict:
        """Return available resources."""
        return {"resources": get_notes_resources(self.storage)}

    def _handle_resources_read(self, params: dict) -> dict:
        """Read a resource."""
        uri = params["uri"]
        # Parse URI: "notes://note_1" -> "note_1"
        note_id = uri.replace("notes://", "")
        note = self.storage.get(note_id)

        if note:
            return {
                "content": [{"type": "text", "text": note.content}],
                "metadata": {"title": note.title, "tags": note.tags}
            }
        else:
            return {"content": [{"type": "text", "text": f"Note not found: {note_id}"}]}

    def _handle_prompts_list(self, params: dict) -> dict:
        """Return available prompts."""
        return {"prompts": get_notes_prompts()}

    def _handle_prompts_get(self, params: dict) -> dict:
        """Get a rendered prompt."""
        name = params["name"]
        arguments = params.get("arguments", {})

        messages = render_prompt(name, arguments, self.storage)
        return {"messages": messages}


# =============================================================================
# Integration Test
# =============================================================================

def run_integration_test():
    """Run a full integration test of the notes server."""
    server = NotesMCPServer()

    print("=" * 60)
    print("MCP Notes Server Integration Test")
    print("=" * 60)

    # Test all the operations
    tests = [
        ("initialize", {}),
        ("tools/list", {}),
        ("tools/call", {"name": "create_note", "arguments": {"title": "Test", "content": "Content"}}),
        ("resources/list", {}),
        ("resources/read", {"uri": "notes://note_1"}),
        ("prompts/list", {}),
        ("prompts/get", {"name": "brainstorm", "arguments": {"topic": "AI", "count": 3}}),
    ]

    for i, (method, params) in enumerate(tests, 1):
        print(f"\n{i}. Testing {method}...")
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": i,
            "method": method,
            "params": params
        }))
        data = json.loads(response)
        if "error" in data:
            print(f"   Error: {data['error']}")
        else:
            print(f"   Success: {str(data['result'])[:100]}...")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_test()
