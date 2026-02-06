"""
MCP Server Implementation

Build a complete (simulated) MCP server with tools, resources, and prompts.
This exercises the full MCP concept without requiring the actual SDK.

Scenario: You're building a "Notes" MCP server that lets AI:
- Create, read, update, delete notes (Tools)
- Browse existing notes (Resources)
- Use pre-built prompts for common tasks (Prompts)
"""
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import json
import re


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Note:
    """A note with content and metadata."""
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
# EXERCISE 1: Notes Storage
# =============================================================================

class NotesStorage:
    """
    In-memory storage for notes.

    Example:
        >>> storage = NotesStorage()
        >>> note = storage.create("My Note", "Some content", ["work"])
        >>> note.title
        'My Note'
        >>> storage.get(note.id).content
        'Some content'
        >>> storage.search("content")
        [Note(...)]
    """

    def __init__(self):
        self._notes: dict = {}  # id -> Note
        self._next_id = 1

    def create(self, title: str, content: str, tags: list = None) -> Note:
        """
        Create a new note and return it.
        Auto-generates a unique ID like "note_1", "note_2", etc.
        """
        # YOUR CODE HERE
        pass

    def get(self, note_id: str) -> Optional[Note]:
        """Get a note by ID. Returns None if not found."""
        # YOUR CODE HERE
        pass

    def update(self, note_id: str, title: str = None, content: str = None,
               tags: list = None) -> Note:
        """
        Update a note. Only updates fields that are provided (not None).
        Updates the updated_at timestamp.
        Raises ValueError if note not found.
        """
        # YOUR CODE HERE
        pass

    def delete(self, note_id: str) -> bool:
        """Delete a note. Returns True if deleted, False if not found."""
        # YOUR CODE HERE
        pass

    def list_all(self) -> list:
        """Return all notes as a list."""
        # YOUR CODE HERE
        pass

    def search(self, query: str) -> list:
        """
        Search notes by title or content (case-insensitive).
        Returns list of matching notes.
        """
        # YOUR CODE HERE
        pass

    def get_by_tag(self, tag: str) -> list:
        """Return all notes with a specific tag."""
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 2: Tool Definitions for Notes Server
# =============================================================================

def get_notes_tools() -> list:
    """
    Return the tool definitions for the notes server.

    Should define these tools:
    1. create_note - Create a new note
       - title (string, required): Note title
       - content (string, required): Note content
       - tags (array of strings, optional): Tags for the note

    2. get_note - Get a note by ID
       - id (string, required): Note ID

    3. update_note - Update an existing note
       - id (string, required): Note ID
       - title (string, optional): New title
       - content (string, optional): New content
       - tags (array, optional): New tags

    4. delete_note - Delete a note
       - id (string, required): Note ID

    5. search_notes - Search notes
       - query (string, required): Search query

    6. list_notes - List all notes (no parameters)

    Return format:
    [
        {
            "name": "tool_name",
            "description": "What it does",
            "inputSchema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        },
        ...
    ]
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Resource Definitions for Notes Server
# =============================================================================

def get_notes_resources(storage: NotesStorage) -> list:
    """
    Generate resource definitions for all notes in storage.

    Each note becomes a resource with:
    - uri: "notes://{note_id}"
    - name: The note's title
    - description: First 50 chars of content + "..."
    - mimeType: "text/plain"

    Example:
        >>> storage = NotesStorage()
        >>> storage.create("Meeting", "Discuss Q1 goals")
        >>> get_notes_resources(storage)
        [{'uri': 'notes://note_1', 'name': 'Meeting', 'description': 'Discuss Q1 goals...', 'mimeType': 'text/plain'}]
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Prompt Definitions
# =============================================================================

def get_notes_prompts() -> list:
    """
    Return prompt definitions for the notes server.

    Define these prompts:
    1. summarize_note - Summarize a note's content
       Arguments: note_id (required)

    2. brainstorm - Generate ideas based on a topic
       Arguments: topic (required), count (optional, default 5)

    3. meeting_notes - Template for meeting notes
       Arguments: meeting_title (required), attendees (optional)

    Return format:
    [
        {
            "name": "prompt_name",
            "description": "What this prompt does",
            "arguments": [
                {"name": "arg_name", "description": "...", "required": True/False}
            ]
        },
        ...
    ]
    """
    # YOUR CODE HERE
    pass


def render_prompt(prompt_name: str, arguments: dict, storage: NotesStorage) -> list:
    """
    Render a prompt into messages.

    Returns list of messages like:
    [{"role": "user", "content": "..."}]

    For summarize_note:
        Fetch the note content and create a summarization prompt.

    For brainstorm:
        Create a prompt asking for N ideas about the topic.

    For meeting_notes:
        Create a template for meeting notes.

    Example:
        >>> render_prompt("brainstorm", {"topic": "app features", "count": 3}, storage)
        [{"role": "user", "content": "Generate 3 creative ideas about: app features..."}]
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 5: Complete Notes MCP Server
# =============================================================================

class NotesMCPServer:
    """
    A complete MCP server for notes management.

    Handles:
    - initialize
    - tools/list, tools/call
    - resources/list, resources/read
    - prompts/list, prompts/get

    Example:
        >>> server = NotesMCPServer()
        >>> server.handle('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
        '{"jsonrpc": "2.0", "id": 1, "result": {"name": "notes-server", ...}}'
    """

    def __init__(self):
        self.name = "notes-server"
        self.version = "1.0.0"
        self.storage = NotesStorage()

        # Pre-populate with sample notes
        self.storage.create("Welcome", "Welcome to your notes!", ["intro"])
        self.storage.create("Todo", "1. Learn MCP\n2. Build something cool", ["tasks"])

    def handle(self, message: str) -> str:
        """
        Handle an incoming JSON-RPC message.
        Parse, route to handler, return response.
        """
        # YOUR CODE HERE
        pass

    def _create_response(self, request_id: int, result: Any) -> str:
        """Create a success response."""
        # YOUR CODE HERE
        pass

    def _create_error(self, request_id: int, code: int, message: str) -> str:
        """Create an error response."""
        # YOUR CODE HERE
        pass

    # Handler methods
    def _handle_initialize(self, params: dict) -> dict:
        """Return server capabilities."""
        # YOUR CODE HERE
        pass

    def _handle_tools_list(self, params: dict) -> dict:
        """Return available tools."""
        # YOUR CODE HERE
        pass

    def _handle_tools_call(self, params: dict) -> dict:
        """
        Execute a tool and return result.
        params: {"name": "tool_name", "arguments": {...}}
        """
        # YOUR CODE HERE
        pass

    def _handle_resources_list(self, params: dict) -> dict:
        """Return available resources."""
        # YOUR CODE HERE
        pass

    def _handle_resources_read(self, params: dict) -> dict:
        """
        Read a resource.
        params: {"uri": "notes://note_1"}
        """
        # YOUR CODE HERE
        pass

    def _handle_prompts_list(self, params: dict) -> dict:
        """Return available prompts."""
        # YOUR CODE HERE
        pass

    def _handle_prompts_get(self, params: dict) -> dict:
        """
        Get a rendered prompt.
        params: {"name": "prompt_name", "arguments": {...}}
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 6: Integration Test Scenario
# =============================================================================

def run_integration_test():
    """
    Run a full integration test of the notes server.
    This simulates how Claude Desktop would interact with your server.
    """
    server = NotesMCPServer()

    print("=" * 60)
    print("MCP Notes Server Integration Test")
    print("=" * 60)

    # Step 1: Initialize
    print("\n1. Initialize connection...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }))
    print(f"   Response: {response[:100]}...")

    # Step 2: List available tools
    print("\n2. List available tools...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }))
    result = json.loads(response)
    tools = result.get("result", {}).get("tools", [])
    print(f"   Found {len(tools)} tools: {[t['name'] for t in tools]}")

    # Step 3: Create a note
    print("\n3. Create a new note...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "create_note",
            "arguments": {
                "title": "MCP Learning",
                "content": "MCP is a protocol for connecting AI to tools and data.",
                "tags": ["learning", "mcp"]
            }
        }
    }))
    print(f"   Response: {response}")

    # Step 4: List resources (should include new note)
    print("\n4. List resources...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "resources/list",
        "params": {}
    }))
    result = json.loads(response)
    resources = result.get("result", {}).get("resources", [])
    print(f"   Found {len(resources)} resources:")
    for r in resources:
        print(f"     - {r['uri']}: {r['name']}")

    # Step 5: Read a resource
    print("\n5. Read a resource...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "resources/read",
        "params": {"uri": "notes://note_1"}
    }))
    print(f"   Response: {response}")

    # Step 6: Search notes
    print("\n6. Search for 'MCP'...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "search_notes",
            "arguments": {"query": "MCP"}
        }
    }))
    print(f"   Response: {response}")

    # Step 7: List prompts
    print("\n7. List prompts...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "prompts/list",
        "params": {}
    }))
    result = json.loads(response)
    prompts = result.get("result", {}).get("prompts", [])
    print(f"   Found {len(prompts)} prompts: {[p['name'] for p in prompts]}")

    # Step 8: Get a prompt
    print("\n8. Get brainstorm prompt...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "prompts/get",
        "params": {
            "name": "brainstorm",
            "arguments": {"topic": "MCP server ideas", "count": 3}
        }
    }))
    print(f"   Response: {response}")

    # Step 9: Update a note
    print("\n9. Update a note...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "update_note",
            "arguments": {
                "id": "note_1",
                "content": "Welcome to your notes! Updated with more info."
            }
        }
    }))
    print(f"   Response: {response}")

    # Step 10: Delete a note
    print("\n10. Delete a note...")
    response = server.handle(json.dumps({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "delete_note",
            "arguments": {"id": "note_2"}
        }
    }))
    print(f"   Response: {response}")

    print("\n" + "=" * 60)
    print("Integration test complete!")
    print("=" * 60)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    run_integration_test()
