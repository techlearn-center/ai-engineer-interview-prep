"""Tests for MCP Server exercises."""
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from p2_mcp_server import (
    Note, NotesStorage, get_notes_tools, get_notes_resources,
    get_notes_prompts, render_prompt, NotesMCPServer
)


# =============================================================================
# Note Tests
# =============================================================================

class TestNote:
    def test_note_creation(self):
        note = Note(id="1", title="Test", content="Content")
        assert note.id == "1"
        assert note.title == "Test"
        assert note.content == "Content"

    def test_note_to_dict(self):
        note = Note(id="1", title="Test", content="Content", tags=["a", "b"])
        d = note.to_dict()
        assert d["id"] == "1"
        assert d["title"] == "Test"
        assert d["tags"] == ["a", "b"]


# =============================================================================
# Exercise 1: NotesStorage Tests
# =============================================================================

class TestNotesStorage:
    def test_create_note(self):
        storage = NotesStorage()
        note = storage.create("My Title", "My content", ["tag1"])
        assert note.title == "My Title"
        assert note.content == "My content"
        assert "tag1" in note.tags
        assert note.id is not None

    def test_get_note(self):
        storage = NotesStorage()
        created = storage.create("Test", "Content")
        retrieved = storage.get(created.id)
        assert retrieved is not None
        assert retrieved.title == "Test"

    def test_get_nonexistent_note(self):
        storage = NotesStorage()
        assert storage.get("nonexistent") is None

    def test_update_note(self):
        storage = NotesStorage()
        note = storage.create("Original", "Original content")
        updated = storage.update(note.id, title="Updated Title")
        assert updated.title == "Updated Title"
        assert updated.content == "Original content"

    def test_update_nonexistent_note(self):
        storage = NotesStorage()
        with pytest.raises(ValueError):
            storage.update("nonexistent", title="New")

    def test_delete_note(self):
        storage = NotesStorage()
        note = storage.create("Test", "Content")
        assert storage.delete(note.id) is True
        assert storage.get(note.id) is None

    def test_delete_nonexistent_note(self):
        storage = NotesStorage()
        assert storage.delete("nonexistent") is False

    def test_list_all(self):
        storage = NotesStorage()
        storage.create("Note 1", "Content 1")
        storage.create("Note 2", "Content 2")
        notes = storage.list_all()
        assert len(notes) == 2

    def test_search(self):
        storage = NotesStorage()
        storage.create("Python Tutorial", "Learn Python basics")
        storage.create("JavaScript Guide", "Learn JS")
        storage.create("Python Advanced", "Advanced Python")

        results = storage.search("python")
        assert len(results) == 2

    def test_get_by_tag(self):
        storage = NotesStorage()
        storage.create("Work Note", "Content", ["work"])
        storage.create("Personal Note", "Content", ["personal"])
        storage.create("Another Work Note", "Content", ["work", "important"])

        work_notes = storage.get_by_tag("work")
        assert len(work_notes) == 2


# =============================================================================
# Exercise 2: Tool Definitions Tests
# =============================================================================

class TestNotesTools:
    def test_get_notes_tools(self):
        tools = get_notes_tools()
        assert len(tools) >= 6

        tool_names = [t["name"] for t in tools]
        assert "create_note" in tool_names
        assert "get_note" in tool_names
        assert "update_note" in tool_names
        assert "delete_note" in tool_names
        assert "search_notes" in tool_names
        assert "list_notes" in tool_names

    def test_create_note_tool_schema(self):
        tools = get_notes_tools()
        create_tool = next(t for t in tools if t["name"] == "create_note")

        assert "inputSchema" in create_tool
        schema = create_tool["inputSchema"]
        assert "title" in schema["properties"]
        assert "content" in schema["properties"]
        assert "title" in schema.get("required", [])
        assert "content" in schema.get("required", [])


# =============================================================================
# Exercise 3: Resource Definitions Tests
# =============================================================================

class TestNotesResources:
    def test_get_notes_resources(self):
        storage = NotesStorage()
        storage.create("Meeting Notes", "Discuss Q1 goals")
        storage.create("Todo List", "Tasks for today")

        resources = get_notes_resources(storage)
        assert len(resources) == 2

        uris = [r["uri"] for r in resources]
        assert any("note_" in uri for uri in uris)

    def test_resource_format(self):
        storage = NotesStorage()
        storage.create("Test Note", "This is the content of the note")

        resources = get_notes_resources(storage)
        assert len(resources) == 1

        resource = resources[0]
        assert "uri" in resource
        assert "name" in resource
        assert "mimeType" in resource


# =============================================================================
# Exercise 4: Prompt Tests
# =============================================================================

class TestNotesPrompts:
    def test_get_notes_prompts(self):
        prompts = get_notes_prompts()
        assert len(prompts) >= 3

        prompt_names = [p["name"] for p in prompts]
        assert "summarize_note" in prompt_names
        assert "brainstorm" in prompt_names
        assert "meeting_notes" in prompt_names

    def test_render_brainstorm_prompt(self):
        storage = NotesStorage()
        messages = render_prompt(
            "brainstorm",
            {"topic": "app features", "count": 5},
            storage
        )

        assert len(messages) >= 1
        assert messages[0]["role"] == "user"
        assert "app features" in messages[0]["content"]
        assert "5" in messages[0]["content"]

    def test_render_summarize_prompt(self):
        storage = NotesStorage()
        note = storage.create("Test", "This is important content to summarize")

        messages = render_prompt(
            "summarize_note",
            {"note_id": note.id},
            storage
        )

        assert len(messages) >= 1
        assert "summarize" in messages[0]["content"].lower() or "content" in messages[0]["content"]


# =============================================================================
# Exercise 5: NotesMCPServer Tests
# =============================================================================

class TestNotesMCPServer:
    def test_initialize(self):
        server = NotesMCPServer()
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }))
        data = json.loads(response)
        assert "result" in data
        assert data["result"]["name"] == "notes-server"

    def test_tools_list(self):
        server = NotesMCPServer()
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }))
        data = json.loads(response)
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) >= 6

    def test_tools_call_create_note(self):
        server = NotesMCPServer()
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "create_note",
                "arguments": {
                    "title": "New Note",
                    "content": "Note content",
                    "tags": ["test"]
                }
            }
        }))
        data = json.loads(response)
        assert "result" in data
        # Should have the note in the result
        result_str = json.dumps(data["result"])
        assert "New Note" in result_str or "note_" in result_str

    def test_tools_call_search_notes(self):
        server = NotesMCPServer()
        # Server has pre-populated notes
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_notes",
                "arguments": {"query": "Welcome"}
            }
        }))
        data = json.loads(response)
        assert "result" in data

    def test_resources_list(self):
        server = NotesMCPServer()
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/list",
            "params": {}
        }))
        data = json.loads(response)
        assert "resources" in data["result"]
        # Should have pre-populated notes as resources
        assert len(data["result"]["resources"]) >= 2

    def test_resources_read(self):
        server = NotesMCPServer()
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "notes://note_1"}
        }))
        data = json.loads(response)
        assert "result" in data
        # Should contain the note content
        result_str = json.dumps(data["result"])
        assert "Welcome" in result_str

    def test_prompts_list(self):
        server = NotesMCPServer()
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/list",
            "params": {}
        }))
        data = json.loads(response)
        assert "prompts" in data["result"]
        assert len(data["result"]["prompts"]) >= 3

    def test_prompts_get(self):
        server = NotesMCPServer()
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {
                "name": "brainstorm",
                "arguments": {"topic": "testing", "count": 3}
            }
        }))
        data = json.loads(response)
        assert "result" in data
        assert "messages" in data["result"]

    def test_error_handling(self):
        server = NotesMCPServer()
        response = server.handle(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method",
            "params": {}
        }))
        data = json.loads(response)
        assert "error" in data
        assert data["error"]["code"] == -32601
