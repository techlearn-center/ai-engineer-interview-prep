# Learn: MCP (Model Context Protocol) - From Zero to Hero

MCP is Anthropic's open standard for connecting AI assistants to external data and tools.
Think of it as a "USB port" for AI - a universal way to plug in capabilities.

---

## 1. What Problem Does MCP Solve?

**Before MCP:**
```
Every AI app builds its own integrations:

Claude App ──────► Custom Slack code
                 ► Custom GitHub code
                 ► Custom Database code

ChatGPT App ─────► Different Slack code
                 ► Different GitHub code
                 ► Different Database code

100 apps × 50 tools = 5,000 custom integrations 😱
```

**With MCP:**
```
One standard protocol, everyone uses it:

┌─────────────────────────────────────────────────┐
│                    MCP Protocol                  │
└─────────────────────────────────────────────────┘
        ▲           ▲           ▲
        │           │           │
   Claude App   ChatGPT App  Other Apps
        │           │           │
        ▼           ▼           ▼
┌─────────────────────────────────────────────────┐
│                    MCP Protocol                  │
└─────────────────────────────────────────────────┘
        │           │           │
        ▼           ▼           ▼
   Slack Server  GitHub Server  DB Server

Any app can use any server! 🎉
```

---

## 2. MCP Architecture: The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP HOST                              │
│  (Claude Desktop, VS Code, your app)                        │
│                                                              │
│    ┌────────────────────────────────────────────────────┐   │
│    │                  MCP CLIENT                         │   │
│    │  (Connects to servers, manages communication)       │   │
│    └────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ MCP Protocol (JSON-RPC)
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ MCP Server  │    │ MCP Server  │    │ MCP Server  │
│  (Files)    │    │  (GitHub)   │    │ (Database)  │
│             │    │             │    │             │
│ - Tools     │    │ - Tools     │    │ - Tools     │
│ - Resources │    │ - Resources │    │ - Resources │
│ - Prompts   │    │ - Prompts   │    │ - Prompts   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Key terms:**
- **Host:** The application running the AI (Claude Desktop, VS Code, your app)
- **Client:** The part that connects to MCP servers (built into the host)
- **Server:** A program that provides tools/data to the AI
- **Protocol:** JSON-RPC over stdio or HTTP (how they communicate)

---

## 3. What Can MCP Servers Provide?

MCP servers expose three types of capabilities:

### Tools (Actions the AI can take)
```python
# Example: A file system server might provide these tools:
tools = [
    "read_file",      # Read a file's contents
    "write_file",     # Write to a file
    "list_directory", # List files in a folder
    "search_files",   # Search for files by name
]

# The AI can call these tools when needed:
# User: "What's in config.json?"
# AI: *calls read_file("config.json")* → returns contents
```

### Resources (Data the AI can access)
```python
# Example: A database server might expose these resources:
resources = [
    "db://users",     # The users table
    "db://orders",    # The orders table
    "db://products",  # The products table
]

# Resources have URIs and can be read by the AI
# User: "Show me recent orders"
# AI: *reads db://orders resource* → gets order data
```

### Prompts (Pre-built prompt templates)
```python
# Example: A code review server might have these prompts:
prompts = [
    "review_code",     # Template for code review
    "explain_error",   # Template for explaining errors
    "suggest_refactor", # Template for refactoring suggestions
]

# User can invoke: "Use the review_code prompt on main.py"
# Server provides a well-crafted prompt template
```

---

## 4. Your First MCP Server (Python)

Let's build a simple MCP server that provides a calculator tool.

```python
# calculator_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio

# Create the server
server = Server("calculator")

# Define available tools
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="calculate",
            description="Perform a math calculation",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression like '2 + 2' or '10 * 5'"
                    }
                },
                "required": ["expression"]
            }
        )
    ]

# Handle tool calls
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "calculate":
        expression = arguments["expression"]
        try:
            # Safety: only allow safe math operations
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return [TextContent(type="text", text="Error: Invalid characters")]

            result = eval(expression)
            return [TextContent(type="text", text=f"Result: {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]

# Run the server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. How the Communication Works

MCP uses JSON-RPC 2.0 over stdio (standard input/output):

```
Host (Claude)                    MCP Server (Calculator)
     │                                    │
     │  ──── Initialize ────────────────► │
     │  {"jsonrpc":"2.0",                 │
     │   "method":"initialize",           │
     │   "params":{...}}                  │
     │                                    │
     │  ◄──── Server Info ─────────────── │
     │  {"jsonrpc":"2.0",                 │
     │   "result":{"name":"calculator"}}  │
     │                                    │
     │  ──── List Tools ────────────────► │
     │  {"method":"tools/list"}           │
     │                                    │
     │  ◄──── Available Tools ─────────── │
     │  {"result":{"tools":[              │
     │     {"name":"calculate",...}       │
     │   ]}}                              │
     │                                    │
     │  ──── Call Tool ─────────────────► │
     │  {"method":"tools/call",           │
     │   "params":{"name":"calculate",    │
     │             "arguments":{          │
     │               "expression":"2+2"   │
     │             }}}                    │
     │                                    │
     │  ◄──── Result ──────────────────── │
     │  {"result":{"content":[            │
     │     {"type":"text",                │
     │      "text":"Result: 4"}           │
     │   ]}}                              │
     │                                    │
```

---

## 6. Installing MCP for Python

```bash
# Install the MCP Python SDK
pip install mcp

# For server development
pip install "mcp[server]"

# For client development (connecting to servers)
pip install "mcp[client]"
```

---

## 7. MCP Server with Resources

Resources let you expose data that the AI can read:

```python
# notes_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent
import asyncio

server = Server("notes")

# In-memory notes storage
NOTES = {
    "meeting": "Discuss Q1 goals with team",
    "todo": "1. Review PRs\n2. Update docs\n3. Deploy v2",
    "ideas": "- Add dark mode\n- Improve search\n- Mobile app",
}

# List available resources
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri=f"notes://{name}",
            name=f"Note: {name}",
            description=f"Contents of the {name} note",
            mimeType="text/plain"
        )
        for name in NOTES.keys()
    ]

# Read a specific resource
@server.read_resource()
async def read_resource(uri: str):
    # Parse the URI: "notes://meeting" -> "meeting"
    note_name = uri.replace("notes://", "")

    if note_name in NOTES:
        return TextContent(type="text", text=NOTES[note_name])

    raise ValueError(f"Note not found: {note_name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

Now Claude can say: "Let me check your meeting notes" and read `notes://meeting`.

---

## 8. MCP Server with Prompts

Prompts are reusable templates that users can invoke:

```python
# prompts_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Prompt, PromptMessage, TextContent, PromptArgument
import asyncio

server = Server("code-helper")

@server.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="explain_code",
            description="Explain what a piece of code does",
            arguments=[
                PromptArgument(
                    name="code",
                    description="The code to explain",
                    required=True
                ),
                PromptArgument(
                    name="language",
                    description="Programming language",
                    required=False
                )
            ]
        ),
        Prompt(
            name="review_code",
            description="Review code for bugs and improvements",
            arguments=[
                PromptArgument(
                    name="code",
                    description="The code to review",
                    required=True
                )
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict):
    if name == "explain_code":
        code = arguments.get("code", "")
        language = arguments.get("language", "unknown")
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Please explain this {language} code step by step:

```{language}
{code}
```

Break down:
1. What each part does
2. The overall purpose
3. Any important patterns used"""
                )
            )
        ]

    elif name == "review_code":
        code = arguments.get("code", "")
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Review this code for:
- Bugs or errors
- Security issues
- Performance improvements
- Code style

```
{code}
```"""
                )
            )
        ]

    raise ValueError(f"Unknown prompt: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. Configuring MCP Servers in Claude Desktop

To use your MCP server with Claude Desktop, add it to the config:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["C:/path/to/calculator_server.py"]
    },
    "notes": {
      "command": "python",
      "args": ["C:/path/to/notes_server.py"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token-here"
      }
    }
  }
}
```

After saving, restart Claude Desktop. Your servers will be available!

---

## 10. Popular Pre-Built MCP Servers

You don't have to build everything yourself. These are ready to use:

| Server | What it does | Install |
|--------|--------------|---------|
| **filesystem** | Read/write local files | `@modelcontextprotocol/server-filesystem` |
| **github** | Manage repos, issues, PRs | `@modelcontextprotocol/server-github` |
| **postgres** | Query PostgreSQL databases | `@modelcontextprotocol/server-postgres` |
| **sqlite** | Query SQLite databases | `@modelcontextprotocol/server-sqlite` |
| **brave-search** | Web search | `@modelcontextprotocol/server-brave-search` |
| **puppeteer** | Browser automation | `@modelcontextprotocol/server-puppeteer` |
| **slack** | Read/send Slack messages | `@modelcontextprotocol/server-slack` |

**Using a pre-built server:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/Documents"
      ]
    }
  }
}
```

---

## 11. Building an MCP Client

If you're building your own AI app, you need an MCP client to connect to servers:

```python
# mcp_client_example.py
from mcp.client import Client
from mcp.client.stdio import stdio_client
import asyncio

async def main():
    # Connect to an MCP server
    async with stdio_client("python", ["calculator_server.py"]) as (read, write):
        client = Client()
        await client.connect(read, write)

        # Initialize
        await client.initialize()

        # List available tools
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # Call a tool
        result = await client.call_tool("calculate", {"expression": "42 * 2"})
        print(f"\nResult: {result}")

        # List resources (if server has any)
        resources = await client.list_resources()
        for resource in resources:
            print(f"  - {resource.uri}: {resource.name}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 12. Real-World Example: File Search Server

A practical server that lets Claude search your codebase:

```python
# file_search_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio
import os
import fnmatch

server = Server("file-search")

# Configuration
SEARCH_ROOT = os.path.expanduser("~/projects")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="find_files",
            description="Find files matching a pattern",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern like '*.py' or '**/*.ts'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum files to return (default 20)",
                        "default": 20
                    }
                },
                "required": ["pattern"]
            }
        ),
        Tool(
            name="search_content",
            description="Search for text within files",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Only search in files matching this pattern",
                        "default": "*"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="read_file",
            description="Read the contents of a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to search root)"
                    }
                },
                "required": ["path"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "find_files":
        pattern = arguments["pattern"]
        max_results = arguments.get("max_results", 20)

        matches = []
        for root, dirs, files in os.walk(SEARCH_ROOT):
            # Skip hidden and common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.')
                       and d not in ['node_modules', '__pycache__', 'venv']]

            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    rel_path = os.path.relpath(os.path.join(root, file), SEARCH_ROOT)
                    matches.append(rel_path)
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

        if matches:
            return [TextContent(type="text", text="\n".join(matches))]
        return [TextContent(type="text", text="No files found")]

    elif name == "search_content":
        query = arguments["query"]
        file_pattern = arguments.get("file_pattern", "*")

        results = []
        for root, dirs, files in os.walk(SEARCH_ROOT):
            dirs[:] = [d for d in dirs if not d.startswith('.')
                       and d not in ['node_modules', '__pycache__', 'venv']]

            for file in files:
                if not fnmatch.fnmatch(file, file_pattern):
                    continue

                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if query.lower() in line.lower():
                                rel_path = os.path.relpath(filepath, SEARCH_ROOT)
                                results.append(f"{rel_path}:{i}: {line.strip()[:100]}")
                                if len(results) >= 20:
                                    break
                except:
                    pass

                if len(results) >= 20:
                    break
            if len(results) >= 20:
                break

        if results:
            return [TextContent(type="text", text="\n".join(results))]
        return [TextContent(type="text", text="No matches found")]

    elif name == "read_file":
        path = arguments["path"]
        full_path = os.path.join(SEARCH_ROOT, path)

        # Security: ensure path is within search root
        if not os.path.abspath(full_path).startswith(os.path.abspath(SEARCH_ROOT)):
            return [TextContent(type="text", text="Error: Path outside allowed directory")]

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [TextContent(type="text", text=content)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading file: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 13. Transport Types

MCP supports multiple ways to communicate:

### stdio (Standard Input/Output)
```
Host ←──stdio──► Server (runs as subprocess)

Best for: Local servers, simple setup
```

### HTTP/SSE (Server-Sent Events)
```
Host ←──HTTP──► Server (runs anywhere)

Best for: Remote servers, web deployment
```

**HTTP Server Example:**
```python
from mcp.server import Server
from mcp.server.sse import sse_server
from starlette.applications import Starlette
from starlette.routing import Route

server = Server("my-http-server")

# ... define tools, resources, prompts ...

app = Starlette(
    routes=[
        Route("/mcp", endpoint=sse_server(server), methods=["GET", "POST"]),
    ]
)

# Run with: uvicorn server:app --port 8000
```

---

## 14. Error Handling

Good MCP servers handle errors gracefully:

```python
from mcp.types import TextContent, ErrorContent
from mcp.server.errors import McpError

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "risky_operation":
            # Do something that might fail
            result = perform_operation(arguments)
            return [TextContent(type="text", text=result)]

    except ValueError as e:
        # Return user-friendly error
        return [TextContent(type="text", text=f"Invalid input: {e}")]

    except PermissionError:
        # Return permission error
        return [TextContent(type="text", text="Permission denied")]

    except Exception as e:
        # Log and return generic error
        print(f"Error in {name}: {e}")
        return [TextContent(type="text", text="An unexpected error occurred")]
```

---

## 15. Best Practices for MCP Servers

### Security
```python
# 1. Validate all inputs
def validate_path(path: str) -> bool:
    # Prevent path traversal attacks
    abs_path = os.path.abspath(path)
    return abs_path.startswith(ALLOWED_ROOT)

# 2. Limit scope
ALLOWED_OPERATIONS = ["read", "list"]  # No write/delete

# 3. Use environment variables for secrets
API_KEY = os.environ.get("API_KEY")
```

### Performance
```python
# 1. Use async for I/O operations
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# 2. Limit result sizes
MAX_RESULTS = 100
results = results[:MAX_RESULTS]

# 3. Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_computation(input_data):
    # ...
```

### User Experience
```python
# 1. Clear tool descriptions
Tool(
    name="search",
    description="Search for documents by keyword. Returns up to 10 results with title and snippet."
)

# 2. Helpful error messages
return [TextContent(
    type="text",
    text="File not found. Available files: config.json, data.csv, readme.md"
)]

# 3. Progress for long operations
return [TextContent(
    type="text",
    text="Processing... found 150 files, analyzed 50 so far..."
)]
```

---

## 16. MCP vs Other Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **MCP** | Universal standard, reusable servers, clean separation | Newer, still evolving |
| **Function Calling** | Simple, built into APIs | One-off, not reusable |
| **LangChain Tools** | Rich ecosystem | Tied to LangChain |
| **Custom APIs** | Full control | Lots of boilerplate |

**When to use MCP:**
- Building tools that multiple AI apps should use
- Want clean separation between AI and integrations
- Need to share servers across team/organization
- Building a tool ecosystem

---

## 17. Debugging MCP Servers

### Check if your server runs
```bash
# Run directly and check for errors
python my_server.py

# Should see no output (waiting for JSON-RPC)
# Press Ctrl+C to exit
```

### Test with MCP Inspector
```bash
# Install the inspector
npx @modelcontextprotocol/inspector python my_server.py

# Opens a web UI to test your server
```

### Add logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    logger.debug(f"Tool called: {name} with {arguments}")
    # ...
```

---

## 18. Quick Reference: MCP Server Template

Copy this template to start a new server:

```python
"""
My MCP Server
Provides: [describe what it does]
"""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource, Prompt, TextContent
import asyncio

# Create server with a name
server = Server("my-server-name")

# ============== TOOLS ==============
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="my_tool",
            description="What this tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                },
                "required": ["param1"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "my_tool":
        param1 = arguments["param1"]
        result = f"You said: {param1}"
        return [TextContent(type="text", text=result)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]

# ============== RESOURCES (optional) ==============
@server.list_resources()
async def list_resources():
    return []  # Add resources here

@server.read_resource()
async def read_resource(uri: str):
    raise ValueError(f"Unknown resource: {uri}")

# ============== PROMPTS (optional) ==============
@server.list_prompts()
async def list_prompts():
    return []  # Add prompts here

@server.get_prompt()
async def get_prompt(name: str, arguments: dict):
    raise ValueError(f"Unknown prompt: {name}")

# ============== MAIN ==============
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 19. MCP in the Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Application Stack                         │
├─────────────────────────────────────────────────────────────────┤
│  Your App / Claude Desktop / VS Code / Other IDE                │
├─────────────────────────────────────────────────────────────────┤
│  LangChain / LlamaIndex / Direct API calls                      │
├─────────────────────────────────────────────────────────────────┤
│  MCP Client (connects to servers)                                │
├─────────────────────────────────────────────────────────────────┤
│                        MCP Protocol                              │
├─────────────────────────────────────────────────────────────────┤
│  MCP Servers  │  MCP Servers  │  MCP Servers  │  Your Servers   │
│  (Official)   │  (Community)  │  (Internal)   │  (Custom)       │
├─────────────────────────────────────────────────────────────────┤
│  External Services: GitHub, Slack, DBs, APIs, File Systems      │
└─────────────────────────────────────────────────────────────────┘
```

---

---

# Part 2: MCP Server Architecture & Scalability (Production)

This section covers how to architect MCP servers for production — the interview question about whether you deploy one big Docker container or split into multiple services.

---

## 20. The Core Architecture Question: Monolith vs Microservices

When you build MCP servers, the first decision is **how to package them**. There are two main patterns:

### Pattern 1: Monolith (Single Server, All Tools)

```
┌──────────────────────────────────────────────┐
│           Single MCP Server Container         │
│                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ GitHub   │ │ Database │ │ File System  │ │
│  │ Tools    │ │ Tools    │ │ Tools        │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Slack    │ │ Search   │ │ Analytics    │ │
│  │ Tools    │ │ Tools    │ │ Tools        │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
│                                               │
│  One process, one Docker image, one port      │
└──────────────────────────────────────────────┘
```

**How it works:** You bundle ALL your tools, resources, and prompts into a single MCP server process. The AI client connects to one endpoint and gets access to everything.

**Advantages:**
- **Simple to deploy** — one Docker image, one container, one URL to configure
- **Simple to develop** — all code in one place, easy to test locally
- **Low latency** — no network hops between tools, shared memory
- **Easy local development** — just run `python server.py` and everything works

**Disadvantages:**
- **Scaling is all-or-nothing** — if your database tool gets heavy traffic, you have to scale the entire server (including the barely-used Slack tools)
- **Single point of failure** — if one tool crashes the process, ALL tools go down
- **Deployment coupling** — updating your GitHub tool means redeploying the database tool too
- **Resource contention** — a memory-hungry search tool starves the other tools

### Pattern 2: Microservices (Separate Servers per Domain)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  GitHub MCP  │  │ Database MCP │  │ Search MCP   │
│   Server     │  │   Server     │  │   Server     │
│              │  │              │  │              │
│  - repos     │  │  - query     │  │  - semantic  │
│  - issues    │  │  - schema    │  │  - keyword   │
│  - PRs       │  │  - migrate   │  │  - index     │
│              │  │              │  │              │
│  Container 1 │  │  Container 2 │  │  Container 3 │
└──────────────┘  └──────────────┘  └──────────────┘
        ▲                ▲                ▲
        │                │                │
        └────────────────┼────────────────┘
                         │
              ┌──────────────────┐
              │   MCP Client     │
              │ (connects to all │
              │  servers)        │
              └──────────────────┘
```

**How it works:** Each domain (GitHub, database, search, etc.) gets its own MCP server running in its own Docker container. The AI client connects to multiple servers simultaneously.

**Advantages:**
- **Independent scaling** — scale your search server to 10 replicas while keeping one GitHub server
- **Fault isolation** — if the database server crashes, GitHub and search tools still work
- **Independent deployment** — update the GitHub server without touching anything else
- **Team ownership** — different teams can own different servers
- **Technology freedom** — your search server can use Rust for performance, others use Python

**Disadvantages:**
- **More complex infrastructure** — service discovery, networking, multiple Docker images
- **Higher operational overhead** — more containers to monitor, log, and maintain
- **Network latency** — cross-service calls add milliseconds
- **Configuration complexity** — the AI client needs to know about all server endpoints

---

## 21. When to Use Which Pattern

This is the key interview question — there's no universally correct answer. It depends on your context:

### Start with Monolith When:
- **Early stage** — you're prototyping or have < 5 tools
- **Small team** — 1-3 developers working on everything
- **Low traffic** — internal tool or small user base
- **Fast iteration** — you need to ship and change things quickly
- **Local/desktop use** — running alongside Claude Desktop or VS Code

### Move to Microservices When:
- **Scale pressure** — one tool gets 100x more traffic than others
- **Team growth** — multiple teams need to own and deploy independently
- **Reliability requirements** — you can't afford one buggy tool taking down everything
- **Different resource profiles** — your RAG search needs GPUs while your CRUD tools need minimal compute
- **Compliance boundaries** — some tools touch sensitive data and need isolated environments

### The Hybrid Pattern (Most Common in Production)

In practice, most teams use a hybrid approach — grouping related tools into a few servers rather than going fully monolith or fully micro:

```
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│  Data MCP Server   │  │  DevOps MCP Server │  │  AI/ML MCP Server  │
│                    │  │                    │  │                    │
│  - database query  │  │  - github repos    │  │  - RAG search      │
│  - database write  │  │  - github issues   │  │  - embeddings      │
│  - cache lookup    │  │  - CI/CD triggers  │  │  - model inference  │
│  - data export     │  │  - deployment      │  │  - vector store    │
│                    │  │  - monitoring      │  │                    │
│  Container 1       │  │  Container 2       │  │  Container 3 (GPU) │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

**Why hybrid works best:** You group by domain and scaling characteristics. Data tools scale together, DevOps tools scale together, and compute-heavy AI tools get their own GPU-enabled containers. You get most of the microservices benefits without the overhead of managing 20 individual containers.

---

## 22. Docker Deployment Strategies for MCP Servers

### Single Container (Monolith)

You package everything into one Docker image. The MCP server listens on a single port (typically using HTTP/SSE transport for production rather than stdio).

**How it works in production:**
1. Your Dockerfile installs all dependencies for all tools
2. The MCP server starts and registers all tools/resources/prompts
3. You expose one port (e.g., 8080) for the MCP protocol
4. The AI client connects to `http://mcp-server:8080`
5. A load balancer can put multiple replicas behind one URL

**When traffic grows:** You scale by running multiple identical copies of the same container behind a load balancer. Every replica has all tools.

### Multiple Containers (Microservices)

Each MCP server gets its own Docker image, optimized for its needs.

**How it works in production:**
1. Each server has its own Dockerfile with only the dependencies it needs
2. Each server runs in its own container with its own port
3. A service mesh or API gateway routes requests to the right server
4. The AI client is configured with multiple server endpoints
5. Each server scales independently based on demand

**Container orchestration** (Kubernetes, ECS, Cloud Run) handles:
- **Service discovery** — servers find each other by name, not IP address
- **Health checks** — automatically restart failed containers
- **Auto-scaling** — spin up more replicas when CPU/memory thresholds are hit
- **Rolling deployments** — update one server without downtime

### The MCP Gateway Pattern

For microservices deployments, a common production pattern is an **MCP Gateway** — a thin proxy that sits between the AI client and your MCP servers:

```
AI Client
    │
    ▼
┌────────────────────────────────┐
│        MCP Gateway             │
│  (routes tool calls to the     │
│   correct backend server)      │
│                                │
│  - Authentication              │
│  - Rate limiting               │
│  - Request routing             │
│  - Tool registry               │
│  - Logging & monitoring        │
└────────────────────────────────┘
    │           │           │
    ▼           ▼           ▼
 Server A   Server B   Server C
```

**Why use a gateway:** The AI client only needs one connection URL. The gateway handles routing each tool call to the correct backend server. It also centralizes auth, rate limiting, and logging — things you'd otherwise duplicate in every server.

---

## 23. Scaling Strategies Explained

### Horizontal Scaling (Adding More Replicas)

The most straightforward scaling approach. You run multiple copies of the same MCP server behind a load balancer.

**How it works:** When traffic increases, the orchestrator spins up additional identical containers. Each request goes to whichever replica is least busy. This works well for **stateless** MCP servers — servers that don't store data between requests (they call external databases/APIs instead).

**Challenge with stateful servers:** If your MCP server maintains in-memory state (like a cache or session data), you need **sticky sessions** (routing the same client to the same replica) or **shared state** (using Redis/database so all replicas see the same data).

### Vertical Scaling (Bigger Machines)

Sometimes the right answer is just giving your server more CPU/RAM/GPU rather than adding replicas.

**When vertical makes sense:**
- GPU-bound work (embeddings, inference) — you need a bigger GPU, not more small ones
- Memory-bound work (large vector stores in memory) — need more RAM
- Single-threaded bottlenecks — more CPUs won't help if the code isn't concurrent

### Auto-Scaling Triggers

In production, you set rules for when to scale. Common triggers:

| Trigger | What It Means | Example |
|---------|---------------|---------|
| **CPU utilization** | Server is compute-bound | Scale up when CPU > 70% for 5 minutes |
| **Memory utilization** | Server needs more RAM | Scale up when memory > 80% |
| **Request queue depth** | Requests are waiting | Scale up when queue > 50 pending |
| **Response latency** | Users are waiting too long | Scale up when p95 latency > 2 seconds |
| **Custom metrics** | Domain-specific signals | Scale up when embedding queue > 100 |

### Scaling to Zero

For MCP servers with sporadic traffic, **scale-to-zero** is powerful. Services like Cloud Run, AWS Lambda, or Azure Container Apps can shut down your server when there's no traffic and cold-start it when a request arrives.

**Trade-off:** Cold start latency (1-5 seconds for containers, 100ms-1s for serverless) vs cost savings from not running idle containers.

---

## 24. Production Architecture Patterns

### Pattern 1: Simple Production (Small Team)

```
┌──────────────────────────────────────────────────┐
│                Cloud Run / ECS / ACA              │
│                                                   │
│   ┌────────────────────────────────────────┐     │
│   │     MCP Server (Monolith)              │     │
│   │     - All tools in one process         │     │
│   │     - HTTP/SSE transport               │     │
│   │     - 2-4 replicas behind LB           │     │
│   └────────────────────────────────────────┘     │
│                                                   │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│   │  Redis    │  │ Postgres  │  │ Secrets   │   │
│   │  (cache)  │  │  (data)   │  │ Manager   │   │
│   └───────────┘  └───────────┘  └───────────┘   │
└──────────────────────────────────────────────────┘
```

**Works for:** Most startups, internal tools, small-medium traffic. You get 80% of the production benefits with 20% of the complexity.

### Pattern 2: Domain-Split Production (Growing Team)

```
┌─────────────────────────────────────────────────────┐
│                  Kubernetes / ECS                     │
│                                                      │
│  ┌──────────┐                                       │
│  │  MCP     │  ← AI clients connect here            │
│  │  Gateway │                                       │
│  └────┬─────┘                                       │
│       │                                              │
│  ┌────┴──────────┬───────────────┬──────────────┐   │
│  ▼               ▼               ▼              │   │
│ ┌──────────┐ ┌──────────┐ ┌──────────────┐     │   │
│ │ Data     │ │ DevOps   │ │ AI/ML        │     │   │
│ │ Server   │ │ Server   │ │ Server (GPU) │     │   │
│ │ 3 reps   │ │ 2 reps   │ │ 5 reps       │     │   │
│ └──────────┘ └──────────┘ └──────────────┘     │   │
│                                                      │
│  Shared: Redis, Postgres, Vector DB, Secrets Manager │
└─────────────────────────────────────────────────────┘
```

**Works for:** Companies with multiple teams, varying traffic patterns, and different compute needs per domain.

### Pattern 3: Serverless / Event-Driven

Instead of always-running containers, each MCP tool invocation triggers a serverless function:

**How it works:**
1. MCP Gateway receives a tool call
2. Gateway routes to the appropriate Lambda/Cloud Function
3. Function executes, returns result
4. Function shuts down (or stays warm for a few minutes)

**Best for:** Sporadic traffic, cost optimization, tools that are called infrequently but need to be available 24/7.

**Not great for:** Low-latency requirements, tools that need persistent connections (WebSockets, database connection pools), or GPU workloads.

---

## 25. Security Considerations for Production MCP

When running MCP servers in production, security becomes critical:

### Authentication & Authorization

- **Server-to-server auth**: Use mutual TLS (mTLS) or API keys between the gateway and backend MCP servers
- **Client auth**: The MCP client should authenticate with the gateway using OAuth tokens or API keys
- **Tool-level permissions**: Not every user should access every tool. The gateway can enforce role-based access — "analyst" role gets read-only database tools, "admin" role gets write tools too
- **Scope limiting**: Each MCP server should only have credentials for the services it needs. The GitHub server shouldn't have database credentials.

### Network Security

- **Internal networking**: MCP servers should NOT be publicly accessible. Only the gateway is exposed.
- **Encryption**: All communication should use TLS, even internal traffic in production
- **Network policies**: In Kubernetes, use network policies to restrict which servers can talk to which services

### Audit & Compliance

- **Log every tool call**: Who called what tool, with what arguments, and what was returned
- **Sensitive data masking**: Don't log database query results or file contents that might contain secrets
- **Rate limiting**: Prevent runaway AI agents from making thousands of tool calls per minute

---

## 26. Monitoring & Observability

In production, you need to know what your MCP servers are doing:

### Key Metrics to Track

| Metric | Why It Matters |
|--------|---------------|
| **Tool call latency** (p50, p95, p99) | Are users waiting too long? |
| **Tool call success rate** | Are tools failing? |
| **Active connections** | How many AI clients are connected? |
| **Tool call volume** (per tool) | Which tools are hot? Which are unused? |
| **Error rate by tool** | Is one specific tool causing problems? |
| **Container resource usage** | CPU, memory, network — for scaling decisions |

### Observability Stack

A typical production setup uses:
- **Structured logging** — JSON logs with tool name, duration, status, and correlation IDs
- **Distributed tracing** — trace a request from AI client → gateway → MCP server → external API and back
- **Dashboards** — Grafana/Datadog showing real-time tool call patterns
- **Alerting** — PagerDuty/Slack alerts when error rate spikes or latency degrades

### Health Checks

Every MCP server should expose health check endpoints:
- **Liveness**: "Is the process alive?" — restart if not
- **Readiness**: "Can it handle requests?" — stop routing traffic if not (e.g., database connection lost)
- **Startup**: "Has it finished initializing?" — don't send traffic until ready

---

## 27. MCP Architecture Decision Guide

Use this flowchart when deciding how to architect your MCP servers:

```
Start: How many tools/domains do you have?
│
├─ < 5 tools, single domain
│  → Monolith (single container)
│  → Scale: horizontal replicas behind load balancer
│
├─ 5-15 tools, 2-3 domains
│  → Hybrid (2-3 servers grouped by domain)
│  → Scale: independently per domain server
│
├─ 15+ tools, many domains
│  → Microservices with MCP Gateway
│  → Scale: per-server with auto-scaling policies
│
└─ Sporadic/unpredictable traffic
   → Serverless with gateway routing
   → Scale: automatic, pay-per-invocation
```

### Quick Comparison Table

| Factor | Monolith | Hybrid (2-4 servers) | Full Microservices |
|--------|----------|---------------------|-------------------|
| **Complexity** | Low | Medium | High |
| **Deployment speed** | Fast | Medium | Slower (per service) |
| **Scaling granularity** | All-or-nothing | Per domain group | Per individual server |
| **Fault isolation** | None | Partial | Full |
| **Team independence** | Low | Medium | High |
| **Operational overhead** | Low | Medium | High |
| **Best for** | Prototypes, small teams | Most production use cases | Large orgs, high scale |

---

## 28. Interview Quick Reference: MCP Architecture

> **"How do you architect your MCP servers — one Docker container or split?"**

**Sample answer:** "It depends on the scale and team structure. I typically start with a monolith — one Docker container with all tools — because it's simple to develop, test, and deploy. As the system grows, I move to a hybrid approach where I group related tools by domain into 2-4 separate containers. For example, data tools in one container, DevOps tools in another, and compute-heavy AI tools in their own GPU-enabled container. This gives me independent scaling and fault isolation where it matters most, without the operational overhead of managing 20 separate microservices.

For production, I use an MCP Gateway pattern — a thin proxy that the AI client connects to. The gateway handles auth, rate limiting, and routes tool calls to the correct backend server. This way, the AI client only needs one connection URL regardless of how many backend servers exist.

The key scaling decision is: are your tools stateless or stateful? Stateless tools are easy — just add more replicas behind a load balancer. Stateful tools need shared state (Redis for sessions, database for persistence) or sticky sessions. I also configure auto-scaling based on CPU utilization and request queue depth, with scale-to-zero for rarely-used tools to save costs."

> **"How do you handle scalability for MCP servers?"**

**Sample answer:** "There are three dimensions. First, horizontal scaling — running multiple replicas of the same server behind a load balancer. This handles increased traffic. Second, domain splitting — separating servers by domain so each can scale independently. If my RAG search server is getting 10x the traffic of my GitHub server, I scale them separately. Third, auto-scaling rules — I set thresholds on CPU, memory, and queue depth so the orchestrator (Kubernetes, ECS, Cloud Run) automatically adds or removes replicas. For sporadic workloads, I use scale-to-zero services like Cloud Run so I'm not paying for idle containers."

---

## 29. What's Next?

Now try the practice problems:

1. `p1_mcp_basics.py` - Build simple tools and resources
2. `p2_mcp_server.py` - Build a complete MCP server

```bash
pytest 09_mcp/ -v
```

**Resources:**
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Pre-built Servers](https://github.com/modelcontextprotocol/servers)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
