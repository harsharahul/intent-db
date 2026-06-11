"""MCP (Model Context Protocol) stdio server for IntentDB.

This is the "retrieval is for an LLM" half of the design: any MCP-capable
client (Claude Code, Claude Desktop, local agent frameworks, etc.) can
mount an IntentDB file as a retrieval tool. Pure standard library —
JSON-RPC 2.0 over stdin/stdout, MCP protocol version 2024-11-05.

Claude Code example (.mcp.json)::

    {
      "mcpServers": {
        "intentdb": {
          "command": "intentdb",
          "args": ["serve-mcp", "/path/to/my.intentdb"]
        }
      }
    }
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .db import IntentDB

PROTOCOL_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "intentdb_query",
        "description": (
            "Search the intent-aware database. Optionally retrieve under a "
            "named intent — the same query returns different results for "
            "different intents. If no intent is given, the most plausible "
            "one is inferred from the query."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search text"},
                "intent": {
                    "type": "string",
                    "description": "registered intent to retrieve under (optional)",
                },
                "k": {"type": "integer", "description": "max results", "default": 5},
                "auto_intent": {
                    "type": "boolean",
                    "description": "infer intent when none is given",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "intentdb_add",
        "description": "Store a document (text plus optional metadata and stable key).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "doc_key": {"type": "string", "description": "stable id; re-adding replaces"},
                "metadata": {"type": "object"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "intentdb_register_intent",
        "description": (
            "Register or redefine a retrieval intent (name, description, "
            "exemplar queries). All stored documents are immediately indexed "
            "under it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "exemplars": {"type": "array", "items": {"type": "string"}},
                "instruction": {
                    "type": "string",
                    "description": (
                        "task instruction for instruction-aware embedders "
                        "(defaults to the description)"
                    ),
                },
            },
            "required": ["name", "description"],
        },
    },
    {
        "name": "intentdb_list_intents",
        "description": "List the registered intents.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "intentdb_explain",
        "description": "Show which intent the classifier would infer for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "intentdb_stats",
        "description": "Database statistics (document count, intents, embedder).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def call_tool(db: IntentDB, name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch one tool call; returns a JSON-serializable result."""
    if name == "intentdb_query":
        results = db.query(
            arguments["query"],
            intent=arguments.get("intent"),
            k=int(arguments.get("k", 5)),
            auto_intent=bool(arguments.get("auto_intent", True)),
        )
        return [r.to_dict() for r in results]
    if name == "intentdb_add":
        key = db.add(
            arguments["text"],
            doc_key=arguments.get("doc_key"),
            metadata=arguments.get("metadata"),
        )
        return {"doc_key": key}
    if name == "intentdb_register_intent":
        db.register_intent(
            arguments["name"],
            description=arguments["description"],
            exemplars=arguments.get("exemplars") or [],
            instruction=arguments.get("instruction"),
        )
        return {"registered": arguments["name"]}
    if name == "intentdb_list_intents":
        return db.list_intents()
    if name == "intentdb_explain":
        return db.explain(arguments["query"])
    if name == "intentdb_stats":
        return db.stats()
    raise ValueError(f"unknown tool {name!r}")


def handle_message(db: IntentDB, msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message; returns the response (None for notifications)."""
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "intentdb", "version": "0.1.0"},
            },
        }
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        try:
            result = call_tool(db, params.get("name", ""), params.get("arguments") or {})
            content = [{"type": "text", "text": json.dumps(result, indent=2)}]
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": content, "isError": False},
            }
        except Exception as e:  # surface tool failures as tool results, per MCP
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"error: {e}"}],
                    "isError": True,
                },
            }
    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    return None


def serve(db_path: str) -> None:
    """Run the stdio server until stdin closes."""
    db = IntentDB(db_path)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = handle_message(db, msg)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
    finally:
        db.close()
