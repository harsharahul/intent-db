import json

import pytest

from intentdb import IntentDB
from intentdb.mcp_server import TOOLS, handle_message


@pytest.fixture()
def db(tmp_path):
    db = IntentDB(tmp_path / "mcp.intentdb")
    yield db
    db.close()


def rpc(db, method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return handle_message(db, msg)


def tool_call(db, name, arguments):
    resp = rpc(db, "tools/call", {"name": name, "arguments": arguments})
    assert resp["result"]["isError"] is False, resp
    return json.loads(resp["result"]["content"][0]["text"])


def test_initialize(db):
    resp = rpc(db, "initialize", {"protocolVersion": "2024-11-05"})
    assert resp["result"]["serverInfo"]["name"] == "intentdb"
    assert "tools" in resp["result"]["capabilities"]


def test_notification_returns_none(db):
    assert handle_message(db, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list(db):
    resp = rpc(db, "tools/list")
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {t["name"] for t in TOOLS}
    assert "intentdb_query" in names


def test_tool_workflow(db):
    out = tool_call(db, "intentdb_add", {"text": "Python code and programming.", "doc_key": "py"})
    assert out["doc_key"] == "py"
    tool_call(db, "intentdb_add", {"text": "Python snakes live in the jungle.", "doc_key": "snake"})
    tool_call(
        db,
        "intentdb_register_intent",
        {
            "name": "coding",
            "description": "software programming and source code",
            "exemplars": ["write code"],
        },
    )
    intents = tool_call(db, "intentdb_list_intents", {})
    assert [i["name"] for i in intents] == ["coding"]

    hits = tool_call(db, "intentdb_query", {"query": "python", "intent": "coding", "k": 1})
    assert hits[0]["doc_key"] == "py"

    stats = tool_call(db, "intentdb_stats", {})
    assert stats["documents"] == 2

    explain = tool_call(db, "intentdb_explain", {"query": "programming code"})
    assert explain["inferred_intent"] == "coding"


def test_tool_error_is_reported_not_raised(db):
    resp = rpc(db, "tools/call", {"name": "intentdb_query", "arguments": {"query": "x", "intent": "missing"}})
    assert resp["result"]["isError"] is True
    assert "missing" in resp["result"]["content"][0]["text"]


def test_unknown_method(db):
    resp = rpc(db, "bogus/method")
    assert resp["error"]["code"] == -32601
