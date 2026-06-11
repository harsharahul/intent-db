"""Command-line interface for IntentDB.

Examples::

    intentdb init mydb.intentdb --embedder "hashing:dim=512"
    intentdb add mydb.intentdb "Postgres uses MVCC for concurrency" --key pg-mvcc
    intentdb add mydb.intentdb --file notes.txt
    intentdb intent add mydb.intentdb debugging \
        --description "Diagnosing errors and failures in software" \
        --exemplar "why is my service crashing" --exemplar "stack trace meaning"
    intentdb query mydb.intentdb "postgres locks" --intent debugging -k 3
    intentdb query mydb.intentdb "postgres locks" --json
    intentdb explain mydb.intentdb "why does my app crash"
    intentdb stats mydb.intentdb
    intentdb serve-mcp mydb.intentdb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import IntentDB


def _open(args: argparse.Namespace) -> IntentDB:
    embedder = getattr(args, "embedder", None)
    return IntentDB(args.db, embedder=embedder)


def cmd_init(args: argparse.Namespace) -> int:
    db = IntentDB(args.db, embedder=args.embedder)
    print(json.dumps(db.stats(), indent=2))
    db.close()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    texts: list[str] = []
    if args.text:
        texts.append(args.text)
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
        if args.split_paragraphs:
            texts.extend(p.strip() for p in content.split("\n\n") if p.strip())
        else:
            texts.append(content)
    if args.stdin:
        texts.append(sys.stdin.read())
    if not texts:
        print("nothing to add: pass TEXT, --file, or --stdin", file=sys.stderr)
        return 2

    metadata = json.loads(args.metadata) if args.metadata else {}
    with _open(args) as db:
        if len(texts) == 1:
            keys = [db.add(texts[0], doc_key=args.key, metadata=metadata)]
        else:
            keys = db.add_many([(t, None, metadata) for t in texts])
    for k in keys:
        print(k)
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    with _open(args) as db:
        ok = db.delete(args.key)
    print("deleted" if ok else "not found")
    return 0 if ok else 1


def cmd_intent_add(args: argparse.Namespace) -> int:
    with _open(args) as db:
        db.register_intent(
            args.name,
            description=args.description,
            exemplars=args.exemplar or [],
            instruction=args.instruction,
            lens_strength=args.lens_strength,
        )
        print(f"intent {args.name!r} registered over {db.stats()['documents']} documents")
    return 0


def cmd_intent_list(args: argparse.Namespace) -> int:
    with _open(args) as db:
        print(json.dumps(db.list_intents(), indent=2))
    return 0


def cmd_intent_rm(args: argparse.Namespace) -> int:
    with _open(args) as db:
        ok = db.remove_intent(args.name)
    print("removed" if ok else "not found")
    return 0 if ok else 1


def cmd_query(args: argparse.Namespace) -> int:
    with _open(args) as db:
        results = db.query(
            args.text,
            intent=args.intent,
            k=args.k,
            auto_intent=not args.no_auto_intent,
        )
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0
    if not results:
        print("(no results)")
        return 0
    used = results[0].intent
    if used:
        how = "inferred" if results[0].intent_inferred else "requested"
        print(f"intent: {used} ({how})\n")
    for i, r in enumerate(results, 1):
        snippet = r.text.replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."
        line = f"{i}. [{r.score:+.4f}] {r.doc_key}: {snippet}"
        print(line)
        if used:
            print(
                f"     base={r.base_score:+.4f}  lensed={r.lensed_score:+.4f}"
                f"  affinity={r.intent_affinity:+.4f}"
            )
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    with _open(args) as db:
        print(json.dumps(db.explain(args.text), indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with _open(args) as db:
        print(json.dumps(db.stats(), indent=2))
    return 0


def cmd_serve_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import serve

    serve(args.db)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="intentdb",
        description="IntentDB: a local-first, intent-aware vector database.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_db_arg(sp):
        sp.add_argument("db", help="path to the .intentdb file")

    sp = sub.add_parser("init", help="create a database")
    add_db_arg(sp)
    sp.add_argument(
        "--embedder",
        default="hashing:dim=512",
        help='embedder spec, e.g. "hashing:dim=512" or "ollama:model=nomic-embed-text"',
    )
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("add", help="add a document")
    add_db_arg(sp)
    sp.add_argument("text", nargs="?", help="document text")
    sp.add_argument("--key", help="stable document key (re-adding replaces)")
    sp.add_argument("--file", help="read text from a file")
    sp.add_argument(
        "--split-paragraphs",
        action="store_true",
        help="with --file, store each blank-line-separated paragraph as its own document",
    )
    sp.add_argument("--stdin", action="store_true", help="read text from stdin")
    sp.add_argument("--metadata", help="JSON object of metadata")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("delete", help="delete a document by key")
    add_db_arg(sp)
    sp.add_argument("key")
    sp.set_defaults(func=cmd_delete)

    sp = sub.add_parser("intent", help="manage intents")
    isub = sp.add_subparsers(dest="intent_command", required=True)

    sp2 = isub.add_parser("add", help="register or redefine an intent")
    add_db_arg(sp2)
    sp2.add_argument("name")
    sp2.add_argument("--description", required=True)
    sp2.add_argument(
        "--exemplar",
        action="append",
        help="example query for this intent (repeatable)",
    )
    sp2.add_argument(
        "--instruction",
        help="task instruction for instruction-aware embedders "
        "(defaults to the description)",
    )
    sp2.add_argument("--lens-strength", type=float, default=4.0)
    sp2.set_defaults(func=cmd_intent_add)

    sp2 = isub.add_parser("list", help="list intents")
    add_db_arg(sp2)
    sp2.set_defaults(func=cmd_intent_list)

    sp2 = isub.add_parser("rm", help="remove an intent")
    add_db_arg(sp2)
    sp2.add_argument("name")
    sp2.set_defaults(func=cmd_intent_rm)

    sp = sub.add_parser("query", help="search the database")
    add_db_arg(sp)
    sp.add_argument("text")
    sp.add_argument("--intent", help="retrieve under this registered intent")
    sp.add_argument(
        "--no-auto-intent",
        action="store_true",
        help="disable intent inference when --intent is not given",
    )
    sp.add_argument("-k", type=int, default=5, help="number of results")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("explain", help="show how the intent classifier sees a query")
    add_db_arg(sp)
    sp.add_argument("text")
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("stats", help="database statistics")
    add_db_arg(sp)
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser(
        "serve-mcp",
        help="serve the database as an MCP (Model Context Protocol) stdio server",
    )
    add_db_arg(sp)
    sp.set_defaults(func=cmd_serve_mcp)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
