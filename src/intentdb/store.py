"""SQLite-backed persistence for documents, vectors, intents, and the
precomputed per-(document, intent) statistics that make lensed retrieval a
single matrix-vector product at query time.

Schema
------
- ``meta``        — key/value store: schema version, embedder spec, dim.
- ``documents``   — text, JSON metadata, and the base float32 vector blob.
- ``intents``     — description, exemplars, intent vector, lens gate.
- ``doc_intent``  — per (doc, intent): affinity = cos(doc, intent),
                    precomputed at ingest so retrieval never re-embeds.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY,
    doc_key    TEXT UNIQUE NOT NULL,
    text       TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}',
    vector     BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS intents (
    name          TEXT PRIMARY KEY,
    description   TEXT NOT NULL,
    exemplars     TEXT NOT NULL DEFAULT '[]',
    instruction   TEXT,
    vector        BLOB NOT NULL,
    gate          BLOB NOT NULL,
    lens_strength REAL NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS doc_intent (
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    intent_name TEXT    NOT NULL REFERENCES intents(name) ON DELETE CASCADE,
    affinity    REAL    NOT NULL,
    PRIMARY KEY (doc_id, intent_name)
);
CREATE TABLE IF NOT EXISTS query_log (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    intent     TEXT,
    inferred   INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
"""

#: queries kept in the log; older entries are pruned
QUERY_LOG_CAP = 10_000


def _to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


class Store:
    """Thin, explicit SQLite wrapper. All vectors are float32 blobs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    # -- documents ----------------------------------------------------------

    def upsert_document(
        self, doc_key: str, text: str, metadata: dict, vector: np.ndarray
    ) -> int:
        """Insert or replace a document; returns its row id."""
        self.conn.execute(
            "INSERT INTO documents(doc_key, text, metadata, vector, created_at) "
            "VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(doc_key) DO UPDATE SET "
            "text=excluded.text, metadata=excluded.metadata, "
            "vector=excluded.vector, created_at=excluded.created_at",
            (doc_key, text, json.dumps(metadata), _to_blob(vector), time.time()),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM documents WHERE doc_key=?", (doc_key,)
        ).fetchone()
        return int(row[0])

    def delete_document(self, doc_key: str) -> bool:
        cur = self.conn.execute("DELETE FROM documents WHERE doc_key=?", (doc_key,))
        self.conn.commit()
        return cur.rowcount > 0

    def get_document(self, doc_key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, doc_key, text, metadata, vector, created_at "
            "FROM documents WHERE doc_key=?",
            (doc_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "doc_key": row[1],
            "text": row[2],
            "metadata": json.loads(row[3]),
            "vector": _from_blob(row[4]),
            "created_at": row[5],
        }

    def count_documents(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def load_all_documents(self) -> tuple[list[int], list[str], list[str], list[dict], np.ndarray]:
        """Load every document, ordered by id.

        Returns (ids, doc_keys, texts, metadatas, matrix) where matrix is a
        (n, dim) float32 array of base vectors aligned with the lists.
        """
        rows = self.conn.execute(
            "SELECT id, doc_key, text, metadata, vector FROM documents ORDER BY id"
        ).fetchall()
        ids = [r[0] for r in rows]
        keys = [r[1] for r in rows]
        texts = [r[2] for r in rows]
        metas = [json.loads(r[3]) for r in rows]
        if rows:
            matrix = np.stack([_from_blob(r[4]) for r in rows])
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)
        return ids, keys, texts, metas, matrix

    # -- intents ------------------------------------------------------------

    def upsert_intent(
        self,
        name: str,
        description: str,
        exemplars: list[str],
        instruction: str | None,
        vector: np.ndarray,
        gate: np.ndarray,
        lens_strength: float,
    ) -> None:
        self.conn.execute(
            "INSERT INTO intents(name, description, exemplars, instruction, "
            "vector, gate, lens_strength, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
            "exemplars=excluded.exemplars, instruction=excluded.instruction, "
            "vector=excluded.vector, gate=excluded.gate, "
            "lens_strength=excluded.lens_strength",
            (
                name,
                description,
                json.dumps(exemplars),
                instruction,
                _to_blob(vector),
                _to_blob(gate),
                lens_strength,
                time.time(),
            ),
        )
        self.conn.commit()

    def delete_intent(self, name: str) -> bool:
        cur = self.conn.execute("DELETE FROM intents WHERE name=?", (name,))
        self.conn.commit()
        return cur.rowcount > 0

    def load_all_intents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT name, description, exemplars, instruction, vector, gate, "
            "lens_strength FROM intents ORDER BY name"
        ).fetchall()
        return [
            {
                "name": r[0],
                "description": r[1],
                "exemplars": json.loads(r[2]),
                "instruction": r[3],
                "vector": _from_blob(r[4]),
                "gate": _from_blob(r[5]),
                "lens_strength": r[6],
            }
            for r in rows
        ]

    # -- doc/intent statistics ----------------------------------------------

    def upsert_doc_intent_stats(self, rows: list[tuple[int, str, float]]) -> None:
        """rows: (doc_id, intent_name, affinity)"""
        self.conn.executemany(
            "INSERT INTO doc_intent(doc_id, intent_name, affinity) "
            "VALUES(?, ?, ?) "
            "ON CONFLICT(doc_id, intent_name) DO UPDATE SET "
            "affinity=excluded.affinity",
            rows,
        )
        self.conn.commit()

    def load_intent_affinities(self, intent_name: str) -> dict[int, float]:
        """Return {doc_id: affinity} for one intent."""
        rows = self.conn.execute(
            "SELECT doc_id, affinity FROM doc_intent WHERE intent_name=?",
            (intent_name,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # -- query log ------------------------------------------------------------

    def log_query(self, text: str, intent: str | None, inferred: bool) -> None:
        self.conn.execute(
            "INSERT INTO query_log(text, intent, inferred, created_at) "
            "VALUES(?, ?, ?, ?)",
            (text, intent, int(inferred), time.time()),
        )
        self.conn.execute(
            "DELETE FROM query_log WHERE id <= ("
            "  SELECT MAX(id) - ? FROM query_log)",
            (QUERY_LOG_CAP,),
        )
        self.conn.commit()

    def load_query_log(
        self, undeclared_only: bool = False, limit: int = QUERY_LOG_CAP
    ) -> list[dict]:
        """Most recent first. ``undeclared_only`` keeps queries that ran
        without an explicitly requested intent (inferred or none)."""
        where = "WHERE intent IS NULL OR inferred = 1" if undeclared_only else ""
        rows = self.conn.execute(
            f"SELECT text, intent, inferred, created_at FROM query_log {where} "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "text": r[0],
                "intent": r[1],
                "inferred": bool(r[2]),
                "created_at": r[3],
            }
            for r in rows
        ]

    def count_query_log(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0])

    def commit(self) -> None:
        self.conn.commit()
