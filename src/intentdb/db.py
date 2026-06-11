"""IntentDB core engine.

Retrieval scoring
-----------------
For a query vector ``q`` (unit norm) and a document vector ``d`` (unit
norm), with an active intent ``I`` (vector ``t``, lens gate ``g``):

- ``base    = cos(q, d)``                     — ordinary vector search
- ``lensed  = <q*g, d*g> / |q*g|``            — query-document overlap with
  intent-characteristic dimensions amplified (cosine in the lensed space on
  the query side; the document keeps its base norm so it is not penalized
  for carrying intent-relevant content). Computed as ``<q*g^2, d> / |q*g|``
  so the whole collection costs one matrix-vector product.
- ``affinity = cos(d, t)``                    — how much the document
  belongs to the intent at all, regardless of the query

``score = w_lensed * lensed + w_affinity * affinity + w_base * base``

Without an intent (and with auto-inference off or inconclusive) the score
is plain cosine similarity, i.e. IntentDB degrades gracefully to a normal
vector database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

import numpy as np

from .embedders import Embedder, get_embedder
from .intent import DEFAULT_LENS_STRENGTH, Intent, IntentLens, infer_intent
from .store import Store

#: Default blend of the three scoring signals.
DEFAULT_WEIGHTS = {"lensed": 0.6, "affinity": 0.25, "base": 0.15}


@dataclass
class QueryResult:
    """One retrieval hit."""

    doc_key: str
    text: str
    metadata: dict
    score: float
    base_score: float
    lensed_score: float | None = None
    intent_affinity: float | None = None
    intent: str | None = None
    intent_inferred: bool = False
    intent_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "doc_key": self.doc_key,
            "text": self.text,
            "metadata": self.metadata,
            "score": round(self.score, 6),
            "base_score": round(self.base_score, 6),
        }
        if self.intent is not None:
            out["intent"] = self.intent
            out["intent_inferred"] = self.intent_inferred
            out["lensed_score"] = round(self.lensed_score, 6)
            out["intent_affinity"] = round(self.intent_affinity, 6)
        return out


class IntentDB:
    """A local, intent-aware vector database.

    Parameters
    ----------
    path:
        File path of the database (a single SQLite file). Created if
        missing.
    embedder:
        Either an :class:`~intentdb.embedders.Embedder` instance or a spec
        string such as ``"hashing:dim=512"`` or
        ``"ollama:model=nomic-embed-text"``. On an existing database this
        must match the embedder the store was created with (pass nothing
        to reuse it automatically).
    """

    def __init__(self, path: str | Path, embedder: Embedder | str | None = None):
        self.store = Store(path)
        stored_spec = self.store.get_meta("embedder_spec")

        if embedder is None:
            spec = stored_spec or "hashing:dim=512"
            self.embedder = get_embedder(spec)
        elif isinstance(embedder, str):
            self.embedder = get_embedder(embedder)
        else:
            self.embedder = embedder

        if stored_spec is None:
            self.store.set_meta("embedder_spec", self.embedder.spec)
            self.store.set_meta("dim", str(self.embedder.dim))
        else:
            stored_dim = int(self.store.get_meta("dim") or 0)
            if stored_dim != self.embedder.dim:
                raise ValueError(
                    f"embedder dim {self.embedder.dim} does not match the "
                    f"store's dim {stored_dim} (store was created with "
                    f"{stored_spec!r})"
                )

        # In-memory mirrors, loaded lazily and kept in sync on writes.
        self._loaded = False
        self._ids: list[int] = []
        self._keys: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._matrix: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._intents: dict[str, Intent] = {}
        # per intent: document affinities aligned with matrix rows
        self._intent_affinities: dict[str, np.ndarray] = {}

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "IntentDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        ids, keys, texts, metas, matrix = self.store.load_all_documents()
        self._ids, self._keys, self._texts, self._metas = ids, keys, texts, metas
        if matrix.size:
            self._matrix = matrix
        for row in self.store.load_all_intents():
            intent = Intent(
                name=row["name"],
                description=row["description"],
                exemplars=row["exemplars"],
                instruction=row["instruction"],
                vector=row["vector"],
                lens=IntentLens(gate=row["gate"]),
                lens_strength=row["lens_strength"],
            )
            self._intents[intent.name] = intent
            self._load_intent_stats(intent.name)
        self._loaded = True

    def _load_intent_stats(self, intent_name: str) -> None:
        by_id = self.store.load_intent_affinities(intent_name)
        self._intent_affinities[intent_name] = np.array(
            [by_id.get(i, 0.0) for i in self._ids], dtype=np.float32
        )

    # -- ingest ---------------------------------------------------------------

    def add(
        self,
        text: str,
        doc_key: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Add (or replace, by ``doc_key``) a single document."""
        return self.add_many([(text, doc_key, metadata)])[0]

    def add_many(
        self,
        items: Iterable[tuple[str, str | None, dict | None]],
    ) -> list[str]:
        """Add documents in bulk. Each item is (text, doc_key, metadata)."""
        self._ensure_loaded()
        items = list(items)
        if not items:
            return []
        texts = [t for t, _, _ in items]
        vectors = self.embedder.embed_document_batch(texts)
        out_keys: list[str] = []
        stats_rows: list[tuple[int, str, float]] = []

        for (text, doc_key, metadata), vec in zip(items, vectors):
            key = doc_key or uuid4().hex
            meta = metadata or {}
            doc_id = self.store.upsert_document(key, text, meta, vec)
            out_keys.append(key)

            if key in self._keys:  # replace in the in-memory mirror
                pos = self._keys.index(key)
                self._texts[pos] = text
                self._metas[pos] = meta
                self._matrix[pos] = vec
            else:
                pos = len(self._keys)
                self._ids.append(doc_id)
                self._keys.append(key)
                self._texts.append(text)
                self._metas.append(meta)
                self._matrix = (
                    np.vstack([self._matrix, vec[None, :]])
                    if self._matrix.size
                    else vec[None, :].astype(np.float32)
                )
                for name, aff in self._intent_affinities.items():
                    self._intent_affinities[name] = np.append(aff, 0.0).astype(
                        np.float32
                    )

            for intent in self._intents.values():
                affinity = float(intent.affinity(vec))
                stats_rows.append((doc_id, intent.name, affinity))
                self._intent_affinities[intent.name][pos] = affinity

        if stats_rows:
            self.store.upsert_doc_intent_stats(stats_rows)
        return out_keys

    def delete(self, doc_key: str) -> bool:
        """Delete a document by key. Returns True if it existed."""
        self._ensure_loaded()
        removed = self.store.delete_document(doc_key)
        if removed and doc_key in self._keys:
            pos = self._keys.index(doc_key)
            for lst in (self._ids, self._keys, self._texts, self._metas):
                lst.pop(pos)
            self._matrix = np.delete(self._matrix, pos, axis=0)
            for name, aff in self._intent_affinities.items():
                self._intent_affinities[name] = np.delete(aff, pos)
        return removed

    def get(self, doc_key: str) -> dict | None:
        """Fetch a document (text + metadata) by key."""
        doc = self.store.get_document(doc_key)
        if doc is None:
            return None
        return {
            "doc_key": doc["doc_key"],
            "text": doc["text"],
            "metadata": doc["metadata"],
        }

    # -- intents ----------------------------------------------------------------

    def register_intent(
        self,
        name: str,
        description: str,
        exemplars: list[str] | None = None,
        instruction: str | None = None,
        lens_strength: float = DEFAULT_LENS_STRENGTH,
    ) -> Intent:
        """Register (or redefine) an intent and index all documents under it.

        The intent's vector and lens are derived from the description and
        exemplar queries; every stored document immediately gets its
        affinity precomputed for this intent. ``instruction`` (defaulting
        to the description) conditions query embedding when the embedder is
        instruction-aware.
        """
        self._ensure_loaded()
        intent = Intent.build(
            name=name,
            description=description,
            exemplars=exemplars or [],
            embed_batch=self.embedder.embed_batch,
            instruction=instruction,
            lens_strength=lens_strength,
        )
        self.store.upsert_intent(
            name=intent.name,
            description=intent.description,
            exemplars=intent.exemplars,
            instruction=intent.instruction,
            vector=intent.vector,
            gate=intent.lens.gate,
            lens_strength=lens_strength,
        )
        self._intents[name] = intent

        if self._ids:
            affinities = intent.affinity(self._matrix).astype(np.float32)
            self.store.upsert_doc_intent_stats(
                [
                    (doc_id, name, float(a))
                    for doc_id, a in zip(self._ids, affinities)
                ]
            )
            self._intent_affinities[name] = affinities
        else:
            self._intent_affinities[name] = np.zeros(0, dtype=np.float32)
        return intent

    def remove_intent(self, name: str) -> bool:
        self._ensure_loaded()
        removed = self.store.delete_intent(name)
        self._intents.pop(name, None)
        self._intent_affinities.pop(name, None)
        return removed

    def list_intents(self) -> list[dict]:
        self._ensure_loaded()
        return [
            {
                "name": i.name,
                "description": i.description,
                "exemplars": i.exemplars,
                "instruction": i.instruction,
                "lens_strength": i.lens_strength,
            }
            for i in self._intents.values()
        ]

    # -- retrieval ----------------------------------------------------------------

    def query(
        self,
        text: str,
        intent: str | None = None,
        k: int = 5,
        auto_intent: bool = True,
        weights: dict[str, float] | None = None,
        where: Callable[[dict], bool] | None = None,
        intent_threshold: float = 0.08,
    ) -> list[QueryResult]:
        """Retrieve the top-``k`` documents for a query.

        Parameters
        ----------
        intent:
            Name of a registered intent to retrieve under. ``None`` with
            ``auto_intent=True`` infers the intent from the query itself;
            ``None`` with ``auto_intent=False`` is plain vector search.
        weights:
            Override the blend of ``lensed`` / ``affinity`` / ``base``.
        where:
            Optional metadata predicate, e.g. ``lambda m: m.get("lang") == "en"``.
        """
        self._ensure_loaded()
        if intent is not None and intent not in self._intents:
            raise KeyError(
                f"unknown intent {intent!r}; registered: {sorted(self._intents)}"
            )
        if not self._ids:
            return []

        q = self.embedder.embed_query(text)
        active: Intent | None = None
        inferred = False
        intent_scores: dict[str, float] = {}

        if intent is not None:
            active = self._intents[intent]
            intent_scores = {
                i.name: float(i.affinity(q)) for i in self._intents.values()
            }
        elif auto_intent and self._intents:
            active, intent_scores = infer_intent(
                q, list(self._intents.values()), threshold=intent_threshold
            )
            inferred = active is not None

        base = (self._matrix @ q).astype(np.float64)

        if active is None:
            scores = base
            lensed = affinities = None
        else:
            w = dict(DEFAULT_WEIGHTS)
            if weights:
                w.update(weights)
            # Instruction-aware embedders re-vectorize the query under the
            # active intent (INSTRUCTOR-style); others reuse the plain vector.
            q_active = q
            if self.embedder.supports_instructions and active.instruction:
                q_active = self.embedder.embed_query(
                    text, instruction=active.instruction
                )
            q_lensed_norm = max(float(active.lens.lensed_norms(q_active)), 1e-9)
            raw = self._matrix @ (q_active * active.lens.gate_sq)
            lensed = (raw / q_lensed_norm).astype(np.float64)
            affinities = self._intent_affinities[active.name].astype(np.float64)
            scores = (
                w["lensed"] * lensed
                + w["affinity"] * affinities
                + w["base"] * base
            )

        order = np.argsort(scores)[::-1]
        results: list[QueryResult] = []
        for pos in order:
            if where is not None and not where(self._metas[pos]):
                continue
            results.append(
                QueryResult(
                    doc_key=self._keys[pos],
                    text=self._texts[pos],
                    metadata=self._metas[pos],
                    score=float(scores[pos]),
                    base_score=float(base[pos]),
                    lensed_score=float(lensed[pos]) if lensed is not None else None,
                    intent_affinity=(
                        float(affinities[pos]) if affinities is not None else None
                    ),
                    intent=active.name if active else None,
                    intent_inferred=inferred,
                    intent_scores=intent_scores,
                )
            )
            if len(results) >= k:
                break
        return results

    # -- introspection ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "path": str(self.store.path),
            "documents": len(self._ids),
            "intents": sorted(self._intents),
            "embedder": self.embedder.spec,
            "dim": self.embedder.dim,
        }

    def explain(self, text: str) -> dict[str, Any]:
        """Show how the intent classifier sees a query (no retrieval)."""
        self._ensure_loaded()
        q = self.embedder.embed_query(text)
        active, scores = infer_intent(q, list(self._intents.values()))
        return {
            "query": text,
            "inferred_intent": active.name if active else None,
            "intent_scores": {
                k: round(v, 6)
                for k, v in sorted(scores.items(), key=lambda kv: -kv[1])
            },
        }
