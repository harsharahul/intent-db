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

from . import fusion
from .chunking import chunk_text
from .embedders import Embedder, get_embedder
from .intent import (
    DEFAULT_LENS_STRENGTH,
    Intent,
    IntentLens,
    infer_intent,
    standardize,
)
from .lexical import BM25Index, rrf_fuse
from .mining import IntentSuggestion, mine_intents
from .rerank import DEFAULT_RERANKER_SPEC, Reranker, get_reranker
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
    lexical_score: float | None = None
    rerank_score: float | None = None
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
        if self.lexical_score is not None:
            out["lexical_score"] = round(self.lexical_score, 6)
        if self.rerank_score is not None:
            out["rerank_score"] = round(self.rerank_score, 6)
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
        self._bm25 = BM25Index()
        self._intents: dict[str, Intent] = {}
        # per intent: document affinities aligned with matrix rows
        self._intent_affinities: dict[str, np.ndarray] = {}
        # per intent: mean lensed norm of standardized docs (score scale)
        self._lens_scale: dict[str, float] = {}
        # per intent: fusion weights learned from relevance feedback
        self._fusion_weights: dict[str, dict[str, float]] = {}
        # rerankers built on demand, cached by spec (models load once)
        self._rerankers: dict[str, Reranker] = {}

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
        for key, text in zip(keys, texts):
            self._bm25.add(key, text)
        dim = self.embedder.dim
        for row in self.store.load_all_intents():
            intent = Intent(
                name=row["name"],
                description=row["description"],
                exemplars=row["exemplars"],
                instruction=row["instruction"],
                vector=row["vector"],
                lens=IntentLens(gate=row["gate"]),
                lens_strength=row["lens_strength"],
                # intents from pre-standardization stores fall back to the
                # raw basis (mu=0, sigma=1), matching how they were fit
                mu=row["mu"] if row["mu"] is not None else np.zeros(dim, np.float32),
                sigma=row["sigma"]
                if row["sigma"] is not None
                else np.ones(dim, np.float32),
            )
            self._intents[intent.name] = intent
            self._load_intent_stats(intent.name)
            self._lens_scale[intent.name] = self._compute_lens_scale(intent)
        self._fusion_weights = self.store.load_fusion_weights()
        self._loaded = True

    #: corpus-stat shrinkage: stats only reach full strength once the
    #: collection is a few hundred documents — mu/sigma estimated from a
    #: handful of docs would make "absence of a term" a matchable feature
    STATS_PSEUDO_COUNT = 250

    def _corpus_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Shrunk per-dimension (mu, sigma) of the stored document vectors.

        The raw estimates are blended toward the identity basis
        (mu=0, sigma=1) by ``lam = n / (n + STATS_PSEUDO_COUNT)``: tiny
        collections keep the raw embedding basis, large collections get the
        full anisotropy correction.
        """
        dim = self.embedder.dim
        n = len(self._ids)
        if n == 0:
            return np.zeros(dim), np.ones(dim)
        lam = n / (n + self.STATS_PSEUDO_COUNT)
        mu = lam * self._matrix.astype(np.float64).mean(axis=0)
        sigma = (1.0 - lam) + lam * self._matrix.astype(np.float64).std(axis=0)
        return mu, np.maximum(sigma, 1e-3)

    def _compute_lens_scale(self, intent: Intent) -> float:
        """Mean lensed norm of standardized documents under an intent.

        Dividing lensed scores by this keeps them in a cosine-like range so
        the three signals blend on comparable scales, while still not
        normalizing per document (which would penalize intent-rich docs).
        """
        if not self._ids:
            return 1.0
        docs_s = standardize(self._matrix, intent.mu, intent.sigma)
        norms = np.linalg.norm(docs_s * intent.lens.gate, axis=1)
        return float(max(norms.mean(), 1e-9))

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
            self._bm25.add(key, text)

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

    def add_chunked(
        self,
        text: str,
        doc_key: str,
        metadata: dict | None = None,
        max_chars: int = 1200,
        overlap: int = 200,
    ) -> list[str]:
        """Split a long text into overlapping chunks and store each one.

        Chunks get keys ``{doc_key}#0``, ``{doc_key}#1``, ... and inherit
        ``metadata`` plus ``parent`` (the doc_key) and ``chunk`` (its index).
        """
        chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
        items = [
            (chunk, f"{doc_key}#{i}", {**(metadata or {}), "parent": doc_key, "chunk": i})
            for i, chunk in enumerate(chunks)
        ]
        return self.add_many(items)

    def delete(self, doc_key: str) -> bool:
        """Delete a document by key. Returns True if it existed."""
        self._ensure_loaded()
        removed = self.store.delete_document(doc_key)
        if removed:
            self._bm25.remove(doc_key)
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
            corpus_stats=self._corpus_stats(),
        )
        self.store.upsert_intent(
            name=intent.name,
            description=intent.description,
            exemplars=intent.exemplars,
            instruction=intent.instruction,
            vector=intent.vector,
            gate=intent.lens.gate,
            mu=intent.mu,
            sigma=intent.sigma,
            lens_strength=lens_strength,
        )
        self._lens_scale[name] = self._compute_lens_scale(intent)
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
        self._lens_scale.pop(name, None)
        self._fusion_weights.pop(name, None)
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
        hybrid: bool = False,
        prf: bool = False,
        prf_depth: int = 5,
        rerank: bool | str | Reranker = False,
        rerank_depth: int = 20,
        log: bool = True,
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
        hybrid:
            Also rank with BM25 and fuse the dense and lexical rankings via
            Reciprocal Rank Fusion. ``score`` then holds the RRF value;
            the per-signal fields keep their usual meanings.
        prf:
            Pseudo-relevance feedback (Rocchio): after a first scoring
            pass, pull the query vector toward the top-``prf_depth``
            on-intent documents (and away from off-intent ones when an
            intent is active), then rescore. Costs one extra pass; no
            training, no index changes.
        rerank:
            Re-score the top ``rerank_depth`` candidates with a
            cross-encoder and order results by that score. ``True`` uses
            the default reranker (flashrank, an optional dependency:
            ``pip install intentdb[rerank]``); a spec string such as
            ``"crossencoder:model=..."`` or a
            :class:`~intentdb.rerank.Reranker` instance selects another.
            When an intent is active its instruction is prefixed to the
            query for the cross-encoder — joint (query, doc) scoring is
            where small models actually use intent text. ``score`` then
            holds the reranker's value (as with ``hybrid`` and RRF);
            only reranked candidates are returned.
        log:
            Record the query in the query log (used by
            :meth:`suggest_intents`). Disable for automated traffic.
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

        # Weight precedence: explicit override > weights learned from this
        # intent's relevance feedback > defaults.
        w = dict(DEFAULT_WEIGHTS)
        if active is not None and active.name in self._fusion_weights:
            w.update(self._fusion_weights[active.name])
        if weights:
            w.update(weights)
        # Instruction-aware embedders re-vectorize the query under the
        # active intent (INSTRUCTOR-style); others reuse the plain vector.
        q_active = q
        if (
            active is not None
            and self.embedder.supports_instructions
            and active.instruction
        ):
            q_active = self.embedder.embed_query(text, instruction=active.instruction)

        base, lensed, affinities, scores = self._score_pass(q, q_active, active, w)

        if prf and len(self._ids) > 1:
            depth = min(prf_depth, len(self._ids))
            top = np.argpartition(scores, -depth)[-depth:]
            top = top[scores[top] > 0]  # feedback only from actual matches
            if len(top):
                fb_scores = scores[top]
                q = self._rocchio(q, top, fb_scores, active)
                q_active = (
                    q
                    if active is None
                    else self._rocchio(q_active, top, fb_scores, active)
                )
                base, lensed, affinities, scores = self._score_pass(
                    q, q_active, active, w
                )

        lexical = None
        if hybrid:
            key_to_pos = {key: pos for pos, key in enumerate(self._keys)}
            lexical = self._bm25.scores(text, key_to_pos, len(self._keys))
            dense_order = np.argsort(scores)[::-1]
            # rank only documents that actually matched a query term
            lex_order = np.argsort(lexical)[::-1]
            lex_order = lex_order[lexical[lex_order] > 0]
            scores = rrf_fuse([dense_order, lex_order], len(self._keys))

        if log:
            self.store.log_query(text, active.name if active else None, inferred)

        order = np.argsort(scores)[::-1]
        if where is not None:
            order = np.array(
                [pos for pos in order if where(self._metas[pos])], dtype=int
            )

        rerank_by_pos: dict[int, float] = {}
        if rerank:
            reranker = self._resolve_reranker(rerank)
            window = order[: max(rerank_depth, k)]
            rerank_text = text
            if active is not None and active.instruction:
                rerank_text = f"{active.instruction.strip()}: {text}"
            rerank_scores = reranker.scores(
                rerank_text, [self._texts[pos] for pos in window]
            )
            rerank_by_pos = {
                int(pos): float(s) for pos, s in zip(window, rerank_scores)
            }
            order = window[np.argsort(rerank_scores)[::-1]]

        results: list[QueryResult] = []
        for pos in order:
            results.append(
                QueryResult(
                    doc_key=self._keys[pos],
                    text=self._texts[pos],
                    metadata=self._metas[pos],
                    score=rerank_by_pos.get(int(pos), float(scores[pos])),
                    base_score=float(base[pos]),
                    lensed_score=float(lensed[pos]) if lensed is not None else None,
                    intent_affinity=(
                        float(affinities[pos]) if affinities is not None else None
                    ),
                    lexical_score=(
                        float(lexical[pos]) if lexical is not None else None
                    ),
                    rerank_score=rerank_by_pos.get(int(pos)),
                    intent=active.name if active else None,
                    intent_inferred=inferred,
                    intent_scores=intent_scores,
                )
            )
            if len(results) >= k:
                break
        return results

    def _resolve_reranker(self, rerank: bool | str | Reranker) -> Reranker:
        """Turn the ``rerank`` argument into a Reranker, caching by spec."""
        if isinstance(rerank, Reranker):
            return rerank
        spec = DEFAULT_RERANKER_SPEC if rerank is True else str(rerank)
        if spec not in self._rerankers:
            self._rerankers[spec] = get_reranker(spec)
        return self._rerankers[spec]

    def _score_pass(
        self,
        q: np.ndarray,
        q_active: np.ndarray,
        active: Intent | None,
        w: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
        """Compute (base, lensed, affinities, fused) over the collection.

        The lensed signal is computed in the intent's standardized basis
        (vectors centered and scaled by the corpus stats captured at
        registration), so the gate measures intent-specific structure
        rather than anisotropy artifacts. Only the query is transformed:
        with ``q_s = (q - mu)/sigma`` and ``v = gate^2 * q_s / sigma``,

            <q_s*g, d_s*g> = <d, v> - <mu, v>

        which is one matrix-vector product for the whole collection.
        """
        base = (self._matrix @ q).astype(np.float64)
        if active is None:
            return base, None, None, base
        q_s = standardize(q_active, active.mu, active.sigma)
        v = (active.lens.gate_sq * q_s) / active.sigma
        raw = self._matrix @ v.astype(np.float64) - float(active.mu @ v)
        q_norm = max(float(np.linalg.norm(q_s * active.lens.gate)), 1e-9)
        scale = self._lens_scale.get(active.name, 1.0)
        lensed = (raw / (q_norm * scale)).astype(np.float64)
        affinities = self._intent_affinities[active.name].astype(np.float64)
        fused = w["lensed"] * lensed + w["affinity"] * affinities + w["base"] * base
        return base, lensed, affinities, fused

    def _rocchio(
        self,
        q: np.ndarray,
        top_positions: np.ndarray,
        fb_scores: np.ndarray,
        active: Intent | None,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
    ) -> np.ndarray:
        """Intent-aware Rocchio pseudo-relevance feedback on the query.

        Moves the query toward the score-weighted centroid of on-intent
        feedback documents and (when an intent is active) away from
        off-intent ones — the classic Rocchio update with the first-pass
        retrieval score standing in for graded relevance and the intent
        affinity splitting positive from negative feedback. Pure vector
        arithmetic over vectors already in memory; the index is untouched.
        """
        vecs = self._matrix[top_positions].astype(np.float64)
        weights = fb_scores.astype(np.float64)
        if active is not None:
            aff = self._intent_affinities[active.name][top_positions]
            on, off = aff > 0, aff <= 0
        else:
            on = np.ones(len(vecs), dtype=bool)
            off = ~on

        def centroid(mask: np.ndarray) -> np.ndarray | None:
            if not mask.any():
                return None
            ws = weights[mask]
            return (ws[:, None] * vecs[mask]).sum(axis=0) / ws.sum()

        q2 = alpha * q.astype(np.float64)
        on_c, off_c = centroid(on), centroid(off)
        if on_c is not None:
            q2 = q2 + beta * on_c
        if off_c is not None:
            q2 = q2 - gamma * off_c
        norm = np.linalg.norm(q2)
        return (q2 / norm if norm > 0 else q2).astype(np.float32)

    # -- relevance feedback and learned fusion ---------------------------------

    def record_feedback(
        self,
        query: str,
        doc_key: str,
        useful: bool = True,
        intent: str | None = None,
    ) -> None:
        """Record whether a retrieved document was actually useful.

        This is the learning signal for :meth:`learn_fusion_weights`: an
        LLM (or human) consuming results reports which documents it used.
        ``intent`` should be the intent the query ran under, if any.
        """
        self.store.add_feedback(query, doc_key, intent, useful)

    def learn_fusion_weights(
        self,
        intent: str | None = None,
        min_pairs: int = fusion.MIN_PAIRS,
    ) -> dict[str, dict[str, float] | None]:
        """Learn per-intent fusion weights from accumulated feedback.

        For every feedback query, the three signals (lensed, affinity,
        base) are computed for the marked documents; useful documents are
        paired against non-useful ones (or sampled implicit negatives when
        only positives were recorded), and a small logistic model fits a
        convex weight blend (Bruch et al., TOIS 2023: tuned linear fusion
        beats rank-only fusion and is sample-efficient). Learned weights
        are persisted and applied automatically on subsequent queries;
        intents without enough feedback keep the defaults (``None`` in the
        returned map).
        """
        self._ensure_loaded()
        if intent is not None and intent not in self._intents:
            raise KeyError(f"unknown intent {intent!r}")
        names = [intent] if intent is not None else sorted(self._intents)
        key_to_pos = {k: i for i, k in enumerate(self._keys)}
        rng = np.random.default_rng(0)
        results: dict[str, dict[str, float] | None] = {}

        for name in names:
            act = self._intents[name]
            by_query: dict[str, dict[str, bool]] = {}
            for r in self.store.load_feedback(name):
                by_query.setdefault(r["query_text"], {})[r["doc_key"]] = r["useful"]

            pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for qtext, marks in by_query.items():
                q = self.embedder.embed_query(qtext)
                q_active = q
                if self.embedder.supports_instructions and act.instruction:
                    q_active = self.embedder.embed_query(
                        qtext, instruction=act.instruction
                    )
                base, lensed, aff, _ = self._score_pass(
                    q, q_active, act, DEFAULT_WEIGHTS
                )

                def signals(pos: int) -> np.ndarray:
                    return np.array([lensed[pos], aff[pos], base[pos]])

                positives = [
                    signals(key_to_pos[k])
                    for k, u in marks.items()
                    if u and k in key_to_pos
                ]
                negatives = [
                    signals(key_to_pos[k])
                    for k, u in marks.items()
                    if not u and k in key_to_pos
                ]
                if positives and not negatives:
                    # only positives recorded: sample unmarked docs as
                    # implicit negatives (standard implicit-feedback move)
                    others = [
                        i for i, k in enumerate(self._keys) if k not in marks
                    ]
                    if others:
                        chosen = rng.choice(
                            others, size=min(3, len(others)), replace=False
                        )
                        negatives = [signals(int(i)) for i in chosen]
                pairs.extend(fusion.build_preference_pairs(positives, negatives))

            learned = fusion.learn_weights(
                pairs, DEFAULT_WEIGHTS, min_pairs=min_pairs
            )
            if learned is not None:
                self.store.upsert_fusion_weights(name, learned, len(pairs))
                self._fusion_weights[name] = learned
            results[name] = learned
        return results

    def fusion_weights(self) -> dict[str, dict[str, float]]:
        """Currently learned per-intent fusion weights."""
        self._ensure_loaded()
        return dict(self._fusion_weights)

    # -- intent mining -------------------------------------------------------

    def suggest_intents(
        self,
        k: int = 3,
        min_cluster_size: int = 3,
        undeclared_only: bool = True,
    ) -> list[IntentSuggestion]:
        """Mine the query log for recurring themes that could be intents.

        Clusters logged queries (by default only those that ran without an
        explicitly requested intent) and returns up to ``k`` suggestions,
        each with representative queries to use as exemplars for
        :meth:`register_intent`.
        """
        self._ensure_loaded()
        entries = self.store.load_query_log(undeclared_only=undeclared_only)
        texts = [e["text"] for e in entries]
        if not texts:
            return []
        vectors = np.stack([self.embedder.embed_query(t) for t in texts])
        return mine_intents(
            texts, vectors, k=k, min_cluster_size=min_cluster_size
        )

    # -- introspection ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "path": str(self.store.path),
            "documents": len(self._ids),
            "intents": sorted(self._intents),
            "embedder": self.embedder.spec,
            "dim": self.embedder.dim,
            "logged_queries": self.store.count_query_log(),
            "feedback": self.store.count_feedback(),
            "learned_intents": sorted(self._fusion_weights),
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
