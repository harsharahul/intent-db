# IntentDB — Academic Research Report

*Deep-research synthesis, June 2026. Five parallel literature sweeps
(metric learning · instruction-conditioned embeddings · document
representations · intent mining · fusion & evaluation), ~25 primary
sources, plus a later targeted novelty sweep (~40 sources, folded into §1)
that revised the novelty claims downward where the literature had
already moved. Claims corroborated by two independent sweeps are marked ✓✓.*

---

## 1. Verdict: how IntentDB stands against the literature

The core bet — **documents embedded once, intent applied at query time** —
is the architecture the field converged on independently: TART-dual,
E5-Mistral, Promptriever, and GritLM all condition only the query against
a fixed document index, explicitly so the corpus never needs re-embedding
per task ✓✓. Doc-side intent conditioning (INSTRUCTOR-style document
instructions) was abandoned by later work because it costs
O(corpus × intents).

Each IntentDB mechanism has a direct academic ancestor:

| IntentDB mechanism | Closest literature | Status |
|---|---|---|
| Diagonal lens, mean²/(var+ε) | Fisher score (Gu+ UAI'11); Schultz & Joachims NIPS'03 diagonal Mahalanobis; Xing+ NIPS'02 | Sound for few exemplars — diagonal is the *only* statistically valid regime at d≈10³ with tens of examples (Verma & Branson NIPS'15) |
| Affinity prior cos(d, t) | Rocchio relevance feedback (1971); session-intent priors (Bennett+ SIGIR'12) | Supported; literature says blend it as a re-ranking feature, not a hard filter |
| Instruction-conditioned queries | INSTRUCTOR ACL'23 (+3.4% avg); TART ACL'23; Promptriever ICLR'25 | Architecture right; **model caveat below is the biggest finding** |
| BM25 + RRF hybrid | Cormack+ SIGIR'09; Pyserini hybrid (+1.8–3.8 MRR/nDCG pts) ✓✓ | Right default with zero training data; beatable (see §3.1) |
| k-means intent mining | MTEB's own clustering protocol is mini-batch k-means | Reasonable baseline; large documented headroom (§3.4) |
| Named intent registry | TnT-LLM (KDD'24, Bing Copilot); MADRM named aspects (KDD'22) | Validated pattern; registry also makes intent-conditioning cacheable in a way ad-hoc instructions are not |

**Where the novelty actually sits (revised by the later sweep).**
Query-side diagonal reweighting of frozen embeddings is *not* new on its
own: DIME (SIGIR'24) does it per query, ECLIPSE (ECIR'25) and
Learning-to-Select (2026) extend it, and Conditional Similarity Networks
(CVPR'17) learn per-condition diagonal masks. What appears unoccupied is
the *granularity and packaging* — a registry of named, persistent,
**per-intent** Fisher-score lenses over a vector store, sitting between
DIME's per-query weighting and Search-Adaptor / Chroma's per-dataset
adapters. Two claims survive scrutiny: (a) the **diagonal-vs-low-rank
sample-efficiency ablation on dense retrieval still appears unpublished**
(the closest, Drift-Adapter EMNLP'25, compares adapter families for model
migration but has no diagonal rung), which IntentDB can produce; and (b)
a Fisher-diagonal-anchored, residual low-rank promotion governed by a
held-out statistical test is unattested (Jain+ JMLR'12 give the
`I + ULUᵀ` identity-plus-low-rank Mahalanobis form to build on). The
theoretical grounding is now firmer too: Weller et al. (ICLR'26) prove a
single fixed embedding geometry cannot realize all relevance orderings —
the formal case for "one embedding, many query-conditioned geometries."

## 2. The three findings that challenge our design

**2.1 Small bi-encoders don't really follow instructions ✓✓ (high confidence).**
FollowIR (arXiv:2403.15246): essentially all embedding bi-encoders score
~zero or *negative* p-MRR — they use instructions as "basic keywords."
InstructIR (arXiv:2402.14334): instruction-tuned retrievers (INSTRUCTOR,
TART) *underperform their own base models* on natural user-style
instructions. Genuine instruction-following appears only in ≥3–7B models
trained with instruction negatives (Promptriever, FollowIR-7B).
→ For IntentDB with a local nomic/e5-class embedder, intent instructions
act as a **soft topical bias** (still useful — that's what our lens and
affinity already provide), not semantic filtering. Honest docs should say
so, and the upgrade path is a reranker (§3.2), not a bigger prompt.

**2.2 Raw per-dimension statistics on dense embeddings are corrupted by
"rogue dimensions" (high confidence).** 1–3 dimensions dominate cosine
similarity and they are not the semantically important ones (Timkey & van
Schijndel EMNLP'21); embedding spaces are anisotropic cones (Ethayarajh
EMNLP'19). Whitening/standardization fixes this and *itself improves
retrieval* (Su+ 2021). → Our Fisher lens fitted on raw dimensions partly
measures anisotropy artifacts. **Fix: standardize the space against corpus
statistics before fitting lenses, and shrink lenses toward identity when
exemplars are few (ITML's LogDet-to-identity is the template).**

**2.3 RRF is the right zero-data default but loses to tuned linear fusion
(high confidence).** Bruch, Gai & Ingber (TOIS 2023): a convex combination
of normalized scores beats RRF in- and out-of-domain and needs only a
handful of labeled examples to tune; Pyserini's reference hybrid is a
single tuned α, not RRF. → Keep RRF as the no-training default; add a
learned-weight mode fed by relevance feedback.

## 3. Improvement proposals, ranked by evidence strength × local-first cost

| # | Proposal | Evidence | Cost |
|---|---|---|---|
| 1 | **Whiten/standardize embeddings once; shrink lens toward identity by exemplar count** | Rogue-dimensions + whitening literature (high) | Tiny — numpy, no API change |
| 2 | **Rocchio vector-PRF with an intent twist**: `q' = α·q + β·mean(top-k on-intent vecs) − γ·mean(off-intent vecs)` | Rocchio '71; vector-PRF TOIS'23 — gains with *untuned* params, zero training, query-side only (high) | Tiny — ~10 lines over vectors already in RAM |
| 3 | **Tiny cross-encoder rerank stage with intent injected into the pair** — score `("[intent] query", doc)` | +4 nDCG@10 over SOTA bi-encoders on BEIR (Rosa+ '22); 22M-param MiniLM hits 39.0 MRR@10; FlashRank nano is 4MB CPU-only (high) | Small — optional dependency, top-k only |
| 4 | **Intent mining upgrades**: LLM labels per query then re-cluster (IDAS +7.4%); LLM-named taxonomy + distilled classifier (TnT-LLM); log *which results get used* and cluster on co-selection (Beeferman & Berger KDD'00 ✓✓) | High (replicated pattern) | Medium — needs an LLM in the loop, fits the MCP design |
| 5 | **Learned linear fusion of the four signals** from accumulated feedback | Bruch+ TOIS'23 — sample-efficient, beats RRF (high) | Small once feedback exists |
| 6 | **Per-intent low-rank query adapters** (auto-promote from diagonal lens when an intent accumulates ~10²–10³ feedback pairs) | Search-Adaptor ACL'24: +5% nDCG@10 on 14 BEIR sets, API-only embeddings; Chroma: ~1.5k pairs → large gains (med-high) | Medium — pure numpy least-squares is viable |
| 7 | **Multi-vector docs for known intents** (MADRM-style: embed chunk prepended with intent context, k vectors/doc), candidate via Anthropic contextual-retrieval results (−35% failure) | MVR ACL'22 / MADRM KDD'22 (high on direction) | High — storage ×k, index rebuild on new intent |
| 8 | **HyDE-style intent-conditioned query rewriting** by the consuming LLM | HyDE ACL'23 (61.3 vs 44.5 nDCG@10 DL19); query2doc +3–15% (high) | Zero in-DB — belongs in the MCP client prompt, document the pattern |

Deliberately *not* proposed: full ColBERT token-level multi-vector
(storage ×10, PLAID-class engineering) and LoRA-tuning embedders
(~10⁵ pairs + GPU-hours) — wrong side of the local-first budget.

## 4. Recommended evaluation protocol

The claim "same query, different intent → different correct results" maps
to the FollowIR/InstructIR paired-intent design with intent-conditional
qrels:

1. **Test set**: query–intent pairs where each query appears under ≥2
   intents with separate relevance judgments per intent. Off-the-shelf:
   InstructIR (9,906 instance-wise instructions over MS MARCO), FollowIR
   via `mteb -m $MODEL -t {Robust04,Core17,News21}InstructionRetrieval`,
   or TREC Web Track diversity subtopics as declared intents.
2. **Metrics**: nDCG@10 per (query, intent) against that intent's qrels,
   plus a **p-MRR-style paired delta** — does doc-relevant-to-A move up
   only when intent A is declared? (Designed exactly to defeat the
   "instruction as keyword soup" confound.)
3. **Ablation grid** (the real scientific control): full system vs.
   lens-only vs. affinity-only vs. plain cosine vs. BM25 vs.
   un-conditioned hybrid. The intent claim holds only if conditioning
   beats the un-conditioned hybrid *on intent-conditional qrels*. Plus
   3–5 BEIR datasets as a no-regression check.
4. **Statistics**: ≥50 query–intent pairs (Voorhees & Buckley SIGIR'02);
   paired t-test on per-query nDCG@10 (Smucker CIKM'07, Urbano SIGIR'19);
   Holm–Bonferroni across the grid; report mean delta + 95% CI.

## 5. Key sources

Metric learning: [Xing+ NIPS'02](https://proceedings.neurips.cc/paper/2002/hash/c3e4035af2a1cde9f21e1ae1951ac80b-Abstract.html) ·
[Schultz & Joachims NIPS'03](https://papers.nips.cc/paper/2366-learning-a-distance-metric-from-relative-comparisons) ·
[ITML ICML'07](https://www.cs.utexas.edu/~inderjit/public_papers/itml_icml07.pdf) ·
[Verma & Branson NIPS'15](https://arxiv.org/abs/1505.02729) ·
[Generalized Fisher Score UAI'11](https://arxiv.org/pdf/1202.3725) ·
[Timkey & van Schijndel EMNLP'21](https://aclanthology.org/2021.emnlp-main.372/) ·
[Ethayarajh EMNLP'19](https://aclanthology.org/D19-1006/) ·
[Whitening, Su+ '21](https://arxiv.org/abs/2103.15316) ·
[Search-Adaptor ACL'24](https://aclanthology.org/2024.acl-long.661/) ·
[Chroma Embedding Adapters](https://research.trychroma.com/embedding-adapters) ·
[Jain+ JMLR'12 (I+ULUᵀ Mahalanobis)](https://arxiv.org/abs/0910.5932) ·
[DIME SIGIR'24](https://dl.acm.org/doi/10.1145/3626772.3657691) ·
[Conditional Similarity Networks CVPR'17](https://arxiv.org/abs/1603.07810) ·
[Drift-Adapter EMNLP'25](https://arxiv.org/abs/2509.23471) ·
[Embedding-retrieval limits, Weller+ ICLR'26](https://arxiv.org/abs/2508.21038)

Instruction conditioning: [INSTRUCTOR ACL'23](https://arxiv.org/abs/2212.09741) ·
[TART](https://arxiv.org/abs/2211.09260) ·
[Promptriever](https://arxiv.org/abs/2409.11136) ·
[GritLM ICLR'24](https://arxiv.org/abs/2402.09906) ·
[E5-Mistral ACL'24](https://arxiv.org/abs/2401.00368) ·
[FollowIR](https://arxiv.org/abs/2403.15246) ·
[InstructIR](https://arxiv.org/abs/2402.14334) ·
[MAIR EMNLP'24](https://arxiv.org/abs/2410.10127) ·
[IFIR NAACL'25](https://aclanthology.org/2025.naacl-long.511/)

Representations & feedback: [ColBERTv2 NAACL'22](https://arxiv.org/abs/2112.01488) ·
[PLAID CIKM'22](https://arxiv.org/abs/2205.09707) ·
[MVR ACL'22](https://aclanthology.org/2022.acl-long.414/) ·
[MADRM KDD'22](https://dl.acm.org/doi/10.1145/3534678.3539137) ·
[HyDE ACL'23](https://arxiv.org/abs/2212.10496) ·
[Query2doc EMNLP'23](https://aclanthology.org/2023.emnlp-main.585/) ·
[Vector-PRF TOIS'23](https://arxiv.org/abs/2108.11044) ·
[ANCE-PRF CIKM'21](https://arxiv.org/abs/2108.13454) ·
[Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval) ·
[Cross-encoders, Rosa+ '22](https://arxiv.org/pdf/2212.06121)

Intent mining & personalization: [Broder SIGIR Forum '02](https://dl.acm.org/doi/pdf/10.1145/792550.792552) ·
[Rose & Levinson WWW'04](https://dblp.org/rec/conf/www/RoseL04.html) ·
[Beeferman & Berger KDD'00](https://dl.acm.org/doi/10.1145/347090.347176) ·
[Wen+ TOIS'02](https://dl.acm.org/doi/10.1145/503104.503108) ·
[DeepAligned AAAI'21](https://arxiv.org/abs/2012.08987) ·
[MTP-CLNN ACL'22](https://aclanthology.org/2022.acl-long.21/) ·
[IDAS '23](https://aclanthology.org/2023.nlp4convai-1.7/) ·
[ClusterLLM EMNLP'23](https://aclanthology.org/2023.emnlp-main.858/) ·
[TnT-LLM KDD'24](https://arxiv.org/abs/2403.12173) ·
[Teevan+ SIGIR'05](https://www.microsoft.com/en-us/research/publication/personalizing-search-via-automated-analysis-of-interests-and-activities/) ·
[Bennett+ SIGIR'12](https://dl.acm.org/doi/10.1145/2348283.2348312)

Fusion & evaluation: [RRF SIGIR'09](https://dl.acm.org/doi/10.1145/1571941.1572114) ·
[Bruch+ TOIS'23](https://arxiv.org/abs/2210.11934) ·
[Pyserini hybrid](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert.md) ·
[BEIR NeurIPS'21](https://arxiv.org/abs/2104.08663) ·
[MTEB EACL'23](https://arxiv.org/abs/2210.07316) ·
[TREC'09 Web Track](https://trec.nist.gov/pubs/trec18/papers/WEB09.OVERVIEW.pdf) ·
[Smucker+ CIKM'07](https://dl.acm.org/doi/10.1145/1321440.1321528) ·
[Urbano+ SIGIR'19](https://arxiv.org/pdf/1905.11096)
