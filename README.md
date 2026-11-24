# hybrid-retriever

A single-file SQLite hybrid retriever: **BM25 + vectors + graph**,
fused by weighted reciprocal rank fusion. P1 from
`project-build-plans-all.md` — the substrate every later plan plugs
into, and the retrieval layer for larp v1.4.

The pattern is DocuSearch / QUANTA (report section 1.6): BGE-style
vectors + BM25 (SQLite FTS5) + knowledge-graph neighbour expansion,
fused by weighted RRF (0.50 / 0.35 / 0.15, k=60), optional
cross-encoder rerank, MMR diversity 0.65. It is the stack Hindsight,
agentmemory and MemPalace all converged on.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# ONNX embedding + rerank models (from Hugging Face, ~180 MB total):
#   models/all-minilm-l6-v2-onnx/{model.onnx, tokenizer.json}
#   models/ms-marco-minilm-l6-v2-onnx/{model.onnx, tokenizer.json}
# (Xenova/all-MiniLM-L6-v2 and Xenova/ms-marco-MiniLM-L-6-v2)

# Index a JSONL corpus (one {"text": ...} per line) and ask a question:
.venv/bin/python -m hybrid_retriever.cli index --db memory.db --corpus eval_data/synthetic_corpus.jsonl
.venv/bin/python -m hybrid_retriever.cli query --db memory.db --query "how does Maya maintain her bread culture"

# Run the evals:
.venv/bin/python -m hybrid_retriever.cli eval --dataset synthetic
.venv/bin/python -m hybrid_retriever.cli eval --dataset eval_data/longmemeval_s_cleaned.json --limit 50 --rerank
.venv/bin/python -m hybrid_retriever.cli eval --trace   # TRACE split eval (P2)

# P2 lifecycle commands:
.venv/bin/python -m hybrid_retriever.cli credit --db memory.db --query "..." --answer "..."
.venv/bin/python -m hybrid_retriever.cli sleep --db memory.db --dry-run
.venv/bin/python -m hybrid_retriever.cli merges --db memory.db

# P3 segment consolidation + query planner:
.venv/bin/python -m hybrid_retriever.cli consolidate --db memory.db --transcript turns.jsonl [--dry-run] [--fixed-window 2]
.venv/bin/python -m hybrid_retriever.cli plan --db memory.db --query "how does Maya maintain her bread culture"
.venv/bin/python -m hybrid_retriever.cli eval --segments   # boundaries, consolidation cost, retrieval, planner

# P18 structured context eviction:
.venv/bin/python -m hybrid_retriever.cli ingest --db memory.db --session s1 --transcript turns.jsonl
.venv/bin/python -m hybrid_retriever.cli evict --db memory.db --session s1 --budget 2000 [--policy naive] [--dry-run]
.venv/bin/python -m hybrid_retriever.cli episodes --db memory.db --session s1 [--explain 12]
.venv/bin/python -m hybrid_retriever.cli eval --eviction   # long session vs naive vs isolated

# P19 type-aware tool-output compression (inside the eviction loop):
.venv/bin/python -m hybrid_retriever.cli evict --db memory.db --session s1 --budget 2000 --compress
.venv/bin/python -m hybrid_retriever.cli eval --compression   # compress+evict vs evict-only A/B
```

## Architecture

```
query ──┬── BM25 (FTS5) ────────────┐
        ├── vector (sqlite-vec) ─────┤ weighted RRF (0.50/0.35/0.15, k=60)
        └── graph neighbours ────────┘        │
                              optional cross-encoder rerank (top 20)
                              optional MMR diversity (lambda 0.65)
                                              ▼
                                    ranked records, with legs
```

- `hybrid_retriever/store.py` — SQLite schema: `records`,
  `records_fts` (FTS5, trigger-synced), vectors in a `vec0` table
  when `sqlite-vec` loads, otherwise as BLOBs with exact NumPy
  search (`vec_mode` reports which). `edges` is the minimal graph.
- `hybrid_retriever/embeddings.py` — `Embedder` protocol.
  Resolution order: sentence-transformers (MiniLM) → ONNX Runtime
  (same model, no torch) → `HashedEmbedder` placeholder. The
  placeholder is deterministic token-hash vectors: lexical, not
  semantic, and every report says which embedder was used.
- `hybrid_retriever/retrieval.py` — `rrf_fuse`, `mmr`, `retrieve`.
  Legs return ranked id lists; a leg that returns nothing
  contributes nothing. Vector search drops cosine ≤ 0 rows so the
  vector leg never returns the whole store on a no-overlap query.
  `candidate_ids` restricts all legs to a subset (used by the
  LongMemEval harness, where each question has its own haystack).
- `hybrid_retriever/rerank.py` — optional cross-encoder rerank:
  `OnnxReranker` (preferred, no torch) or `CrossEncoderReranker`
  (sentence-transformers). Never a hard dependency.
- `hybrid_retriever/eval.py` — metrics (P@10, R@10, nDCG@10,
  any-hit@5, grounding@10 = any-hit@10) and the LongMemEval-S +
  synthetic runners, plus `run_synthetic_trace` (P2) and
  `run_segment_eval` (P3: boundary F1, consolidation cost,
  retrieval after consolidation, planner routes).
- `hybrid_retriever/lifecycle.py` — P2: TRACE-UCB credit
  (`record_use`), the judge-gated merge (`propose_merge` /
  `rollback_merge` / `certificate_for_merge`), merge candidates, and
  the `sleep` job. No LLM call anywhere on the write path; the judge
  is rules + the same MiniLM vectors the store already holds.
- `hybrid_retriever/segments.py` — P3: Lychee-style lazy segment
  consolidation (`detect_boundaries`, `classify_segment`,
  `consolidate_transcript` / `consolidate_records`) and the query
  planner (`plan_query` / `retrieve_planned`). Consolidation reuses
  the P2 merge machinery with the segment profile
  (`max_length_ratio=1.0`, type override); the planner picks one
  route per query — temporal, typed, keyword, vector, hybrid —
  from text cues plus cheap corpus probes, and returns the choice
  in the `Plan` so it can be logged and evaluated.
- `hybrid_retriever/eviction.py` — P18: CWL structured context
  eviction (`annotate_episode` / `ingest_session`,
  `evict_to_budget`, `explain_episode`). Episodes are the context
  (typed, dependency-linked, in `episodes`); records are the
  memory. Eviction removes from context, never deletes a record;
  every eviction lands in the `evictions` audit table with its
  reason. Tokens are estimated deterministically
  (`ceil(len(text) / 4)`); no LLM, no model, no tokenizer.
  `evict_to_budget(..., compress=True)` (P19) compresses a
  `tool_result` victim before evicting it.
- `hybrid_retriever/compress.py` — P19: type-aware tool-output
  compressor (`detect_type` / `compress_text` /
  `verify_no_invention`). JSON / table / log / code / text
  compressors, extractive-only, no LLM, no model. Compressed
  text lives on the episode row (`compressed_text` /
  `compressed_tokens`); the record is never rewritten; every
  compression lands in the `compressions` audit table.

## P2 — MEMOIR-style lifecycle (TRACE-UCB + judge-gated merge + sleep)

`lifecycle.py` implements MEMOIR (arXiv:2611.07631) on top of the P1
store. Records carry `importance` / `credit` / `uses` / `last_used`
plus a `status` (`live` / `superseded` / `pending` / `pruned` /
`archived`); search legs only ever return `live` records. v1
databases are migrated in place (columns added by `ALTER TABLE`).

- **TRACE-UCB** — after each answer, `record_use` attributes credit
  from lexical overlap with the answer, times a leave-one-out
  uniqueness factor (the share of the overlap no other retrieved
  record also explains), times a small legs-agreement multiplier.
  Importance is mean credit + a UCB exploration bonus
  (`c * sqrt(ln(N+1) / uses)`, neutral prior 0.5). `retrieve(...,
  importance_weight=w)` multiplies the fused RRF score by
  `1 + w * normalised_importance`; w = 0 is exactly P1 behaviour.
- **Judge-gated merge** — `propose_merge` drafts a merged text by
  sentence union (no LLM, so faithfulness holds by construction)
  and runs the sufficiency certificate: faithfulness ≥ 0.95,
  coverage ≥ 0.45, length ≤ 0.90, mean source similarity ≥ 0.30.
  On pass the merged record goes live and the sources are marked
  `superseded` (never deleted); on fail only the rejected audit row
  is written. Every proposal lands in the `merges` table;
  `rollback_merge` restores the sources. `hr merges` lists the audit
  table.
- **Sleep** (`hr sleep`) — the offline job: propose merges for
  near-duplicate clusters (word-set Jaccard ≥ 0.9, or cosine ≥ 0.90
  for pairs sharing half their tokens), prune dead memories (never
  used, zero credit, older than `--prune-days`, default 90), and
  promote co-recall pairs (used together for one answer, count ≥ 2)
  to `co-recall` graph edges. `--dry-run` reports without writing.

### Measured results — TRACE split (synthetic, labelled)

Warm-up on the 48 keyword queries (credit attributed against the
gold records' text as the answer proxy), eval on the 48 paraphrase +
8 trap queries. The split tests whether credit learned from one
phrasing transfers to another phrasing of the same facts; warming up
on the eval queries themselves would be training on the test set.
Embedder: `onnx-all-MiniLM-L6-v2`. Reproduce with
`hr eval --trace [--ucb-c 0.3]`.

| mode | P@10 | R@10 | nDCG@10 | any-hit@5 | grounding@10 |
|---|---|---|---|---|---|
| vector (bare cosine) | 0.223 | 0.744 | 0.626 | 0.875 | 0.911 |
| hybrid | 0.118 | 0.393 | 0.297 | 0.589 | 0.768 |
| hybrid+trace w=0.5, ucb_c=1.0 | 0.136 | 0.452 | 0.325 | 0.571 | 0.911 |
| hybrid+trace w=1.0, ucb_c=1.0 | 0.141 | 0.470 | 0.330 | 0.536 | 0.946 |
| hybrid+trace w=0.5, ucb_c=0.3 | 0.138 | 0.458 | 0.340 | **0.661** | 0.893 |
| hybrid+trace w=1.0, ucb_c=0.3 | 0.145 | 0.482 | 0.341 | 0.571 | 0.929 |
| hybrid+trace w=2.0, ucb_c=0.3 | 0.139 | 0.464 | 0.325 | 0.607 | 0.929 |

**Honest reading:** TRACE works as designed but does not close the
gap to bare cosine on this corpus. With the textbook exploration
constant (ucb_c=1.0) the bonus swamps the credit signal after a
single warm-up pass — every once-retrieved record, distractor or
not, gets boosted, so recall@5 *falls* while R@10 rises. With
ucb_c=0.3 the credit signal dominates and recall@5 improves
0.589 → 0.661 (+7.2pp) at w=0.5, with R@10 up 0.393 → 0.458.
The weight sweet spot is 0.5; heavier weights push more gold into
the top 10 but not the top 5. And bare cosine still wins outright
(0.875) — the same P1 finding: on this paraphrase-heavy corpus the
vector leg dominates and the fixed-weight hybrid dilutes it. TRACE
is a partial fix for the hybrid, not a cure for the weighting
problem. The code default stays at ucb_c=1.0 (standard UCB1); the
CLI exposes `--ucb-c` because the measurement says 0.3 is better
for this workload.

The P1 limitation inherits: on LongMemEval-S the records *are*
whole sessions truncated to 256 tokens, so TRACE's credit signal is
only as good as the truncated embeddings that produced the
retrieval. P2 works at record level and does not fix that — chunked
or summarised sessions (the P1 follow-up) are still the fix.

### Measured results — merge audit (`scripts/audit_merges.py`)

100 proposals from the synthetic corpus (96 near-duplicate merges +
4 negatives that must be rejected). Both embedders:

| embedder | proposed | committed | rejected | losses | unexpected |
|---|---|---|---|---|---|
| onnx-all-MiniLM-L6-v2 | 100 | 96 | 4 | **0** | 0 |
| hashed-384 | 100 | 96 | 4 | **0** | 0 |

The 4 rejections are the 4 negatives (unrelated sources → length,
invented content → faithfulness, mixed types, single source), each
with the expected reason. Zero losses means every sentence of every
source survived into its merged record.

### Measured results — LongMemEval-S regression (real, 50Q)

Re-run after P2 with importance off: numbers are **identical** to
the P1 table (bm25 R@10 1.000, vector 0.860, hybrid 0.980,
hybrid+mmr 0.860). The schema migration, the live-only filtering,
and the new columns changed nothing on the standard path.

## Measured results

All numbers below were measured on this VM (1.5 shared CPUs, ~6 GB
RAM). Embedder: `onnx-all-MiniLM-L6-v2` unless stated. Nothing here
is a vendor claim — the scripts reproduce every number.

### Synthetic corpus (labelled, 160 records, 104 queries)

8 people × 6 attributes × 3 records + 16 lexical-trap records.
Queries: 48 keyword, 48 paraphrase, 8 trap. Built by
`scripts/build_synthetic_corpus.py` (seed 42, deterministic).

| mode | P@10 | R@10 | nDCG@10 | any-hit@5 | grounding@10 |
|---|---|---|---|---|---|
| bm25 | 0.181 | 0.603 | 0.546 | 0.740 | 0.827 |
| vector | 0.252 | 0.840 | 0.756 | 0.933 | 0.952 |
| hybrid (paper weights) | 0.189 | 0.631 | 0.513 | 0.779 | 0.875 |
| hybrid+mmr | 0.242 | 0.808 | 0.729 | 0.933 | 0.962 |

With the hashed placeholder embedder instead of MiniLM, hybrid
(0.188 / 0.625 / 0.527 / 0.779 / 0.875) still beats both single legs —
the fusion works even when the vector leg is only lexical.

### Weight sensitivity (`scripts/weight_sweep.py`, same corpus)

| weights (bm25/vector/graph) | P@10 | R@10 | nDCG@10 | hit@5 | grounding@10 |
|---|---|---|---|---|---|
| paper 0.50/0.35/0.15 | 0.189 | 0.631 | 0.513 | 0.779 | 0.875 |
| vector-heavy 0.25/0.60/0.15 | 0.222 | 0.740 | 0.546 | 0.846 | 0.952 |
| balanced 0.40/0.40/0.20 | 0.187 | 0.622 | 0.332 | 0.519 | 0.904 |
| no graph 0.50/0.50/0.00 | 0.237 | 0.788 | 0.679 | 0.837 | 0.952 |
| bm25-heavy 0.70/0.20/0.10 | 0.187 | 0.622 | 0.575 | 0.788 | 0.865 |
| vector only | 0.252 | 0.840 | 0.756 | 0.933 | 0.952 |

**Honest reading:** on this paraphrase-heavy corpus the vector leg
dominates and the paper weights underperform vector-only. Two
reasons, both fixable in later plans, not by tuning constants here:

1. The graph leg as implemented chains *all* same-person records,
   most of which are not gold for any given query — indiscriminate
   edges add noise. Raising the graph weight to 0.20 makes nDCG
   *worse* (0.332). Meaningful edges (co-recall, P4/REALM) are the
   fix, not a bigger weight.
2. Fixed weights can't adapt to a corpus where one leg is much
   stronger. The defaults stay at the paper's 0.50/0.35/0.15, but
   `weight_sweep.py` exists to re-tune per corpus, and per-query
   weight adaptation is a P2 follow-up.

### LongMemEval-S (real, 50 of 500 questions, 2,375 sessions)

Dataset: `xiaowu0162/longmemeval-cleaned`,
`longmemeval_s_cleaned.json` (SHA-256 verified
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`).
Abstention questions excluded, as in the official harness. Records
are whole sessions; each question is scored against its own ~50
session haystack via `candidate_ids`. Run took ~9 min on this VM.

| mode | P@10 | R@10 | nDCG@10 | any-hit@5 | grounding@10 |
|---|---|---|---|---|---|
| bm25 | 0.100 | 1.000 | 0.961 | 1.000 | 1.000 |
| vector | 0.154 | 0.860 | 0.715 | 0.800 | 0.860 |
| hybrid (paper weights) | 0.098 | 0.980 | 0.908 | 0.920 | 0.980 |
| hybrid+mmr | 0.086 | 0.860 | 0.699 | 0.820 | 0.860 |
| hybrid+rerank | 0.092 | 0.920 | 0.738 | 0.760 | 0.920 |

**Honest reading — the opposite of the synthetic result, same lesson:**
BM25 alone is the strongest leg here, and that matches published
work (an independent BM25 implementation reports 95% R@5 on this
benchmark). LongMemEval questions share distinctive keywords with
the gold session, so lexical matching wins; the vector leg loses
because whole sessions are truncated to 256 tokens and the answer
detail often sits past the cut. The fixed-weight hybrid dilutes the
strong leg with the weak one (R@10 0.980 vs BM25's 1.000), MMR hurts
because the gold session resembles its neighbours, and the
cross-encoder rerank hurts (any-hit@5 0.760 vs 0.920) because it
also only sees the first 512 tokens of each session.

So: the fusion machinery works, but on this benchmark the *inputs*
to the vector and rerank legs are wrong — they need chunked or
summarised sessions, not truncated wholes. That is the concrete
next step for the larp v1.4 integration (chunk records before
embedding; rerank on the chunk, not the session), and it is why the
synthetic corpus — where records are short — shows the opposite
ranking. Both results are real; they disagree because the corpora
differ, and the README says so instead of picking the flattering
one. The full 500-question run is the obvious follow-up once a
faster machine is available.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q   # 79 tests, all passing
```

Covers: store (both vector backends), FTS5 + candidate filtering,
RRF fusion math, MMR diversity, end-to-end hybrid retrieval, graph
expansion, the ONNX embedder's paraphrase gap, and the empty-query
edge case (zero-vector queries return nothing, not the whole store).
P2 adds TRACE credit, the merge certificate (including rejections),
rollback, sleep and v1 migration; P3 adds boundaries, segment
typing, consolidation and planner routing; P18 adds episode
annotation, eviction tiers and protections, budget enforcement,
explainability, and the naive-vs-CWL contrast.
P2 adds: TRACE credit (used vs unused, importance ordering and
decay, co-recall logging, the importance boost reordering
retrieval), the merge gate (commit, three rejection reasons,
rollback, sentence-union drafting), the sleep job (merge + prune +
co-recall promotion, dry-run writes nothing), and the v1 database
migration. P3 adds: boundary detection (topic switch found,
min/max-turns guards, semantic beats fixed-window on two topics),
segment typing (all four types, decision wins ties), consolidation
(closed segments merged and typed, sources superseded, dry-run
writes nothing, single-turn segments skipped, fixed-window
control), the type-override default (P2 merges unchanged), and the
query planner (temporal / typed / keyword / vector routes, typed
fallback, importance prior, `retrieve_planned`).

## P3 — Lychee-style lazy segment consolidation

`segments.py` implements LycheeMemory V2 (arXiv:2608.12990) on top of
the P1 store, with the P2 merge machinery reused for consolidation.
A transcript is buffered as `episode` records; embedding drift
(cosine between adjacent turns below a threshold) cuts it into
topic segments; only *closed* segments are consolidated — the last
segment keeps buffering — into one typed record each (fact /
decision / error-fix / preference, rule-based cues, no model). The
query planner picks one retrieval route per query.

One honest substitution vs the paper: Lychee's saving is LLM
*construction tokens* (−86% vs per-turn consolidation). This module
has no LLM — the "summary" of a segment is the sentence union of
its turns — so the measured saving is in consolidation
*operations* (one per segment instead of one per turn) and records
created, not tokens. The eval reports those numbers, not the
paper's.

- **Boundaries** — `detect_boundaries` cuts when adjacent-turn
  cosine drops below τ, with a min-turns guard (a lone turn never
  forms its own segment) and a max-turns force-cut. τ=0.30 was
  tuned on 3 dev transcripts (sweep 0.15–0.60, best F1 0.952).
- **Consolidation** — `hr consolidate` ingests a transcript JSONL
  file and consolidates closed segments through `propose_merge`
  with the segment profile: `max_length_ratio=1.0` (a segment merge
  compresses *operations*, N turns → 1 record, not words — the
  sentence union of distinct turns cannot be shorter than its
  inputs) and `type_override` from `classify_segment`. The
  certificate's faithfulness and coverage checks still apply, and
  every proposal lands in the `merges` audit table. `--dry-run`
  judges without writing; `--fixed-window N` runs the control.
- **Planner** — `plan_query` routes by text cues (temporal words →
  recent records; decision/preference words → that type; quoted
  phrase → BM25) plus a cheap 5-hit BM25/vector probe: no hits on
  one leg → the other; legs disagree → hybrid; legs agree →
  keyword only when *every* content term appears in the corpus,
  else vector. Per-record importance/credit feeds the P2 hook
  (`importance_weight=0.5` when the probe's hits average > 0.6).
  Two rules were fixed after the first eval run (see below):
  the keyword bar was raised from "most terms appear" (≥ 0.6) to
  "all terms appear" (= 1.0), and error-fix type cues were dropped
  — "fix"/"bug" are topic words in queries, not type signals, and
  routing on them restricted candidates away from the gold record.
  The planner is opt-in (`retrieve_planned` / `hr plan`), not the
  default: it trades a few accuracy points for ~3× fewer leg runs.

### Measured results — segment eval (synthetic, labelled)

6 transcripts built from the corpus (116 turns, 8 people × 6
attributes, runs of 2–3 turns; traps excluded), 3 dev + 3 held-out.
Embedder: `onnx-all-MiniLM-L6-v2`. Reproduce with
`hr eval --segments`. The paper's headline (89.22% vs 82.40%
LoCoMo, semantic vs fixed-window) holds here in boundary F1:

| boundaries (held-out) | P | R | F1 |
|---|---|---|---|
| semantic (τ=0.30) | 1.000 | 0.905 | **0.949** |
| fixed-window (window=2) | 0.261 | 0.333 | 0.292 |

| consolidation | ops | live records |
|---|---|---|
| per-turn (baseline) | 116 | 116 |
| semantic | **40** (2.9× fewer) | 56 |
| fixed-window | 54 | 62 |

Retrieval after consolidation (hybrid, no MMR, all 104 queries).
**Read R@10 with care:** consolidation shrinks the gold set (3
turn records → 1 segment record), so hitting the single gold
record scores R@10 1.0 — the numbers are not comparable across
stores. any-hit@5 is the apples-to-apples metric. Consolidation
helps it because each attribute's evidence is concentrated in one
record, where single turns matched only part of a query:

| any-hit@5 | paraphrase | keyword | trap |
|---|---|---|---|
| raw turns | 0.812 | 1.000 | 0.625 |
| semantic-consolidated | **0.958** | 1.000 | **0.750** |
| fixed-window-consolidated | 0.896 | 1.000 | 0.750 |

Planner vs plain hybrid on the semantic store (same 104 queries):

| any-hit@5 | paraphrase | keyword | trap |
|---|---|---|---|
| hybrid | 0.958 | 1.000 | 0.750 |
| planner | 0.896 | 1.000 | 0.625 |

Routes chosen: paraphrase → vector 42, keyword 2, temporal 2,
typed 1, hybrid 1; keyword → keyword 32, vector 16; trap →
vector 5, keyword 2, hybrid 1. The first version of the rule
(hit_frac ≥ 0.6, error-fix type cues) was much worse — paraphrase
hit@5 0.771, trap 0.500 — because it misrouted paraphrase queries
to BM25, which cannot match synonyms. The fixed rule keeps the
planner within a few points of hybrid while running one leg
instead of three on most queries. Trap queries stay the weak
spot: they are built to have high lexical overlap, so any
lexical signal routes them to the leg they are designed to fool.

### Measured results — LongMemEval-S regression (real, 50Q)

Re-run after P3: numbers are **identical** to the P1/P2 table
(bm25 R@10 1.000, vector 0.860, hybrid 0.980, hybrid+mmr 0.860).
Consolidation is a separate offline job, so the standard retrieval
path is untouched — the regression check confirms it.

## What P3 hooks into (P4)

- `Store` — segment records are typed and carry `importance` /
  `credit` / `uses` exactly like other records; the `merges`
  table already records which turns each segment absorbed, so P4's
  graph can link segments (co-recall between segments) without a
  new table.
- `plan_query` — the planner's route log is the signal P4's
  adaptive search can learn from: which routes win per query kind.
- `eval.py` — `run_segment_eval` is the fixed target P4 must not
  regress: boundary F1 0.949, consolidation 40 ops, semantic
  any-hit@5 0.958 / 1.000 / 0.750.

## P18 — CWL structured context eviction

`eviction.py` implements Beyond Compaction / CWL (arXiv:2606.11213)
on top of the P1–P3 store. One delimiter tool annotates the
trajectory as typed, dependency-linked episodes as work proceeds;
a deterministic, LLM-free policy evicts in priority order when a
token budget is exceeded.

**Episodes, not turns.** `hr ingest` stores each turn as an
`episode` record plus an `episodes` row: a type (`user` /
`reasoning` / `tool_call` / `tool_result` / `decision` /
`error_fix` / `preference` / `fact` / `chatter`), a dependency
list, a `persisted_by` record id, and a token estimate. Role
defaults apply (`user` → `user`, `tool` → `tool_result`,
assistant → `reasoning` unless the text carries decision /
error-fix / preference cues); an explicit `episode_type` on the
turn always wins. A `tool_result` with no stated dependency is
linked to the latest preceding `tool_call` — a result without its
call is meaningless. `mark_session_persisted_by_merges` wires P3
consolidation in: once a segment record absorbs a turn, that
turn's episode is marked persisted by it.

**Eviction tiers** (lower evicted first): 0 persisted
`tool_result`, 1 `chatter`, 2 persisted `tool_call`, 3 any
persisted episode, 4 unpersisted `reasoning`, 5 unpersisted
`decision` / `error_fix`, 6 unpersisted `tool_call` /
`tool_result` (evicting loses information — last resort). Within
a tier: older first, then lower importance, then fewer uses.
Hard protections, never broken for a budget: user turns, the
active window (last 3 episodes), and any episode a still-live
episode depends on (checked dynamically, so evicting the
dependent first frees its dependency). If the protections make
the budget unreachable, the policy evicts everything evictable
and reports `under_budget=False` with the overage. Every
eviction is written to the `evictions` audit table with its
reason; `hr episodes --explain <id>` explains any episode from
the log, or why a live one stays. `--dry-run` reports
`would_evict` and writes nothing. The naive control
(`--policy naive`) is oldest-first truncation with the same
active window but no tiers, no user-turn protection and no
dependency protection.

### Measured results — eviction demo (synthetic, labelled)

40 tasks (20 keyword + 20 paraphrase queries, distinct
attributes), 7 turns each, 3229 tokens total, budget 1076 (33%).
After each task the session consolidates the task's non-user
turns into one segment record (user turns are the conversation,
not facts to merge), marks absorbed episodes persisted, and
evicts to budget. Reproduce with `hr eval --eviction`. Embedder:
`onnx-all-MiniLM-L6-v2`.

| arm | task hit@5 | user-turn recall | user turns | live tokens | dep. violations |
|---|---|---|---|---|---|
| isolated (per task) | 1.000 | 1.000 | — | — | — |
| CWL | **1.000** | **1.000** | 40/40 | 1076 | 0 |
| naive (oldest-first) | 1.000 | 0.275 | 11/40 | 1073 | 1 |

CWL evicted 194 episodes (tier 0: 118, tier 1: 39, tier 2: 37)
— every one of them persisted, so nothing was lost. Naive
evicted 201 with no tier structure.

The honest reading: task accuracy holds in *both* long arms
because P3 consolidation preserves the facts in segment records
— eviction is not what saves the facts, consolidation is. The
policy's value is in the conversation, not the facts: CWL keeps
every user turn in context (recall 1.000 vs 0.275) and never
strands a dependent (0 vs 1 violation), while naive truncation
destroys the conversation it is supposed to be working in. That
is exactly the paper's claim in the form that applies here: a
long session inside a fixed budget with no accuracy degradation
vs isolated sessions.

One bug was caught by the first demo run and fixed: the first
version scored task questions by matching stored turn text
against corpus gold texts, but stored turns carry a `"role: "`
prefix the corpus texts do not have, so every long-arm task
scored 0.000. Gold is now tracked by record id at ingest time,
and the demo smoke test pins the fixed behaviour.

### Regressions (re-run after P18)

- Segment eval: identical to P3 — boundaries semantic F1 0.949,
  fixed-window 0.292; consolidation 40 semantic ops; semantic
  any-hit@5 0.958 / 1.000 / 0.750.
- TRACE split: identical to P2 (vector 0.875, hybrid 0.589,
  hybrid+trace w=0.5 0.571 at ucb_c=1.0).
- LongMemEval-S 50Q: identical to P1/P2/P3 (bm25 R@10 1.000,
  vector 0.860, hybrid 0.980, hybrid+mmr 0.860).

## P19 — Type-aware tool-output compressor

`compress.py` implements a minimal, honest Headroom: compress
tool_result episodes by output type *before* eviction sheds
them, inside the eviction loop (`evict_to_budget(...,
compress=True)`; `hr evict --compress`). When the next victim
is a `tool_result`, its output is compressed first and the loop
re-evaluates — the episode is evicted only if the session still
does not fit after compression. Default off, so P18 numbers are
identical with `compress=False`.

**Compressors, by detected type** (`detect_type`): JSON —
schema/key pruning (drop null/empty values, collapse
homogeneous lists to head + tail + a marker naming the shared
keys and the count, dedupe identical items, truncate long
strings with a marker); tables — drop all-empty columns,
dedupe rows, keep head + tail rows with a count marker;
logs — fold repeated lines (`line (xN)`), keep head + tail;
Python code — AST pruning (strip docstrings/comments, truncate
long string literals); text — extractive keep (dedupe
sentences, keep head + tail). All deterministic, no LLM, no
model. A compression only counts when it saves ≥10 tokens and
≥10% of the original; anything smaller is returned unchanged.

**Originals are never touched.** Compression lives on the
*episode* (the context), never on the *record* (the memory):
`set_episode_compressed` stores the compressed text beside the
episode and shrinks its token count; the record keeps the
original, retrievable on demand (`episode_context_text`). Every
compression is written to the `compressions` audit table (type,
tokens before/after, units before/after, latency). An episode
is compressed at most once — a second encounter evicts it
instead of re-counting the same savings (a first version
re-compressed already-compressed episodes and logged phantom
savings; the regression test pins the fix).

**Faithfulness is checkable by construction.** The compressor
is extractive-only: output is built from pieces of the original
plus count markers, nothing is invented. Each compressor
reports units kept / units seen (JSON leaf paths, table cells,
distinct log lines, code lines, sentences), and
`verify_no_invention` proves no line of a compressed output is
unaccounted for. Turn text carries a `"role: "` prefix;
detection and compression run on the body, the prefix is
re-attached.

### Measured results — compression A/B (synthetic, labelled)

40 tasks (20 keyword + 20 paraphrase), 7 turns each, with the 3
tool_result turns carrying realistic outputs (JSON API response,
log tail, markdown table) embedding the gold facts; gold
position rotates per task so it is sometimes in the region the
compressor keeps and sometimes in the region it collapses.
64,733 tokens total, budget 21,577 (33%). Reproduce with
`hr eval --compression`. Embedder: `onnx-all-MiniLM-L6-v2`.

| arm | task hit@5 | user-turn recall | gold survival (end) | gold survival (mean over time) | live tokens | evicted |
|---|---|---|---|---|---|---|
| isolated (per task) | — | — | 1.000 | — | — | — |
| evict (CWL) | 0.350 | 1.000 | 0.317 | 0.681 | 21,292 | 82 |
| compress (CWL) | 0.350 | 1.000 | 0.317 | 0.682 | 21,514 | 81 |

Compress arm: 82 compressions, **29,514 tokens saved by
compression** (46% of the session total, ~360 per
compression), 1 eviction avoided (82 → 81), faithfulness
**0.486** (units kept / units seen), **0 no-invention
violations**, latency mean 0.79 ms / p95 0.62 ms per call
(a few slow outliers pull the mean above p95). Both arms under
budget, 0 dependency violations.

The honest reading: the per-episode savings are large and
real, but the end-state benefit in this setup is ~zero — 1
fewer eviction, identical end survival, identical task hit@5,
and mean-over-time survival 0.682 vs 0.681 (noise). The session
grows monotonically and the budget is a cap, so compression
delays eviction by about one task rather than preventing it: a
compressed episode survives one more eviction round, then gets
evicted as the oldest tier-0 victim in the next round, and the
saved tokens are absorbed by the next task's growth. The
faithfulness price is also real: the compressor keeps about
half the countable units, dominated by JSON list collapse (12
items → head + tail + marker). And task hit@5 is 0.350 in *both*
long arms vs 1.000 in the P18 demo — not a compression
regression (both arms are equal), but a property of this
workload: P3 consolidation merges the long tool outputs into
segment records too long for the 256-token embedding window,
so retrieval dilutes. Compression is a token lever, not a
recall lever, on this workload — publish the harness, not the
headline (the same lesson as Headroom's 21–57% claim vs the
independent ~2% test).

### Regressions (re-run after P19)

- Eviction eval (P18 fixed target): CWL task hit@5 1.000,
  user-turn recall 1.000, 40/40 user turns, 0 dependency
  violations, at budget 1076 — identical.
- Segment eval: identical to P3 — boundaries semantic F1 0.949,
  fixed-window 0.292; consolidation 40 semantic ops; semantic
  any-hit@5 0.958 / 1.000 / 0.750.
- TRACE split: identical to P2 (vector 0.875, hybrid 0.589,
  hybrid+trace w=0.5 0.571 at ucb_c=1.0).
- LongMemEval-S 50Q: identical to P1/P2/P3 (bm25 R@10 1.000,
  vector 0.860, hybrid 0.980, hybrid+mmr 0.860).

## What P19 hooks into (P20/P21)

- `run_compression_eval` is the A/B harness P20's
  verify-by-doing evaluator and P21's mini-TOBench can reuse:
  fixed task set, four measured metrics, no LLM in the loop.
- The `compressions` audit table is the measurement log for
  any future policy that compresses before evicting (e.g. a
  per-type budget, or compressing `reasoning` episodes too —
  currently only `tool_result` is compressed).

## P20 — Verify-by-Doing (VbD) harness

The paper's idea (arXiv:2611.15276): judge a memory by whether it
helps an agent *do* the task, not by text similarity — score
memories by task success with them vs without them, so the harness
measures causal usefulness, not retrieval overlap. The verifier
checks by doing: it re-parses the raw tool output and recomputes
the expected answer itself, and reads the agent's answer back
from the artifact file the agent wrote — it never reads the
actor's reasoning or its memories.

Two substitutions, stated plainly: there is no LLM anywhere.
The actor is a deterministic solver that parses a tool-output
artifact, applies the first rule memory it was given, and writes
an answer artifact. The paper's Actor prompt + skill files become
the rule registry in `hybrid_retriever/verify.py`, versioned in
git; the adoption gate becomes the leave-one-out ablation (a
rule memory is adopted only if removing it from the recalled set
lowers task success outside noise).

Tasks reuse the P19 artifact shapes (JSON list with a duplicated
item, log tail with repeated heartbeats, table with a duplicated
row) and the P1 synthetic corpus (160 facts) as background
memories. The task wrappers are new — neither fixture ships a
verifiable *doing* task — but every artifact shape and every
background memory is reused. Each task asks for a computed
answer (sum of scores, count of distinct messages); every task
is a trap for the naive solver, and the handling is stated in
exactly one rule memory in the store. The store also holds six
distractor rule memories that prescribe the *wrong* handling
and share more vocabulary with the question than the correct
rule does — a lexical trap, by design. Reproduce with
`hr eval --vbd --tasks 24`. Embedder: `onnx-all-MiniLM-L6-v2`.

### Measured results — VbD (synthetic, labelled)

24 tasks (8 JSON + 8 log + 8 table), store 169 records (160
background facts + 9 rule memories), k=5 retrieved memories per
task. Lift is paired McNemar vs the no-memory baseline on the
same tasks (lift ± 95% CI); "outside" means |lift| > 2 SE.

| arm | success | rate | lift vs no-memory |
|---|---|---|---|
| no-memory (baseline) | 0/24 | 0.000 | — |
| bm25 | 8/24 | 0.333 | +0.333 ± 0.231 (outside noise) |
| vector | 8/24 | 0.333 | +0.333 ± 0.231 (outside noise) |
| hybrid | 8/24 | 0.333 | +0.333 ± 0.231 (outside noise) |
| oracle (correct rule handed directly) | 24/24 | 1.000 | +1.000 ± 0.400 (outside noise) |

Per-kind success rate: log 1.000 on every retrieval arm;
JSON 0.000 and table 0.000 on every retrieval arm; oracle
1.000 on all three kinds. k=10 gives identical results.

Adoption gate (hybrid minus one correct rule memory):
`log-distinct-only` 8/24 → 0/24, causal lift +0.333 ± 0.231
(adopt); `json-dedup-by-id` and `table-dedup-rows` 8/24 → 8/24,
lift +0.000 (reject — the memory never fires, so removing it
changes nothing).

The honest reading: the harness works — the baseline is 0/24,
the ceiling is 24/24, and the retrieval arms beat the baseline
outside noise — but the win comes entirely from the log kind,
and the reason is instructive. The correct JSON and table rules
*are* in the top 5 in most leg/kind combinations (vector ranks
the JSON rule 3rd, all three legs rank the table rule 3rd);
they lose because the solver applies the *first* firing rule
and a distractor outranks them in every leg (JSON: "include
every item" ranks 1st–2nd; table: "sum only the first ten
rows" ranks 1st). Similarity ranks by topic, not by the
correctness of the prescribed action — a wrong rule about the
same sum is exactly as similar as the right rule, and the
fused RRF scores across the whole top 5 sit within 0.001 of
each other, so the ranking among the rule memories is nearly
a coin flip. The log kind passes only because its correct rule
happens to share the question's exact words ("distinct",
"messages") and ranks 1st. So on this task set, retrieval
lifts success from 0 to 1/3, and the remaining 2/3 is not a
retrieval-recall problem — it is a rule-adjudication problem:
given two contradictory memories, similarity cannot tell
which one to follow. Closing it needs the feedback the
paper's co-evolution describes — the verifier's verdicts
feeding back into which memories get used (P2 TRACE credit
is the mechanism this repo already has), not a better
retriever. Publish the harness, not the headline: a memory
that never fires has zero causal value no matter how
retrievable it is, and the adoption gate is what catches
that — the two rejected rules look identical to the adopted
one in every retrieval metric.

### Regressions (re-run after P20)

- Eviction eval (P18 fixed target): CWL task hit@5 1.000,
  user-turn recall 1.000, 40/40 user turns, 0 dependency
  violations, at budget 1076 — identical.
- Segment eval: identical to P3 — boundaries semantic F1 0.949,
  fixed-window 0.292; consolidation 40 semantic ops; semantic
  any-hit@5 0.958 / 1.000 / 0.750.
- TRACE split: identical to P2 (vector 0.875, hybrid 0.589,
  hybrid+trace w=0.5 0.571 at ucb_c=1.0).
- LongMemEval-S 50Q: identical to P1/P2/P3 (bm25 R@10 1.000,
  vector 0.860, hybrid 0.980, hybrid+mmr 0.860).

## What P20 hooks into (P21)

- `run_vbd_eval` is the task-success harness P21's
  mini-TOBench builds on: swap the deterministic solver for a
  second agent and the computed-answer tasks for tool-use tasks,
  and the verifier / lift / adoption-gate machinery carries over
  unchanged.
- The adoption gate's verdicts are the input P2's TRACE credit
  would consume: a rule memory whose causal lift is zero should
  lose importance, and one whose lift is positive and outside
  noise should gain it — the VbD loop is what makes TRACE
  credit causal instead of correlational.

## P21 — Mini-TOBench: verifier-first evals for a workspace-assistant domain

TOBench (arXiv:2605.16909) scores agents on executable tasks over
real tool surfaces, with programmatic verifiers that inspect
artifacts and tool-call traces instead of judging final text. This
module (`hybrid_retriever/tobench.py`) is the small deterministic
version of that idea, built to be the harness P20's VbD loop and
P23's self-improving loop plug into.

Three servers, one domain (a workspace assistant over the hybrid
store), all real tool surfaces the agents only touch through
`call_tool`:

- `memory` — recall / remember (with supersedes) / forget / list,
  backed by the real `Store` (SQLite) from P1.
- `files` — read / write / append / list, backed by a real
  directory; paths cannot escape the workspace root.
- `tracker` — create / complete / add-note / list, backed by a
  real `tasks.json` in the workspace.

Each server also has a thin stdio JSON-RPC 2.0 wrapper
(`python -m hybrid_retriever.tobench serve --server files --root DIR`,
methods `call_tool` / `list_tools`), so they can be driven over a
pipe like a real MCP server.

12 tasks, 4 kinds x 3 variants, each with a grounded verifier that
checks the artifact (recomputed from the environment, not trusted
from the agent), the tool-call trace (ordering, required calls,
forbidden tools, call budget), and the store state. Each failed
check carries a detail string — expected vs got, which call was
missing, which order was violated — so a failure is diagnosable
from the verifier output alone. Reproduce with
`hr eval --tobench` (no embedder needed).

Substitutions, stated plainly: there is no LLM on this VM and no
model API key exists, so the "two models" are two DETERMINISTIC
agent policies, not models — `grounded` (recall first, read
sources, compute from the real artifacts with dedup, tracker work
in the right order) and `sloppy` (no recall, naive compute without
dedup, tracker tasks completed before the deliverable exists,
corrections stored without superseding). No model names or scores
are invented. The servers run in-process, so there is no
transport, auth, or concurrency to test. Memory recall is BM25
only (no embedder); the vector leg is exercised by the P1/P20
evals. Tasks are a fixed 12-task set (seed 7), not a sample from
a task distribution.

### Measured results — mini-TOBench (synthetic, labelled)

| agent | total | summarize | correct | compute | deliver |
|---|---|---|---|---|---|
| grounded | 12/12 (1.000) | 3/3 | 3/3 | 3/3 | 3/3 |
| sloppy | 1/12 (0.083) | 1/3 | 0/3 | 0/3 | 0/3 |

Sloppy's single pass is S3, the one prompt-contained task (the
facts are in the task text, no recall needed). Every other sloppy
failure names its cause: S1/S2 write the distractor ("tea",
"March 3") instead of the recalled facts and never call recall;
C1–C3 leave the wrong record live (`old record #1 status=live,
superseded_by=None`); M1 writes 375.25 where the deduped total
is 254.75, M2 writes 46.00 vs 41.50, M3 counts 5 rows where
3 are distinct; D1–D3 complete the tracker task at trace 0,
before the deliverable write at trace 1, and never call recall.

The honest reading: the harness discriminates — 12/12 vs 1/12
with a per-task table and a named cause per failure — but the
scores say nothing about any LLM. Both policies were written
with knowledge of the task structure, so `grounded` passing
everything is the harness working as designed, not evidence
that a model would. What the scores do prove is narrower and
is the point of the build: a task's failure is diagnosable from
the verifier output alone (the M1 detail alone — "expected
254.75 (deduped from expenses.json), got 375.25" — tells you
the agent summed without dedup), and the interesting failures
are wrong tool use, not wrong text. The ceiling here is the
policy, not the model: drop a real model policy into `AGENTS`
and the same tasks and verifiers score it unchanged. The
reference frame is the full TOBench: best model 41.0%, human
94.0% — this 12-task set is far easier by construction, so
do not compare the 1.000 to either number.

### Regressions (re-run after P21)

- TRACE split: identical to P2 (vector 0.875, hybrid 0.589,
  hybrid+trace w=0.5 0.571 at ucb_c=1.0).
- Segment eval: identical to P3 — boundaries semantic F1 0.949,
  fixed-window 0.292; consolidation 40 semantic ops; semantic
  any-hit@5 0.958 / 1.000 / 0.750.
- Eviction eval (P18 fixed target): CWL task hit@5 1.000,
  user-turn recall 1.000, 40/40 user turns, 0 dependency
  violations, at budget 1076 — identical.
- Compression eval: identical to P19 (hit@5 0.350, survival
  mean-over-time 0.681/0.682, 29,514 tokens saved, faithfulness
  0.486, 0 violations).
- LongMemEval-S 50Q: identical to P1/P2/P3 (bm25 R@10 1.000,
  vector 0.860, hybrid 0.980, hybrid+mmr 0.860).

## What P21 hooks into (P22/P23)

- P23's GEPA-style loop A/B-tests prompt/skill edits on this
  fixed task set: the per-task table is the metric, and an edit
  is kept only if it wins on it.
- P20's adoption gate gains a second consumer: a rule memory
  whose lift the VbD harness measured can be scored here too,
  as the policy the agent follows when the memory is recalled.

## P22 — Durable agent loop on real DBOS (SQLite system DB)

DBOS (the durable-execution library: annotate a workflow and its
steps, and a crash mid-run resumes from the journal instead of
restarting) wraps the P21 grounded policy in
`hybrid_retriever/durable.py`: one DBOS step per tool call, each
step's output checkpointed in the system database, and a fresh
process auto-recovers the PENDING workflow on launch. The
human-approval gate is a file flag (`approval.txt`) polled with
durable `DBOS.sleep` — completed sleeps are skipped on recovery,
so the pause survives a restart.

Substitutions, stated plainly: there is no Postgres on this VM
(no server, no docker, no psql), and DBOS 3.6.0's DEFAULT system
database is SQLite (`system_database_url` defaults to
`sqlite:///{name}`), so this runs REAL DBOS — same
`@DBOS.workflow` / `@DBOS.step` decorators, same journal tables
(`workflow_status`, `operation_outputs`), same recovery path —
with the system database in a SQLite file instead of Postgres.
That is the configuration DBOS itself ships as the default, not
a reimplementation; Postgres is the production default and the
only change needed to move there is the URL. There is no LLM:
the loop's work is the deterministic P21 grounded policy,
decomposed into one step per tool call. Steps carry an
artificial `step_delay` (default 0) before their side effect,
ONLY so the kill demo/tests can land the SIGKILL mid-run on a
workload whose real work takes milliseconds.

The guarantee is not "DBOS makes it exactly-once" — it does
not. DBOS steps are at-least-once: a step killed after its side
effect but before its output is checkpointed re-executes on
recovery (verified in the smoke test — a step killed mid-write
ran twice in the raw log). So every step body here is
IDEMPOTENT — remember/create/complete check before they act,
file writes overwrite — and the seed step is idempotent too
(it reuses the same check-then-act helpers, plus a `seed.json`
marker). The durable trace (`trace.jsonl`, one line per executed
step keyed by DBOS step id) dedupes by step id on rebuild, so a
re-executed step leaves exactly one trace entry. The
no-duplicated-side-effects claim is proved by the demo's
assertions (live-record counts, tracker state, trace step ids,
artifact bytes vs a plain P21 grounded run), not assumed from
the engine.

### Measured results — kill and resume (D1, real run)

`hr durable demo --task D1 --step-delay 1.0 --kill-after-steps 2`
(SIGKILL after 2 trace lines, fresh process recovers):

- At kill: workflow PENDING, 2 trace lines (recall + deliverable
  write) with 2–3 steps checkpointed in `operation_outputs` —
  the exact count depends on where the kill lands relative to
  the in-flight step's return: a step killed after appending
  its trace line but before returning re-executes on recovery
  (idempotent overwrite, trace line deduped by step id).
- Recovery log: `Recovering 1 workflows from application version
  142dddcb1b63f114356bda847a897d46`, then `STATUS: SUCCESS`,
  `RESULT: {'task_id': 'D1', 'result': 'release 1.4 ships
  semantic recall\nrelease 1.4 fixes the torn-tail bug'}`.
- P21 verifier on the rebuilt bundle (store + durable trace):
  all checks pass. Exactly-once assertions: zero violations —
  each seeded memory live exactly once, the tracker task
  completed exactly once, the rebuilt trace holds one event per
  distinct step id, and both artifacts are byte-identical to a
  plain P21 grounded run of the same task.

### Measured results — approval pause survives a restart (D1)

`hr durable approval-demo --task D1`: the run writes the
deliverable, pauses at the gate (tracker task still open), is
SIGKILLed mid-pause, and a fresh DBOS instance recovers it —
still PENDING, tracker still open after several sleep cycles —
then the operator creates `approval.txt` and the run completes
with `STATUS: SUCCESS`, verifier passing, zero violations.

### Regressions (re-run after P22)

- TRACE split: identical to P2 (vector 0.875, hybrid 0.589,
  hybrid+trace w=0.5 0.571 at ucb_c=1.0).
- Segment eval: identical to P3 — boundaries semantic F1 0.949,
  fixed-window 0.292; consolidation 40 semantic ops; semantic
  any-hit@5 0.958 / 1.000 / 0.750.
- Eviction eval (P18 fixed target): CWL task hit@5 1.000,
  user-turn recall 1.000, 40/40 user turns, 0 dependency
  violations, at budget 1076 — identical.
- Compression eval: identical to P19 (hit@5 0.350, survival
  mean-over-time 0.681/0.682, 82 compressions, 29,514 tokens
  saved, faithfulness 0.486, 0 violations). Latency this run
  1.74 ms mean vs the README's 0.79 ms — load-dependent, same
  as P20/P21 noted; the functional numbers are identical.
- LongMemEval-S 50Q: identical to P1/P2/P3 (bm25 R@10 1.000,
  vector 0.860, hybrid 0.980, hybrid+mmr 0.860).

### The honest reading

What resume guarantees here: a workflow killed after at least
one step output is checkpointed resumes from the journal in a
fresh process, completed steps are not re-executed, and the
side effects of the run happen exactly once — because the step
bodies are idempotent, not because the engine promises it.
What it does not guarantee: a step that is not idempotent
duplicates its side effect when it re-executes in the crash
window (the smoke test caught this: a step killed after its
write but before its checkpoint ran twice); a side effect
performed outside a step — in the workflow body, or by an
external process — is invisible to the journal and is neither
replayed nor deduped; a crash inside a step's own write (e.g.
half a file flushed) leaves the partial write, and only the
step's idempotency (overwrite, check-then-act) repairs it on
re-execution; and a kill before the first step output is
checkpointed leaves the workflow ENQUEUED, which
`recover_workflows` does not pick up (it only recovers
PENDING) — the run is orphaned with zero side effects, which
the test suite pins as the observed edge, not a success. The
approval gate's `DBOS.sleep` loop records one sleep operation
per poll cycle, so a pause of hours would bloat the journal;
for that, DBOS `recv`/`send` events are the right primitive
and the file flag is the demo's inspectable stand-in. None of
this is a reason to hand-roll a journal instead: the 2026
consensus in the plan (use the engine, don't build your own
retry/checkpoint logic) held — the engine did the hard part
(checkpointing, recovery, sleep replay) and the application
did the small part (idempotent steps).

## What P22 hooks into (P23)

- P23's GEPA-style loop is the first consumer that needs this:
  a nightly diagnoser that proposes an edit, A/B-tests it, and
  keeps it only if it wins must survive a crash mid-A/B without
  double-counting a task or losing the verdict — that is a
  durable workflow over the P21 harness, and this module is the
  pattern for it.
- Any agent loop built on the P21 servers (memory/files/tracker)
  can be wrapped the same way: one step per tool call, idempotent
  step bodies, the durable trace for verification.
