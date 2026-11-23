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

## What P18 hooks into (P19)

- `evict_to_budget` runs on any session store, so P19's
  type-aware compressor can sit *inside* the eviction loop: when
  the budget is exceeded, compress a tool_result episode's JSON
  first (P19) and only evict it if it still does not fit.
- `context_record_ids` is the retrieval surface P19's A/B
  harness should score against — compressed vs raw recall over
  the same live context.
- The `evictions` audit table is the measurement log for P19:
  tokens freed per eviction vs tokens saved per compression, on
  the same sessions.
- `eval.py` — `run_eviction_eval` is the fixed target P19 must
  not regress: CWL task hit@5 1.000, user-turn recall 1.000,
  40/40 user turns, 0 dependency violations, at budget 1076.
