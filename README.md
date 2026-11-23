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
  synthetic runners.

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
.venv/bin/python -m pytest tests/ -q   # 27 tests, all passing
```

Covers: store (both vector backends), FTS5 + candidate filtering,
RRF fusion math, MMR diversity, end-to-end hybrid retrieval, graph
expansion, the ONNX embedder's paraphrase gap, and the empty-query
edge case (zero-vector queries return nothing, not the whole store).

## What P2 hooks into

- `Store` — add `importance` / `credit` columns for TRACE-UCB, and a
  `merges` audit table for judge-gated consolidation. The sleep-phase
  job calls `store` directly; no retrieval changes needed.
- `retrieve(..., candidate_ids=...)` — P2's lifecycle needs scoped
  recall (e.g. "only memories from this project"); the filter is
  already there.
- `Retrieved.legs` — P2's credit attribution needs to know which leg
  surfaced each memory; the set is already returned per record.
- `eval.py` — the LongMemEval runner is the fixed target P2 must
  improve on; re-run it after every lifecycle change.
