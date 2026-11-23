"""Retrieval: BM25 + vector + graph legs fused by weighted RRF.

The DocuSearch / QUANTA pattern (report section 1.6):
- BM25 leg (weight 0.50), vector leg (0.35), graph neighbour-expansion
  leg (0.15), fused by reciprocal rank fusion with k=60.
- Optional cross-encoder rerank on the fused top-N.
- Optional MMR diversity (lambda=0.65) on the final list.

RRF score for a document d: sum over legs of
``weight_leg / (k_rrf + rank_leg(d))`` with 1-based ranks. A leg that
returns nothing contributes nothing; weights are *not* renormalised,
so a missing leg can only lower scores, never invent them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .store import Store

DEFAULT_WEIGHTS = (0.50, 0.35, 0.15)  # bm25, vector, graph
DEFAULT_K_RRF = 60
DEFAULT_MMR_LAMBDA = 0.65


@dataclass
class Retrieved:
    id: int
    text: str
    score: float
    legs: set = field(default_factory=set)


def rrf_fuse(
    ranked_lists: list[list[int]],
    weights: list[float] | tuple[float, ...],
    k_rrf: int = DEFAULT_K_RRF,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Weighted reciprocal rank fusion.

    ``ranked_lists`` is one id list per leg, best-first. Returns
    [(id, fused_score)] sorted by score descending.
    """
    scores: dict[int, float] = {}
    for leg, (ids, w) in enumerate(zip(ranked_lists, weights)):
        if w == 0:
            continue
        for rank, rid in enumerate(ids, start=1):
            scores[rid] = scores.get(rid, 0.0) + w / (k_rrf + rank)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return ranked[:top_n] if top_n is not None else ranked


def mmr(
    candidates: list[tuple[int, float]],
    query_vec: np.ndarray,
    doc_vecs: dict[int, np.ndarray],
    lambda_: float = DEFAULT_MMR_LAMBDA,
    top_n: int = 10,
) -> list[int]:
    """Maximal marginal relevance.

    Iteratively picks the candidate maximising
    ``lambda * sim(query, doc) - (1 - lambda) * max sim(doc, picked)``.
    ``candidates`` is [(id, base_score)] in fused order; the base score
    is used only as a tiebreak — the diversity term uses vectors.
    Candidates without a vector are kept in fused order at the end.
    """
    if not candidates:
        return []
    with_vec = [c for c in candidates if c[0] in doc_vecs]
    without_vec = [c[0] for c in candidates if c[0] not in doc_vecs]
    if not with_vec:
        return [c[0] for c in candidates[:top_n]] + without_vec[: max(0, top_n - len(candidates[:top_n]))]
    picked: list[int] = []
    remaining = list(with_vec)
    while remaining and len(picked) < top_n:
        best_id, best_val = None, None
        for rid, base in remaining:
            qsim = float(doc_vecs[rid] @ query_vec)
            if picked:
                dsim = max(float(doc_vecs[rid] @ doc_vecs[p]) for p in picked)
            else:
                dsim = 0.0
            val = lambda_ * qsim - (1.0 - lambda_) * dsim
            if best_val is None or val > best_val:
                best_id, best_val = rid, val
        picked.append(best_id)  # type: ignore[arg-type]
        remaining = [c for c in remaining if c[0] != best_id]
    # Any vector-less candidates fill remaining slots in fused order.
    picked.extend(without_vec[: max(0, top_n - len(picked))])
    return picked[:top_n]


def retrieve(
    store: Store,
    embedder,
    query: str,
    k: int = 10,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    k_rrf: int = DEFAULT_K_RRF,
    leg_k: int = 50,
    use_mmr: bool = True,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    reranker=None,
    rerank_n: int = 20,
    candidate_ids: list[int] | None = None,
) -> list[Retrieved]:
    """Run the three legs, fuse with weighted RRF, optionally rerank + MMR.

    ``candidate_ids`` restricts all legs to a subset (used by the
    LongMemEval harness, where each question has its own haystack).
    Returns at most ``k`` records, best first.
    """
    bm25_ids = [rid for rid, _ in store.bm25_search(query, leg_k, candidate_ids)]
    vec_ids: list[int] = []
    query_vec = None
    if embedder is not None:
        query_vec = embedder.embed([query])[0]
        # A zero query vector (no embeddable tokens) makes cosine
        # undefined; the vector leg is meaningless for such a query.
        if float(np.linalg.norm(query_vec)) > 0:
            vec_ids = [
                rid
                for rid, _ in store.vector_search(query_vec, leg_k, candidate_ids)
            ]
        else:
            query_vec = None
    seed_ids = bm25_ids[:10] + vec_ids[:10]
    graph_ids = [rid for rid, _ in store.neighbours(seed_ids, leg_k)]

    fused = rrf_fuse([bm25_ids, vec_ids, graph_ids], weights, k_rrf)
    leg_map: dict[int, set] = {}
    for leg_name, ids in (
        ("bm25", bm25_ids),
        ("vector", vec_ids),
        ("graph", graph_ids),
    ):
        for rid in ids:
            leg_map.setdefault(rid, set()).add(leg_name)

    if reranker is not None and fused:
        top = fused[:rerank_n]
        texts = [store.get_record(rid)["text"] for rid, _ in top]
        new_order = reranker.rerank(query, [rid for rid, _ in top], texts)
        # reranker returns ids in new order; keep fused scores for the rest
        score_of = dict(fused)
        fused = [(rid, score_of[rid]) for rid in new_order] + [
            (rid, s) for rid, s in fused[rerank_n:]
        ]

    if use_mmr and query_vec is not None and fused:
        doc_vecs = {}
        for rid, _ in fused[: max(k * 3, 20)]:
            v = store.get_vector(rid)
            if v is not None:
                doc_vecs[rid] = v
        ordered = mmr(fused, query_vec, doc_vecs, mmr_lambda, top_n=k)
    else:
        ordered = [rid for rid, _ in fused[:k]]

    score_of = dict(fused)
    out = []
    for rid in ordered:
        rec = store.get_record(rid)
        if rec is None:
            continue
        out.append(
            Retrieved(
                id=rid,
                text=rec["text"],
                score=score_of.get(rid, 0.0),
                legs=leg_map.get(rid, set()),
            )
        )
    return out
