"""Eval harness: P@10 / R@10 / nDCG@10 / any-hit@5 / grounding@10.

Two corpora:
- LongMemEval-S (real): ``xiaowu0162/longmemeval-cleaned``,
  ``longmemeval_s_cleaned.json``. Each question has a haystack of
  sessions and gold ``answer_session_ids``. Abstention questions
  (``_abs`` suffix) are excluded from retrieval scoring, as in the
  official harness. Records are sessions; retrieval is restricted to
  each question's haystack via ``candidate_ids``.
- Synthetic (labelled as such): ``eval_data/synthetic_corpus.jsonl``
  + ``eval_data/synthetic_queries.json``. Same metrics, same code path.

"Grounding@10" here means the share of queries with at least one gold
record in the top 10 — the retriever surfaced evidence the answer can
be grounded in. It is the same number as any-hit@10, reported under
both names so the DocuSearch comparison is explicit.

Modes compared: bm25, vector, hybrid (RRF), hybrid+mmr,
hybrid+rerank (only when a reranker is supplied).
"""

from __future__ import annotations

import json
import math
import os
import tempfile

from .embeddings import get_embedder
from .lifecycle import record_use
from .retrieval import retrieve
from .store import Store

MODES = ("bm25", "vector", "hybrid", "hybrid+mmr", "hybrid+rerank")


def precision_at_k(ranked: list[int], gold: set[int], k: int = 10) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in gold) / len(top)


def recall_at_k(ranked: list[int], gold: set[int], k: int = 10) -> float:
    if not gold:
        return 0.0
    return sum(1 for r in ranked[:k] if r in gold) / len(gold)


def ndcg_at_k(ranked: list[int], gold: set[int], k: int = 10) -> float:
    dcg = 0.0
    for i, r in enumerate(ranked[:k], start=1):
        if r in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def any_hit_at_k(ranked: list[int], gold: set[int], k: int = 10) -> float:
    return 1.0 if any(r in gold for r in ranked[:k]) else 0.0


def score_modes(
    ranked_by_mode: dict[str, list[int]], gold: set[int]
) -> dict[str, dict[str, float]]:
    out = {}
    for mode, ranked in ranked_by_mode.items():
        out[mode] = {
            "P@10": precision_at_k(ranked, gold),
            "R@10": recall_at_k(ranked, gold),
            "nDCG@10": ndcg_at_k(ranked, gold),
            "any-hit@5": any_hit_at_k(ranked, gold, 5),
            "grounding@10": any_hit_at_k(ranked, gold, 10),
        }
    return out


def average_scores(
    per_question: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    modes = per_question[0].keys() if per_question else []
    metrics = ["P@10", "R@10", "nDCG@10", "any-hit@5", "grounding@10"]
    return {
        mode: {
            m: sum(q[mode][m] for q in per_question) / len(per_question)
            for m in metrics
        }
        for mode in modes
    }


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def flatten_session(session: list[dict]) -> str:
    return "\n".join(
        "%s: %s" % (turn.get("role", "?"), turn.get("content", ""))
        for turn in session
    )


def load_longmemeval(path: str, limit: int | None = None) -> list[dict]:
    """Return [{qid, question, haystack: [(session_id, text)], gold: set}].

    Abstention questions are skipped (official harness does the same).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    questions = []
    skipped_abs = 0
    for item in raw:
        qid = item.get("question_id", "")
        if qid.endswith("_abs"):
            skipped_abs += 1
            continue
        haystack = [
            (sid, flatten_session(sess))
            for sid, sess in zip(
                item["haystack_session_ids"], item["haystack_sessions"]
            )
        ]
        questions.append(
            {
                "qid": qid,
                "question": item["question"],
                "haystack": haystack,
                "gold": set(item["answer_session_ids"]),
            }
        )
        if limit is not None and len(questions) >= limit:
            break
    return questions


def load_synthetic(corpus_path: str, queries_path: str) -> tuple[list[dict], list[dict]]:
    with open(corpus_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)
    return records, queries


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def _retrieve_mode(
    store, embedder, question, mode, candidate_ids, reranker,
    importance_weight: float = 0.0,
):
    if mode == "bm25":
        res = retrieve(
            store, None, question, k=10, weights=(1.0, 0.0, 0.0),
            use_mmr=False, candidate_ids=candidate_ids,
        )
    elif mode == "vector":
        res = retrieve(
            store, embedder, question, k=10, weights=(0.0, 1.0, 0.0),
            use_mmr=False, candidate_ids=candidate_ids,
        )
    elif mode == "hybrid":
        res = retrieve(
            store, embedder, question, k=10, use_mmr=False,
            candidate_ids=candidate_ids,
            importance_weight=importance_weight,
        )
    elif mode == "hybrid+mmr":
        res = retrieve(
            store, embedder, question, k=10, use_mmr=True,
            candidate_ids=candidate_ids,
            importance_weight=importance_weight,
        )
    elif mode == "hybrid+rerank":
        res = retrieve(
            store, embedder, question, k=10, use_mmr=False,
            reranker=reranker, candidate_ids=candidate_ids,
        )
    else:
        raise ValueError("unknown mode %r" % mode)
    return [r.id for r in res]


def run_synthetic_trace(
    corpus_path: str,
    queries_path: str,
    embedder=None,
    importance_weights: tuple[float, ...] = (0.5, 1.0, 2.0),
    verbose: bool = True,
) -> dict:
    """TRACE eval on the synthetic corpus, with a train/test split.

    Warm-up on the 48 keyword queries (credit attributed against the
    gold records' text as the answer proxy — the simulated agent
    answer grounded in gold evidence), eval on the 48 paraphrase + 8
    trap queries. The split tests whether credit learned from one
    phrasing transfers to another phrasing of the same facts.
    Warming up on the eval queries themselves would be training on
    the test set, so it is not done. Headline metric: evidence
    recall@5 = any-hit@5, the MEMOIR paper's metric, vs bare cosine
    (vector-only) and plain hybrid.
    """
    embedder = embedder or get_embedder()
    records, queries = load_synthetic(corpus_path, queries_path)
    tmp = tempfile.mkdtemp(prefix="hr-eval-")
    store = Store(os.path.join(tmp, "eval.db"), embedder=embedder)
    texts = [r["text"] for r in records]
    rids = store.add_texts(texts, type_="fact", embedder=embedder)
    rid_of_key = {r["key"]: rid for r, rid in zip(records, rids)}
    by_group: dict[str, list[int]] = {}
    for r, rid in zip(records, rids):
        by_group.setdefault(r.get("group", ""), []).append(rid)
    edges = []
    for group_rids in by_group.values():
        for a, b in zip(group_rids, group_rids[1:]):
            edges.append((a, b, 1.0, "related"))
            edges.append((b, a, 1.0, "related"))
    store.add_edges(edges)
    warmup = [q for q in queries if q["kind"] == "keyword"]
    heldout = [q for q in queries if q["kind"] != "keyword"]
    if verbose:
        print(
            "indexed %d synthetic records (vec_mode=%s); warm-up on %d "
            "keyword queries, eval on %d paraphrase+trap queries"
            % (store.count(), store.vec_mode, len(warmup), len(heldout))
        )

    def score(questions, mode, importance_weight=0.0):
        metrics = ["P@10", "R@10", "nDCG@10", "any-hit@5", "grounding@10"]
        per = []
        for q in questions:
            gold = {rid_of_key[k] for k in q["gold_keys"] if k in rid_of_key}
            ranked = _retrieve_mode(
                store, embedder, q["question"], mode, None, None,
                importance_weight,
            )
            per.append(score_modes({mode: ranked}, gold)[mode])
        return {
            m: sum(p[m] for p in per) / len(per) for m in metrics
        }

    scores = {
        "vector (bare cosine)": score(heldout, "vector"),
        "hybrid": score(heldout, "hybrid"),
    }
    for q in warmup:
        gold = {rid_of_key[k] for k in q["gold_keys"] if k in rid_of_key}
        res = retrieve(store, embedder, q["question"], k=10, use_mmr=False)
        answer = "\n".join(
            store.get_record(g)["text"] for g in sorted(gold)
        )
        record_use(store, res, answer)
    if verbose:
        print("warm-up done (TRACE credit recorded)")
    for w in importance_weights:
        scores["hybrid+trace w=%.1f" % w] = score(
            heldout, "hybrid", importance_weight=w
        )
    store.close()
    return {
        "corpus": "synthetic (labelled) — TRACE split",
        "n_questions": len(heldout),
        "n_warmup": len(warmup),
        "n_records": len(records),
        "embedder": embedder.name,
        "scores": scores,
    }


def print_report(result: dict) -> None:
    print("\n== %s ==" % result["corpus"])
    print(
        "questions: %d   records: %d   embedder: %s"
        % (result["n_questions"], result["n_records"], result["embedder"])
    )
    if "n_warmup" in result:
        print("warm-up queries: %d" % result["n_warmup"])
    print(
        "%-22s %6s %6s %8s %10s %12s"
        % ("mode", "P@10", "R@10", "nDCG@10", "any-hit@5", "grounding@10")
    )
    for mode, m in result["scores"].items():
        print(
            "%-22s %6.3f %6.3f %8.3f %10.3f %12.3f"
            % (mode, m["P@10"], m["R@10"], m["nDCG@10"], m["any-hit@5"], m["grounding@10"])
        )


def run_longmemeval(
    path: str,
    limit: int = 50,
    embedder=None,
    reranker=None,
    modes: tuple[str, ...] = MODES,
    verbose: bool = True,
) -> dict:
    """Build one store from the unique sessions of the sampled questions.

    Returns {"n_questions": int, "n_records": int, "scores": {mode: metrics}}.
    """
    embedder = embedder or get_embedder()
    questions = load_longmemeval(path, limit=limit)
    if verbose:
        print("loaded %d questions from %s" % (len(questions), path))

    # Unique sessions across the sampled questions.
    sessions: dict[str, str] = {}
    for q in questions:
        for sid, text in q["haystack"]:
            sessions.setdefault(sid, text)
    if verbose:
        print("indexing %d unique sessions ..." % len(sessions))

    tmp = tempfile.mkdtemp(prefix="hr-eval-")
    db_path = os.path.join(tmp, "eval.db")
    store = Store(db_path, embedder=embedder)
    sid_list = list(sessions.keys())
    texts = [sessions[s] for s in sid_list]
    record_ids = store.add_texts(texts, type_="episode", embedder=embedder)
    rid_of_sid = dict(zip(sid_list, record_ids))
    if verbose:
        print("indexed %d records (vec_mode=%s)" % (store.count(), store.vec_mode))

    per_question = []
    for i, q in enumerate(questions):
        cand = [rid_of_sid[s] for s in dict(q["haystack"]).keys()]
        gold = {rid_of_sid[s] for s in q["gold"] if s in rid_of_sid}
        ranked = {}
        for mode in modes:
            if mode == "hybrid+rerank" and reranker is None:
                continue
            ranked[mode] = _retrieve_mode(
                store, embedder, q["question"], mode, cand, reranker
            )
        per_question.append(score_modes(ranked, gold))
        if verbose and (i + 1) % 10 == 0:
            print("  ... %d/%d questions" % (i + 1, len(questions)))
    store.close()
    return {
        "corpus": "LongMemEval-S (real)",
        "n_questions": len(questions),
        "n_records": len(sessions),
        "embedder": embedder.name,
        "scores": average_scores(per_question),
    }


def run_synthetic(
    corpus_path: str,
    queries_path: str,
    embedder=None,
    reranker=None,
    modes: tuple[str, ...] = MODES,
    verbose: bool = True,
) -> dict:
    embedder = embedder or get_embedder()
    records, queries = load_synthetic(corpus_path, queries_path)
    tmp = tempfile.mkdtemp(prefix="hr-eval-")
    store = Store(os.path.join(tmp, "eval.db"), embedder=embedder)
    texts = [r["text"] for r in records]
    rids = store.add_texts(texts, type_="fact", embedder=embedder)
    rid_of_key = {r["key"]: rid for r, rid in zip(records, rids)}
    # Chain same-group records so the graph leg has something to expand.
    by_group: dict[str, list[int]] = {}
    for r, rid in zip(records, rids):
        by_group.setdefault(r.get("group", ""), []).append(rid)
    edges = []
    for group_rids in by_group.values():
        for a, b in zip(group_rids, group_rids[1:]):
            edges.append((a, b, 1.0, "related"))
            edges.append((b, a, 1.0, "related"))
    store.add_edges(edges)
    if verbose:
        print(
            "indexed %d synthetic records (vec_mode=%s)"
            % (store.count(), store.vec_mode)
        )
    per_question = []
    for q in queries:
        gold = {rid_of_key[k] for k in q["gold_keys"] if k in rid_of_key}
        ranked = {}
        for mode in modes:
            if mode == "hybrid+rerank" and reranker is None:
                continue
            ranked[mode] = _retrieve_mode(
                store, embedder, q["question"], mode, None, reranker
            )
        per_question.append(score_modes(ranked, gold))
    store.close()
    return {
        "corpus": "synthetic (labelled)",
        "n_questions": len(queries),
        "n_records": len(records),
        "embedder": embedder.name,
        "scores": average_scores(per_question),
    }
