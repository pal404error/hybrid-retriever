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
    ucb_c: float = 1.0,
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
        record_use(store, res, answer, ucb_c=ucb_c)
    if verbose:
        print("warm-up done (TRACE credit recorded, ucb_c=%.2f)" % ucb_c)
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
        "ucb_c": ucb_c,
        "scores": scores,
    }


def print_report(result: dict) -> None:
    print("\n== %s ==" % result["corpus"])
    print(
        "questions: %d   records: %d   embedder: %s"
        % (result["n_questions"], result["n_records"], result["embedder"])
    )
    if "n_warmup" in result:
        print(
            "warm-up queries: %d   ucb_c: %s"
            % (result["n_warmup"], result.get("ucb_c", "-"))
        )
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


# ----------------------------------------------------------------------
# P3 segment eval
# ----------------------------------------------------------------------
def build_transcripts(
    records: list[dict],
    n_transcripts: int = 6,
    runs_per_transcript: int = 8,
    run_len: tuple[int, int] = (2, 3),
    seed: int = 42,
) -> list[dict]:
    """Turn the synthetic corpus into transcripts with known boundaries.

    A topic run is one attribute (3 records); each run keeps 2–3 of
    them, in original order. Runs are shuffled into transcripts.
    Trap records are excluded — they are distractors for queries,
    not topics. Returns [{"turns": [{"key", "text"}], "boundaries":
    {positions where a new run starts}}]. Every attribute is used
    exactly once, so the transcripts cover the whole corpus.
    """
    import random

    rng = random.Random(seed)
    by_attr: dict[str, list[dict]] = {}
    for r in records:
        if r["key"].startswith("trap-"):
            continue
        by_attr.setdefault(r["key"].rsplit("-", 1)[0], []).append(r)
    runs = []
    for attr in sorted(by_attr):
        recs = by_attr[attr]
        n = rng.randint(*run_len)
        runs.append(rng.sample(recs, min(n, len(recs))))
    rng.shuffle(runs)
    transcripts = []
    for t in range(n_transcripts):
        chunk = runs[
            t * runs_per_transcript : (t + 1) * runs_per_transcript
        ]
        turns: list[dict] = []
        boundaries: set[int] = set()
        for run in chunk:
            if turns:
                boundaries.add(len(turns))
            turns.extend(
                {"key": r["key"], "text": r["text"]} for r in run
            )
        transcripts.append({"turns": turns, "boundaries": boundaries})
    return transcripts


def run_segment_eval(
    corpus_path: str,
    queries_path: str,
    embedder=None,
    threshold: float | None = None,
    window: int | None = None,
    verbose: bool = True,
) -> dict:
    """P3 eval: boundaries, consolidation cost, retrieval after it.

    Threshold tuning: the first half of the transcripts is the dev
    set (sweep tau 0.15–0.60, best mean F1 wins); the second half is
    held out. The fixed-window control uses window = the mean true
    segment length, rounded. Retrieval compares three stores over
    the same 104 queries: raw turns, semantic-consolidated, and
    fixed-window-consolidated. The planner is scored on the
    semantic store: route distribution per query kind, and
    per-kind accuracy vs plain hybrid on the same store.
    """
    from .segments import (
        boundary_f1,
        boundary_positions,
        consolidate_transcript,
        detect_boundaries,
        detect_boundaries_fixed_window,
        retrieve_planned,
    )

    embedder = embedder or get_embedder()
    records, queries = load_synthetic(corpus_path, queries_path)
    transcripts = build_transcripts(records)
    dev, heldout = transcripts[:3], transcripts[3:]

    # --- threshold sweep on dev -------------------------------------
    sweep = {}
    for tau in [round(0.15 + 0.05 * i, 2) for i in range(10)]:
        f1s = []
        for tr in dev:
            texts = [t["text"] for t in tr["turns"]]
            pred = boundary_positions(
                detect_boundaries(texts, embedder, threshold=tau)
            )
            f1s.append(boundary_f1(pred, tr["boundaries"])["f1"])
        sweep[tau] = sum(f1s) / len(f1s)
    best_tau = max(sweep, key=lambda t: sweep[t])
    tau = threshold if threshold is not None else best_tau

    # --- boundaries on held-out --------------------------------------
    def _boundary_scores(transcripts, fn):
        per = []
        for tr in transcripts:
            texts = [t["text"] for t in tr["turns"]]
            pred = boundary_positions(fn(texts))
            per.append(boundary_f1(pred, tr["boundaries"]))
        return {
            m: sum(p[m] for p in per) / len(per)
            for m in ("precision", "recall", "f1")
        }

    mean_seg_len = sum(
        len(tr["turns"]) / (len(tr["boundaries"]) + 1)
        for tr in transcripts
    ) / len(transcripts)
    win = window if window is not None else max(2, round(mean_seg_len))
    sem_scores = _boundary_scores(
        heldout,
        lambda texts: detect_boundaries(texts, embedder, threshold=tau),
    )
    fw_scores = _boundary_scores(
        heldout,
        lambda texts: detect_boundaries_fixed_window(len(texts), win),
    )
    if verbose:
        print(
            "threshold sweep (dev, %d transcripts): best tau=%.2f "
            "F1=%.3f; using tau=%.2f; fixed window=%d (mean true "
            "segment length %.2f)"
            % (len(dev), best_tau, sweep[best_tau], tau, win, mean_seg_len)
        )

    # --- three stores: raw / semantic / fixed-window -----------------
    def _build(kind: str):
        tmp = tempfile.mkdtemp(prefix="hr-seg-")
        store = Store(os.path.join(tmp, "eval.db"), embedder=embedder)
        key_to_rid: dict[str, int] = {}
        summaries = []
        for tr in transcripts:
            turns = [{"text": t["text"]} for t in tr["turns"]]
            if kind == "raw":
                from .segments import ingest_transcript

                ids = ingest_transcript(store, turns)
                summaries.append({"turns_ingested": len(ids)})
            else:
                s = consolidate_transcript(
                    store,
                    embedder,
                    turns,
                    threshold=tau,
                    fixed_window=win if kind == "fixed" else None,
                )
                summaries.append(s)
                # The turns just ingested are the last len(turns)
                # episode records, in order.
                ids = [
                    row[0]
                    for row in store.conn.execute(
                        "SELECT id FROM records WHERE type = 'episode' "
                        "ORDER BY id DESC LIMIT ?",
                        (len(turns),),
                    ).fetchall()
                ][::-1]
            for t, rid in zip(tr["turns"], ids):
                key_to_rid[t["key"]] = rid
        # Gold mapping: a turn key maps to its live record — the
        # segment record that absorbed it, or the turn itself when
        # it was left open / single-turn / raw.
        gold_map: dict[str, set[int]] = {}
        seg_of_turn: dict[int, int] = {}
        for m in store.list_merges(status="committed"):
            for src in m["merged_ids"]:
                seg_of_turn[src] = m["merged_record_id"]
        for k, rid in key_to_rid.items():
            gold_map[k] = {seg_of_turn.get(rid, rid)}
        stats = {
            "turns": sum(s["turns_ingested"] for s in summaries),
            "segments_consolidated": sum(
                s.get("consolidated", 0) for s in summaries
            ),
            "rejected": sum(s.get("rejected", 0) for s in summaries),
            "live_records": store.count_live(),
        }
        return store, gold_map, stats

    def _score_store(store, gold_map, use_planner=False):
        per_kind: dict[str, list[dict]] = {}
        routes: dict[str, dict[str, int]] = {}
        for q in queries:
            gold: set[int] = set()
            for k in q["gold_keys"]:
                gold |= gold_map.get(k, set())
            if use_planner:
                res, plan = retrieve_planned(
                    store, embedder, q["question"], k=10
                )
                routes.setdefault(q["kind"], {}).setdefault(
                    plan.route, 0
                )
                routes[q["kind"]][plan.route] += 1
                ranked = [r.id for r in res]
            else:
                ranked = [
                    r.id
                    for r in retrieve(
                        store, embedder, q["question"], k=10, use_mmr=False
                    )
                ]
            per_kind.setdefault(q["kind"], []).append(
                {
                    "R@10": recall_at_k(ranked, gold),
                    "any-hit@5": any_hit_at_k(ranked, gold, 5),
                }
            )
        scores = {
            kind: {
                m: sum(p[m] for p in per) / len(per)
                for m in ("R@10", "any-hit@5")
            }
            for kind, per in per_kind.items()
        }
        return scores, routes

    built = {kind: _build(kind) for kind in ("raw", "semantic", "fixed")}
    retrieval = {}
    for kind in ("raw", "semantic", "fixed"):
        store, gold_map, _ = built[kind]
        retrieval[kind], _ = _score_store(store, gold_map)
    # Planner on the semantic store, while it is still open.
    sem_store, sem_gold, _ = built["semantic"]
    planner_scores, routes = _score_store(
        sem_store, sem_gold, use_planner=True
    )
    for store, _, _ in built.values():
        store.close()
    stats = {kind: built[kind][2] for kind in built}

    return {
        "corpus": "synthetic transcripts (labelled)",
        "n_transcripts": len(transcripts),
        "n_turns": stats["raw"]["turns"],
        "embedder": embedder.name,
        "threshold": tau,
        "threshold_sweep": {str(k): round(v, 3) for k, v in sweep.items()},
        "fixed_window": win,
        "boundary_semantic": {
            k: round(v, 3) for k, v in sem_scores.items()
        },
        "boundary_fixed_window": {
            k: round(v, 3) for k, v in fw_scores.items()
        },
        "consolidation": {
            "per_turn_ops": stats["raw"]["turns"],
            "semantic_ops": stats["semantic"]["segments_consolidated"],
            "fixed_ops": stats["fixed"]["segments_consolidated"],
            "semantic_live_records": stats["semantic"]["live_records"],
            "fixed_live_records": stats["fixed"]["live_records"],
            "raw_live_records": stats["raw"]["live_records"],
        },
        "retrieval": retrieval,
        "planner": {"scores": planner_scores, "routes": routes},
    }


def print_segment_report(result: dict) -> None:
    print("\n== %s ==" % result["corpus"])
    print(
        "transcripts: %d   turns: %d   embedder: %s   tau=%.2f   "
        "fixed window=%d"
        % (
            result["n_transcripts"],
            result["n_turns"],
            result["embedder"],
            result["threshold"],
            result["fixed_window"],
        )
    )
    print(
        "boundaries (held-out): semantic P=%.3f R=%.3f F1=%.3f | "
        "fixed-window P=%.3f R=%.3f F1=%.3f"
        % (
            result["boundary_semantic"]["precision"],
            result["boundary_semantic"]["recall"],
            result["boundary_semantic"]["f1"],
            result["boundary_fixed_window"]["precision"],
            result["boundary_fixed_window"]["recall"],
            result["boundary_fixed_window"]["f1"],
        )
    )
    c = result["consolidation"]
    print(
        "consolidation ops: per-turn=%d  semantic=%d  fixed-window=%d"
        "  (live records: raw=%d semantic=%d fixed=%d)"
        % (
            c["per_turn_ops"],
            c["semantic_ops"],
            c["fixed_ops"],
            c["raw_live_records"],
            c["semantic_live_records"],
            c["fixed_live_records"],
        )
    )
    for kind in ("raw", "semantic", "fixed"):
        print("retrieval (%s):" % kind)
        for qkind, m in result["retrieval"][kind].items():
            print(
                "  %-10s R@10=%.3f any-hit@5=%.3f"
                % (qkind, m["R@10"], m["any-hit@5"])
            )
    print("planner (on semantic store):")
    for qkind, m in result["planner"]["scores"].items():
        print(
            "  %-10s R@10=%.3f any-hit@5=%.3f  routes=%s"
            % (
                qkind,
                m["R@10"],
                m["any-hit@5"],
                result["planner"]["routes"].get(qkind, {}),
            )
        )


# ----------------------------------------------------------------------
# P18 — CWL eviction demo: one long session vs isolated sessions
# ----------------------------------------------------------------------
def _eviction_task_turns(question: str, gold_texts: list[str]) -> list[dict]:
    """One task as 7 annotated turns.

    user (the question) -> reasoning -> tool_call -> 3 x tool_result
    (the gold facts) -> chatter (ack). The user turn is the
    conversation, not a fact: it is deliberately excluded from the
    per-task consolidation below, so it survives as a live episode
    record and the user-turn-recall metric can see it.
    """
    turns = [
        {"role": "user", "text": question},
        {
            "role": "assistant",
            "text": "Let me look up the records for this question.",
        },
        {
            "role": "assistant",
            "text": "lookup records",
            "episode_type": "tool_call",
        },
    ]
    turns.extend({"role": "tool", "text": t} for t in gold_texts)
    turns.append(
        {"role": "assistant", "text": "Noted.", "episode_type": "chatter"}
    )
    return turns


def run_eviction_eval(
    corpus_path: str,
    queries_path: str,
    embedder=None,
    n_tasks: int = 40,
    budget_frac: float = 1.0 / 3.0,
    verbose: bool = True,
) -> dict:
    """P18 eval: a long session under a token budget, three arms.

    Tasks are synthetic queries (first n_tasks//2 keyword + first
    n_tasks//2 paraphrase, file order — deterministic). Each task
    is 7 turns; after each task the session consolidates the
    task's non-user turns into one segment record (P3 machinery
    via ``propose_merge``), marks the absorbed episodes persisted,
    and evicts to budget. Arms: ``isolated`` (each task alone, no
    budget, no eviction), ``cwl`` (tiered policy), ``naive``
    (oldest-first truncation, same active window).

    Two accuracies, both any-hit@5:
    - task questions, scored over context + memory (live episode
      records + live segment records). Consolidation preserves
      facts, so both long arms are expected near isolated here —
      that is a finding, not a failure.
    - user-turn recall, scored over live episode records only:
      can the agent still find what the user asked? CWL never
      evicts user turns; naive truncation does. This is where the
      policy shows.
    """
    from .eviction import (
        dependency_violations,
        estimate_tokens,
        evict_to_budget,
        ingest_session,
        mark_session_persisted_by_merges,
    )
    from .lifecycle import draft_merge_text, propose_merge
    from .segments import classify_segment

    embedder = embedder or get_embedder()
    records, queries = load_synthetic(corpus_path, queries_path)
    by_key = {r["key"]: r for r in records}
    # Candidate pool in file order (keyword first, then
    # paraphrase); keep the first n_tasks with a distinct gold
    # attribute, so each task's gold set is unambiguous.
    tasks = []
    seen_gold: set[tuple[str, ...]] = set()
    pool = [q for q in queries if q["kind"] == "keyword"] + [
        q for q in queries if q["kind"] == "paraphrase"
    ]
    for q in pool:
        key = tuple(sorted(q["gold_keys"]))
        if key in seen_gold:
            continue
        seen_gold.add(key)
        tasks.append(q)
        if len(tasks) >= n_tasks:
            break
    for q in tasks:
        q["_gold_texts"] = [
            by_key[k]["text"] for k in q["gold_keys"] if k in by_key
        ]
    # Budget from the full session's tokens, before running it.
    total_tokens = sum(
        estimate_tokens(t["text"])
        for q in tasks
        for t in _eviction_task_turns(q["question"], q["_gold_texts"])
    )
    budget = max(1, int(total_tokens * budget_frac))
    if verbose:
        print(
            "eviction eval: %d tasks, %d turns, %d tokens total, "
            "budget %d (%.0f%%)"
            % (
                len(tasks),
                len(tasks) * 7,
                total_tokens,
                budget,
                budget_frac * 100,
            )
        )

    def _consolidate_task(store, episode_ids):
        """Merge a finished task's non-user turns into one segment.

        User turns are excluded: they are the conversation, not
        facts to merge. Returns the merged record id or None.
        """
        eps = [store.get_episode(e) for e in episode_ids]
        srcs = [
            e for e in eps if e is not None and e["type"] != "user"
        ]
        if len(srcs) < 2:
            return None
        ids = [e["record_id"] for e in srcs]
        texts = [store.get_record(i)["text"] for i in ids]
        union = draft_merge_text(texts)
        res = propose_merge(
            store,
            embedder,
            ids,
            merged_text=union,
            type_override=classify_segment(union),
            max_length_ratio=1.0,
        )
        return res.merged_record_id

    def _score(store, question, gold, candidates):
        if not candidates:
            return 0.0
        ranked = [
            r.id
            for r in retrieve(
                store, embedder, question, k=10, use_mmr=False,
                candidate_ids=candidates,
            )
        ]
        return any_hit_at_k(ranked, gold, 5)

    def _run_long(policy: str):
        tmp = tempfile.mkdtemp(prefix="hr-evict-")
        store = Store(os.path.join(tmp, "eval.db"), embedder=embedder)
        session = "long-%s" % policy
        ev_summaries = []
        # Per-task gold, tracked by record id at ingest time —
        # never by text matching: stored turn text carries a
        # "role: " prefix the corpus gold texts do not have.
        task_gold_rids: list[set[int]] = []
        task_user_rids: list[int] = []
        for q in tasks:
            turns = _eviction_task_turns(q["question"], q["_gold_texts"])
            ep_ids = ingest_session(store, session, turns)
            gold: set[int] = set()
            user_rid: int | None = None
            for eid in ep_ids:
                ep = store.get_episode(eid)
                if ep["type"] == "tool_result":
                    gold.add(ep["record_id"])
                elif ep["type"] == "user":
                    user_rid = ep["record_id"]
            task_gold_rids.append(gold)
            task_user_rids.append(user_rid)
            _consolidate_task(store, ep_ids)
            mark_session_persisted_by_merges(store, session)
            ev_summaries.append(
                evict_to_budget(
                    store, session, budget, policy=policy
                )
            )
        # Gold mapping through the merges table, as in the P3 eval:
        # a consolidated turn's gold is the segment record that
        # absorbed it; a rejected merge leaves the turn itself.
        seg_of_turn: dict[int, int] = {}
        for m in store.list_merges(status="committed"):
            for src in m["merged_ids"]:
                seg_of_turn[src] = m["merged_record_id"]
        task_scores: list[float] = []
        recall_scores: list[float] = []
        live_ep_recs = [
            e["record_id"]
            for e in store.list_episodes(session, status="live")
        ]
        for q, gold_rids, user_rid in zip(
            tasks, task_gold_rids, task_user_rids
        ):
            gold = {seg_of_turn.get(r, r) for r in gold_rids}
            task_scores.append(
                _score(
                    store,
                    q["question"],
                    gold,
                    store.context_record_ids(session),
                )
            )
            if user_rid is not None:
                recall_scores.append(
                    _score(
                        store,
                        q["question"],
                        {user_rid},
                        live_ep_recs,
                    )
                )
        live = store.list_episodes(session, status="live")
        user_live = sum(1 for e in live if e["type"] == "user")
        last = ev_summaries[-1] if ev_summaries else {}
        by_tier: dict[int, int] = {}
        for s in ev_summaries:
            for t, n in s.get("evicted_by_tier", {}).items():
                by_tier[t] = by_tier.get(t, 0) + n
        result = {
            "task_any_hit5": (
                sum(task_scores) / len(task_scores) if task_scores else 0.0
            ),
            "user_turn_recall": (
                sum(recall_scores) / len(recall_scores)
                if recall_scores
                else 0.0
            ),
            "user_turns_live": user_live,
            "user_turns_total": len(tasks),
            "live_tokens": store.session_token_total(session),
            "total_tokens": total_tokens,
            "budget": budget,
            "evicted_total": sum(
                s.get("n_evicted", 0) for s in ev_summaries
            ),
            "evicted_by_tier": by_tier,
            "dependency_violations": len(
                dependency_violations(store, session)
            ),
            "under_budget": bool(last.get("under_budget", True)),
        }
        store.close()
        return result

    def _run_isolated():
        scores: list[float] = []
        recall_scores: list[float] = []
        for q in tasks:
            tmp = tempfile.mkdtemp(prefix="hr-iso-")
            store = Store(
                os.path.join(tmp, "eval.db"), embedder=embedder
            )
            session = "isolated"
            turns = _eviction_task_turns(q["question"], q["_gold_texts"])
            ep_ids = ingest_session(store, session, turns)
            gold = {
                store.get_episode(e)["record_id"]
                for e in ep_ids
                if store.get_episode(e)["type"] == "tool_result"
            }
            all_recs = [
                e["record_id"]
                for e in store.list_episodes(session)
            ]
            scores.append(
                _score(store, q["question"], gold, all_recs)
            )
            user_rec = next(
                (
                    store.get_episode(e)["record_id"]
                    for e in ep_ids
                    if store.get_episode(e)["type"] == "user"
                ),
                None,
            )
            if user_rec is not None:
                recall_scores.append(
                    _score(
                        store, q["question"], {user_rec}, all_recs
                    )
                )
            store.close()
        return {
            "task_any_hit5": (
                sum(scores) / len(scores) if scores else 0.0
            ),
            "user_turn_recall": (
                sum(recall_scores) / len(recall_scores)
                if recall_scores
                else 0.0
            ),
        }

    arms = {
        "isolated": _run_isolated(),
        "cwl": _run_long("cwl"),
        "naive": _run_long("naive"),
    }
    return {
        "corpus": "synthetic tasks (labelled)",
        "n_tasks": len(tasks),
        "total_tokens": total_tokens,
        "budget": budget,
        "budget_frac": budget_frac,
        "embedder": embedder.name,
        "arms": arms,
    }


def print_eviction_report(result: dict) -> None:
    print("\n== %s ==" % result["corpus"])
    print(
        "tasks: %d   total tokens: %d   budget: %d (%.0f%%)   "
        "embedder: %s"
        % (
            result["n_tasks"],
            result["total_tokens"],
            result["budget"],
            result["budget_frac"] * 100,
            result["embedder"],
        )
    )
    print(
        "%-10s %12s %16s %14s %12s %10s"
        % (
            "arm",
            "task hit@5",
            "user-turn rec.",
            "user turns",
            "live toks",
            "dep. viol.",
        )
    )
    for arm, m in result["arms"].items():
        user_turns = (
            "%d/%d" % (m.get("user_turns_live", "-"), m.get("user_turns_total", "-"))
            if "user_turns_live" in m
            else "-"
        )
        live_toks = str(m.get("live_tokens", "-"))
        viol = str(m.get("dependency_violations", "-"))
        print(
            "%-10s %12.3f %16.3f %14s %12s %10s"
            % (
                arm,
                m["task_any_hit5"],
                m["user_turn_recall"],
                user_turns,
                live_toks,
                viol,
            )
        )
    for arm in ("cwl", "naive"):
        m = result["arms"][arm]
        print(
            "%s: evicted %d (by tier %s), under budget: %s"
            % (
                arm,
                m.get("evicted_total", 0),
                m.get("evicted_by_tier", {}),
                m.get("under_budget", "-"),
            )
        )


# ----------------------------------------------------------------------
# P19 — compression A/B harness
# ----------------------------------------------------------------------
def _compression_task_turns(
    question: str, gold_texts: list[str], task_idx: int
) -> list[dict]:
    """One task as 7 annotated turns, like ``_eviction_task_turns``,
    but the 3 tool_result turns carry realistic tool outputs
    (JSON API response, log tail, markdown table) with the gold
    facts embedded in them.

    The gold fact's position inside each output rotates with the
    task index (``task_idx % size``), so the gold is sometimes in
    the region the compressor keeps (head/tail) and sometimes in
    the region it collapses (middle). That is deliberate: the
    survival metric must measure what compression actually
    keeps, not a best case where the gold is always first.
    """
    turns = [
        {"role": "user", "text": question},
        {
            "role": "assistant",
            "text": "Let me look up the records for this question.",
        },
        {
            "role": "assistant",
            "text": "lookup records",
            "episode_type": "tool_call",
        },
    ]
    # JSON: a homogeneous list of 12 items; the gold is one note.
    items = [
        {
            "id": i,
            "note": "unrelated entry %d about topic %d" % (i, i % 5),
            "score": round(0.1 * i, 1),
            "detail": "x" * 200,
        }
        for i in range(12)
    ]
    gold_pos = task_idx % 12
    items[gold_pos]["note"] = gold_texts[0] if gold_texts else "no gold"
    items.append(dict(items[3]))  # exact duplicate, for dedup
    turns.append(
        {
            "role": "tool",
            "text": json.dumps(
                {
                    "status": "ok",
                    "count": len(items),
                    "results": items,
                    "empty": None,
                }
            ),
        }
    )
    # Log: 30 distinct lines + 5 repeated heartbeats + 1 error;
    # the gold is one line's message.
    log_lines = []
    for i in range(30):
        msg = "fetched record %d from shard %d" % (i, i % 4)
        if i == task_idx % 30 and len(gold_texts) > 1:
            msg = gold_texts[1]
        log_lines.append(
            "2026-11-24T10:%02d:00Z INFO %s" % (i % 60, msg)
        )
    for _ in range(5):
        log_lines.append("2026-11-24T10:00:00Z INFO heartbeat ok")
    log_lines.append("2026-11-24T10:00:00Z ERROR timeout on shard 7")
    turns.append({"role": "tool", "text": "\n".join(log_lines)})
    # Table: 15 rows; the gold is one row's note.
    rows = ["| id | note | score |", "|---|---|---|"]
    for i in range(15):
        note = "unrelated entry %d" % i
        if i == task_idx % 15 and len(gold_texts) > 2:
            note = gold_texts[2]
        rows.append("| %d | %s | %.1f |" % (i, note, 0.1 * i))
    rows.append("| 3 | unrelated entry 3 | 0.3 |")  # duplicate row
    turns.append({"role": "tool", "text": "\n".join(rows)})
    turns.append(
        {"role": "assistant", "text": "Noted.", "episode_type": "chatter"}
    )
    return turns


def _norm_text(text: str) -> str:
    return " ".join(text.lower().split())


def _gold_survival_context(store: Store, session: str, gold_texts) -> float:
    """Fraction of gold facts present in the live context text.

    The context is what the agent actually sees: live episodes'
    effective text (compressed when compressed, original when
    not). Consolidated segment records are deliberately excluded
    — they preserve facts by design (P18's finding), and this
    metric is about what compression + eviction leave in the
    context itself.
    """
    if not gold_texts:
        return 1.0
    context = []
    for ep in store.list_episodes(session, status="live"):
        text = store.episode_context_text(ep["id"])
        if text:
            context.append(text)
    blob = _norm_text("\n".join(context))
    hits = sum(
        1 for g in gold_texts if _norm_text(g) in blob
    )
    return hits / len(gold_texts)


def run_compression_eval(
    corpus_path: str,
    queries_path: str,
    embedder=None,
    n_tasks: int = 40,
    budget_frac: float = 1.0 / 3.0,
    verbose: bool = True,
) -> dict:
    """P19 eval: compress+evict vs evict-only, same budget.

    Same task set, budget and consolidation as
    ``run_eviction_eval``; the only difference is the tool
    outputs (JSON / log / table, via ``_compression_task_turns``)
    and the arms: ``evict`` (CWL, compress off) vs ``compress``
    (CWL, compress on), plus ``isolated`` (no budget) as the
    ceiling. Four metrics, all measured on this run:

    * tokens saved — tokens the compressor removed, and
      evictions avoided vs the evict-only arm;
    * recall preserved — gold survival in the live context text,
      plus the P18 retrieval scores (task hit@5, user-turn
      recall) as the regression check;
    * faithfulness — units kept / units seen over the
      compression audit rows, plus a no-invention violation
      count (every compressed line checked against its
      original);
    * latency — ms per compression call, from the audit rows.
    """
    from .compress import verify_no_invention
    from .eviction import (
        dependency_violations,
        estimate_tokens,
        evict_to_budget,
        ingest_session,
        mark_session_persisted_by_merges,
    )
    from .lifecycle import draft_merge_text, propose_merge
    from .segments import classify_segment

    embedder = embedder or get_embedder()
    records, queries = load_synthetic(corpus_path, queries_path)
    by_key = {r["key"]: r for r in records}
    tasks = []
    seen_gold: set[tuple[str, ...]] = set()
    pool = [q for q in queries if q["kind"] == "keyword"] + [
        q for q in queries if q["kind"] == "paraphrase"
    ]
    for q in pool:
        key = tuple(sorted(q["gold_keys"]))
        if key in seen_gold:
            continue
        seen_gold.add(key)
        tasks.append(q)
        if len(tasks) >= n_tasks:
            break
    for q in tasks:
        q["_gold_texts"] = [
            by_key[k]["text"] for k in q["gold_keys"] if k in by_key
        ]
    total_tokens = sum(
        estimate_tokens(t["text"])
        for qi, q in enumerate(tasks)
        for t in _compression_task_turns(
            q["question"], q["_gold_texts"], qi
        )
    )
    budget = max(1, int(total_tokens * budget_frac))
    if verbose:
        print(
            "compression eval: %d tasks, %d turns, %d tokens "
            "total, budget %d (%.0f%%)"
            % (
                len(tasks),
                len(tasks) * 7,
                total_tokens,
                budget,
                budget_frac * 100,
            )
        )

    def _consolidate_task(store, episode_ids):
        eps = [store.get_episode(e) for e in episode_ids]
        srcs = [
            e for e in eps if e is not None and e["type"] != "user"
        ]
        if len(srcs) < 2:
            return None
        ids = [e["record_id"] for e in srcs]
        texts = [store.get_record(i)["text"] for i in ids]
        union = draft_merge_text(texts)
        res = propose_merge(
            store,
            embedder,
            ids,
            merged_text=union,
            type_override=classify_segment(union),
            max_length_ratio=1.0,
        )
        return res.merged_record_id

    def _score(store, question, gold, candidates):
        if not candidates:
            return 0.0
        ranked = [
            r.id
            for r in retrieve(
                store, embedder, question, k=10, use_mmr=False,
                candidate_ids=candidates,
            )
        ]
        return any_hit_at_k(ranked, gold, 5)

    def _run_long(compress: bool):
        tmp = tempfile.mkdtemp(prefix="hr-compress-")
        store = Store(os.path.join(tmp, "eval.db"), embedder=embedder)
        session = "long-compress" if compress else "long-evict"
        ev_summaries = []
        task_gold_rids: list[set[int]] = []
        task_user_rids: list[int] = []
        task_gold_texts: list[list[str]] = []
        # Gold survival sampled after EVERY task's eviction, over
        # the gold introduced so far: the end-of-session number
        # alone washes compression out (both arms end at the
        # budget cap holding the newest episodes), while the
        # time-averaged number shows what the agent could still
        # see while the session was running.
        survival_samples: list[float] = []
        for qi, q in enumerate(tasks):
            turns = _compression_task_turns(
                q["question"], q["_gold_texts"], qi
            )
            ep_ids = ingest_session(store, session, turns)
            gold: set[int] = set()
            user_rid: int | None = None
            for eid in ep_ids:
                ep = store.get_episode(eid)
                if ep["type"] == "tool_result":
                    gold.add(ep["record_id"])
                elif ep["type"] == "user":
                    user_rid = ep["record_id"]
            task_gold_rids.append(gold)
            task_user_rids.append(user_rid)
            task_gold_texts.append(q["_gold_texts"])
            _consolidate_task(store, ep_ids)
            mark_session_persisted_by_merges(store, session)
            ev_summaries.append(
                evict_to_budget(
                    store, session, budget, compress=compress
                )
            )
            survival_samples.append(
                _gold_survival_context(
                    store,
                    session,
                    [
                        g
                        for texts in task_gold_texts
                        for g in texts
                    ],
                )
            )
        seg_of_turn: dict[int, int] = {}
        for m in store.list_merges(status="committed"):
            for src in m["merged_ids"]:
                seg_of_turn[src] = m["merged_record_id"]
        task_scores: list[float] = []
        recall_scores: list[float] = []
        live_ep_recs = [
            e["record_id"]
            for e in store.list_episodes(session, status="live")
        ]
        for q, gold_rids, user_rid in zip(
            tasks, task_gold_rids, task_user_rids
        ):
            gold = {seg_of_turn.get(r, r) for r in gold_rids}
            task_scores.append(
                _score(
                    store,
                    q["question"],
                    gold,
                    store.context_record_ids(session),
                )
            )
            if user_rid is not None:
                recall_scores.append(
                    _score(
                        store,
                        q["question"],
                        {user_rid},
                        live_ep_recs,
                    )
                )
        survival = sum(
            _gold_survival_context(store, session, texts)
            for texts in task_gold_texts
        ) / max(1, len(task_gold_texts))
        # Faithfulness + latency from the compression audit rows.
        comp_rows = store.list_compressions(session_id=session)
        units_before = sum(r["units_before"] for r in comp_rows)
        units_after = sum(r["units_after"] for r in comp_rows)
        latencies = sorted(r["latency_ms"] for r in comp_rows)
        violations = 0
        for r in comp_rows:
            ep = store.get_episode(r["episode_id"])
            rec = store.get_record(ep["record_id"]) if ep else None
            if ep is None or rec is None:
                continue
            # Eviction never clears compressed_text, so even an
            # episode evicted after compression still carries its
            # compressed rendering for this check.
            compressed = ep.get("compressed_text")
            if compressed and verify_no_invention(
                rec["text"], compressed
            ):
                violations += 1
        live = store.list_episodes(session, status="live")
        user_live = sum(1 for e in live if e["type"] == "user")
        last = ev_summaries[-1] if ev_summaries else {}
        result = {
            "task_any_hit5": (
                sum(task_scores) / len(task_scores) if task_scores else 0.0
            ),
            "user_turn_recall": (
                sum(recall_scores) / len(recall_scores)
                if recall_scores
                else 0.0
            ),
            "user_turns_live": user_live,
            "user_turns_total": len(tasks),
            "live_tokens": store.session_token_total(session),
            "total_tokens": total_tokens,
            "budget": budget,
            "evicted_total": sum(
                s.get("n_evicted", 0) for s in ev_summaries
            ),
            "compressed_total": sum(
                s.get("n_compressed", 0) for s in ev_summaries
            ),
            "tokens_saved_by_compression": sum(
                s.get("tokens_saved_by_compression", 0)
                for s in ev_summaries
            ),
            "gold_survival_context": survival,
            "gold_survival_mean_over_time": (
                sum(survival_samples) / len(survival_samples)
                if survival_samples
                else 1.0
            ),
            "faithfulness": (
                units_after / units_before if units_before else 1.0
            ),
            "no_invention_violations": violations,
            "compression_calls": len(comp_rows),
            "latency_ms_mean": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            "latency_ms_p95": (
                latencies[int(0.95 * (len(latencies) - 1))]
                if latencies
                else 0.0
            ),
            "dependency_violations": len(
                dependency_violations(store, session)
            ),
            "under_budget": bool(last.get("under_budget", True)),
        }
        store.close()
        return result

    def _run_isolated():
        survival = 0.0
        for qi, q in enumerate(tasks):
            tmp = tempfile.mkdtemp(prefix="hr-iso-c-")
            store = Store(
                os.path.join(tmp, "eval.db"), embedder=embedder
            )
            session = "isolated"
            turns = _compression_task_turns(
                q["question"], q["_gold_texts"], qi
            )
            ingest_session(store, session, turns)
            survival += _gold_survival_context(
                store, session, q["_gold_texts"]
            )
            store.close()
        return {
            "gold_survival_context": (
                survival / len(tasks) if tasks else 1.0
            ),
        }

    arms = {
        "isolated": _run_isolated(),
        "evict": _run_long(compress=False),
        "compress": _run_long(compress=True),
    }
    return {
        "corpus": "synthetic tool-output tasks (labelled)",
        "n_tasks": len(tasks),
        "total_tokens": total_tokens,
        "budget": budget,
        "budget_frac": budget_frac,
        "embedder": embedder.name,
        "arms": arms,
    }


def print_compression_report(result: dict) -> None:
    print("\n== %s ==" % result["corpus"])
    print(
        "tasks: %d   total tokens: %d   budget: %d (%.0f%%)   "
        "embedder: %s"
        % (
            result["n_tasks"],
            result["total_tokens"],
            result["budget"],
            result["budget_frac"] * 100,
            result["embedder"],
        )
    )
    print(
        "%-10s %12s %16s %14s %12s %12s"
        % (
            "arm",
            "task hit@5",
            "user-turn rec.",
            "gold survival",
            "live toks",
            "evicted",
        )
    )
    for arm, m in result["arms"].items():
        print(
            "%-10s %12.3f %16s %14.3f %12s %12s"
            % (
                arm,
                m.get("task_any_hit5", float("nan")),
                ("%.3f" % m["user_turn_recall"])
                if "user_turn_recall" in m
                else "-",
                m["gold_survival_context"],
                str(m.get("live_tokens", "-")),
                str(m.get("evicted_total", "-")),
            )
        )
    c = result["arms"]["compress"]
    e = result["arms"]["evict"]
    print(
        "gold survival, mean over time (sampled after each "
        "task): evict %.3f, compress %.3f"
        % (
            e["gold_survival_mean_over_time"],
            c["gold_survival_mean_over_time"],
        )
    )
    print(
        "compress: %d compressions, %d tokens saved by "
        "compression, %d evictions avoided vs evict-only "
        "(%d -> %d), faithfulness %.3f, no-invention "
        "violations %d, latency mean %.2f ms / p95 %.2f ms"
        % (
            c["compressed_total"],
            c["tokens_saved_by_compression"],
            e["evicted_total"] - c["evicted_total"],
            e["evicted_total"],
            c["evicted_total"],
            c["faithfulness"],
            c["no_invention_violations"],
            c["latency_ms_mean"],
            c["latency_ms_p95"],
        )
    )
    print(
        "under budget: evict %s, compress %s; dependency "
        "violations: evict %s, compress %s"
        % (
            e["under_budget"],
            c["under_budget"],
            e["dependency_violations"],
            c["dependency_violations"],
        )
    )
