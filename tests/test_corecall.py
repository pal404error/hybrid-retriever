"""P4 — REALM-style co-recall graph tests.

Uses HashedEmbedder (no torch) so the tests run anywhere, as in
test_retrieval.py. The ONNX-backed accuracy numbers live in the
eval (``hr eval --corecall``), not here.
"""

import time

import pytest

from hybrid_retriever.corecall import (
    co_recall_clusters,
    compose_expansion,
    expand_from_edges,
    expand_from_log,
    expand_temporal,
    expand_typed,
    graph_stats,
    log_co_use,
    make_expander,
    reconsolidate,
    retrieve_expanded,
)
from hybrid_retriever.embeddings import HashedEmbedder
from hybrid_retriever.retrieval import Retrieved, retrieve
from hybrid_retriever.store import Store

EMB = HashedEmbedder()

DOCS = [
    "Maya feeds her sourdough starter every morning at six.",
    "Maya keeps her sourdough starter in a glass jar on the counter.",
    "The corner cafe orders croissants every Tuesday.",
    "Dev rides his bike along the river path to work.",
    "Dev trains four mornings a week with intervals on Tuesdays.",
    "Priya waters the tomato beds at dawn every day.",
]


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "r.db"), embedder=EMB)
    s.add_texts(DOCS)
    yield s
    s.close()


def _retrieved(*ids, credits=None):
    """Fake Retrieved list with per-id credit, for log_co_use."""
    out = []
    for i in ids:
        out.append(Retrieved(id=i, text="x", score=0.0, legs={"vector"}))
    return out


def test_store_creates_corecall_tables(tmp_path):
    s = Store(str(tmp_path / "fresh.db"))
    rows = {
        r[0]
        for r in s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "corecall" in rows
    assert "reconsolidations" in rows
    s.close()


def test_log_co_use_weights_by_min_credit(store):
    res = _retrieved(1, 2, 3)
    credits = {1: 0.8, 2: 0.5, 3: 0.0}  # 3 retrieved but not used
    assert log_co_use(store, res, credits) == 1  # only the (1, 2) pair
    assert store.corecall_pairs() == [(1, 2, 1, 0.5)]
    assert store.corecall_weight(1, 2) == pytest.approx(0.5)
    assert store.corecall_weight(1, 3) == 0.0


def test_log_corecall_ignores_self_and_nonpositive(store):
    store.log_corecall(1, 1, 1.0)
    store.log_corecall(1, 2, 0.0)
    store.log_corecall(1, 2, -1.0)
    assert store.corecall_pairs() == []


def test_log_co_use_accumulates_count_and_wsum(store):
    res = _retrieved(1, 2)
    log_co_use(store, res, {1: 0.8, 2: 0.5})
    log_co_use(store, res, {1: 0.6, 2: 0.9})
    assert store.corecall_pairs() == [(1, 2, 2, 1.1)]
    assert store.corecall_pairs(min_wsum=1.5) == []
    assert store.corecall_pairs(min_wsum=1.0) == [(1, 2, 2, 1.1)]


def test_expand_from_edges_threshold_and_kind(store):
    # Co-recall edges (promoted) + one loud 'related' edge that must
    # never be followed: the P1 mistake is indiscriminate chaining.
    store.add_edge(1, 2, weight=2.0, kind="co-recall")
    store.add_edge(2, 1, weight=2.0, kind="co-recall")
    store.add_edge(1, 3, weight=1.8, kind="co-recall")
    store.add_edge(3, 1, weight=1.8, kind="co-recall")
    store.add_edge(1, 4, weight=5.0, kind="related")
    store.add_edge(4, 1, weight=5.0, kind="related")
    out = expand_from_edges(store, [1])
    assert [r for r, _ in out] == [2, 3]
    # Below the retrieval floor: the 2-3 edge at 0.4 is not followed.
    store.add_edge(2, 3, weight=0.4, kind="co-recall")
    assert expand_from_edges(store, [2], min_weight=0.5) == [
        (1, 2.0),
    ]
    # Top-k per seed: only the best neighbour survives.
    assert expand_from_edges(store, [1], max_per_seed=1) == [(2, 2.0)]


def test_expand_from_edges_bounded_hops(store):
    store.add_edge(1, 2, weight=2.0, kind="co-recall")
    store.add_edge(2, 1, weight=2.0, kind="co-recall")
    store.add_edge(2, 4, weight=2.0, kind="co-recall")
    store.add_edge(4, 2, weight=2.0, kind="co-recall")
    assert expand_from_edges(store, [1], max_hops=1) == [(2, 2.0)]
    out = expand_from_edges(store, [1], max_hops=2)
    assert [r for r, _ in out] == [2, 4]
    # The 2-hop neighbour is decayed, never outranking a direct edge.
    assert dict(out)[4] == pytest.approx(1.0)


def test_expand_from_edges_skips_non_live(store):
    store.add_edge(1, 2, weight=2.0, kind="co-recall")
    store.add_edge(2, 1, weight=2.0, kind="co-recall")
    store.set_status(2, "superseded", superseded_by=1)
    assert expand_from_edges(store, [1]) == []


def test_expand_from_log_threshold(store):
    store.log_corecall(1, 2, 1.2)
    store.log_corecall(1, 2, 1.2)  # wsum 2.4, count 2
    store.log_corecall(1, 3, 0.4)
    assert expand_from_log(store, [1], min_wsum=1.0) == [(2, 2.4)]
    assert expand_from_log(store, [1], min_wsum=0.0) == [
        (2, 2.4),
        (3, 0.4),
    ]


def test_expand_temporal_window(store):
    now = time.time()
    store.conn.execute(
        "UPDATE records SET created_at = ? WHERE id = ?", (now, 1)
    )
    store.conn.execute(
        "UPDATE records SET created_at = ? WHERE id = ?", (now, 2)
    )
    store.conn.execute(
        "UPDATE records SET created_at = ? WHERE id = ?",
        (now - 30 * 86400.0, 6),
    )
    store.conn.commit()
    out = expand_temporal(store, [1], window_days=7.0)
    ids = [r for r, _ in out]
    assert 2 in ids
    assert 6 not in ids
    assert 1 not in ids  # seeds are never returned


def test_expand_typed_same_type_by_importance(store):
    pref = store.add_record(
        "Maya prefers rye flour over white flour.", type_="preference"
    )
    other = store.add_record(
        "Dev prefers morning rides over evening rides.", type_="preference"
    )
    store.record_credit_update(pref, uses=1, credit=0.9, importance=0.9)
    store.record_credit_update(other, uses=1, credit=0.1, importance=0.1)
    out = expand_typed(store, [pref])
    assert [r for r, _ in out] == [other]


def test_compose_expansion_dedupes_and_rejects_unknown(store):
    store.add_edge(1, 2, weight=2.0, kind="co-recall")
    store.add_edge(2, 1, weight=2.0, kind="co-recall")
    out = compose_expansion(store, [1], atoms=("neighbour", "typed"))
    ids = [r for r, _ in out]
    assert ids.count(2) == 1
    with pytest.raises(ValueError):
        compose_expansion(store, [1], atoms=("nope",))


def test_make_expander_unknown_source(store):
    with pytest.raises(ValueError):
        make_expander(store, source="nope")


def test_retrieve_expanded_empty_graph_is_hybrid(store):
    # No co-recall edges at all: the expander contributes nothing,
    # so the result must be exactly the P1 hybrid result.
    base = retrieve(store, EMB, "Maya sourdough starter", k=5, use_mmr=False)
    expanded = retrieve_expanded(
        store, EMB, "Maya sourdough starter", k=5
    )
    assert [r.id for r in expanded] == [r.id for r in base]


def test_retrieve_expanded_surfaces_co_recalled_record(store):
    # Record 3 shares no words with the query, so the bm25/vector
    # legs cannot find it — only the co-recall edge from record 1 can.
    store.add_edge(1, 3, weight=2.0, kind="co-recall")
    store.add_edge(3, 1, weight=2.0, kind="co-recall")
    base = retrieve(
        store, EMB, "Maya sourdough starter morning six", k=3,
        weights=(0.50, 0.35, 0.0), use_mmr=False,
    )
    assert 3 not in [r.id for r in base]
    expanded = retrieve_expanded(
        store, EMB, "Maya sourdough starter morning six", k=3
    )
    ids = [r.id for r in expanded]
    assert 3 in ids
    assert "graph" in next(r for r in expanded if r.id == 3).legs


def test_retrieve_expanded_candidate_ids_restrict_graph(store):
    store.add_edge(1, 3, weight=2.0, kind="co-recall")
    store.add_edge(3, 1, weight=2.0, kind="co-recall")
    out = retrieve_expanded(
        store, EMB, "Maya sourdough starter morning six", k=5,
        candidate_ids=[1, 2],
    )
    assert all(r.id in (1, 2) for r in out)


def test_co_recall_clusters(store):
    store.log_corecall(1, 2, 1.0)
    store.log_corecall(1, 2, 1.0)
    store.log_corecall(2, 3, 1.0)
    store.log_corecall(2, 3, 1.0)
    store.log_corecall(4, 5, 1.0)
    assert co_recall_clusters(store, min_wsum=1.5) == [[1, 2, 3]]
    # Below the gate the 4-5 pair (wsum 1.0, count 1) forms a cluster
    # only when the count floor is lowered too.
    assert co_recall_clusters(store, min_wsum=0.5, min_count=1) == [
        [1, 2, 3],
        [4, 5],
    ]


def test_reconsolidate_promotes_prunes_and_closes(store):
    # (1,2) meets the gate; (1,3) does not; (2,4) meets it via 2 hops
    # to (1,4) closure; a stale hand-added edge must be pruned.
    store.log_corecall(1, 2, 1.0)
    store.log_corecall(1, 2, 1.0)
    store.log_corecall(2, 4, 1.0)
    store.log_corecall(2, 4, 1.0)
    store.log_corecall(1, 3, 0.4)
    store.add_edge(5, 6, weight=0.3, kind="co-recall")
    store.add_edge(6, 5, weight=0.3, kind="co-recall")
    summary = reconsolidate(store, embedder=EMB)
    assert summary["promoted"] == 2
    assert summary["closure_added"] == 1  # (1, 4) via record 2
    assert summary["pruned"] == 2  # the stale (5,6) pair, both ways
    assert summary["clusters"] == 1
    assert summary["merges_proposed"] == 0
    edges = {(s, d): w for s, d, w, _ in store.list_edges(kind="co-recall")}
    assert edges[(1, 2)] == pytest.approx(2.0)
    assert edges[(1, 4)] == pytest.approx(1.0)  # 0.5 decay x 2.0
    assert (1, 3) not in edges
    assert (5, 6) not in edges
    assert len(store.list_reconsolidations()) == 1


def test_reconsolidate_dry_run_writes_nothing(store):
    store.log_corecall(1, 2, 1.0)
    store.log_corecall(1, 2, 1.0)
    summary = reconsolidate(store, embedder=EMB, dry_run=True)
    assert summary["promoted"] == 1
    assert summary["dry_run"] is True
    assert store.list_edges(kind="co-recall") == []
    assert store.list_reconsolidations() == []


def test_reconsolidate_merges_near_duplicates_in_cluster(store):
    dup = store.add_record(
        "Maya feeds her sourdough starter every morning at six."
    )
    store.log_corecall(1, dup, 1.0)
    store.log_corecall(1, dup, 1.0)
    summary = reconsolidate(store, embedder=EMB)
    assert summary["merges_proposed"] == 1
    assert summary["merges_committed"] == 1
    assert store.get_record(1)["status"] == "superseded"
    assert store.get_record(dup)["status"] == "superseded"
    # The merged record is live and no edge points at a dead record.
    assert store.count_live() == len(DOCS)  # 6 originals - 2 + 1 merged
    assert all(
        store.get_record(s)["status"] == "live"
        and store.get_record(d)["status"] == "live"
        for s, d, _w, _k in store.list_edges(kind="co-recall")
    )


def test_graph_stats(store):
    store.log_corecall(1, 2, 1.0)
    store.log_corecall(1, 2, 1.0)
    stats = graph_stats(store)
    assert stats["corecall_pairs"] == 1
    assert stats["corecall_pairs_ge_1.5"] == 1
    assert stats["live_records"] == len(DOCS)


def test_run_corecall_eval_smoke(tmp_path):
    from hybrid_retriever.eval import run_corecall_eval

    result = run_corecall_eval(
        "eval_data/synthetic_corpus.jsonl",
        "eval_data/synthetic_queries.json",
        embedder=EMB,
        warmup_rounds=2,
        verbose=False,
    )
    assert result["n_questions"] == 56
    assert result["n_warmup"] == 48
    assert set(result["scores"]) == {
        "hybrid",
        "vector",
        "hybrid+corecall-naive",
        "hybrid+corecall-raw",
        "hybrid+corecall",
        "vector+corecall",
    }
    for scores in result["scores"].values():
        assert set(scores) == {
            "P@10", "R@10", "nDCG@10", "any-hit@5", "grounding@10",
        }
    # Two warm-up rounds put every co-credited pair at count 2, so
    # the promotion gate can fire and the audit row exists.
    assert result["reconsolidation"]["promoted"] > 0
    assert result["graph_after"]["edges_by_kind"].get("co-recall", 0) > 0
