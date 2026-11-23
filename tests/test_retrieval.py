"""End-to-end retrieval tests on a tiny in-memory store.

Uses HashedEmbedder (no torch) so the tests run anywhere.
"""

import pytest

from hybrid_retriever.embeddings import HashedEmbedder
from hybrid_retriever.retrieval import retrieve
from hybrid_retriever.store import Store

DOCS = [
    "Maya feeds her sourdough starter every morning at six.",
    "Maya delivers croissants to the corner cafe on Tuesdays.",
    "Dev rides his bike along the river path to work.",
    "Dev trains four mornings a week with intervals on Tuesdays.",
    "Priya waters the tomato beds at dawn every day.",
]


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "r.db"), embedder=HashedEmbedder())
    s.add_texts(DOCS)
    # Chain the two Maya records so the graph leg has an edge to follow.
    s.add_edge(1, 2, weight=1.0)
    s.add_edge(2, 1, weight=1.0)
    yield s
    s.close()


def test_hybrid_finds_gold(store):
    res = retrieve(store, HashedEmbedder(), "Maya sourdough starter", k=3)
    assert res and res[0].id == 1
    assert "bm25" in res[0].legs or "vector" in res[0].legs


def test_bm25_only_mode(store):
    res = retrieve(
        store, None, "croissants cafe", k=3, weights=(1.0, 0.0, 0.0),
        use_mmr=False,
    )
    assert res[0].id == 2


def test_vector_only_mode(store):
    res = retrieve(
        store, HashedEmbedder(), "Maya feeds sourdough starter morning",
        k=3, weights=(0.0, 1.0, 0.0), use_mmr=False,
    )
    assert res[0].id == 1


def test_graph_leg_surfaces_linked_record(store):
    # Query that matches record 1 should also surface linked record 2
    # via the graph leg, even though record 2 shares no query words.
    res = retrieve(
        store, HashedEmbedder(), "sourdough starter six morning",
        k=5, weights=(0.5, 0.35, 0.15), use_mmr=False,
    )
    ids = [r.id for r in res]
    assert 1 in ids
    assert 2 in ids
    rec2 = next(r for r in res if r.id == 2)
    assert "graph" in rec2.legs


def test_candidate_ids_restrict_all_legs(store):
    res = retrieve(
        store, HashedEmbedder(), "Maya", k=10, candidate_ids=[3, 4],
    )
    assert all(r.id in (3, 4) for r in res)


def test_empty_query_returns_nothing(store):
    assert retrieve(store, HashedEmbedder(), "!!!", k=5) == []
