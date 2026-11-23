"""Store tests: records, FTS5 BM25, vectors (both backends), edges."""

import numpy as np
import pytest

from hybrid_retriever.embeddings import HashedEmbedder
from hybrid_retriever.store import Store

DOCS = [
    "Maya feeds her sourdough starter every morning.",
    "Maya delivers croissants to the corner cafe.",
    "Dev rides his bike along the river path.",
]


@pytest.fixture(params=["sqlite-vec", "numpy"])
def store(request, tmp_path):
    if request.param == "numpy":
        # Force the NumPy fallback path even though sqlite-vec is installed.
        orig = Store._detect_vec_mode
        Store._detect_vec_mode = lambda self: "numpy"  # noqa: ARG005
        s = Store(str(tmp_path / "t.db"), embedder=HashedEmbedder())
        Store._detect_vec_mode = orig
        assert s.vec_mode == "numpy"
    else:
        s = Store(str(tmp_path / "t.db"), embedder=HashedEmbedder())
        assert s.vec_mode == "sqlite-vec"
    s.add_texts(DOCS)
    yield s
    s.close()


def test_add_and_get(store):
    assert store.count() == 3
    rec = store.get_record(1)
    assert rec["text"] == DOCS[0]
    assert rec["type"] == "fact"
    assert store.get_record(999) is None


def test_bm25_finds_keyword(store):
    hits = store.bm25_search("sourdough starter", k=10)
    assert hits and hits[0][0] == 1
    # No tokens -> no hits, not an error.
    assert store.bm25_search("!!!", k=10) == []


def test_bm25_candidate_filter(store):
    hits = store.bm25_search("Maya", k=10, candidate_ids=[2])
    assert [h[0] for h in hits] == [2]


def test_vector_search_self_match(store):
    emb = HashedEmbedder()
    q = emb.embed(["Maya feeds her sourdough starter every morning."])[0]
    hits = store.vector_search(q, k=3)
    assert hits[0][0] == 1
    assert hits[0][1] > 0.99  # identical text -> cosine ~1


def test_vector_candidate_filter(store):
    emb = HashedEmbedder()
    q = emb.embed(["Maya"])[0]
    hits = store.vector_search(q, k=10, candidate_ids=[1])
    assert [h[0] for h in hits] == [1]
    # A candidate with zero similarity is excluded, not returned with 0.
    hits = store.vector_search(q, k=10, candidate_ids=[3])
    assert hits == []


def test_vector_roundtrip(store):
    v = store.get_vector(1)
    assert v is not None and v.shape == (384,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    assert store.get_vector(999) is None


def test_edges_and_neighbours(tmp_path):
    s = Store(str(tmp_path / "e.db"))
    a = s.add_record("alpha")
    b = s.add_record("beta")
    c = s.add_record("gamma")
    s.add_edge(a, b, weight=2.0)
    s.add_edge(a, c, weight=1.0)
    assert s.neighbours([a], k=10) == [(b, 2.0), (c, 1.0)]
    # Seeds are excluded; unknown seeds give nothing.
    assert s.neighbours([b], k=10) == []
    assert s.neighbours([], k=10) == []
    s.close()
