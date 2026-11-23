"""P3 segment tests: boundary detection, typing, consolidation, planner.

Uses a deterministic topic-vector test double, not the hashed
embedder: hashed vectors of short turns are too sparse to separate
topics (a Maya->Dev switch scored 0.24, above some within-topic
pairs at 0.22). The double maps each text to a topic by keyword
and returns unit vectors with cosine 1.0 within a topic and 0.01
across — it tests the detector / consolidator / planner mechanics,
while the real embedder's separation is measured by the eval
(``hr eval --segments`` with the ONNX model).
"""

import numpy as np
import pytest

from hybrid_retriever.embeddings import Embedder
from hybrid_retriever.lifecycle import propose_merge
from hybrid_retriever.retrieval import retrieve
from hybrid_retriever.segments import (
    boundary_f1,
    boundary_positions,
    classify_segment,
    consolidate_records,
    consolidate_transcript,
    detect_boundaries,
    detect_boundaries_fixed_window,
    plan_query,
    retrieve_planned,
)
from hybrid_retriever.store import Store


class TopicEmbedder(Embedder):
    """Test double: topic by keyword, cosine 1.0 / 0.01.

    Unknown topics get the zero vector (no similarity at all),
    matching how the real code treats unembeddable text.
    """

    _TOPICS = (("maya", 0), ("dev", 1), ("pipes", 2), ("copper", 2))

    def __init__(self) -> None:
        self.dim = 8
        self.name = "topic-stub"

    def _topic(self, text: str) -> int | None:
        low = text.lower()
        for keyword, topic in self._TOPICS:
            if keyword in low:
                return topic
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            topic = self._topic(text)
            if topic is None:
                continue
            out[i, 0] = 0.1
            out[i, 1 + topic] = float(np.sqrt(0.99))
        return out


EMB = TopicEmbedder()

MAYA = [
    "Maya feeds her sourdough starter with rye flour every morning at six.",
    "The starter Maya keeps in the pantry bubbled over after four days without feeding.",
    "Maya's starter, begun in 2019, now leavens twelve loaves a week.",
]
DEV = [
    "Dev's road bike has a cracked chainstay that needs welding before spring.",
    "The bike Dev rides to work has tubeless tyres that seal small punctures.",
    "Dev fitted a new saddle to his bike and the numbness on long rides stopped.",
]


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "s.db"), embedder=EMB)
    yield s
    s.close()


# ----------------------------------------------------------------------
# Boundary detection
# ----------------------------------------------------------------------
def test_boundaries_find_topic_switch():
    turns = MAYA + DEV
    segments = detect_boundaries(turns, EMB, threshold=0.30)
    assert boundary_positions(segments) == {3}
    assert [len(s) for s in segments] == [3, 3]


def test_boundaries_no_switch_without_drift():
    segments = detect_boundaries(MAYA, EMB, threshold=0.30)
    assert segments == [[0, 1, 2]]


def test_boundaries_min_turns_blocks_lone_turn_segment():
    # A lone off-topic turn never forms its own segment: the cut
    # before it happens (the segment before it is long enough), but
    # it is absorbed into the following segment instead of splitting
    # one off on its own.
    turns = MAYA[:2] + [DEV[0]] + MAYA[2:]
    segments = detect_boundaries(turns, EMB, threshold=0.30, min_turns=2)
    assert segments == [[0, 1], [2, 3]]


def test_boundaries_max_turns_forces_cut():
    turns = MAYA * 3
    segments = detect_boundaries(turns, EMB, threshold=0.0, max_turns=3)
    assert [len(s) for s in segments] == [3, 3, 3]


def test_boundaries_empty_and_single():
    assert detect_boundaries([], EMB) == []
    assert detect_boundaries(["only one turn"], EMB) == [[0]]


def test_fixed_window_control():
    assert detect_boundaries_fixed_window(7, 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert detect_boundaries_fixed_window(0, 3) == []


def test_boundary_f1_math():
    assert boundary_f1({3}, {3}) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert boundary_f1({3, 6}, {3})["precision"] == 0.5
    assert boundary_f1(set(), {3})["recall"] == 0.0


def test_semantic_beats_fixed_window_on_two_topics():
    turns = MAYA + DEV
    truth = {3}
    sem = boundary_f1(
        boundary_positions(detect_boundaries(turns, EMB, threshold=0.30)),
        truth,
    )
    fw = boundary_f1(
        boundary_positions(detect_boundaries_fixed_window(len(turns), 2)),
        truth,
    )
    assert sem["f1"] == 1.0
    assert sem["f1"] > fw["f1"]


# ----------------------------------------------------------------------
# Segment typing
# ----------------------------------------------------------------------
def test_classify_segment_types():
    assert (
        classify_segment("We decided to go with the copper pipes.")
        == "decision"
    )
    assert (
        classify_segment("I prefer card payments over cash, always.")
        == "preference"
    )
    assert (
        classify_segment("The boiler broke down and we fixed the leak.")
        == "error-fix"
    )
    assert (
        classify_segment("Maya feeds her starter every morning.")
        == "fact"
    )
    # Decision cues win over preference cues in the same text.
    assert (
        classify_segment("We decided to go with what I prefer.")
        == "decision"
    )


# ----------------------------------------------------------------------
# Consolidation
# ----------------------------------------------------------------------
def test_consolidate_closed_segments(store):
    summary = consolidate_transcript(store, EMB, MAYA + DEV)
    assert summary["turns_ingested"] == 6
    assert summary["segments"] == 2
    assert summary["closed_segments"] == 1
    assert summary["consolidated"] == 1
    assert summary["left_open"] == 3  # the Dev segment is still open
    # The merged record is live and typed; its sources are superseded.
    merged_id = summary["merged_record_ids"][0]
    merged = store.get_record(merged_id)
    assert merged["status"] == "live" and merged["type"] == "fact"
    assert "Maya feeds" in merged["text"]
    for rid in (1, 2, 3):
        assert store.get_record(rid)["status"] == "superseded"
    for rid in (4, 5, 6):
        assert store.get_record(rid)["status"] == "live"
    # The audit row is in the merges table, committed.
    merges = store.list_merges(status="committed")
    assert len(merges) == 1
    assert merges[0]["merged_record_id"] == merged_id
    assert merges[0]["merged_ids"] == [1, 2, 3]
    # The merged record is retrievable; the sources are not.
    res = retrieve(store, EMB, "Maya sourdough starter", k=10, use_mmr=False)
    assert res[0].id == merged_id
    assert not ({1, 2, 3} & {r.id for r in res})


def test_consolidate_dry_run_writes_nothing(store):
    summary = consolidate_transcript(store, EMB, MAYA + DEV, dry_run=True)
    assert summary["consolidated"] == 1
    assert store.count() == 6  # only the 6 turn records
    assert store.list_merges() == []
    assert store.count_live() == 6


def test_consolidate_single_turn_segment_skipped(store):
    # Two turns, one segment, but it is the last (open) one.
    summary = consolidate_transcript(store, EMB, MAYA[:2])
    assert summary["segments"] == 1
    assert summary["consolidated"] == 0
    assert summary["left_open"] == 2
    # A one-turn closed segment is skipped, not merged.
    ids = [store.add_record(t, type_="episode") for t in MAYA + DEV]
    summary2 = consolidate_records(
        store, EMB, ids, threshold=0.30, min_turns=2
    )
    assert summary2["segments"] == 2
    assert summary2["consolidated"] == 1


def test_consolidate_typed_segment(store):
    # Every turn names the pipes/copper topic (the test double maps
    # topic by keyword), so the boundary detector sees one segment.
    turns = [
        "We decided to replace the lead pipes with copper pipes.",
        "The decision about the copper pipes was made after the leak.",
        "We will replace the lead pipes in the cottage with copper.",
    ]
    summary = consolidate_transcript(store, EMB, turns + MAYA)
    assert summary["consolidated"] == 1
    merged = store.get_record(summary["merged_record_ids"][0])
    assert merged["type"] == "decision"


def test_consolidate_fixed_window_control(store):
    summary = consolidate_transcript(
        store, EMB, MAYA + DEV, fixed_window=2
    )
    # Fixed window 2 cuts [0,1], [2,3], [4,5]; the last is open.
    assert summary["segments"] == 3
    assert summary["consolidated"] == 2
    assert summary["left_open"] == 2


def test_type_override_does_not_change_p2_merge(store):
    # Near-duplicates with shared sentences, so the sentence union
    # actually compresses (length ratio 0.75) and the gate commits.
    a = store.add_record(
        "Maya feeds her starter every morning. "
        "Maya feeds her starter with rye flour.",
        type_="fact",
    )
    b = store.add_record(
        "Maya feeds her starter every morning. "
        "The starter bubbled over after four days.",
        type_="fact",
    )
    res = propose_merge(store, EMB, [a, b])
    assert res.status == "committed"
    assert store.get_record(res.merged_record_id)["type"] == "fact"


# ----------------------------------------------------------------------
# Query planner
# ----------------------------------------------------------------------
def _seed(store):
    store.add_texts(MAYA + DEV, type_="fact")
    store.add_record(
        "We decided to replace the lead pipes with copper.",
        type_="decision",
    )


def test_plan_temporal_route(store):
    _seed(store)
    plan = plan_query(store, "what did we do recently", EMB)
    assert plan.route == "temporal"
    assert plan.candidate_ids  # the just-added records are recent


def test_plan_typed_route(store):
    _seed(store)
    plan = plan_query(store, "what decision did we make about the pipes", EMB)
    assert plan.route == "typed"
    assert plan.candidate_ids == [7]


def test_plan_typed_falls_back_when_type_absent(store):
    _seed(store)
    # "prefer" is a type cue, but no preference records exist — and
    # no word of the query appears in the corpus either, so the
    # probe has nothing to route on. The typed route must not fire.
    plan = plan_query(store, "does anyone prefer brunch", EMB)
    assert plan.route == "hybrid"


def test_plan_quoted_phrase_is_keyword(store):
    _seed(store)
    plan = plan_query(store, '"sourdough starter" feeding', EMB)
    assert plan.route == "keyword"
    assert plan.weights == (1.0, 0.0, 0.0)


def test_plan_no_tokens_is_vector(store):
    _seed(store)
    plan = plan_query(store, "!!!", EMB)
    assert plan.route == "vector"


def test_plan_lexical_query_is_keyword(store):
    _seed(store)
    plan = plan_query(store, "Maya sourdough starter feeding", EMB)
    assert plan.route == "keyword"


def test_plan_paraphrase_query_is_vector(store):
    _seed(store)
    plan = plan_query(store, "how does Maya maintain her bread culture", EMB)
    assert plan.route == "vector"


def test_plan_importance_prior(store):
    _seed(store)
    # Make the Maya records high-importance; the probe should notice.
    for rid in (1, 2, 3):
        store.record_credit_update(rid, uses=1, credit=1.0, importance=0.9)
    plan = plan_query(store, "Maya sourdough starter feeding", EMB)
    assert plan.route == "keyword"  # lexical route still wins
    # ...but on a hybrid-routed query the prior fires.
    plan2 = plan_query(store, "what happened with the starter", EMB)
    assert plan2.importance_weight == 0.5


def test_retrieve_planned_returns_plan_and_results(store):
    _seed(store)
    results, plan = retrieve_planned(
        store, EMB, "how does Maya maintain her bread culture"
    )
    assert plan.route == "vector"
    assert results and results[0].id in (1, 2, 3)
