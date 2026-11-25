"""Tests for P6 LoRA weight memory (Route to Remember).

Uses HashedEmbedder (no torch) so the tests run anywhere, as in
test_corecall.py. The ONNX-backed accuracy numbers live in the
eval (``hr eval --lora``), not here — hashed embeddings are
lexical, so no accuracy ordering is asserted on them, only the
machinery: distillation, determinism, versioned artifacts,
staleness, and verify-then-regenerate incl. the planted conflict.
"""

import os

import numpy as np
import pytest

from hybrid_retriever.embeddings import HashedEmbedder
from hybrid_retriever.eval import load_synthetic
from hybrid_retriever.lora import (
    PLANTED_CORRECTED_FACT,
    PLANTED_CORRECTED_MARKER,
    PLANTED_CORRECTED_TEXTS,
    PLANTED_FACT,
    PLANTED_QUESTION,
    FactIndex,
    answer_student,
    artifact_path,
    audit_answer,
    build_distillation_examples,
    build_fact_index,
    is_stale,
    load_student,
    run_lora_eval,
    run_planted_conflict,
    save_student,
    teacher_answer,
    train_student,
)

EMB = HashedEmbedder()
CORPUS = os.path.join(
    os.path.dirname(__file__), "..", "eval_data", "synthetic_corpus.jsonl"
)
QUERIES = os.path.join(
    os.path.dirname(__file__), "..", "eval_data", "synthetic_queries.json"
)


def _setup(tmp_path):
    """A store + fact index + trained student on the synthetic corpus."""
    from hybrid_retriever.store import Store

    records, queries = load_synthetic(CORPUS, QUERIES)
    store = Store(str(tmp_path / "l.db"), embedder=EMB)
    rids = store.add_texts(
        [r["text"] for r in records], type_="fact", embedder=EMB
    )
    index = build_fact_index(records, queries, rids)
    texts, labels = build_distillation_examples(index, queries)
    student = train_student(
        texts, labels, EMB, rank=32, fact_set_hash=index.fact_set_hash()
    )
    return store, index, student, records, queries


# --- Fact index ------------------------------------------------------------


def test_fact_index_has_48_facts_and_traps_map_to_none(tmp_path):
    records, queries = load_synthetic(CORPUS, QUERIES)
    index = FactIndex()
    assert index.fact_set_hash()  # empty index still hashes
    store_index = build_fact_index(records, queries, list(range(1, 161)))
    assert len(store_index.facts) == 48
    assert len(store_index.fact_texts["maya-oven"]) == 3
    assert "Maya's deck oven runs at 240 degrees" in (
        store_index.representative_text("maya-oven")
        + " ".join(store_index.fact_texts["maya-oven"])
    )
    trap_rids = [
        rid
        for rid, fact in store_index.rid_to_fact.items()
        if fact is None
    ]
    assert trap_rids  # trap records name no fact


def test_distillation_examples_are_keyword_plus_teacher_answers(tmp_path):
    records, queries = load_synthetic(CORPUS, QUERIES)
    index = build_fact_index(records, queries, list(range(1, 161)))
    texts, labels = build_distillation_examples(index, queries)
    assert len(texts) == 48 * 4  # 1 keyword question + 3 answer texts
    assert len(set(labels)) == 48
    assert all("trap" not in label for label in labels)
    # No held-out (paraphrase/trap) question is a training text.
    heldout = {q["question"] for q in queries if q["kind"] != "keyword"}
    assert not (set(texts) & heldout)


# --- Student training ------------------------------------------------------


def test_train_student_is_deterministic(tmp_path):
    records, queries = load_synthetic(CORPUS, QUERIES)
    index = build_fact_index(records, queries, list(range(1, 161)))
    texts, labels = build_distillation_examples(index, queries)
    s1 = train_student(texts, labels, EMB, rank=32)
    s2 = train_student(texts, labels, EMB, rank=32)
    assert np.allclose(s1.a, s2.a) and np.allclose(s1.b, s2.b)
    assert s1.audit_threshold == s2.audit_threshold
    assert s1.fact_ids == s2.fact_ids
    assert s1.n_train == len(texts) == 192


def test_train_student_rank_and_param_count(tmp_path):
    records, queries = load_synthetic(CORPUS, QUERIES)
    index = build_fact_index(records, queries, list(range(1, 161)))
    texts, labels = build_distillation_examples(index, queries)
    student = train_student(texts, labels, EMB, rank=16)
    assert student.rank == 16
    assert student.a.shape == (384, 16)
    assert student.b.shape == (16, 48)
    assert student.n_params == 384 * 16 + 16 * 48
    # The student predicts a known fact for every training question.
    fact, conf, logits = student.predict(EMB, "Maya walnut loaf recipe hydration")
    assert fact in student.fact_ids
    assert logits.shape == (48,)


def test_student_predicts_its_training_questions(tmp_path):
    store, index, student, records, queries = _setup(tmp_path)
    try:
        keyword = [q for q in queries if q["kind"] == "keyword"]
        correct = 0
        for q in keyword:
            fact, _, _ = student.predict(EMB, q["question"])
            if fact == q["gold_keys"][0].rsplit("-", 1)[0]:
                correct += 1
        # Closed-form prototypes fit the training phrasing well
        # even under the lexical hashed embedder.
        assert correct >= len(keyword) // 2
    finally:
        store.close()


# --- Versioned artifact ----------------------------------------------------


def test_artifact_path_is_versioned_per_project():
    assert artifact_path("synthetic", 1, "/tmp") == "/tmp/lora_synthetic_v1.npz"
    assert artifact_path("synthetic", 2, "/tmp") == "/tmp/lora_synthetic_v2.npz"
    assert artifact_path("other-project", 1, "/tmp") != artifact_path(
        "synthetic", 1, "/tmp"
    )


def test_save_load_round_trip_preserves_predictions(tmp_path):
    store, index, student, records, queries = _setup(tmp_path)
    try:
        path = str(tmp_path / "lora_synthetic_v1.npz")
        assert save_student(student, path) == path
        loaded = load_student(path)
        assert loaded.project == student.project
        assert loaded.version == student.version
        assert loaded.rank == student.rank
        assert loaded.fact_set_hash == student.fact_set_hash
        assert loaded.audit_threshold == student.audit_threshold
        assert loaded.fact_ids == student.fact_ids
        for q in queries[:6]:
            assert student.predict(EMB, q["question"])[0] == loaded.predict(
                EMB, q["question"]
            )[0]
    finally:
        store.close()


def test_is_stale_after_correction_but_not_before(tmp_path):
    store, index, student, records, queries = _setup(tmp_path)
    try:
        assert not is_stale(student, index)
        index.add_fact("brand-new-fact", ["A brand new fact text."])
        assert is_stale(student, index)
    finally:
        store.close()


# --- Answering + verify-then-regenerate ------------------------------------


def test_student_alone_reads_nothing(tmp_path):
    store, index, student, records, queries = _setup(tmp_path)
    try:
        student.audit_threshold = -1e9  # force the gated audit off
        ans = answer_student(
            store, EMB, student, queries[0]["question"], index, audit="gated"
        )
        assert ans["records_read"] == 0
        assert ans["tokens_read"] == 0
        assert not ans["audited"]
        assert ans["final_fact"] == ans["student_fact"]
    finally:
        store.close()


def test_always_audit_reads_five_records(tmp_path):
    store, index, student, records, queries = _setup(tmp_path)
    try:
        ans = answer_student(
            store, EMB, student, queries[0]["question"], index, audit="always"
        )
        assert ans["audited"]
        assert ans["records_read"] == 5
        assert ans["tokens_read"] > 0
    finally:
        store.close()


def test_unknown_audit_policy_raises(tmp_path):
    store, index, student, records, queries = _setup(tmp_path)
    try:
        with pytest.raises(ValueError):
            answer_student(
                store, EMB, student, queries[0]["question"], index,
                audit="sometimes",
            )
    finally:
        store.close()


def test_teacher_answer_is_hybrid_top1(tmp_path):
    store, index, student, records, queries = _setup(tmp_path)
    try:
        ans = teacher_answer(store, EMB, queries[0]["question"], index)
        assert ans["records_read"] == 10
        assert ans["tokens_read"] > 0
        assert ans["text"]
    finally:
        store.close()


def test_audit_answer_returns_vector_leg_facts(tmp_path):
    store, index, student, records, queries = _setup(tmp_path)
    try:
        aud = audit_answer(store, EMB, queries[0]["question"], index, k=5)
        assert aud["records_read"] == 5
        assert len(aud["facts"]) == 5
        assert all(f is None or f in index.fact_texts for f in aud["facts"])
    finally:
        store.close()


# --- Eval + planted conflict -----------------------------------------------


def test_lora_eval_arms_and_read_cost_ordering():
    result = run_lora_eval(CORPUS, QUERIES, embedder=EMB, verbose=False)
    assert result["n_questions"] == 56
    assert result["n_facts"] == 48
    assert set(result["arms"]) == {
        "retrieval",
        "student",
        "student+verify-always",
        "student+verify-gated",
    }
    for arm in result["arms"].values():
        assert 0.0 <= arm["accuracy"] <= 1.0
        assert arm["correct"] <= arm["n"] == 56
    # Read cost: student reads nothing; the gated audit reads less
    # than retrieval; always-audit reads a fixed 5 per query.
    assert result["arms"]["student"]["records_per_query"] == 0
    assert (
        result["arms"]["student+verify-gated"]["records_per_query"]
        < result["arms"]["retrieval"]["records_per_query"]
    )
    assert result["arms"]["student+verify-always"]["records_per_query"] == 5.0
    assert result["arms"]["retrieval"]["records_per_query"] == 10.0


def test_planted_conflict_is_caught_and_regenerated():
    result = run_planted_conflict(CORPUS, QUERIES, embedder=EMB)
    assert result["student_fact"] == PLANTED_FACT
    assert result["superseded_records"] == 3
    assert result["artifact_stale_after_correction"]
    assert result["audited"]
    assert result["conflict_caught"]
    assert result["regenerated"]
    assert result["correct_regeneration"]
    assert result["final_fact"] == PLANTED_CORRECTED_FACT
    assert PLANTED_CORRECTED_MARKER in result["final_text"]
    assert PLANTED_CORRECTED_TEXTS[0] in result["final_text"] or (
        PLANTED_CORRECTED_MARKER in result["final_text"]
    )
