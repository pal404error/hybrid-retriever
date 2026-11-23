"""Tests for P20 — Verify-by-Doing harness.

Uses HashedEmbedder (no torch) so the tests run anywhere. The
measured leg comparison lives in the full eval run (ONNX MiniLM),
not here: these tests pin the harness mechanics — traps, solver,
verifier independence, lift math, determinism.
"""

from __future__ import annotations

import json
import os

import pytest

from hybrid_retriever.embeddings import HashedEmbedder
from hybrid_retriever.verify import (
    RULE_MEMORIES,
    RULES,
    _paired_lift,
    build_vbd_tasks,
    expected_answer,
    matching_rule,
    run_vbd_eval,
    solve_task,
    verify_artifact,
    write_answer_artifact,
)

CORPUS = os.path.join(
    os.path.dirname(__file__), "..", "eval_data", "synthetic_corpus.jsonl"
)


def test_tasks_are_traps_and_verifier_agrees_with_generator():
    tasks = build_vbd_tasks(n_tasks=24)
    assert len(tasks) == 24
    kinds = [t["kind"] for t in tasks]
    assert kinds.count("json") == 8
    assert kinds.count("log") == 8
    assert kinds.count("table") == 8
    for task in tasks:
        # The trap must change the answer, or the task measures nothing.
        assert task["ground_truth"] != task["naive_answer"]
        # The verifier recomputes from the raw artifact, independently
        # of the generator's stored ground truth — they must agree.
        assert expected_answer(task) == task["ground_truth"]


def test_task_generation_is_deterministic():
    a = build_vbd_tasks(n_tasks=24, seed=7)
    b = build_vbd_tasks(n_tasks=24, seed=7)
    assert [t["artifact"] for t in a] == [t["artifact"] for t in b]
    assert [t["ground_truth"] for t in a] == [t["ground_truth"] for t in b]


def test_matching_rule_triggers():
    json_rules = {r["key"]: r for r in RULES["json"]}
    assert matching_rule("json", RULE_MEMORIES["json-dedup-by-id"]) is json_rules["json-dedup-by-id"]
    assert matching_rule("json", RULE_MEMORIES["json-dedup-by-note"]) is json_rules["json-dedup-by-note"]
    assert matching_rule("json", "the cat sits on the mat") is None
    # A memory about another kind's rule does not fire here.
    assert matching_rule("json", RULE_MEMORIES["table-dedup-rows"]) is None
    # "deduplicate" alone is not enough: all trigger words are required.
    assert matching_rule("json", "please deduplicate the results") is None


def test_solver_no_memory_fails_every_task(tmp_path):
    tasks = build_vbd_tasks(n_tasks=24)
    for task in tasks:
        result = solve_task(task, [])
        assert result["answer"] == task["naive_answer"]
        assert result["used_rule"] is None
        path = write_answer_artifact(str(tmp_path), "no-memory", result)
        verdict = verify_artifact(path, task)
        assert verdict["passed"] is False
        assert verdict["got"] == task["naive_answer"]
        assert verdict["expected"] == task["ground_truth"]


def test_solver_oracle_passes_every_task(tmp_path):
    tasks = build_vbd_tasks(n_tasks=24)
    for task in tasks:
        result = solve_task(task, [RULE_MEMORIES[task["rule_key"]]])
        assert result["used_rule"] == task["rule_key"]
        path = write_answer_artifact(str(tmp_path), "oracle", result)
        verdict = verify_artifact(path, task)
        assert verdict["passed"] is True, task["task_id"]


def test_distractor_rules_fire_and_fail(tmp_path):
    tasks = build_vbd_tasks(n_tasks=24)
    for task in tasks:
        for rule in RULES[task["kind"]]:
            if rule["correct"]:
                continue
            result = solve_task(task, [RULE_MEMORIES[rule["key"]]])
            assert result["used_rule"] == rule["key"]
            path = write_answer_artifact(str(tmp_path), "distractor", result)
            verdict = verify_artifact(path, task)
            assert verdict["passed"] is False, (
                task["task_id"],
                rule["key"],
                result["answer"],
            )


def test_first_firing_rule_wins():
    tasks = build_vbd_tasks(n_tasks=3)
    task = tasks[0]  # json
    correct = RULE_MEMORIES[task["rule_key"]]
    wrong = RULE_MEMORIES["json-include-all"]
    assert solve_task(task, [correct, wrong])["used_rule"] == task["rule_key"]
    assert solve_task(task, [wrong, correct])["used_rule"] == "json-include-all"
    # A non-firing memory is skipped, the next firing one applies.
    assert solve_task(task, ["the cat sits on the mat", correct])["used_rule"] == task["rule_key"]


def test_verifier_rejects_wrong_value_missing_and_malformed(tmp_path):
    task = build_vbd_tasks(n_tasks=1)[0]
    good = solve_task(task, [RULE_MEMORIES[task["rule_key"]]])
    path = write_answer_artifact(str(tmp_path), "oracle", good)
    assert verify_artifact(path, task)["passed"] is True

    tampered = dict(good, answer=good["answer"] + 1)
    bad_path = os.path.join(str(tmp_path), "tampered.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump(tampered, f)
    verdict = verify_artifact(bad_path, task)
    assert verdict["passed"] is False
    assert verdict["reason"] == "wrong value"

    assert verify_artifact(
        os.path.join(str(tmp_path), "does-not-exist.json"), task
    )["reason"] == "no answer artifact"

    broken = os.path.join(str(tmp_path), "broken.json")
    with open(broken, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert "unreadable" in verify_artifact(broken, task)["reason"]

    no_answer = os.path.join(str(tmp_path), "no-answer.json")
    with open(no_answer, "w", encoding="utf-8") as f:
        json.dump({"task_id": task["task_id"]}, f)
    assert verify_artifact(no_answer, task)["reason"] == "answer missing or not a number"

    wrong_task = os.path.join(str(tmp_path), "wrong-task.json")
    with open(wrong_task, "w", encoding="utf-8") as f:
        json.dump({"task_id": "other-99", "answer": task["ground_truth"]}, f)
    assert verify_artifact(wrong_task, task)["reason"] == "artifact is for another task"


def test_verifier_accepts_float_tolerance(tmp_path):
    task = build_vbd_tasks(n_tasks=1)[0]
    good = solve_task(task, [RULE_MEMORIES[task["rule_key"]]])
    near = dict(good, answer=float(good["answer"]) + 1e-7)
    path = os.path.join(str(tmp_path), "near.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(near, f)
    assert verify_artifact(path, task)["passed"] is True
    far = dict(good, answer=float(good["answer"]) + 0.01)
    path2 = os.path.join(str(tmp_path), "far.json")
    with open(path2, "w", encoding="utf-8") as f:
        json.dump(far, f)
    assert verify_artifact(path2, task)["passed"] is False


def test_paired_lift_math():
    # Arm wins 6 tasks the baseline loses, loses none (n=8): lift 0.75.
    lift = _paired_lift(
        [True] * 6 + [False] * 2, [False] * 8
    )
    assert lift["lift"] == pytest.approx(0.75)
    assert lift["arm_only_wins"] == 6
    assert lift["base_only_wins"] == 0
    assert lift["outside_noise"] is True
    # Identical arms: lift exactly 0, no spread to measure.
    lift = _paired_lift([True, False], [True, False])
    assert lift["lift"] == 0.0
    assert lift["se"] == 0.0
    assert lift["outside_noise"] is False
    # One win each way: lift 0, but the disagreement is real spread.
    lift = _paired_lift([True, False], [False, True])
    assert lift["lift"] == 0.0
    assert lift["se"] > 0
    assert lift["outside_noise"] is False
    # Arm strictly worse on all 8 tasks: negative lift, outside noise.
    lift = _paired_lift([False] * 8, [True] * 8)
    assert lift["lift"] == pytest.approx(-1.0)
    assert lift["outside_noise"] is True


def test_run_vbd_eval_hashed_is_deterministic():
    emb = HashedEmbedder()
    a = run_vbd_eval(CORPUS, embedder=emb, n_tasks=24, verbose=False)
    b = run_vbd_eval(CORPUS, embedder=emb, n_tasks=24, verbose=False)
    for name in a["arms"]:
        assert a["arms"][name]["successes"] == b["arms"][name]["successes"]
        assert a["arms"][name]["per_task"] == b["arms"][name]["per_task"]
    assert a["lifts_vs_no_memory"] == b["lifts_vs_no_memory"]
    assert a["ablations"] == b["ablations"]


def test_run_vbd_eval_extremes_and_ablation_monotonic():
    result = run_vbd_eval(
        CORPUS, embedder=HashedEmbedder(), n_tasks=24, verbose=False
    )
    assert set(result["arms"]) == {"no-memory", "bm25", "vector", "hybrid", "oracle"}
    # Baseline: naive solver fails every trap by construction.
    assert result["arms"]["no-memory"]["successes"] == 0
    # Ceiling: the correct rule, handed directly, solves every task.
    assert result["arms"]["oracle"]["successes"] == 24
    # Retrieval arms sit between baseline and ceiling.
    for name in ("bm25", "vector", "hybrid"):
        assert 0 <= result["arms"][name]["successes"] <= 24
    # Adoption gate: removing a correct rule memory can never help
    # the hybrid arm — the tasks it was solving now fall to a
    # distractor or to the naive computation, both of which fail.
    hybrid_n = result["arms"]["hybrid"]["successes"]
    assert set(result["ablations"]) == {
        "json-dedup-by-id",
        "log-distinct-only",
        "table-dedup-rows",
    }
    for key, ab in result["ablations"].items():
        assert ab["successes"] <= hybrid_n
        assert ab["causal_lift"]["lift"] >= 0
    # Per-kind rates are consistent with the arm totals.
    for kind, rates in result["per_kind"].items():
        assert set(rates) == set(result["arms"])
        for rate in rates.values():
            assert 0.0 <= rate <= 1.0
