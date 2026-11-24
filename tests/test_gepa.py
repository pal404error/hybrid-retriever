"""Tests for P23 GEPA-style self-improving loop: trace store, diagnoser,
skill file, A/B gate, nightly loop, full demo."""

import os

import pytest

from hybrid_retriever.gepa import (
    RULES,
    ab_test,
    apply_rule,
    diagnose,
    load_skill_rules,
    propose_edit,
    remove_rule,
    render_skill,
    run_gepa_demo,
    run_nightly,
    run_skill_agent,
    run_skill_eval,
    skill_version,
    TraceStore,
)
from hybrid_retriever.tobench import build_tobench_tasks


def _log_eval(store, session_id, result, rules):
    """Log every task of a run_skill_eval result into the trace store."""
    tasks = build_tobench_tasks()
    version = skill_version(sorted(rules))
    for task, verdict in zip(tasks, result["per_task"]):
        store.log_run(session_id, task, verdict["trace_events"], verdict,
                      "skill-agent", version)


def _empty_store(tmp_path):
    return TraceStore(str(tmp_path / "traces.db"))


# --- trace store --------------------------------------------------------


def test_trace_store_logs_runs_failures_and_calls(tmp_path):
    store = _empty_store(tmp_path)
    result = run_skill_eval(set())
    sid = store.new_session("skill-agent", skill_version([]), notes="t")
    _log_eval(store, sid, result, set())
    failures = store.failures(sid)
    assert failures  # the empty skill fails 11 of 12 tasks
    assert all(f["check"] for f in failures)
    assert any(f["kind"] == "compute" and f["check"] == "value_matches"
               for f in failures)
    # Calls were logged, and only the agent's calls (no verifier reads).
    rows = store.conn.execute(
        "SELECT COUNT(*) FROM trace_calls"
        " JOIN traces t ON t.id = trace_calls.trace_id"
        " WHERE t.session_id = ?", (sid,)).fetchone()[0]
    assert rows > 0
    assert store.conn.execute(
        "SELECT COUNT(*) FROM trace_calls WHERE server = 'agent'"
    ).fetchone()[0] == 0
    store.close()


def test_trace_store_fts_searches_prompt_and_failures(tmp_path):
    store = _empty_store(tmp_path)
    result = run_skill_eval(set())
    sid = store.new_session("skill-agent", skill_version([]))
    _log_eval(store, sid, result, set())
    hits = store.search_failures("March", session_id=sid)
    assert hits and all("March" in h["prompt"] or "March" in h["failures"]
                        for h in hits)
    hits = store.search_failures("deduped", session_id=sid)
    assert hits and all("deduped" in h["failures"] for h in hits)
    assert store.search_failures("zzzznope", session_id=sid) == []
    store.close()


def test_skill_edits_audit_roundtrip(tmp_path):
    store = _empty_store(tmp_path)
    sid = store.new_session("skill-agent", "abc123")
    store.record_edit(sid, "dedup-before-sum", "compute/value_matches", 3,
                      1, 4, ["M1", "M2", "M3"], [], "kept")
    edits = store.list_edits()
    assert len(edits) == 1
    assert edits[0]["rule_id"] == "dedup-before-sum"
    assert edits[0]["fixed_tasks"] == ["M1", "M2", "M3"]
    assert edits[0]["verdict"] == "kept"
    store.close()


# --- skill file ---------------------------------------------------------


def test_skill_render_load_apply_remove_roundtrip(tmp_path):
    path = str(tmp_path / "SKILL.md")
    assert load_skill_rules(path) == []
    assert apply_rule(path, "dedup-before-sum") is True
    assert load_skill_rules(path) == ["dedup-before-sum"]
    assert apply_rule(path, "dedup-before-sum") is False  # idempotent
    assert load_skill_rules(path) == ["dedup-before-sum"]
    assert "dedup-before-sum" in open(path).read()
    assert open(path).read().startswith("---\nname: tobench-agent")
    assert remove_rule(path, "dedup-before-sum") is True
    assert load_skill_rules(path) == []
    assert remove_rule(path, "dedup-before-sum") is False
    assert render_skill([]).count("(none yet") == 1


def test_skill_version_is_deterministic_and_order_free():
    assert skill_version(["a", "b"]) == skill_version(["b", "a"])
    assert skill_version(["a"]) != skill_version(["a", "b"])


# --- skill agent --------------------------------------------------------


def test_skill_agent_no_rules_matches_sloppy():
    result = run_skill_eval(set())
    assert result["passed"] == 1
    assert [t["task_id"] for t in result["per_task"] if t["passed"]] == ["S3"]


def test_skill_agent_all_rules_matches_grounded():
    rules = {"recall-before-write", "supersede-on-correct",
             "dedup-before-sum", "recall-deliver-in-order"}
    result = run_skill_eval(rules)
    assert result["passed"] == 12


@pytest.mark.parametrize(
    "rules, expected_passed, expected_ids",
    [
        ({"dedup-before-sum"}, 4, ["M1", "M2", "M3", "S3"]),
        ({"supersede-on-correct"}, 4, ["C1", "C2", "C3", "S3"]),
        ({"recall-before-write"}, 3, ["S1", "S2", "S3"]),
        ({"recall-deliver-in-order"}, 4, ["D1", "D2", "D3", "S3"]),
    ],
)
def test_each_rule_fixes_exactly_its_kind(rules, expected_passed, expected_ids):
    result = run_skill_eval(rules)
    assert result["passed"] == expected_passed
    assert sorted(t["task_id"] for t in result["per_task"] if t["passed"]) == expected_ids


def test_skill_agent_unknown_kind_raises(tmp_path):
    from hybrid_retriever.tobench import _seed_env
    task = {"id": "X1", "kind": "teleport", "goal": "go nowhere",
            "context": {}, "setup": {}}
    with pytest.raises(ValueError, match="unknown kind"):
        run_skill_agent(task, _seed_env(task, str(tmp_path)), set())


# --- diagnoser ----------------------------------------------------------


def test_diagnose_clusters_and_ranks_deterministically(tmp_path):
    store = _empty_store(tmp_path)
    result = run_skill_eval(set())
    sid = store.new_session("skill-agent", skill_version([]))
    _log_eval(store, sid, result, set())
    clusters = diagnose(store, sid)
    assert clusters
    # Top cluster: size 3, earliest kind in task order among the size-3
    # clusters (correct < compute < deliver); within the kind, the
    # alphabetically first check wins the tie.
    assert clusters[0]["size"] == 3
    assert clusters[0]["kind"] == "correct"
    assert clusters[0]["check"] == "correction_text"
    assert clusters[0]["rule_id"] == "supersede-on-correct"
    assert clusters[0]["task_ids"] == ["C1", "C2", "C3"]
    # Sizes are non-increasing down the ranking.
    sizes = [c["size"] for c in clusters]
    assert sizes == sorted(sizes, reverse=True)
    store.close()


def test_propose_skips_rules_already_in_skill(tmp_path):
    store = _empty_store(tmp_path)
    result = run_skill_eval(set())
    sid = store.new_session("skill-agent", skill_version([]))
    _log_eval(store, sid, result, set())
    proposal = propose_edit(store, sid, ["supersede-on-correct"])
    assert proposal is not None
    assert proposal["rule_id"] == "dedup-before-sum"  # next size-3 cluster
    assert propose_edit(store, sid, list(RULES)) is None
    store.close()


def test_propose_none_when_no_failures(tmp_path):
    store = _empty_store(tmp_path)
    rules = {"recall-before-write", "supersede-on-correct",
             "dedup-before-sum", "recall-deliver-in-order"}
    result = run_skill_eval(rules)
    sid = store.new_session("skill-agent", skill_version(sorted(rules)))
    _log_eval(store, sid, result, rules)
    assert diagnose(store, sid) == []
    assert propose_edit(store, sid, sorted(rules)) is None
    store.close()


# --- A/B gate -----------------------------------------------------------


def test_ab_gate_keeps_a_winning_edit():
    gate = ab_test(set(), "dedup-before-sum", "compute")
    assert gate["win"] is True
    assert gate["control_passed"] == 1
    assert gate["treatment_passed"] == 4
    assert gate["fixed_tasks"] == ["M1", "M2", "M3"]
    assert gate["regressed_tasks"] == []


def test_ab_gate_rejects_complete_first_after_full_skill():
    rules = {"recall-before-write", "supersede-on-correct",
             "dedup-before-sum", "recall-deliver-in-order"}
    gate = ab_test(rules, "complete-first", "deliver")
    assert gate["win"] is False
    assert gate["control_passed"] == 12
    assert gate["treatment_passed"] == 9
    assert gate["fixed_tasks"] == []
    assert gate["regressed_tasks"] == ["D1", "D2", "D3"]


def test_ab_gate_rejects_an_edit_that_changes_nothing():
    gate = ab_test(set(), "complete-first", "deliver")
    assert gate["win"] is False
    assert gate["control_passed"] == gate["treatment_passed"] == 1
    assert gate["fixed_tasks"] == []


# --- nightly loop + demo ------------------------------------------------


def test_one_night_keeps_the_top_cluster_rule(tmp_path):
    store = _empty_store(tmp_path)
    skill = str(tmp_path / "SKILL.md")
    with open(skill, "w", encoding="utf-8") as fh:
        fh.write(render_skill([]))
    night = run_nightly(store, skill)
    assert night["eval_passed"] == 1
    assert night["proposal"]["rule_id"] == "supersede-on-correct"
    assert night["verdict"] == "kept"
    assert night["ab"]["control_passed"] == 1
    assert night["ab"]["treatment_passed"] == 4
    assert load_skill_rules(skill) == ["supersede-on-correct"]
    assert len(store.list_edits()) == 1
    assert store.list_edits()[0]["verdict"] == "kept"
    store.close()


def test_full_demo_four_nights_four_kept_one_rejected(tmp_path):
    demo = run_gepa_demo(nights=4)
    assert [n["verdict"] for n in demo["nights"]] == ["kept"] * 4
    assert [n["proposal"]["rule_id"] for n in demo["nights"]] == [
        "supersede-on-correct",
        "dedup-before-sum",
        "recall-deliver-in-order",
        "recall-before-write",
    ]
    assert demo["final_rules"] == sorted(
        ["recall-before-write", "supersede-on-correct",
         "dedup-before-sum", "recall-deliver-in-order"])
    assert demo["final_eval"]["passed"] == 12
    assert len(demo["kept_edits"]) == 4
    assert len(demo["rejected_edits"]) == 1
    bad = demo["bad_edit_trial"]
    assert bad["candidate_rule"] == "complete-first"
    assert bad["verdict"] == "rejected"
    assert bad["control_passed"] == 12
    assert bad["treatment_passed"] == 9
    assert bad["regressed_tasks"] == ["D1", "D2", "D3"]
    # The demo wrote real files (skill + trace db exist on disk).
    assert os.path.exists(demo["skill_path"])
    assert os.path.exists(demo["traces_db"])
