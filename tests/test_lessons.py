"""Tests for P5 IBM-style lessons miner: extraction, dedup/merge
with provenance, guidelines file round-trip, retrieval, A/B arms,
and the gate (backfire rejection + good-candidate keep)."""

import os

import pytest

from hybrid_retriever.gepa import TraceStore, run_skill_eval, skill_version
from hybrid_retriever.lessons import (
    Guideline,
    _stem,
    ab_test_guideline,
    extract_lessons,
    merge_lessons,
    mine_from_eval,
    mine_from_trace_store,
    planted_backfire_guideline,
    read_guidelines,
    retrieve_guidelines,
    run_lessons_demo,
    run_lessons_eval,
    write_guidelines,
)
from hybrid_retriever.tobench import build_tobench_tasks


def _mined_guidelines():
    """Guidelines mined from the no-rule (sloppy) run: 27 lessons -> 4."""
    mined = mine_from_eval(set())
    return merge_lessons(mined["lessons"]), mined


def _log_eval_to_store(store, session_id, result, rules):
    tasks = build_tobench_tasks()
    version = skill_version(sorted(rules))
    for task, verdict in zip(tasks, result["per_task"]):
        store.log_run(session_id, task, verdict.pop("trace_events"),
                      verdict, "skill-agent", version)


# --- Phase 1: extraction --------------------------------------------------


def test_failed_episode_yields_up_to_3_lessons_naming_check_and_fix():
    result = run_skill_eval(set())
    tasks = build_tobench_tasks()
    by_id = {t["id"]: (t, v) for t, v in zip(tasks, result["per_task"])}
    task, verdict = by_id["S1"]
    lessons = extract_lessons(task, verdict)
    assert len(lessons) == 3  # S1 fails exactly 3 checks
    assert all(le.polarity == "failed" for le in lessons)
    assert {le.check for le in lessons} == {
        "recall_before_write", "gold_facts_present", "no_forbidden_text",
    }
    # Every lesson names its fix: the rule text for a rule-mapped check.
    assert all("recall" in le.text.lower() for le in lessons)
    assert all(le.rule_id == "recall-before-write" for le in lessons)
    assert all(le.evidence for le in lessons)  # verifier detail kept
    assert len({le.lesson_id for le in lessons}) == 3


def test_passed_episode_yields_one_worked_lesson_naming_a_real_rule():
    result = run_skill_eval(
        {"recall-before-write", "supersede-on-correct",
         "dedup-before-sum", "recall-deliver-in-order"})
    assert result["passed"] == 12
    tasks = build_tobench_tasks()
    for task, verdict in zip(tasks, result["per_task"]):
        lessons = extract_lessons(task, verdict)
        if task["id"] == "S3":
            # Prompt-contained: no primary check in its verdict, so
            # no worked lesson — see test_prompt_contained_episode.
            assert lessons == []
            continue
        assert len(lessons) == 1
        assert lessons[0].polarity == "worked"
        assert lessons[0].text.startswith("What worked: ")
        assert lessons[0].rule_id is not None


def test_prompt_contained_episode_yields_zero_lessons():
    """S3 passes with no recall and no rule-mapped check in its
    verdict — there is nothing attributable to learn from it, so
    the miner emits nothing instead of inventing a lesson."""
    result = run_skill_eval(set())
    tasks = build_tobench_tasks()
    task = next(t for t in tasks if t["id"] == "S3")
    verdict = next(v for t, v in zip(tasks, result["per_task"])
                   if t["id"] == "S3")
    assert verdict["passed"]
    assert extract_lessons(task, verdict) == []


# --- Phase 2: merge + provenance ------------------------------------------


def test_merge_counts_support_and_provenance():
    guidelines, _mined = _mined_guidelines()
    assert len(guidelines) == 4
    by_rule = {g.rule_id: g for g in guidelines}
    assert by_rule["supersede-on-correct"].support == 9
    assert by_rule["supersede-on-correct"].provenance == ["C1", "C2", "C3"]
    assert by_rule["recall-deliver-in-order"].support == 9
    assert by_rule["recall-deliver-in-order"].provenance == ["D1", "D2", "D3"]
    assert by_rule["recall-before-write"].support == 6
    assert by_rule["recall-before-write"].provenance == ["S1", "S2"]
    assert by_rule["dedup-before-sum"].support == 3
    assert by_rule["dedup-before-sum"].provenance == ["M1", "M2", "M3"]
    # The checks that produced each guideline survive the merge too.
    assert by_rule["supersede-on-correct"].checks == [
        "correction_text", "old_record_superseded", "recall_before_remember",
    ]


def test_merge_preserves_provenance():
    """The paper's failure mode: provenance silently dropped in
    consolidation. Merge in two stages and through a second
    merge with a new episode — every episode id must survive."""
    mined = mine_from_eval(set())["lessons"]
    first_half = [le for le in mined if le.episode_id in ("C1", "C2")]
    stage1 = merge_lessons(first_half)
    assert len(stage1) == 1
    assert stage1[0].provenance == ["C1", "C2"]
    assert stage1[0].support == 6
    # Second merge: the same lessons plus a fresh episode's lesson.
    extra = [le for le in mined if le.episode_id == "C3"]
    stage2 = merge_lessons(first_half + extra)
    assert len(stage2) == 1
    assert stage2[0].provenance == ["C1", "C2", "C3"]
    assert stage2[0].support == 9
    # And merging the union never drops an episode seen earlier.
    assert set(stage1[0].provenance) <= set(stage2[0].provenance)


def test_worked_lessons_merge_separately_from_failed():
    mined = mine_from_eval(
        {"recall-before-write", "supersede-on-correct",
         "dedup-before-sum", "recall-deliver-in-order"})
    assert len(mined["lessons"]) == 11  # all but S3 (no primary check)
    guidelines = merge_lessons(mined["lessons"])
    assert len(guidelines) == 4
    assert all(g.polarity == "worked" for g in guidelines)
    by_rule = {g.rule_id: g for g in guidelines}
    assert by_rule["recall-before-write"].support == 2  # S1, S2 only
    assert by_rule["supersede-on-correct"].support == 3
    assert by_rule["dedup-before-sum"].support == 3
    assert by_rule["recall-deliver-in-order"].support == 3
    assert sorted(e for g in guidelines for e in g.provenance) == sorted(
        t["id"] for t in build_tobench_tasks() if t["id"] != "S3"
    )


def test_guidelines_file_round_trip(tmp_path):
    guidelines, _mined = _mined_guidelines()
    path = str(tmp_path / "guidelines.md")
    write_guidelines(path, guidelines)
    assert os.path.exists(path)
    back = read_guidelines(path)
    assert len(back) == len(guidelines)
    for orig, rt in zip(guidelines, back):
        assert rt.guideline_id == orig.guideline_id
        assert rt.text == orig.text
        assert rt.kind == orig.kind
        assert rt.support == orig.support
        assert rt.provenance == orig.provenance
        assert rt.checks == orig.checks
        assert rt.rule_id == orig.rule_id
        assert rt.polarity == orig.polarity
        assert rt.planted is False
    # The file itself names the provenance: grep-able by a human.
    content = open(path, encoding="utf-8").read()
    assert "episodes: C1, C2, C3" in content
    assert "support: 9" in content


# --- Retrieval ------------------------------------------------------------


def test_stemmer_equates_word_forms():
    assert _stem("deduplicating") == _stem("deduplicate")
    assert _stem("totalling") == _stem("total")
    assert _stem("tasks") == _stem("task")
    assert _stem("memories") == _stem("memory")
    assert _stem("writing") == _stem("write")


def test_retrieval_picks_the_kind_rule_for_each_task():
    guidelines, _mined = _mined_guidelines()
    expected = {
        "summarize": "recall-before-write",
        "correct": "supersede-on-correct",
        "compute": "dedup-before-sum",
        "deliver": "recall-deliver-in-order",
    }
    for task in build_tobench_tasks():
        if task["id"] == "S3":
            continue  # prompt-contained: no rule needed, any pick is fine
        text = f"{task['goal']} {task['context'].get('query', '')}"
        top = retrieve_guidelines(guidelines, text, k=1)
        assert top, f"nothing retrieved for {task['id']}"
        assert top[0].rule_id == expected[task["kind"]], task["id"]


def test_retrieval_returns_nothing_with_no_overlap():
    guidelines, _mined = _mined_guidelines()
    assert retrieve_guidelines(guidelines, "zebra xylophone quasar", k=2) == []


# --- Phase 3: injection arms ----------------------------------------------


def test_eval_arms_none_retrieved_full():
    guidelines, _mined = _mined_guidelines()
    none = run_lessons_eval(guidelines, "none")
    full = run_lessons_eval(guidelines, "full")
    ret1 = run_lessons_eval(guidelines, "retrieved", k=1)
    ret2 = run_lessons_eval(guidelines, "retrieved", k=2)
    assert none["passed"] == 1
    assert full["passed"] == 12
    assert ret1["passed"] == 12
    assert ret2["passed"] == 12
    # The subset matches full injection at a fraction of the cost.
    assert none["injected_tokens"] == 0
    assert 0 < ret1["injected_tokens"] < ret2["injected_tokens"]
    assert ret2["injected_tokens"] < full["injected_tokens"]
    assert ret1["injected_tokens"] <= full["injected_tokens"] // 3


# --- The A/B gate -----------------------------------------------------------


def test_backfire_guideline_is_rejected_despite_high_support():
    """The done condition's second half: a guideline that
    contradicts a verifier rule is caught before it ships, even
    though its support (12) is higher than any real guideline's."""
    guidelines, _mined = _mined_guidelines()
    backfire = planted_backfire_guideline()
    assert backfire.planted
    assert backfire.support > max(g.support for g in guidelines)
    assert backfire.provenance == ["planted-negative"]
    trial = ab_test_guideline(guidelines, backfire, "deliver")
    assert trial["win"] is False
    assert trial["verdict"] == "rejected"
    assert trial["control_passed"] == 12
    assert trial["treatment_passed"] == 9
    assert trial["fixed_tasks"] == []
    assert trial["regressed_tasks"] == ["D1", "D2", "D3"]


def test_good_candidate_is_kept_and_duplicate_is_rejected():
    """The gate must be able to say yes as well as no — and a
    candidate that adds nothing (its rule is already injected)
    is not a win, because the treatment is not strictly better."""
    guidelines, _mined = _mined_guidelines()
    first = guidelines[0]
    good = ab_test_guideline([], first, first.kind)
    assert good["win"] is True
    assert good["verdict"] == "kept"
    assert good["control_passed"] == 1
    assert good["treatment_passed"] == 4
    assert good["regressed_tasks"] == []
    duplicate = Guideline(
        guideline_id="GX-dup", text=first.text, kind=first.kind,
        check=first.check, checks=list(first.checks),
        polarity=first.polarity, rule_id=first.rule_id,
        support=1, provenance=["C1"], lesson_ids=["dup"],
    )
    dup_trial = ab_test_guideline(guidelines, duplicate, first.kind)
    assert dup_trial["win"] is False
    assert dup_trial["verdict"] == "rejected"
    assert dup_trial["control_passed"] == dup_trial["treatment_passed"] == 12


# --- Trace-store mining -----------------------------------------------------


def test_mine_from_trace_store_agrees_on_failure_lessons(tmp_path):
    store = TraceStore(str(tmp_path / "traces.db"))
    result = run_skill_eval(set())
    sid = store.new_session("skill-agent", skill_version([]), notes="t")
    _log_eval_to_store(store, sid, result, set())
    in_memory = mine_from_eval(set())["lessons"]
    from_store = mine_from_trace_store(store, sid)
    mem_fail = sorted((le.episode_id, le.check) for le in in_memory
                      if le.polarity == "failed")
    store_fail = sorted((le.episode_id, le.check) for le in from_store
                        if le.polarity == "failed")
    assert mem_fail == store_fail
    # The store path yields one extra lesson: S3 passed, so its
    # worked lesson is inferred from the kind's primary check
    # (in-memory extraction finds no such check in S3's verdict).
    assert len(from_store) == len(in_memory) + 1
    assert any(le.episode_id == "S3" and le.polarity == "worked"
               for le in from_store)
    store.close()


# --- End to end -------------------------------------------------------------


def test_lessons_demo_end_to_end():
    result = run_lessons_demo()
    assert result["mining_agrees"] is True
    assert result["n_lessons"] == 27
    assert len(result["guidelines"]) == 4
    assert result["eval_none"]["passed"] == 1
    assert result["eval_full"]["passed"] == 12
    assert result["eval_retrieved"]["passed"] == 12
    assert result["eval_retrieved_k1"]["passed"] == 12
    assert result["backfire_trial"]["verdict"] == "rejected"
    assert result["backfire_trial"]["regressed_tasks"] == ["D1", "D2", "D3"]
    assert result["good_trial"]["verdict"] == "kept"
    # The written file round-trips with provenance intact.
    assert len(result["round_tripped"]) == 4
    assert result["round_tripped"][0].provenance
