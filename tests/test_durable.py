"""P22 — Durable agent loop on real DBOS (3.6.0, SQLite system DB).

Tests: idempotent step helpers (no DBOS), in-process workflow
runs, the SIGKILL-and-resume demo, and the approval pause
surviving a restart. DBOS is a process-global singleton: every
test that configures it destroys it afterwards, and each test
uses a fresh temp workspace (fresh system DB).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from hybrid_retriever import durable
from hybrid_retriever.durable import (
    append_trace,
    approve,
    assert_no_duplicate_side_effects,
    checkpointed_step_count,
    complete_task_once,
    configure,
    create_task_once,
    load_trace,
    remember_once,
    run_approval_demo,
    run_demo,
    run_durable_task,
    seed_once,
    shutdown,
    start_run,
    verify_durable_run,
    wait_for_status,
    workflow_statuses,
)
from hybrid_retriever.tobench import (
    build_tobench_tasks,
    build_servers,
    run_grounded,
    verify_task,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _task(task_id: str) -> dict:
    return next(t for t in build_tobench_tasks() if t["id"] == task_id)


def _bundle(root: str) -> dict:
    return build_servers(root)


# ----------------------------------------------------------------------
# Idempotent helpers: no DBOS needed.
# ----------------------------------------------------------------------


def test_remember_once_is_idempotent(tmp_path):
    bundle = _bundle(str(tmp_path))
    try:
        first = remember_once(bundle, "the user prefers rust")
        second = remember_once(bundle, "the user prefers rust")
        assert first["id"] == second["id"]
        live = bundle["servers"]["memory"].call_tool("list_records")["result"]
        assert sum(1 for r in live if r["text"] == "the user prefers rust") == 1
    finally:
        bundle["store"].close()


def test_remember_once_correction_supersedes_once(tmp_path):
    bundle = _bundle(str(tmp_path))
    try:
        wrong = remember_once(bundle, "the user prefers python")
        first = remember_once(
            bundle, "the user prefers rust", "correction", wrong["id"]
        )
        second = remember_once(
            bundle, "the user prefers rust", "correction", wrong["id"]
        )
        assert first["id"] == second["id"]
        live = bundle["servers"]["memory"].call_tool("list_records")["result"]
        assert [r["text"] for r in live] == ["the user prefers rust"]
    finally:
        bundle["store"].close()


def test_create_task_once_is_idempotent(tmp_path):
    bundle = _bundle(str(tmp_path))
    try:
        first = create_task_once(bundle, "prepare release notes")
        second = create_task_once(bundle, "prepare release notes")
        assert first["id"] == second["id"]
        tasks = bundle["servers"]["tracker"].call_tool("list_tasks")["result"]
        assert len(tasks) == 1
    finally:
        bundle["store"].close()


def test_complete_task_once_is_idempotent(tmp_path):
    bundle = _bundle(str(tmp_path))
    try:
        tid = create_task_once(bundle, "do the thing")["id"]
        first = complete_task_once(bundle, tid)
        second = complete_task_once(bundle, tid)
        assert first == second == {"id": tid, "status": "completed"}
        tasks = bundle["servers"]["tracker"].call_tool("list_tasks")["result"]
        assert len(tasks) == 1 and tasks[0]["status"] == "completed"
    finally:
        bundle["store"].close()


def test_seed_once_is_idempotent(tmp_path):
    root = str(tmp_path)
    task = _task("D1")
    first = seed_once(root, task)
    second = seed_once(root, task)
    assert first == second
    bundle = _bundle(root)
    try:
        live = bundle["servers"]["memory"].call_tool("list_records")["result"]
        for text in task["setup"]["memories"]:
            assert sum(1 for r in live if r["text"] == text) == 1
        tasks = bundle["servers"]["tracker"].call_tool("list_tasks")["result"]
        assert len(tasks) == 1
    finally:
        bundle["store"].close()


def test_trace_dedupes_repeated_step_id(tmp_path):
    root = str(tmp_path)
    append_trace(root, "memory", "recall", {"query": "rust", "k": 5})
    append_trace(root, "memory", "recall", {"query": "rust", "k": 5})
    append_trace(root, "files", "write_file", {"path": "a.md"})
    trace = load_trace(root)
    assert len(trace) == 2
    assert trace.events[0] == {
        "server": "memory", "tool": "recall",
        "args": {"query": "rust", "k": 5},
    }
    assert trace.events[1]["tool"] == "write_file"


# ----------------------------------------------------------------------
# In-process DBOS runs (no kill): one task per kind.
# ----------------------------------------------------------------------


@pytest.fixture
def dbos_root(tmp_path):
    root = str(tmp_path)
    configure(root)
    yield root
    shutdown()


@pytest.mark.parametrize("task_id", ["S1", "C1", "M1", "D1"])
def test_workflow_completes_and_verifies(dbos_root, task_id):
    handle = start_run(task_id, dbos_root)
    result = handle.get_result()
    assert result["task_id"] == task_id
    statuses = workflow_statuses(dbos_root)
    assert len(statuses) == 1 and statuses[0]["status"] == "SUCCESS"
    verification = verify_durable_run(dbos_root, task_id)
    assert verification["passed"], verification
    assert assert_no_duplicate_side_effects(dbos_root, task_id) == []


def test_workflow_result_matches_plain_grounded(dbos_root):
    """Same task, same artifact bytes as the P21 grounded policy."""
    handle = start_run("D1", dbos_root)
    handle.get_result()
    task = _task("D1")
    with tempfile.TemporaryDirectory(prefix="durable-oracle-") as oroot:
        from hybrid_retriever.tobench import _seed_env
        bundle = _seed_env(task, oroot)
        try:
            run_grounded(task, bundle)
        finally:
            bundle["store"].close()
        for name in ("release-notes.md", "status.md"):
            with open(os.path.join(oroot, name), "rb") as fh:
                expected = fh.read()
            with open(os.path.join(dbos_root, name), "rb") as fh:
                assert fh.read() == expected, name


# ----------------------------------------------------------------------
# Kill-and-resume: real SIGKILL in a subprocess, fresh-process recovery.
# ----------------------------------------------------------------------


def test_kill_and_resume_no_duplicate_side_effects(tmp_path):
    root = str(tmp_path)
    report = run_demo(task_id="D1", step_delay=1.0,
                      kill_after_steps=2, workdir=root)
    assert report["verification_passed"], report["verification"]
    assert report["violations"] == [], report["violations"]
    at_kill = report["at_kill"]
    assert any(s["status"] == "PENDING" for s in at_kill["statuses"])
    assert at_kill["checkpointed_steps"] >= 2
    assert any(s["status"] == "SUCCESS" for s in report["final_statuses"])


def test_kill_before_first_checkpoint(tmp_path):
    """Kill during the FIRST step's delay: no step output is
    checkpointed yet. What happens next depends on whether the
    queue executor had dequeued the workflow yet — PENDING means
    launch auto-recovers it to a clean SUCCESS; ENQUEUED means
    recovery finds nothing and the run stays orphaned, but with
    zero side effects either way. Both outcomes are asserted
    here; see the README honest reading for the edge.
    """
    root = str(tmp_path)
    log_path = os.path.join(root, "run1.log")
    proc = durable._spawn([
        "run", "--task", "D1", "--root", root, "--step-delay", "5.0",
    ], log_path)
    try:
        time.sleep(2.0)  # inside the seed step's delay
        assert proc.poll() is None
        proc.kill()
        proc.wait()
        assert checkpointed_step_count(root) == 0
        assert not os.path.exists(os.path.join(root, "trace.jsonl"))
        assert not os.path.exists(os.path.join(root, "seed.json"))
        statuses = workflow_statuses(root)
        assert len(statuses) == 1, statuses
        status = statuses[0]["status"]
        assert status in ("ENQUEUED", "PENDING"), statuses
        configure(root)
        try:
            if status == "PENDING":
                # launch() auto-recovers; wait for it to finish.
                deadline = time.time() + 60.0
                final = None
                while time.time() < deadline:
                    cur = workflow_statuses(root)
                    if cur and cur[0]["status"] in ("SUCCESS", "ERROR"):
                        final = cur[0]["status"]
                        break
                    time.sleep(0.5)
                assert final == "SUCCESS", workflow_statuses(root)
                assert verify_durable_run(root, "D1")["passed"]
                assert assert_no_duplicate_side_effects(root, "D1") == []
            else:
                handles = durable.DBOS.recover_workflows(["local"])
                assert handles == []
                assert checkpointed_step_count(root) == 0
                assert not os.path.exists(os.path.join(root, "trace.jsonl"))
                assert not os.path.exists(os.path.join(root, "seed.json"))
        finally:
            shutdown()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_approval_pause_survives_restart(tmp_path):
    root = str(tmp_path)
    report = run_approval_demo(
        task_id="D1", step_delay=1.0, approval_poll_sec=2.0, workdir=root
    )
    assert report["verification_passed"], report["verification"]
    assert report["violations"] == [], report["violations"]
    assert report["final_status"] == "SUCCESS"
    assert report["mid_restart"]["tracker_open"] is True
    assert any(
        s["status"] == "PENDING" for s in report["mid_restart"]["statuses"]
    )


def test_approve_then_complete_without_kill(dbos_root):
    """The gate releases on the flag with no crash involved."""
    root = dbos_root
    result: dict = {}

    def _run():
        handle = start_run("D1", root, require_approval=True,
                           step_delay=0.0, approval_poll_sec=1.0)
        result["handle"] = handle
        result["out"] = handle.get_result()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    deadline = time.time() + 30.0
    artifact = os.path.join(root, "release-notes.md")
    while time.time() < deadline:
        tasks = durable._tracker_state(root)
        if os.path.exists(artifact) and tasks and all(
            t["status"] == "open" for t in tasks
        ):
            break
        time.sleep(0.2)
    assert os.path.exists(artifact), "run never reached the gate"
    tasks = durable._tracker_state(root)
    assert all(t["status"] == "open" for t in tasks)
    approve(root)
    thread.join(timeout=30.0)
    assert not thread.is_alive()
    assert result["out"]["task_id"] == "D1"
    assert verify_durable_run(root, "D1")["passed"]
    assert assert_no_duplicate_side_effects(root, "D1") == []
