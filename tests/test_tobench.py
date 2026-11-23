"""Tests for P21 mini-TOBench: servers, verifiers, agents, scoring."""

import json
import os
import subprocess
import sys

import pytest

from hybrid_retriever.tobench import (
    build_servers,
    build_tobench_tasks,
    run_tobench_eval,
    verify_task,
    _compute_from_text,
    _seed_env,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_for(task_id, tmp_path):
    task = next(t for t in build_tobench_tasks() if t["id"] == task_id)
    return task, _seed_env(task, str(tmp_path))


# --- servers ------------------------------------------------------------


def test_memory_recall_remember_supersedes(tmp_path):
    bundle = build_servers(str(tmp_path))
    mem = bundle["servers"]["memory"]
    rid = mem.call_tool("remember", {"text": "the user prefers python"})["result"]["id"]
    found = mem.call_tool("recall", {"query": "prefers"})["result"]
    assert any(r["id"] == rid for r in found)
    new = mem.call_tool(
        "remember",
        {"text": "the user prefers rust", "type": "correction", "supersedes": rid},
    )["result"]
    assert new["superseded"] == rid
    row = bundle["store"].conn.execute(
        "SELECT status, superseded_by FROM records WHERE id = ?", (rid,)
    ).fetchone()
    assert row == ("superseded", new["id"])
    assert mem.call_tool("recall", {"query": "prefers"})["result"][0]["text"] == "the user prefers rust"


def test_memory_rejects_supersede_of_non_live(tmp_path):
    bundle = build_servers(str(tmp_path))
    mem = bundle["servers"]["memory"]
    res = mem.call_tool("remember", {"text": "x", "supersedes": 999})
    assert res["ok"] is False
    assert "not a live record" in res["error"]


def test_unknown_tool_is_an_error_not_a_crash(tmp_path):
    bundle = build_servers(str(tmp_path))
    res = bundle["servers"]["memory"].call_tool("teleport", {})
    assert res["ok"] is False
    assert "unknown tool" in res["error"]


def test_files_server_rejects_escape(tmp_path):
    bundle = build_servers(str(tmp_path))
    files = bundle["servers"]["files"]
    res = files.call_tool("write_file", {"path": "../escape.txt", "content": "x"})
    assert res["ok"] is False
    assert "escapes workspace root" in res["error"]


def test_files_write_read_list(tmp_path):
    bundle = build_servers(str(tmp_path))
    files = bundle["servers"]["files"]
    assert files.call_tool("write_file", {"path": "a/b.txt", "content": "hello"})["ok"]
    assert files.call_tool("read_file", {"path": "a/b.txt"})["result"] == "hello"
    assert files.call_tool("list_files")["result"] == ["a/b.txt"]


def test_tracker_roundtrip_and_persistence(tmp_path):
    bundle = build_servers(str(tmp_path))
    tr = bundle["servers"]["tracker"]
    tid = tr.call_tool("create_task", {"title": "do the thing"})["result"]["id"]
    tr.call_tool("add_note", {"id": tid, "note": "started"})
    tr.call_tool("complete_task", {"id": tid})
    tasks = tr.call_tool("list_tasks")["result"]
    assert tasks[0]["status"] == "completed"
    assert tasks[0]["notes"] == ["started"]
    # The state is a real artifact on disk, not process memory.
    on_disk = json.loads((tmp_path / "tasks.json").read_text())
    assert on_disk["tasks"][0]["status"] == "completed"
    assert tr.call_tool("complete_task", {"id": 999})["ok"] is False


def test_trace_records_calls_in_order(tmp_path):
    bundle = build_servers(str(tmp_path))
    bundle["servers"]["memory"].call_tool("recall", {"query": "rust"})
    bundle["servers"]["files"].call_tool("write_file", {"path": "x.md", "content": "y"})
    trace = bundle["trace"]
    assert len(trace) == 2
    assert trace.first_index("memory", "recall") == 0
    assert trace.first_index("files", "write_file") == 1
    assert trace.has("memory", "recall") is True
    assert trace.has("tracker", "complete_task") is False


# --- verifiers ----------------------------------------------------------


def test_summarize_verifier_pass_and_fail(tmp_path):
    task, bundle = _env_for("S1", tmp_path)
    servers = bundle["servers"]
    servers["memory"].call_tool("recall", {"query": "quarterly planning", "k": 5})
    servers["files"].call_tool(
        "write_file",
        {"path": "summary.md", "content": "quarterly planning is on Thursday at 10am\nagenda covers hiring and the roadmap\n"},
    )
    assert verify_task(task, bundle)["passed"] is True

    _task2, bundle2 = _env_for("S1", tmp_path / "b")
    bundle2["servers"]["files"].call_tool(
        "write_file", {"path": "summary.md", "content": "the user likes tea every morning\n"}
    )
    result = verify_task(_task2, bundle2)
    assert result["passed"] is False
    failed = {c["name"] for c in result["checks"] if not c["passed"]}
    assert failed >= {"gold_facts_present", "no_forbidden_text", "recall_before_write"}


def test_correct_verifier_fails_without_supersede(tmp_path):
    task, bundle = _env_for("C1", tmp_path)
    servers = bundle["servers"]
    servers["memory"].call_tool("recall", {"query": "prefers", "k": 5})
    servers["memory"].call_tool("remember", {"text": "the user prefers rust"})
    result = verify_task(task, bundle)
    assert result["passed"] is False
    failed = {c["name"] for c in result["checks"] if not c["passed"]}
    assert "old_record_superseded" in failed
    assert "correction_text" in failed


def test_compute_verifier_recomputes_deduped_total(tmp_path):
    task, bundle = _env_for("M1", tmp_path)
    ctx = task["context"]
    src = bundle["servers"]["files"].call_tool("read_file", {"path": "expenses.json"})["result"]
    assert _compute_from_text(src, ctx, dedup=True) == 254.75
    assert _compute_from_text(src, ctx, dedup=False) == 375.25
    # Naive sum fails the verifier; the deduped sum passes.
    bundle["servers"]["files"].call_tool("write_file", {"path": "total.txt", "content": "375.25\n"})
    assert verify_task(task, bundle)["passed"] is False
    _t2, bundle2 = _env_for("M1", tmp_path / "b")
    bundle2["servers"]["files"].call_tool("read_file", {"path": "expenses.json"})
    bundle2["servers"]["files"].call_tool("write_file", {"path": "total.txt", "content": "254.75\n"})
    assert verify_task(_t2, bundle2)["passed"] is True


def test_deliver_verifier_fails_wrong_order(tmp_path):
    task, bundle = _env_for("D1", tmp_path)
    servers = bundle["servers"]
    tid = bundle["tracker_ids"]["prepare release notes"]
    # Complete first, deliverable second: the ordering check must name it.
    servers["tracker"].call_tool("complete_task", {"id": tid})
    servers["memory"].call_tool("recall", {"query": "release notes", "k": 5})
    servers["files"].call_tool(
        "write_file",
        {"path": "release-notes.md", "content": "release 1.4 ships semantic recall\nrelease 1.4 fixes the torn-tail bug\n"},
    )
    servers["files"].call_tool(
        "write_file", {"path": "status.md", "content": "Completed: prepare release notes\n"}
    )
    result = verify_task(task, bundle)
    assert result["passed"] is False
    ordering = next(c for c in result["checks"] if c["name"] == "ordering_deliver_then_complete_then_status")
    assert "complete at 0" in ordering["detail"]


def test_verifier_names_the_failure(tmp_path):
    # A failure must be diagnosable from the verifier output alone.
    task, bundle = _env_for("M3", tmp_path)
    bundle["servers"]["files"].call_tool("read_file", {"path": "events.json"})
    bundle["servers"]["files"].call_tool("write_file", {"path": "count.txt", "content": "5\n"})
    result = verify_task(task, bundle)
    assert result["passed"] is False
    value = next(c for c in result["checks"] if c["name"] == "value_matches")
    assert "expected 3" in value["detail"] and "got 5.0" in value["detail"]


# --- agents + scoring ---------------------------------------------------


def test_grounded_passes_all_sloppy_passes_one():
    result = run_tobench_eval()
    assert result["agents"]["grounded"]["passed"] == 12
    assert result["agents"]["sloppy"]["passed"] == 1
    sloppy_passes = [t["task_id"] for t in result["agents"]["sloppy"]["per_task"] if t["passed"]]
    assert sloppy_passes == ["S3"]  # the only prompt-contained task


def test_eval_is_deterministic():
    a = run_tobench_eval()
    b = run_tobench_eval()
    for name in ("grounded", "sloppy"):
        assert [t["passed"] for t in a["agents"][name]["per_task"]] == [
            t["passed"] for t in b["agents"][name]["per_task"]
        ]


def test_crashing_agent_fails_tasks_without_escaping():
    def boom(task, bundle):
        raise RuntimeError("agent exploded")

    import hybrid_retriever.tobench as tb

    tb.AGENTS["boom"] = boom
    try:
        result = run_tobench_eval(agents=("boom",), n_tasks=3)
        assert result["agents"]["boom"]["passed"] == 0
    finally:
        del tb.AGENTS["boom"]


def test_tasks_are_twelve_across_four_kinds():
    tasks = build_tobench_tasks()
    assert len(tasks) == 12
    kinds = [t["kind"] for t in tasks]
    for kind in ("summarize", "correct", "compute", "deliver"):
        assert kinds.count(kind) == 3


def test_stdio_wrapper_serves_files_over_json_rpc(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-m", "hybrid_retriever.tobench", "serve",
         "--server", "files", "--root", str(root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        cwd=REPO_ROOT,
    )
    try:
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "call_tool",
                                     "params": {"name": "write_file", "arguments": {"path": "x.txt", "content": "hi"}}}) + "\n")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "call_tool",
                                     "params": {"name": "read_file", "arguments": {"path": "x.txt"}}}) + "\n")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "list_tools"}) + "\n")
        proc.stdin.close()
        lines = [json.loads(l) for l in proc.stdout if l.strip()]
        assert lines[0]["result"]["ok"] is True
        assert lines[1]["result"]["result"] == "hi"
        assert "read_file" in lines[2]["result"]
    finally:
        proc.wait(timeout=10)
    assert proc.returncode == 0
