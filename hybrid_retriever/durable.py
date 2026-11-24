"""P22 — Durable agent loop on real DBOS (3.6.0), SQLite system DB.

Wraps the P21 grounded-agent policy so a crash mid-run RESUMES
instead of restarting: every tool call is a DBOS step whose output
is checkpointed in the system database, and a fresh process
auto-recovers the PENDING workflow on launch.

Substitutions, stated plainly:

1. No Postgres on this VM (no server, no docker, no psql). DBOS
   3.6.0's default system database is SQLite
   (``system_database_url`` defaults to ``sqlite:///{name}``), so
   this runs REAL DBOS — same ``@DBOS.workflow`` / ``@DBOS.step``
   decorators, same journal tables (``workflow_status``,
   ``operation_outputs``), same recovery path — with the system
   database in a SQLite file instead of Postgres. That is the
   configuration DBOS itself ships as the default; it is not a
   reimplementation. Postgres is the production default and the
   only change needed to move this there is the URL.
2. No LLM anywhere: the loop's work is the deterministic P21
   grounded policy, decomposed into one step per tool call.
3. DBOS steps are at-least-once, not exactly-once: a step killed
   after its side effect but before its output is checkpointed
   re-executes on recovery (verified in the smoke test — a step
   killed mid-write ran twice in the raw log). So every step body
   here is IDEMPOTENT — remember/create/complete check before
   they act, file writes overwrite — and the no-duplicated-side-
   effects guarantee is proved by the demo's assertions, not
   assumed from the engine. A step that is not idempotent, a side
   effect outside the journal, or a crash inside a step's own
   write is NOT protected; see the README honest reading.
4. The human-approval gate is a file flag (``approval.txt`` in
   the workspace) polled with durable ``DBOS.sleep`` — completed
   sleeps are skipped on recovery, so the pause survives a
   restart. For hours-long pauses DBOS ``recv``/``send`` events
   are the better primitive (no sleep-operation per poll); the
   file flag is the demo's explicit, inspectable mechanism.
5. Steps carry an artificial ``step_delay`` (default 0) before
   their side effect, ONLY so the kill demo/tests can land the
   SIGKILL mid-run on a workload whose real work takes
   milliseconds. It changes timing, not semantics.

Layout of a run workspace (``root``):

- ``dbos-system.db`` — the DBOS system database (SQLite).
- ``store.db`` — the P21 memory store (SQLite).
- ``tasks.json`` — the P21 tracker state.
- ``trace.jsonl`` — the durable tool-call trace, one line per
  executed step, keyed by DBOS step id. Rebuilt traces dedupe
  by step id (first occurrence wins), so a step re-executed in
  the crash window leaves exactly one trace entry.
- ``seed.json`` — the seed marker (mem/tracker ids); the seed
  step is itself idempotent, the marker is belt-and-braces.
- ``approval.txt`` — created by the operator to release the
  approval gate.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

from dbos import DBOS, DBOSConfig

from .tobench import (
    Trace,
    _compute_from_text,
    _fmt_number,
    build_servers,
    build_tobench_tasks,
    verify_task,
)

APP_NAME = "hybrid-retriever-durable"
SYSTEM_DB_NAME = "dbos-system.db"
TRACE_NAME = "trace.jsonl"
SEED_NAME = "seed.json"
APPROVAL_NAME = "approval.txt"

_CONFIGURED_ROOT: str | None = None


# ----------------------------------------------------------------------
# Paths and DBOS lifecycle
# ----------------------------------------------------------------------


def system_db_path(root: str) -> str:
    return os.path.join(os.path.abspath(root), SYSTEM_DB_NAME)


def trace_path(root: str) -> str:
    return os.path.join(os.path.abspath(root), TRACE_NAME)


def seed_path(root: str) -> str:
    return os.path.join(os.path.abspath(root), SEED_NAME)


def approval_path(root: str) -> str:
    return os.path.join(os.path.abspath(root), APPROVAL_NAME)


def configure(root: str) -> None:
    """Create (once) and launch the DBOS singleton for this root.

    Launch auto-recovers PENDING workflows of this application
    version — that is the resume path; no explicit recover call
    is needed for the normal case.
    """
    global _CONFIGURED_ROOT
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    if _CONFIGURED_ROOT is not None and _CONFIGURED_ROOT != root:
        raise RuntimeError(
            f"DBOS already configured for {_CONFIGURED_ROOT}; "
            f"call shutdown() before configuring {root}"
        )
    if _CONFIGURED_ROOT is None:
        config: DBOSConfig = {
            "name": APP_NAME,
            "system_database_url": f"sqlite:///{system_db_path(root)}",
        }
        DBOS(config=config)
        _CONFIGURED_ROOT = root
    DBOS.launch()


def shutdown() -> None:
    """Destroy the DBOS singleton (stops background threads)."""
    global _CONFIGURED_ROOT
    DBOS.destroy()
    _CONFIGURED_ROOT = None


def workflow_statuses(root: str) -> list[dict]:
    """Read workflow_status rows straight from the system DB.

    No DBOS instance needed — used by the demo/tests to observe
    state at kill time and while a run is paused.
    """
    path = system_db_path(root)
    if not os.path.exists(path):
        return []
    # A plain (read-write) connection, not mode=ro: right after a
    # SIGKILL the WAL may be uncheckpointed, and a read-only open
    # can fail where a normal open recovers it. Retried, because
    # a live writer may briefly hold the lock.
    last_error: Exception | None = None
    for _ in range(10):
        try:
            con = sqlite3.connect(path, timeout=10.0)
            try:
                rows = con.execute(
                    "SELECT workflow_uuid, status, name, recovery_attempts "
                    "FROM workflow_status ORDER BY created_at"
                ).fetchall()
            finally:
                con.close()
            return [
                {
                    "workflow_id": r[0],
                    "status": r[1],
                    "name": r[2],
                    "recovery_attempts": r[3],
                }
                for r in rows
            ]
        except sqlite3.OperationalError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(
        f"could not read workflow_status in {path}: {last_error}"
    )


def checkpointed_step_count(root: str) -> int:
    """How many step outputs DBOS has journaled for this root."""
    path = system_db_path(root)
    if not os.path.exists(path):
        return 0
    for _ in range(10):
        try:
            con = sqlite3.connect(path, timeout=10.0)
            try:
                return int(
                    con.execute(
                        "SELECT COUNT(*) FROM operation_outputs"
                    ).fetchone()[0]
                )
            finally:
                con.close()
        except sqlite3.OperationalError:
            time.sleep(0.2)
    return 0


# ----------------------------------------------------------------------
# Durable trace: one JSONL line per executed step, keyed by step id.
# ----------------------------------------------------------------------


def append_trace(root: str, server: str, tool: str, args: dict) -> None:
    line = {
        "workflow_id": DBOS.workflow_id,
        "step_id": DBOS.step_id,
        "server": server,
        "tool": tool,
        "args": args,
    }
    with open(trace_path(root), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def load_trace(root: str) -> Trace:
    """Rebuild the P21 Trace from the durable log.

    Dedupe by step id (first occurrence wins): a step re-executed
    in the crash window appends a second line, but the verifier
    sees each tool call exactly once. Lines without a step id
    (direct calls outside a workflow, e.g. in tests) dedupe by
    their full content instead.
    """
    trace = Trace()
    seen: set = set()
    path = trace_path(root)
    if not os.path.exists(path):
        return trace
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            step_id = event.get("step_id")
            key = ("step", step_id) if step_id is not None else (
                "content",
                event.get("server"),
                event.get("tool"),
                json.dumps(event.get("args", {}), sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            trace.record(event["server"], event["tool"], event.get("args", {}))
    return trace


# ----------------------------------------------------------------------
# Idempotent step bodies (pure helpers: unit-testable without DBOS).
# Each one is safe to run twice — the second run is a no-op that
# returns the first run's result.
# ----------------------------------------------------------------------


def _open_bundle(root: str) -> dict:
    return build_servers(root)


def remember_once(bundle: dict, text: str, type_: str = "fact",
                  supersedes: int | None = None) -> dict:
    """Remember, unless an identical live record already exists.

    A correction supersedes its target only if the target is still
    live; if the target is already superseded by a correction with
    the same text, return that correction's id.
    """
    servers = bundle["servers"]
    live = servers["memory"].call_tool("list_records")["result"]
    if supersedes is not None:
        for rec in live:
            if rec["text"] == text and rec["type"] == "correction":
                # Target already superseded by this same correction.
                return {"id": rec["id"], "superseded": supersedes}
        target_live = any(r["id"] == supersedes for r in live)
        if not target_live:
            raise ValueError(f"supersedes target {supersedes} is not live")
    else:
        for rec in live:
            if rec["text"] == text and rec["type"] == type_:
                return {"id": rec["id"], "superseded": None}
    return servers["memory"].call_tool(
        "remember",
        {"text": text, "type": type_, "supersedes": supersedes},
    )["result"]


def create_task_once(bundle: dict, title: str) -> dict:
    """Create a tracker task, unless one with this title exists."""
    servers = bundle["servers"]
    tasks = servers["tracker"].call_tool("list_tasks")["result"]
    for task in tasks:
        if task["title"] == title:
            return {"id": task["id"], "title": title}
    return servers["tracker"].call_tool("create_task", {"title": title})["result"]


def complete_task_once(bundle: dict, task_id: int) -> dict:
    """Complete a tracker task; already-completed is a no-op."""
    servers = bundle["servers"]
    tasks = servers["tracker"].call_tool("list_tasks")["result"]
    for task in tasks:
        if task["id"] == task_id:
            if task["status"] == "completed":
                return {"id": task_id, "status": "completed"}
            break
    return servers["tracker"].call_tool(
        "complete_task", {"id": task_id}
    )["result"]


def seed_once(root: str, task: dict) -> dict:
    """Seed the environment idempotently; return mem/tracker ids.

    Uses the idempotent helpers (not tobench._seed_env, whose
    remember/create calls would duplicate on re-execution), and
    writes a seed.json marker so a re-run can short-circuit.
    """
    marker = seed_path(root)
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            saved = json.load(fh)
        if saved.get("task_id") == task["id"]:
            return saved
        raise ValueError(
            f"workspace already seeded for {saved.get('task_id')}, "
            f"cannot seed {task['id']} in the same root"
        )
    bundle = _open_bundle(root)
    try:
        servers = bundle["servers"]
        setup = task.get("setup", {})
        mem_ids: dict[str, int] = {}
        for text in setup.get("memories", []):
            mem_ids[text] = remember_once(bundle, text)["id"]
        for path, content in setup.get("files", {}).items():
            servers["files"].call_tool(
                "write_file", {"path": path, "content": content}
            )
        tracker_ids: dict[str, int] = {}
        for title in setup.get("tracker_tasks", []):
            tracker_ids[title] = create_task_once(bundle, title)["id"]
    finally:
        bundle["store"].close()
    saved = {"task_id": task["id"], "mem_ids": mem_ids,
             "tracker_ids": tracker_ids}
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump(saved, fh)
    return saved


# ----------------------------------------------------------------------
# DBOS steps: one per tool call. The artificial delay runs BEFORE
# the side effect, so a kill during the delay means the step never
# ran — clean re-execution on recovery. The trace line is appended
# after the tool call, inside the same step.
# ----------------------------------------------------------------------


def _delay(step_delay: float) -> None:
    if step_delay > 0:
        time.sleep(step_delay)


@DBOS.step()
def step_seed(root: str, task_id: str, step_delay: float = 0.0) -> dict:
    _delay(step_delay)
    task = next(t for t in build_tobench_tasks() if t["id"] == task_id)
    return seed_once(root, task)


@DBOS.step()
def step_recall(root: str, query: str, k: int = 5,
                step_delay: float = 0.0) -> list:
    _delay(step_delay)
    bundle = _open_bundle(root)
    try:
        result = bundle["servers"]["memory"].call_tool(
            "recall", {"query": query, "k": k}
        )["result"]
    finally:
        bundle["store"].close()
    append_trace(root, "memory", "recall", {"query": query, "k": k})
    return result


@DBOS.step()
def step_remember(root: str, text: str, type_: str = "fact",
                  supersedes: int | None = None,
                  step_delay: float = 0.0) -> dict:
    _delay(step_delay)
    bundle = _open_bundle(root)
    try:
        result = remember_once(bundle, text, type_, supersedes)
    finally:
        bundle["store"].close()
    append_trace(
        root, "memory", "remember",
        {"text": text, "type": type_, "supersedes": supersedes},
    )
    return result


@DBOS.step()
def step_write_file(root: str, path: str, content: str,
                    step_delay: float = 0.0) -> dict:
    _delay(step_delay)
    bundle = _open_bundle(root)
    try:
        result = bundle["servers"]["files"].call_tool(
            "write_file", {"path": path, "content": content}
        )["result"]
    finally:
        bundle["store"].close()
    append_trace(root, "files", "write_file", {"path": path})
    return result


@DBOS.step()
def step_read_file(root: str, path: str,
                   step_delay: float = 0.0) -> str:
    _delay(step_delay)
    bundle = _open_bundle(root)
    try:
        result = bundle["servers"]["files"].call_tool(
            "read_file", {"path": path}
        )["result"]
    finally:
        bundle["store"].close()
    append_trace(root, "files", "read_file", {"path": path})
    return result


@DBOS.step()
def step_create_task(root: str, title: str,
                     step_delay: float = 0.0) -> dict:
    _delay(step_delay)
    bundle = _open_bundle(root)
    try:
        result = create_task_once(bundle, title)
    finally:
        bundle["store"].close()
    append_trace(root, "tracker", "create_task", {"title": title})
    return result


@DBOS.step()
def step_complete_task(root: str, task_id: int,
                       step_delay: float = 0.0) -> dict:
    _delay(step_delay)
    bundle = _open_bundle(root)
    try:
        result = complete_task_once(bundle, task_id)
    finally:
        bundle["store"].close()
    append_trace(root, "tracker", "complete_task", {"id": task_id})
    return result


# ----------------------------------------------------------------------
# The durable workflow: the P21 grounded policy, one step per tool
# call. Workflow-body code is deterministic and side-effect-free
# (it re-runs from the top on recovery; only steps are skipped).
# ----------------------------------------------------------------------


def _get_task(task_id: str) -> dict:
    task = next(
        (t for t in build_tobench_tasks() if t["id"] == task_id), None
    )
    if task is None:
        raise ValueError(f"unknown tobench task: {task_id}")
    return task


def wait_for_approval(root: str, poll_sec: float = 5.0) -> None:
    """Block until approval.txt appears. Durable: each DBOS.sleep
    is checkpointed and skipped on recovery, so the pause survives
    a restart — the loop just re-checks the file."""
    while not os.path.exists(approval_path(root)):
        DBOS.sleep(poll_sec)


@DBOS.workflow()
def run_durable_task(task_id: str, root: str,
                     require_approval: bool = False,
                     step_delay: float = 0.0,
                     approval_poll_sec: float = 5.0) -> dict:
    task = _get_task(task_id)
    ctx = task["context"]
    kind = task["kind"]
    seed = step_seed(root, task_id, step_delay)

    if kind == "summarize":
        recalled = (
            step_recall(root, ctx["query"], 5, step_delay)
            if ctx["query"] else []
        )
        text = (
            "\n".join(r["text"] for r in recalled)
            if recalled else ctx.get("prompt_text", "")
        )
        step_write_file(root, ctx["artifact"], text + "\n", step_delay)
        return {"task_id": task_id, "result": text}

    if kind == "correct":
        recalled = step_recall(root, ctx["query"], 5, step_delay)
        target = next(
            (r for r in recalled if r["text"] == ctx["wrong"]), None
        )
        if target is None:
            return {"task_id": task_id,
                    "result": "could not find the wrong record"}
        step_remember(
            root, ctx["truth"], "correction", target["id"], step_delay
        )
        return {"task_id": task_id, "result": ctx["truth"]}

    if kind == "compute":
        text = step_read_file(root, ctx["source"], step_delay)
        value = _compute_from_text(text, ctx, dedup=True)
        step_write_file(
            root, ctx["artifact"], _fmt_number(value) + "\n", step_delay
        )
        return {"task_id": task_id, "result": _fmt_number(value)}

    if kind == "deliver":
        recalled = step_recall(root, ctx["query"], 5, step_delay)
        text = "\n".join(r["text"] for r in recalled)
        step_write_file(root, ctx["artifact"], text + "\n", step_delay)
        if require_approval:
            wait_for_approval(root, approval_poll_sec)
        tid = seed["tracker_ids"][ctx["title"]]
        step_complete_task(root, tid, step_delay)
        step_write_file(
            root, ctx["status_file"],
            f"Completed: {ctx['title']}\n", step_delay,
        )
        return {"task_id": task_id, "result": text}

    raise ValueError(f"unknown kind: {kind}")


# ----------------------------------------------------------------------
# Running, waiting, verifying
# ----------------------------------------------------------------------


def start_run(task_id: str, root: str, require_approval: bool = False,
              step_delay: float = 0.0,
              approval_poll_sec: float = 5.0):
    """Configure + launch (auto-recovers PENDING workflows), then
    start the workflow in the background. Returns the handle."""
    configure(root)
    return DBOS.start_workflow(
        run_durable_task, task_id, root, require_approval,
        step_delay, approval_poll_sec,
    )


def wait_for_status(workflow_id: str, want: tuple[str, ...],
                    timeout_sec: float = 120.0,
                    poll_sec: float = 0.5) -> str | None:
    """Poll get_workflow_status until it is one of `want`."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        status = DBOS.get_workflow_status(workflow_id)
        if status is not None and status.status in want:
            return status.status
        time.sleep(poll_sec)
    status = DBOS.get_workflow_status(workflow_id)
    return status.status if status is not None else None


def verify_durable_run(root: str, task_id: str) -> dict:
    """Rebuild the bundle from disk (store + durable trace + the
    seed marker's mem/tracker ids, which the P21 verifier needs)
    and run the P21 verifier on the final state."""
    task = _get_task(task_id)
    bundle = _open_bundle(root)
    bundle["trace"] = load_trace(root)
    marker = seed_path(root)
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            saved = json.load(fh)
        bundle["mem_ids"] = saved.get("mem_ids", {})
        bundle["tracker_ids"] = saved.get("tracker_ids", {})
    try:
        return verify_task(task, bundle)
    finally:
        bundle["store"].close()


def assert_no_duplicate_side_effects(root: str, task_id: str) -> list[str]:
    """Exactly-once assertions over the real side effects.

    Returns a list of violations (empty = clean). Checks: every
    seeded memory text exists exactly once as a live record (or,
    for a corrected record, exactly once superseded by exactly
    one correction with the truth text); every tracker task title
    exists exactly once; the durable trace has unique step ids.
    """
    task = _get_task(task_id)
    violations: list[str] = []
    bundle = _open_bundle(root)
    try:
        servers = bundle["servers"]
        live = servers["memory"].call_tool("list_records")["result"]
        texts = [r["text"] for r in live]
        setup = task.get("setup", {})
        setup_memories = setup.get("memories", [])
        if task["kind"] == "correct":
            # The truth text may ALSO be a seeded background fact
            # (C1/C2: "the user prefers rust" / "home base is
            # Ottawa" are in BACKGROUND_FACTS), so the live count
            # of the truth is background occurrences + 1 correction.
            wrong = task["context"]["wrong"]
            truth = task["context"]["truth"]
            if texts.count(wrong) != 0:
                violations.append(
                    f"wrong record still live: {wrong!r} "
                    f"(count {texts.count(wrong)})"
                )
            expected_truth = (
                sum(1 for m in setup_memories if m == truth) + 1
            )
            if texts.count(truth) != expected_truth:
                violations.append(
                    f"correction text {truth!r} live count "
                    f"{texts.count(truth)}, expected {expected_truth}"
                )
            for text in setup_memories:
                if text in (wrong, truth):
                    continue
                if texts.count(text) != 1:
                    violations.append(
                        f"memory {text!r} live count {texts.count(text)}, "
                        f"expected 1"
                    )
        else:
            for text in setup_memories:
                if texts.count(text) != 1:
                    violations.append(
                        f"memory {text!r} live count {texts.count(text)}, "
                        f"expected 1"
                    )
        tasks = servers["tracker"].call_tool("list_tasks")["result"]
        for title in setup.get("tracker_tasks", []):
            same = [t for t in tasks if t["title"] == title]
            if len(same) != 1:
                violations.append(
                    f"tracker task {title!r} count {len(same)}, expected 1"
                )
            elif task["kind"] == "deliver" and same[0]["status"] != "completed":
                violations.append(
                    f"tracker task {title!r} status "
                    f"{same[0]['status']}, expected completed"
                )
    finally:
        bundle["store"].close()
    trace = load_trace(root)
    # The raw log may hold a step twice (re-executed in the crash
    # window); the rebuilt trace must hold it once.
    raw_step_ids: list = []
    path = trace_path(root)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("step_id") is not None:
                    raw_step_ids.append(event["step_id"])
    # The raw log may hold a step twice (re-executed in the crash
    # window) — that is expected, not a violation. The property
    # that must hold is that the rebuilt trace holds it once.
    if len(set(raw_step_ids)) != len(trace.events):
        violations.append(
            f"rebuilt trace has {len(trace.events)} events for "
            f"{len(set(raw_step_ids))} distinct step ids"
        )
    return violations


# ----------------------------------------------------------------------
# Demos: kill-and-resume, and the approval pause surviving a restart.
# ----------------------------------------------------------------------


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _spawn(args: list[str], log_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env["TMPDIR"] = os.path.expanduser("~/.tmp")
    log_fh = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "hybrid_retriever.durable", *args],
        cwd=_repo_root(),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _trace_line_count(root: str) -> int:
    path = trace_path(root)
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _tracker_state(root: str) -> list[dict]:
    state = _read_json(os.path.join(root, "tasks.json"))
    return state["tasks"] if state else []


def _workflow_id_from_log(log_path: str) -> str | None:
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("WORKFLOW_ID:"):
                return line.split(":", 1)[1].strip()
    return None


def approve(root: str) -> str:
    """The operator's move: create the approval flag."""
    path = approval_path(root)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"approved at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    return path


def run_demo(task_id: str = "D1", step_delay: float = 1.0,
            kill_after_steps: int = 2,
            workdir: str | None = None) -> dict:
    """Kill the run mid-flight (SIGKILL), recover in a fresh
    process, prove no duplicated side effects. Returns a report."""
    root = os.path.abspath(workdir or tempfile.mkdtemp(prefix="durable-demo-"))
    os.makedirs(root, exist_ok=True)
    log_path = os.path.join(root, "run1.log")
    proc = _spawn([
        "run", "--task", task_id, "--root", root,
        "--step-delay", str(step_delay),
    ], log_path)
    try:
        deadline = time.time() + 90.0
        while _trace_line_count(root) < kill_after_steps:
            if proc.poll() is not None:
                raise RuntimeError(
                    "run process exited before the kill point "
                    f"(exit {proc.returncode}); see {log_path}"
                )
            if time.time() > deadline:
                raise RuntimeError(
                    "run never reached the kill point "
                    f"({kill_after_steps} trace lines) in 90s"
                )
            time.sleep(0.1)
        proc.kill()
        proc.wait()
        at_kill = {
            "statuses": workflow_statuses(root),
            "checkpointed_steps": checkpointed_step_count(root),
            "trace_lines": _trace_line_count(root),
        }
        pending = [s for s in at_kill["statuses"] if s["status"] == "PENDING"]
        if not pending:
            raise RuntimeError(
                "expected the killed workflow to be PENDING at kill "
                f"time, got: {at_kill['statuses']}"
            )
        workflow_id = _workflow_id_from_log(log_path)
        if workflow_id is None:
            raise RuntimeError(f"no WORKFLOW_ID in {log_path}")
        # Fresh process: launch auto-recovers the PENDING workflow.
        recover_log = os.path.join(root, "recover.log")
        recover = _spawn([
            "recover", "--root", root, "--workflow-id", workflow_id,
            "--wait", "120",
        ], recover_log)
        rc = recover.wait(timeout=150)
        if rc != 0:
            raise RuntimeError(f"recover exited {rc}; see {recover_log}")
        verification = verify_durable_run(root, task_id)
        violations = assert_no_duplicate_side_effects(root, task_id)
        final = workflow_statuses(root)
        # Oracle comparison: the same task run by the plain P21
        # grounded policy in a fresh dir must produce the same
        # artifact bytes.
        oracle: dict = {}
        with tempfile.TemporaryDirectory(prefix="durable-oracle-") as oroot:
            from .tobench import run_grounded, _seed_env
            task = _get_task(task_id)
            bundle = _seed_env(task, oroot)
            run_grounded(task, bundle)
            for name in os.listdir(oroot):
                if (
                    name.startswith("store.db")
                    or name in (SYSTEM_DB_NAME, TRACE_NAME, SEED_NAME,
                                APPROVAL_NAME, "tasks.json")
                ):
                    continue
                with open(os.path.join(oroot, name), "rb") as fh:
                    oracle[name] = fh.read()
            bundle["store"].close()
        oracle_mismatch = []
        for name, content in oracle.items():
            mine = os.path.join(root, name)
            if not os.path.exists(mine):
                oracle_mismatch.append(f"missing artifact {name}")
            elif open(mine, "rb").read() != content:
                oracle_mismatch.append(f"artifact differs: {name}")
        return {
            "task_id": task_id,
            "root": root,
            "workflow_id": workflow_id,
            "at_kill": at_kill,
            "final_statuses": final,
            "verification_passed": bool(verification["passed"]),
            "verification": verification,
            "violations": violations + oracle_mismatch,
            "recover_log": recover_log,
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def run_approval_demo(task_id: str = "D1", step_delay: float = 1.0,
                      approval_poll_sec: float = 2.0,
                      workdir: str | None = None) -> dict:
    """Pause at the approval gate, SIGKILL mid-pause, recover in
    a fresh process, prove the pause survived, then approve and
    prove the run completes. Recovery runs in this process (a
    fresh DBOS instance after the kill). Returns a report."""
    root = os.path.abspath(workdir or tempfile.mkdtemp(prefix="durable-approval-"))
    os.makedirs(root, exist_ok=True)
    task = _get_task(task_id)
    if task["kind"] != "deliver":
        raise ValueError("the approval gate only exists on deliver tasks")
    log_path = os.path.join(root, "run1.log")
    proc = _spawn([
        "run", "--task", task_id, "--root", root,
        "--step-delay", str(step_delay), "--approval",
        "--approval-poll-sec", str(approval_poll_sec),
    ], log_path)
    try:
        # Wait until the deliverable is written but the tracker
        # task is still open: the run is paused at the gate.
        artifact = os.path.join(root, task["context"]["artifact"])
        deadline = time.time() + 90.0
        paused = False
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    "run process exited before pausing "
                    f"(exit {proc.returncode}); see {log_path}"
                )
            tasks = _tracker_state(root)
            if os.path.exists(artifact) and tasks and all(
                t["status"] == "open" for t in tasks
            ):
                paused = True
                break
            time.sleep(0.2)
        if not paused:
            raise RuntimeError("run never reached the approval gate in 90s")
        proc.kill()
        proc.wait()
        workflow_id = _workflow_id_from_log(log_path)
        if workflow_id is None:
            raise RuntimeError(f"no WORKFLOW_ID in {log_path}")
        at_kill = {
            "statuses": workflow_statuses(root),
            "trace_lines": _trace_line_count(root),
        }
        # Fresh DBOS instance in THIS process: launch auto-recovers
        # the PENDING workflow, which re-enters the approval loop.
        configure(root)
        try:
            time.sleep(approval_poll_sec * 4)
            mid = {
                "statuses": workflow_statuses(root),
                "tracker_open": all(
                    t["status"] == "open" for t in _tracker_state(root)
                ),
            }
            if not any(s["status"] == "PENDING" for s in mid["statuses"]):
                raise RuntimeError(
                    "workflow did not stay PENDING through the "
                    f"restart; statuses: {mid['statuses']}"
                )
            if not mid["tracker_open"]:
                raise RuntimeError("tracker task completed while paused")
            approve(root)
            final_status = wait_for_status(
                workflow_id, ("SUCCESS", "ERROR"), timeout_sec=90.0
            )
            if final_status != "SUCCESS":
                raise RuntimeError(
                    f"workflow ended {final_status} after approval"
                )
            result = DBOS.get_result(workflow_id)
        finally:
            shutdown()
        verification = verify_durable_run(root, task_id)
        violations = assert_no_duplicate_side_effects(root, task_id)
        return {
            "task_id": task_id,
            "root": root,
            "workflow_id": workflow_id,
            "at_kill": at_kill,
            "mid_restart": mid,
            "final_status": final_status,
            "result": result,
            "verification_passed": bool(verification["passed"]),
            "verification": verification,
            "violations": violations,
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ----------------------------------------------------------------------
# CLI: `python -m hybrid_retriever.durable <cmd> ...`
# (`hr durable ...` forwards here.)
# ----------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    configure(root)
    try:
        handle = DBOS.start_workflow(
            run_durable_task, args.task, root, args.approval,
            args.step_delay, args.approval_poll_sec,
        )
        print(f"WORKFLOW_ID: {handle.get_workflow_id()}", flush=True)
        print(f"RESULT: {handle.get_result()}", flush=True)
    finally:
        shutdown()
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    configure(root)
    try:
        if args.workflow_id:
            status = wait_for_status(
                args.workflow_id, ("SUCCESS", "ERROR", "CANCELLED",
                                   "MAX_RECOVERY_ATTEMPTS_EXCEEDED"),
                timeout_sec=args.wait,
            )
            print(f"WORKFLOW_ID: {args.workflow_id} STATUS: {status}",
                  flush=True)
            if status == "SUCCESS":
                print(f"RESULT: {DBOS.get_result(args.workflow_id)}",
                      flush=True)
                return 0
            return 1
        handles = DBOS.recover_workflows(["local"])
        print(f"RECOVERED: {len(handles)}", flush=True)
        for handle in handles:
            print(f"RESULT: {handle.get_result()}", flush=True)
        return 0
    finally:
        shutdown()


def _cmd_approve(args: argparse.Namespace) -> int:
    path = approve(os.path.abspath(args.root))
    print(f"APPROVED: {path}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    print(json.dumps({
        "root": root,
        "workflows": workflow_statuses(root),
        "checkpointed_steps": checkpointed_step_count(root),
        "trace_lines": _trace_line_count(root),
    }, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    report = run_demo(
        task_id=args.task,
        step_delay=args.step_delay,
        kill_after_steps=args.kill_after_steps,
        workdir=args.workdir,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if not report["violations"] and report["verification_passed"] else 1


def _cmd_approval_demo(args: argparse.Namespace) -> int:
    report = run_approval_demo(
        task_id=args.task,
        step_delay=args.step_delay,
        approval_poll_sec=args.approval_poll_sec,
        workdir=args.workdir,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if not report["violations"] and report["verification_passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="durable",
        description="P22: durable agent loop on DBOS (SQLite system DB).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="start a durable task run and block")
    pr.add_argument("--task", required=True, help="tobench task id, e.g. D1")
    pr.add_argument("--root", required=True, help="run workspace dir")
    pr.add_argument("--approval", action="store_true",
                    help="require human approval before tracker completion")
    pr.add_argument("--step-delay", type=float, default=0.0)
    pr.add_argument("--approval-poll-sec", type=float, default=5.0)
    pr.set_defaults(func=_cmd_run)

    pc = sub.add_parser("recover", help="launch DBOS (auto-recovery) and wait")
    pc.add_argument("--root", required=True)
    pc.add_argument("--workflow-id", default=None)
    pc.add_argument("--wait", type=float, default=120.0)
    pc.set_defaults(func=_cmd_recover)

    pa = sub.add_parser("approve", help="create the approval flag")
    pa.add_argument("--root", required=True)
    pa.set_defaults(func=_cmd_approve)

    ps = sub.add_parser("status", help="workflow + journal state for a root")
    ps.add_argument("--root", required=True)
    ps.set_defaults(func=_cmd_status)

    pd = sub.add_parser("demo", help="kill mid-run, recover, prove no dupes")
    pd.add_argument("--task", default="D1")
    pd.add_argument("--step-delay", type=float, default=1.0)
    pd.add_argument("--kill-after-steps", type=int, default=2)
    pd.add_argument("--workdir", default=None)
    pd.set_defaults(func=_cmd_demo)

    p2 = sub.add_parser(
        "approval-demo",
        help="pause at the gate, kill, resume paused, approve, complete",
    )
    p2.add_argument("--task", default="D1")
    p2.add_argument("--step-delay", type=float, default=1.0)
    p2.add_argument("--approval-poll-sec", type=float, default=2.0)
    p2.add_argument("--workdir", default=None)
    p2.set_defaults(func=_cmd_approval_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
