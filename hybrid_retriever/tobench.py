"""P21 — Mini-TOBench: verifier-first evals for a workspace-assistant domain.

TOBench (arXiv:2605.16909) scores agents on executable tasks over
real tool surfaces, with programmatic verifiers that inspect artifacts
and tool-call traces instead of judging final text. This module is the
small, deterministic version of that idea for this repo, with the
substitutions stated plainly:

1. No LLM anywhere, and no model API key exists on this VM. The "two
   models" TOBench runs are two DETERMINISTIC agent policies, not
   models: `grounded` (recall first, read sources, compute from the
   real artifacts, do tracker work in the right order) and `sloppy`
   (no recall, naive compute without dedup, tracker tasks completed
   before the deliverable exists, corrections stored without
   superseding). Their scores measure how well the harness
   discriminates good tool use from bad tool use — they say nothing
   about any LLM's capability, and no model names or scores are
   invented. The harness accepts any agent function, so a real model
   policy can be dropped in later without changing a task or verifier.
2. The three MCP servers are in-process Python objects with a
   `call_tool(name, arguments)` interface (plus a thin stdio JSON-RPC
   2.0 wrapper, `serve_stdio`, so they can also be driven over a pipe
   like a real MCP server). They are real tool surfaces — the agents
   only touch the environment through them, and the verifier reads
   the same surfaces back — but they run in the same process, so
   there is no transport, no auth, and no concurrency to test.
3. The memory server recalls with BM25 only (weights (1, 0, 0), no
   embedder). These tasks turn on whole-word topic matching, which
   BM25 answers deterministically and fast; the vector leg is
   exercised by the P1/P20 evals, not here.
4. Tasks are generated from a fixed seed (default 7) with fixed
   content, not sampled from a corpus — 12 tasks, 4 kinds x 3
   variants. The harness is therefore a regression suite for tool
   use, not a sample from a task distribution; scores generalise
   only to tasks shaped like these.

Domain: a workspace assistant over the hybrid store. Three servers:

- `memory` — recall / remember (with supersedes) / forget / list,
  backed by the real `Store` (SQLite) from P1.
- `files` — read / write / append / list, backed by a real directory
  (the task workspace). Paths cannot escape the root.
- `tracker` — create / complete / add-note / list tasks, backed by
  a real `tasks.json` in the workspace.

Every task is multi-step and the interesting failures are wrong tool
use, not wrong text: a summary that quotes the distractor instead of
the recalled facts, a correction stored without superseding (so the
old fact stays live), a total summed without dedup, a tracker task
completed before its deliverable exists. The verifier checks three
things per task: the artifact (recomputed from the environment, not
trusted from the agent), the tool-call trace (ordering, required
calls, forbidden tools, call budget), and the store state. Each
failed check carries a detail string (expected vs got, which call
was missing, which order was violated) so a failure is diagnosable
from the verifier output alone.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile

from .retrieval import retrieve
from .store import Store

# ----------------------------------------------------------------------
# Trace: the shared call log. Every tool call on every server lands
# here in order, including failed calls — the verifier reads it.
# ----------------------------------------------------------------------


class Trace:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, server: str, tool: str, args: dict) -> None:
        self.events.append({"server": server, "tool": tool, "args": args})

    def calls(self, server: str, tool: str) -> list[dict]:
        return [e for e in self.events if e["server"] == server and e["tool"] == tool]

    def first_index(self, server: str, tool: str) -> int | None:
        for i, e in enumerate(self.events):
            if e["server"] == server and e["tool"] == tool:
                return i
        return None

    def has(self, server: str, tool: str) -> bool:
        return self.first_index(server, tool) is not None

    def __len__(self) -> int:
        return len(self.events)


class ToolServer:
    """Base: dispatch + trace + a uniform result envelope."""

    name = "base"

    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    def tools(self) -> list[str]:
        return sorted(
            name[5:] for name in dir(self) if name.startswith("tool_")
        )

    def call_tool(self, name: str, args: dict | None = None) -> dict:
        args = dict(args or {})
        self.trace.record(self.name, name, args)
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return {"ok": False, "result": None, "error": f"unknown tool: {name}"}
        try:
            return {"ok": True, "result": fn(**args), "error": None}
        except Exception as exc:  # tool errors are data, not crashes
            return {"ok": False, "result": None, "error": str(exc)}


# ----------------------------------------------------------------------
# Server 1 — memory, over the real Store (BM25-only recall).
# ----------------------------------------------------------------------


class MemoryServer(ToolServer):
    name = "memory"

    def __init__(self, trace: Trace, store: Store) -> None:
        super().__init__(trace)
        self.store = store

    def tool_recall(self, query: str, k: int = 5) -> list[dict]:
        ranked = retrieve(
            self.store, None, query, k=k,
            weights=(1.0, 0.0, 0.0), use_mmr=False,
        )
        return [
            {"id": r.id, "text": r.text, "score": round(r.score, 4)}
            for r in ranked
        ]

    def tool_remember(
        self,
        text: str,
        type: str = "fact",
        salience: float = 0.5,
        supersedes: int | None = None,
    ) -> dict:
        if supersedes is not None:
            live_ids = {r["id"] for r in self.store.list_live_records()}
            if supersedes not in live_ids:
                raise ValueError(f"supersedes target {supersedes} is not a live record")
        rid = self.store.add_record(text, type_=type)
        if supersedes is not None:
            self.store.set_status(supersedes, "superseded", superseded_by=rid)
        return {"id": rid, "superseded": supersedes}

    def tool_forget(self, id: int) -> dict:
        live_ids = {r["id"] for r in self.store.list_live_records()}
        if id not in live_ids:
            raise ValueError(f"record {id} is not live")
        self.store.set_status(id, "superseded")
        return {"id": id, "status": "superseded"}

    def tool_list_records(self) -> list[dict]:
        return [
            {"id": r["id"], "text": r["text"], "type": r["type"], "status": r["status"]}
            for r in self.store.list_live_records()
        ]


# ----------------------------------------------------------------------
# Server 2 — files, over a real directory. Paths cannot escape root.
# ----------------------------------------------------------------------


class FilesServer(ToolServer):
    name = "files"

    def __init__(self, trace: Trace, root: str) -> None:
        super().__init__(trace)
        self.root = os.path.abspath(root)

    def _resolve(self, path: str) -> str:
        full = os.path.abspath(os.path.join(self.root, path))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ValueError(f"path escapes workspace root: {path}")
        return full

    def tool_read_file(self, path: str) -> str:
        with open(self._resolve(path), encoding="utf-8") as fh:
            return fh.read()

    def tool_write_file(self, path: str, content: str) -> dict:
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        return {"path": path, "bytes": len(content.encode("utf-8"))}

    def tool_append_file(self, path: str, content: str) -> dict:
        full = self._resolve(path)
        with open(full, "a", encoding="utf-8") as fh:
            fh.write(content)
        return {"path": path, "bytes": len(content.encode("utf-8"))}

    def tool_list_files(self) -> list[str]:
        out = []
        for dirpath, _dirs, files in os.walk(self.root):
            for f in sorted(files):
                if f == "tasks.json" or f.startswith("store.db"):
                    continue
                out.append(os.path.relpath(os.path.join(dirpath, f), self.root))
        return sorted(out)


# ----------------------------------------------------------------------
# Server 3 — tracker, over a real tasks.json in the workspace.
# ----------------------------------------------------------------------


class TrackerServer(ToolServer):
    name = "tracker"

    def __init__(self, trace: Trace, root: str) -> None:
        super().__init__(trace)
        self.path = os.path.join(os.path.abspath(root), "tasks.json")
        if not os.path.exists(self.path):
            self._save({"next_id": 1, "tasks": []})

    def _load(self) -> dict:
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self, state: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    def tool_create_task(self, title: str) -> dict:
        state = self._load()
        task = {"id": state["next_id"], "title": title, "status": "open", "notes": []}
        state["next_id"] += 1
        state["tasks"].append(task)
        self._save(state)
        return {"id": task["id"], "title": title}

    def tool_complete_task(self, id: int) -> dict:
        state = self._load()
        for t in state["tasks"]:
            if t["id"] == id:
                t["status"] = "completed"
                self._save(state)
                return {"id": id, "status": "completed"}
        raise ValueError(f"no task {id}")

    def tool_add_note(self, id: int, note: str) -> dict:
        state = self._load()
        for t in state["tasks"]:
            if t["id"] == id:
                t["notes"].append(note)
                self._save(state)
                return {"id": id, "notes": len(t["notes"])}
        raise ValueError(f"no task {id}")

    def tool_list_tasks(self) -> list[dict]:
        return self._load()["tasks"]


def build_servers(root: str) -> dict:
    """One fresh server bundle per task run: temp store + temp dir."""
    os.makedirs(root, exist_ok=True)
    trace = Trace()
    store = Store(os.path.join(root, "store.db"))
    servers = {
        "memory": MemoryServer(trace, store),
        "files": FilesServer(trace, root),
        "tracker": TrackerServer(trace, root),
    }
    return {"servers": servers, "trace": trace, "store": store, "root": root}


# ----------------------------------------------------------------------
# Tasks: 12, 4 kinds x 3 variants. `setup` seeds the environment;
# `context` is the structured data the policies work from; the
# verifier recomputes ground truth from the seeded environment.
# ----------------------------------------------------------------------

BACKGROUND_FACTS = [
    "the user prefers rust",
    "home base is Ottawa",
    "the dog is named Biscuit",
    "the user flies to Ahmedabad in January",
]


def _summarize_tasks() -> list[dict]:
    return [
        {
            "id": "S1",
            "kind": "summarize",
            "goal": (
                "Write summary.md summarizing what you recall about "
                "quarterly planning. Someone mentioned tea at the last "
                "meeting — include only what is relevant to planning."
            ),
            "context": {
                "query": "quarterly planning",
                "artifact": "summary.md",
                "gold": ["Thursday at 10am", "hiring and the roadmap"],
                "forbidden": ["tea"],
                "require_recall": True,
            },
            "setup": {
                "memories": [
                    "quarterly planning is on Thursday at 10am",
                    "quarterly planning agenda covers hiring and the roadmap",
                    "the user likes tea every morning",
                    *BACKGROUND_FACTS,
                ],
            },
        },
        {
            "id": "S2",
            "kind": "summarize",
            "goal": (
                "Write summary.md summarizing what you recall about "
                "the deployment runbook. The old deploy date was March 3 "
                "— that is stale, do not repeat it."
            ),
            "context": {
                "query": "deployment runbook",
                "artifact": "summary.md",
                "gold": ["migrate script first", "rollback is one command"],
                "forbidden": ["March 3"],
                "require_recall": True,
            },
            "setup": {
                "memories": [
                    "the deployment runbook says run the migrate script first",
                    "the deployment runbook says rollback is one command",
                    "the old deploy date was March 3",
                    *BACKGROUND_FACTS,
                ],
            },
        },
        {
            "id": "S3",
            "kind": "summarize",
            "goal": (
                "Write note.md saying exactly this: the office wifi "
                "password is on the whiteboard, and the standup is at "
                "9:30. No lookup is needed — the facts are in this task."
            ),
            "context": {
                "query": "",
                "artifact": "note.md",
                "gold": ["wifi password is on the whiteboard", "standup is at 9:30"],
                "forbidden": [],
                "require_recall": False,
                "prompt_text": (
                    "the office wifi password is on the whiteboard, "
                    "and the standup is at 9:30"
                ),
            },
            "setup": {"memories": list(BACKGROUND_FACTS)},
        },
    ]


def _correct_tasks() -> list[dict]:
    specs = [
        ("C1", "prefers", "the user prefers python", "the user prefers rust"),
        ("C2", "home base", "home base is Toronto", "home base is Ottawa"),
        ("C3", "deploy region", "deploy region is us-west", "deploy region is eu-central"),
    ]
    tasks = []
    for tid, query, wrong, truth in specs:
        tasks.append(
            {
                "id": tid,
                "kind": "correct",
                "goal": (
                    f"The stored preference is wrong — it says '{wrong}'. "
                    f"Find it and correct it to '{truth}'."
                ),
                "context": {
                    "query": query,
                    "wrong": wrong,
                    "truth": truth,
                    "require_recall": True,
                },
                "setup": {"memories": [wrong, *BACKGROUND_FACTS]},
            }
        )
    return tasks


def _compute_tasks() -> list[dict]:
    return [
        {
            "id": "M1",
            "kind": "compute",
            "goal": "Total the expenses in expenses.json, deduplicating by id, and write the total to total.txt.",
            "context": {
                "source": "expenses.json",
                "artifact": "total.txt",
                "format": "json-sum",
                "id_field": "id",
                "value_field": "amount",
            },
            "setup": {
                "files": {
                    "expenses.json": json.dumps(
                        [
                            {"id": 1, "note": "flight", "amount": 120.50},
                            {"id": 2, "note": "hotel", "amount": 89.00},
                            {"id": 3, "note": "meals", "amount": 45.25},
                            # re-fetch of item 1: same id, different note.
                            {"id": 1, "note": "flight re-fetch", "amount": 120.50},
                        ],
                        indent=2,
                    ),
                },
            },
        },
        {
            "id": "M2",
            "kind": "compute",
            "goal": "Total the sales in sales.csv, deduplicating by id, and write the total to total.txt.",
            "context": {
                "source": "sales.csv",
                "artifact": "total.txt",
                "format": "csv-sum",
                "id_field": "id",
                "value_field": "amount",
            },
            "setup": {
                "files": {
                    "sales.csv": (
                        "id,item,amount\n"
                        "1,book,25.00\n"
                        "2,pen,4.50\n"
                        "3,notebook,12.00\n"
                        "2,pen duplicate row,4.50\n"
                    ),
                },
            },
        },
        {
            "id": "M3",
            "kind": "compute",
            "goal": "Count the DISTINCT messages in events.json and write the count to count.txt.",
            "context": {
                "source": "events.json",
                "artifact": "count.txt",
                "format": "json-distinct",
                "id_field": "message",
                "value_field": "message",
            },
            "setup": {
                "files": {
                    "events.json": json.dumps(
                        [
                            {"ts": "10:00", "message": "heartbeat"},
                            {"ts": "10:01", "message": "deploy started"},
                            {"ts": "10:02", "message": "heartbeat"},
                            {"ts": "10:03", "message": "heartbeat"},
                            {"ts": "10:04", "message": "deploy finished"},
                        ],
                        indent=2,
                    ),
                },
            },
        },
    ]


def _deliver_tasks() -> list[dict]:
    specs = [
        (
            "D1",
            "prepare release notes",
            "release notes",
            "release-notes.md",
            ["release 1.4 ships semantic recall", "release 1.4 fixes the torn-tail bug"],
        ),
        (
            "D2",
            "prepare onboarding doc",
            "onboarding",
            "onboarding.md",
            ["onboarding starts with the repo tour", "onboarding pairs you with a buddy"],
        ),
        (
            "D3",
            "prepare quarterly report",
            "quarterly report",
            "quarterly-report.md",
            ["quarterly revenue is 42k", "quarterly churn is 2 percent"],
        ),
    ]
    tasks = []
    for tid, title, query, artifact, facts in specs:
        tasks.append(
            {
                "id": tid,
                "kind": "deliver",
                "goal": (
                    f"Complete the tracker task '{title}': write the "
                    f"deliverable from what you recall about {query}, "
                    f"mark the task complete, then write status.md "
                    f"listing the completed work."
                ),
                "context": {
                    "query": query,
                    "title": title,
                    "artifact": artifact,
                    "gold": facts,
                    "status_file": "status.md",
                },
                "setup": {
                    "memories": [*facts, *BACKGROUND_FACTS],
                    "tracker_tasks": [title],
                },
            }
        )
    return tasks


def build_tobench_tasks(seed: int = 7) -> list[dict]:
    """12 tasks, deterministic — the seed is accepted for API parity
    with the other evals; the task set is fixed by design."""
    _ = seed
    return _summarize_tasks() + _correct_tasks() + _compute_tasks() + _deliver_tasks()


# ----------------------------------------------------------------------
# Environment setup: seed one fresh temp dir per task run.
# ----------------------------------------------------------------------


def _seed_env(task: dict, root: str) -> dict:
    bundle = build_servers(root)
    servers = bundle["servers"]
    setup = task.get("setup", {})
    mem_ids = {}
    for text in setup.get("memories", []):
        rid = servers["memory"].call_tool("remember", {"text": text})["result"]["id"]
        mem_ids[text] = rid
    for path, content in setup.get("files", {}).items():
        servers["files"].call_tool("write_file", {"path": path, "content": content})
    tracker_ids = {}
    for title in setup.get("tracker_tasks", []):
        tid = servers["tracker"].call_tool("create_task", {"title": title})["result"]["id"]
        tracker_ids[title] = tid
    # Setup calls are not the agent's work: clear the trace.
    bundle["trace"].events.clear()
    return {**bundle, "mem_ids": mem_ids, "tracker_ids": tracker_ids}


# ----------------------------------------------------------------------
# Agents: two deterministic policies. `grounded` does the job the
# right way; `sloppy` is the degraded policy — no recall, naive
# compute, tracker completed before the deliverable, corrections
# without superseding.
# ----------------------------------------------------------------------


def _compute_from_text(text: str, context: dict, dedup: bool) -> float | int:
    fmt = context["format"]
    if fmt == "json-sum":
        items = json.loads(text)
        seen: set = set()
        total = 0.0
        for item in items:
            key = item[context["id_field"]]
            if dedup and key in seen:
                continue
            seen.add(key)
            total += float(item[context["value_field"]])
        return total
    if fmt == "csv-sum":
        rows = list(csv.DictReader(text.splitlines()))
        seen = set()
        total = 0.0
        for row in rows:
            key = row[context["id_field"]]
            if dedup and key in seen:
                continue
            seen.add(key)
            total += float(row[context["value_field"]])
        return total
    if fmt == "json-distinct":
        items = json.loads(text)
        values = [item[context["value_field"]] for item in items]
        return len(set(values)) if dedup else len(values)
    raise ValueError(f"unknown compute format: {fmt}")


def _fmt_number(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def run_grounded(task: dict, bundle: dict) -> str:
    """The good policy: recall first, read sources, compute from the
    real artifacts, do tracker work in the right order."""
    servers = bundle["servers"]
    ctx = task["context"]
    kind = task["kind"]

    if kind == "summarize":
        recalled: list[str] = []
        if ctx["query"]:
            res = servers["memory"].call_tool("recall", {"query": ctx["query"], "k": 5})
            recalled = [r["text"] for r in res["result"]]
        text = "\n".join(recalled) if recalled else ctx.get("prompt_text", "")
        servers["files"].call_tool("write_file", {"path": ctx["artifact"], "content": text + "\n"})
        return text

    if kind == "correct":
        res = servers["memory"].call_tool("recall", {"query": ctx["query"], "k": 5})
        target = next(
            (r for r in res["result"] if r["text"] == ctx["wrong"]),
            None,
        )
        if target is None:
            return "could not find the wrong record"
        servers["memory"].call_tool(
            "remember",
            {"text": ctx["truth"], "type": "correction", "supersedes": target["id"]},
        )
        return ctx["truth"]

    if kind == "compute":
        res = servers["files"].call_tool("read_file", {"path": ctx["source"]})
        value = _compute_from_text(res["result"], ctx, dedup=True)
        servers["files"].call_tool(
            "write_file", {"path": ctx["artifact"], "content": _fmt_number(value) + "\n"}
        )
        return _fmt_number(value)

    if kind == "deliver":
        res = servers["memory"].call_tool("recall", {"query": ctx["query"], "k": 5})
        text = "\n".join(r["text"] for r in res["result"])
        servers["files"].call_tool("write_file", {"path": ctx["artifact"], "content": text + "\n"})
        tid = bundle["tracker_ids"][ctx["title"]]
        servers["tracker"].call_tool("complete_task", {"id": tid})
        servers["files"].call_tool(
            "write_file",
            {"path": ctx["status_file"], "content": f"Completed: {ctx['title']}\n"},
        )
        return text

    raise ValueError(f"unknown kind: {kind}")


def run_sloppy(task: dict, bundle: dict) -> str:
    """The degraded policy: no recall, naive compute, tracker first,
    corrections without superseding. Every shortcut is a real
    failure mode the verifier can name."""
    servers = bundle["servers"]
    ctx = task["context"]
    kind = task["kind"]

    if kind == "summarize":
        if ctx.get("prompt_text"):
            text = ctx["prompt_text"]
        else:
            # No recall: writes the red herring from the goal text.
            text = f"Summary. The user likes tea every morning. The old deploy date was March 3."
        servers["files"].call_tool("write_file", {"path": ctx["artifact"], "content": text + "\n"})
        return text

    if kind == "correct":
        # Stores the truth but never supersedes: the wrong fact stays live.
        servers["memory"].call_tool("remember", {"text": ctx["truth"], "type": "fact"})
        return ctx["truth"]

    if kind == "compute":
        res = servers["files"].call_tool("read_file", {"path": ctx["source"]})
        value = _compute_from_text(res["result"], ctx, dedup=False)
        servers["files"].call_tool(
            "write_file", {"path": ctx["artifact"], "content": _fmt_number(value) + "\n"}
        )
        return _fmt_number(value)

    if kind == "deliver":
        tid = bundle["tracker_ids"][ctx["title"]]
        servers["tracker"].call_tool("complete_task", {"id": tid})
        servers["files"].call_tool(
            "write_file",
            {"path": ctx["artifact"], "content": "Deliverable drafted.\n"},
        )
        servers["files"].call_tool(
            "write_file",
            {"path": ctx["status_file"], "content": f"Completed: {ctx['title']}\n"},
        )
        return "Deliverable drafted."

    raise ValueError(f"unknown kind: {kind}")


AGENTS = {"grounded": run_grounded, "sloppy": run_sloppy}


# ----------------------------------------------------------------------
# Verifiers: artifact + trace + store state. Each check returns
# (name, passed, detail) — the detail must diagnose the failure.
# ----------------------------------------------------------------------


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _read_artifact(bundle: dict, path: str) -> str | None:
    res = bundle["servers"]["files"].call_tool("read_file", {"path": path})
    return res["result"] if res["ok"] else None


def verify_task(task: dict, bundle: dict) -> dict:
    servers = bundle["servers"]
    trace: Trace = bundle["trace"]
    ctx = task["context"]
    kind = task["kind"]
    checks: list[dict] = []
    # Snapshot the agent's call count before the verifier's own reads
    # append to the trace (verifier reads are not agent work).
    n_agent_calls = len(trace)

    # No agent may ever delete a memory to "fix" it.
    checks.append(
        _check(
            "no_forget_used",
            not trace.has("memory", "forget"),
            "forget was called" if trace.has("memory", "forget") else "forget never called",
        )
    )

    if kind == "summarize":
        text = _read_artifact(bundle, ctx["artifact"])
        checks.append(
            _check(
                "artifact_exists",
                text is not None,
                f"{ctx['artifact']} missing" if text is None else f"{ctx['artifact']} written",
            )
        )
        if text is not None:
            missing = [g for g in ctx["gold"] if g not in text]
            checks.append(
                _check(
                    "gold_facts_present",
                    not missing,
                    f"missing gold facts: {missing}" if missing else "all gold facts present",
                )
            )
            leaked = [f for f in ctx["forbidden"] if f in text]
            checks.append(
                _check(
                    "no_forbidden_text",
                    not leaked,
                    f"forbidden text present: {leaked}" if leaked else "no forbidden text",
                )
            )
        if ctx["require_recall"]:
            r = trace.first_index("memory", "recall")
            w = trace.first_index("files", "write_file")
            checks.append(
                _check(
                    "recall_before_write",
                    r is not None and w is not None and r < w,
                    f"recall at trace {r}, first write at trace {w}"
                    if r is not None and w is not None
                    else "recall never called" if r is None else "write never called",
                )
            )

    elif kind == "correct":
        wrong_id = bundle["mem_ids"][ctx["wrong"]]
        # The old record is no longer live: look it up in the store directly.
        row = bundle["store"].conn.execute(
            "SELECT status, superseded_by FROM records WHERE id = ?", (wrong_id,)
        ).fetchone()
        status, superseded_by = row if row else ("missing", None)
        checks.append(
            _check(
                "old_record_superseded",
                status == "superseded" and superseded_by is not None,
                f"old record #{wrong_id} status={status}, superseded_by={superseded_by}",
            )
        )
        corr = None
        if superseded_by is not None:
            row = bundle["store"].conn.execute(
                "SELECT text FROM records WHERE id = ?", (superseded_by,)
            ).fetchone()
            corr = row[0] if row else None
        checks.append(
            _check(
                "correction_text",
                corr is not None and ctx["truth"] in corr,
                f"correction record text: {corr!r}, expected to contain {ctx['truth']!r}",
            )
        )
        r = trace.first_index("memory", "recall")
        m = trace.first_index("memory", "remember")
        checks.append(
            _check(
                "recall_before_remember",
                r is not None and m is not None and r < m,
                f"recall at trace {r}, remember at trace {m}"
                if r is not None and m is not None
                else "recall never called" if r is None else "remember never called",
            )
        )

    elif kind == "compute":
        # Recompute ground truth from the source file, in the verifier.
        src = _read_artifact(bundle, ctx["source"])
        expected = _compute_from_text(src, ctx, dedup=True) if src is not None else None
        text = _read_artifact(bundle, ctx["artifact"])
        checks.append(
            _check(
                "artifact_exists",
                text is not None,
                f"{ctx['artifact']} missing" if text is None else f"{ctx['artifact']} written",
            )
        )
        if text is not None and expected is not None:
            try:
                got = float(text.strip())
                ok = abs(got - float(expected)) < 0.005
            except ValueError:
                got, ok = None, False
            checks.append(
                _check(
                    "value_matches",
                    ok,
                    f"expected {expected} (deduped from {ctx['source']}), got {got!r}",
                )
            )
        r = trace.first_index("files", "read_file")
        w = trace.first_index("files", "write_file")
        checks.append(
            _check(
                "source_read_before_write",
                r is not None and w is not None and r < w,
                f"read at trace {r}, first write at trace {w}"
                if r is not None and w is not None
                else "source never read" if r is None else "write never called",
            )
        )
        checks.append(
            _check(
                "call_budget",
                n_agent_calls <= 8,
                f"{n_agent_calls} agent tool calls (budget 8)",
            )
        )

    elif kind == "deliver":
        text = _read_artifact(bundle, ctx["artifact"])
        checks.append(
            _check(
                "deliverable_exists",
                text is not None,
                f"{ctx['artifact']} missing" if text is None else f"{ctx['artifact']} written",
            )
        )
        if text is not None:
            missing = [g for g in ctx["gold"] if g not in text]
            checks.append(
                _check(
                    "deliverable_content",
                    not missing,
                    f"missing facts in deliverable: {missing}" if missing else "all facts present",
                )
            )
        tasks = servers["tracker"].tool_list_tasks()
        mine = next((t for t in tasks if t["title"] == ctx["title"]), None)
        checks.append(
            _check(
                "task_completed",
                mine is not None and mine["status"] == "completed",
                f"tracker task status: {mine['status'] if mine else 'not found'}",
            )
        )
        status = _read_artifact(bundle, ctx["status_file"])
        checks.append(
            _check(
                "status_mentions_task",
                status is not None and ctx["title"] in status,
                f"status.md content: {status!r}",
            )
        )
        w = trace.first_index("files", "write_file")
        c = trace.first_index("tracker", "complete_task")
        # The last write_file is status.md; order: deliverable, complete, status.
        writes = [i for i, e in enumerate(trace.events) if e["server"] == "files" and e["tool"] == "write_file"]
        status_write = writes[-1] if writes else None
        checks.append(
            _check(
                "ordering_deliver_then_complete_then_status",
                w is not None and c is not None and status_write is not None and w < c < status_write,
                f"deliverable write at {w}, complete at {c}, status write at {status_write}",
            )
        )
        checks.append(
            _check(
                "recall_called",
                trace.has("memory", "recall"),
                "recall called" if trace.has("memory", "recall") else "recall never called",
            )
        )

    passed = all(c["passed"] for c in checks)
    return {"task_id": task["id"], "kind": kind, "passed": passed, "checks": checks}


# ----------------------------------------------------------------------
# Runner + report.
# ----------------------------------------------------------------------


def run_tobench_eval(
    n_tasks: int = 12,
    agents: tuple[str, ...] = ("grounded", "sloppy"),
    seed: int = 7,
) -> dict:
    tasks = build_tobench_tasks(seed=seed)[:n_tasks]
    results: dict[str, dict] = {}
    for name in agents:
        agent_fn = AGENTS[name]
        per_task = []
        for task in tasks:
            with tempfile.TemporaryDirectory(prefix="tobench-") as root:
                bundle = _seed_env(task, root)
                try:
                    agent_fn(task, bundle)
                except Exception as exc:  # a crashed agent fails the task
                    bundle["trace"].events.append(
                        {"server": "agent", "tool": "crash", "args": {"error": str(exc)}}
                    )
                per_task.append(verify_task(task, bundle))
        passed = sum(1 for t in per_task if t["passed"])
        by_kind: dict[str, dict] = {}
        for t in per_task:
            by_kind.setdefault(t["kind"], {"passed": 0, "total": 0})
            by_kind[t["kind"]]["total"] += 1
            by_kind[t["kind"]]["passed"] += 1 if t["passed"] else 0
        results[name] = {
            "passed": passed,
            "total": len(per_task),
            "rate": passed / len(per_task) if per_task else 0.0,
            "per_task": per_task,
            "by_kind": by_kind,
        }
    return {"n_tasks": len(tasks), "agents": results}


def print_tobench_report(result: dict) -> None:
    print("mini-TOBench (synthetic, labelled): 12 tasks, 4 kinds x 3")
    print("reference: full TOBench best model 41.0%, human 94.0%")
    print()
    print(f"{'task':<6}{'kind':<11}" + "".join(f"{a:<10}" for a in result["agents"]))
    print("-" * (17 + 10 * len(result["agents"])))
    tasks = next(iter(result["agents"].values()))["per_task"]
    for i, t in enumerate(tasks):
        row = f"{t['task_id']:<6}{t['kind']:<11}"
        for name in result["agents"]:
            row += ("PASS      " if result["agents"][name]["per_task"][i]["passed"] else "FAIL      ")
        print(row)
    print("-" * (17 + 10 * len(result["agents"])))
    for name, r in result["agents"].items():
        kinds = ", ".join(
            f"{k} {v['passed']}/{v['total']}" for k, v in r["by_kind"].items()
        )
        print(f"{name}: {r['passed']}/{r['total']} ({r['rate']:.3f}) — {kinds}")
    print()
    for name, r in result["agents"].items():
        for t in r["per_task"]:
            if not t["passed"]:
                failed = [c for c in t["checks"] if not c["passed"]]
                print(f"{name} {t['task_id']} failed:")
                for c in failed:
                    print(f"  - {c['name']}: {c['detail']}")


# ----------------------------------------------------------------------
# Minimal stdio JSON-RPC 2.0 wrapper: one request per line, one
# response per line. Methods: "call_tool", "list_tools".
# ----------------------------------------------------------------------


def serve_stdio(server_name: str, root: str, store_path: str | None = None) -> None:
    trace = Trace()
    if server_name == "memory":
        store = Store(store_path or os.path.join(root, "store.db"))
        server: ToolServer = MemoryServer(trace, store)
    elif server_name == "files":
        server = FilesServer(trace, root)
    elif server_name == "tracker":
        server = TrackerServer(trace, root)
    else:
        raise ValueError(f"unknown server: {server_name}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}) + "\n")
            sys.stdout.flush()
            continue
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        if method == "list_tools":
            result = server.tools()
            resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        elif method == "call_tool":
            result = server.call_tool(params.get("name", ""), params.get("arguments") or {})
            resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        else:
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"unknown method: {method}"}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    _p = argparse.ArgumentParser(prog="tobench")
    _p.add_argument("command", choices=["serve"])
    _p.add_argument("--server", required=True, choices=["memory", "files", "tracker"])
    _p.add_argument("--root", required=True)
    _p.add_argument("--store", default=None)
    _a = _p.parse_args()
    serve_stdio(_a.server, _a.root, _a.store)
