"""P23 — GEPA-style self-improving loop: trace, diagnose, propose, A/B, keep.

GEPA (ICLR 2026 Oral) learns at the harness level with no weight
updates: log traces, diagnose *why* tasks fail, propose one
prompt/skill edit, A/B it on a fixed task set, and keep it only if
it wins. The Hermes Agent repo (NousResearch/hermes-agent) is the
same idea at the skill level — skills as Markdown files, agentskills.io
standard — so the edits this loop writes are real SKILL.md rules.

Substitutions, stated plainly:

1. No LLM anywhere, and no model API key exists on this VM. GEPA's
   proposer is an LLM that reads failure traces and writes a new
   system prompt; here the proposer is a DETERMINISTIC diagnoser:
   it clusters failures by (task kind, failed verifier check) and
   maps the top cluster to one rule from a fixed rule registry
   (`RULES`). The mapping is the human-written part — the loop
   decides *which* rule to try and *whether* it wins, not what the
   rule says. A rule the registry does not contain can never be
   proposed, which is a real limitation vs the paper: the loop
   cannot invent a fix for a failure mode nobody wrote a rule for.
   `diagnose` reports that case as "no known fix" instead of
   guessing.
2. The "model" being improved is not a model at all: it is the
   deterministic skill agent (`run_skill_agent`), whose behaviour
   is gated on the rules in its SKILL.md. A rule it does not have,
   it does not follow — so the baseline (empty skill) reproduces
   the P21 `sloppy` policy (1/12) and the full skill reproduces
   `grounded` (12/12). Scores measure whether the loop's edits
   move a fixed policy, not whether any LLM improves.
3. Traces are seeded, not live: the loop runs the fixed P21
   12-task set every night, so the failure clusters come from
   repeatable runs, not production traffic. The trace store is
   the durable record — the diagnoser reads the store, never the
   eval's in-memory result — so the loop is honest about what it
   actually logged.
4. One edit per night, exactly as the plan requires: `run_nightly`
   proposes, A/B-tests and keeps-or-rejects a single edit, then
   stops. The bad edit in the demo is a PLANTED NEGATIVE (the
   P2 merge-audit pattern): a rule that completes the tracker
   task before writing anything, which must regress the deliver
   tasks and be rejected by the gate. It is not something the
   diagnoser proposed — the gate test is that it *cannot* get in.

The A/B gate is the same discipline as P20's adoption gate:
an edit wins only if the treatment passes strictly more tasks
than the control, regresses nothing the control passed, and
fixes at least one task in the cluster it was proposed for.
The metric is the verifier's pass count on the fixed 12-task
set — an objective metric, never the agent's self-assessment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile

from .tobench import _seed_env, build_tobench_tasks, verify_task

# ----------------------------------------------------------------------
# Rule registry: the fixed set of skill edits the diagnoser may
# propose. Each rule is one Markdown rule body; the agent's behaviour
# for one task kind is gated on exactly one rule (see run_skill_agent),
# so each kept edit fixes one failure cluster and nothing else.
# ----------------------------------------------------------------------

RULES: dict[str, str] = {
    "recall-before-write": (
        "Before writing any summary, recall the relevant memories "
        "first and write only what was recalled. Never copy a red "
        "herring from the task text (a mentioned tea break, an old "
        "date) into the summary."
    ),
    "supersede-on-correct": (
        "When correcting a stored fact, recall the wrong record "
        "first and store the correction with supersedes set to the "
        "old record's id, so the old fact stops being live. Never "
        "store a correction as a plain fact — the wrong record "
        "would stay live alongside it."
    ),
    "dedup-before-sum": (
        "When totalling or counting items from a file, deduplicate "
        "by the id field first: a repeated id is a re-fetch of the "
        "same item, not a second item. For distinct counts, count "
        "each message value once."
    ),
    "recall-deliver-in-order": (
        "For a tracker task, work in this order: recall the facts, "
        "write the deliverable from what you recalled, then complete "
        "the tracker task, then write status.md. Never complete the "
        "task before the deliverable exists."
    ),
    # Planted negative (never proposed by the diagnoser): completes
    # the tracker task before doing any other work, which breaks
    # the deliver ordering check. The gate must reject it.
    "complete-first": (
        "Complete the tracker task immediately when you start, "
        "before writing anything else."
    ),
}

# (task kind, failed verifier check) -> rule that fixes it.
# A cluster whose check is not listed falls back to the kind row.
RULE_FOR_CLUSTER: dict[tuple[str, str], str] = {
    ("summarize", "recall_before_write"): "recall-before-write",
    ("summarize", "gold_facts_present"): "recall-before-write",
    ("summarize", "no_forbidden_text"): "recall-before-write",
    ("correct", "old_record_superseded"): "supersede-on-correct",
    ("correct", "correction_text"): "supersede-on-correct",
    ("correct", "recall_before_remember"): "supersede-on-correct",
    ("compute", "value_matches"): "dedup-before-sum",
    ("compute", "call_budget"): "dedup-before-sum",
    ("deliver", "ordering_deliver_then_complete_then_status"): "recall-deliver-in-order",
    ("deliver", "recall_called"): "recall-deliver-in-order",
    ("deliver", "deliverable_content"): "recall-deliver-in-order",
    ("deliver", "task_completed"): "recall-deliver-in-order",
}

KIND_FALLBACK: dict[str, str] = {
    "summarize": "recall-before-write",
    "correct": "supersede-on-correct",
    "compute": "dedup-before-sum",
    "deliver": "recall-deliver-in-order",
}

# Kind order for deterministic cluster tie-breaking (task order).
KIND_ORDER = ["summarize", "correct", "compute", "deliver"]

SKILL_FRONTMATTER = (
    "---\n"
    "name: tobench-agent\n"
    "description: Workspace-assistant policy for the P21 mini-TOBench "
    "tasks. Rules are added one per night by the P23 GEPA loop, kept "
    "only when they win their A/B.\n"
    "---\n"
)


def render_skill(rule_ids: list[str]) -> str:
    """Render the full SKILL.md for a set of rule ids (sorted)."""
    lines = [SKILL_FRONTMATTER, "# tobench-agent\n", "## Rules\n"]
    if not rule_ids:
        lines.append("(none yet — the loop has kept no edits)\n")
    for rid in sorted(rule_ids):
        lines.append(f"- `{rid}`: {RULES[rid]}\n")
    return "\n".join(lines)


_RULE_LINE = re.compile(r"^-\s+`([^`]+)`:", re.MULTILINE)


def load_skill_rules(path: str) -> list[str]:
    """Rule ids present in a SKILL.md, in file order. Missing file -> []."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return _RULE_LINE.findall(fh.read())


def apply_rule(path: str, rule_id: str) -> bool:
    """Append a rule to the skill file. Returns False if already present."""
    rules = load_skill_rules(path)
    if rule_id in rules:
        return False
    rules.append(rule_id)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_skill(rules))
    return True


def remove_rule(path: str, rule_id: str) -> bool:
    """Remove a rule from the skill file. Returns False if absent."""
    rules = load_skill_rules(path)
    if rule_id not in rules:
        return False
    rules.remove(rule_id)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_skill(rules))
    return True


def skill_version(rule_ids: list[str]) -> str:
    """Short deterministic version for a rule set (first 8 of sha1)."""
    digest = hashlib.sha1(",".join(sorted(rule_ids)).encode("utf-8")).hexdigest()
    return digest[:8]


# ----------------------------------------------------------------------
# Phase 1 — trace logging. Sessions in SQLite, FTS5 over the prompt
# and the failure text (same external-content trigger pattern as
# store.py's records_fts).
# ----------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    agent_name TEXT NOT NULL,
    skill_version TEXT NOT NULL,
    notes TEXT DEFAULT '',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    task_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    prompt TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    skill_version TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('pass', 'fail')),
    n_calls INTEGER NOT NULL,
    failure_text TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS trace_calls (
    id INTEGER PRIMARY KEY,
    trace_id INTEGER NOT NULL REFERENCES traces(id),
    seq INTEGER NOT NULL,
    server TEXT NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS trace_failures (
    id INTEGER PRIMARY KEY,
    trace_id INTEGER NOT NULL REFERENCES traces(id),
    check_name TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skill_edits (
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id),
    rule_id TEXT NOT NULL,
    cluster_key TEXT NOT NULL,
    cluster_size INTEGER NOT NULL,
    control_passed INTEGER NOT NULL,
    treatment_passed INTEGER NOT NULL,
    fixed_tasks TEXT NOT NULL DEFAULT '[]',
    regressed_tasks TEXT NOT NULL DEFAULT '[]',
    verdict TEXT NOT NULL CHECK (verdict IN ('kept', 'rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE VIRTUAL TABLE IF NOT EXISTS traces_fts
    USING fts5(prompt, failure_text, content='traces', content_rowid='id');
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS traces_fts_ai AFTER INSERT ON traces BEGIN
  INSERT INTO traces_fts(rowid, prompt, failure_text)
  VALUES (new.id, new.prompt, new.failure_text);
END;
CREATE TRIGGER IF NOT EXISTS traces_fts_ad AFTER DELETE ON traces BEGIN
  INSERT INTO traces_fts(traces_fts, rowid, prompt, failure_text)
  VALUES ('delete', old.id, old.prompt, old.failure_text);
END;
CREATE TRIGGER IF NOT EXISTS traces_fts_au AFTER UPDATE ON traces BEGIN
  INSERT INTO traces_fts(traces_fts, rowid, prompt, failure_text)
  VALUES ('delete', old.id, old.prompt, old.failure_text);
  INSERT INTO traces_fts(rowid, prompt, failure_text)
  VALUES (new.id, new.prompt, new.failure_text);
END;
"""


def _fts_query(query: str) -> str | None:
    """Free text -> safe FTS5 OR query of quoted terms (as in store.py)."""
    terms = [t for t in re.split(r"[^A-Za-z0-9_]+", query) if t]
    if not terms:
        return None
    return " OR ".join('"%s"' % t for t in terms)


class TraceStore:
    """SQLite trace log: sessions, per-task traces with calls and
    failed checks, the skill-edit audit table, FTS5 search."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.executescript(_FTS_TRIGGERS)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def new_session(self, agent_name: str, skill_version_: str, notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (agent_name, skill_version, notes) VALUES (?, ?, ?)",
            (agent_name, skill_version_, notes),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_run(
        self,
        session_id: int,
        task: dict,
        trace_events: list[dict],
        verdict: dict,
        agent_name: str,
        skill_version_: str,
    ) -> int:
        """Log one task run: prompt, tool calls, failed checks, outcome."""
        failed = [c for c in verdict["checks"] if not c["passed"]]
        failure_text = "\n".join(f"{c['name']}: {c['detail']}" for c in failed)
        cur = self.conn.execute(
            "INSERT INTO traces (session_id, task_id, kind, prompt, agent_name,"
            " skill_version, outcome, n_calls, failure_text)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                task["id"],
                task["kind"],
                task["goal"],
                agent_name,
                skill_version_,
                "pass" if verdict["passed"] else "fail",
                len(trace_events),
                failure_text,
            ),
        )
        trace_id = cur.lastrowid
        for seq, event in enumerate(trace_events):
            self.conn.execute(
                "INSERT INTO trace_calls (trace_id, seq, server, tool, args_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (trace_id, seq, event["server"], event["tool"],
                 json.dumps(event["args"], sort_keys=True)),
            )
        for c in failed:
            self.conn.execute(
                "INSERT INTO trace_failures (trace_id, check_name, detail)"
                " VALUES (?, ?, ?)",
                (trace_id, c["name"], c["detail"]),
            )
        self.conn.commit()
        return trace_id

    def failures(self, session_id: int) -> list[dict]:
        """Every failed check logged in a session, with task + kind."""
        rows = self.conn.execute(
            "SELECT t.task_id, t.kind, f.check_name, f.detail"
            " FROM trace_failures f JOIN traces t ON t.id = f.trace_id"
            " WHERE t.session_id = ? ORDER BY t.task_id, f.check_name",
            (session_id,),
        ).fetchall()
        return [
            {"task_id": r[0], "kind": r[1], "check": r[2], "detail": r[3]}
            for r in rows
        ]

    def search_failures(self, query: str, session_id: int | None = None,
                        limit: int = 20) -> list[dict]:
        """FTS5 search over logged prompts + failure text."""
        fts_q = _fts_query(query)
        if fts_q is None:
            return []
        sql = (
            "SELECT t.task_id, t.kind, t.prompt, t.failure_text"
            " FROM traces_fts JOIN traces t ON t.id = traces_fts.rowid"
            " WHERE traces_fts MATCH ?"
        )
        params: list = [fts_q]
        if session_id is not None:
            sql += " AND t.session_id = ?"
            params.append(session_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {"task_id": r[0], "kind": r[1], "prompt": r[2], "failures": r[3]}
            for r in rows
        ]

    def record_edit(self, session_id: int, rule_id: str, cluster_key: str,
                    cluster_size: int, control_passed: int, treatment_passed: int,
                    fixed_tasks: list[str], regressed_tasks: list[str],
                    verdict: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO skill_edits (session_id, rule_id, cluster_key,"
            " cluster_size, control_passed, treatment_passed, fixed_tasks,"
            " regressed_tasks, verdict) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, rule_id, cluster_key, cluster_size,
                control_passed, treatment_passed,
                json.dumps(fixed_tasks), json.dumps(regressed_tasks), verdict,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_edits(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT rule_id, cluster_key, cluster_size, control_passed,"
            " treatment_passed, fixed_tasks, regressed_tasks, verdict"
            " FROM skill_edits ORDER BY id"
        ).fetchall()
        return [
            {
                "rule_id": r[0], "cluster_key": r[1], "cluster_size": r[2],
                "control_passed": r[3], "treatment_passed": r[4],
                "fixed_tasks": json.loads(r[5]),
                "regressed_tasks": json.loads(r[6]), "verdict": r[7],
            }
            for r in rows
        ]


# ----------------------------------------------------------------------
# Phase 2 — diagnoser: cluster failures, propose one edit.
# ----------------------------------------------------------------------


def diagnose(store: TraceStore, session_id: int) -> list[dict]:
    """Cluster a session's failures by (kind, check). Ranked by count
    desc, then kind order, then check name — fully deterministic."""
    counts: dict[tuple[str, str], list[dict]] = {}
    for f in store.failures(session_id):
        counts.setdefault((f["kind"], f["check"]), []).append(f)
    clusters = []
    for (kind, check), items in counts.items():
        rule = RULE_FOR_CLUSTER.get((kind, check), KIND_FALLBACK.get(kind))
        clusters.append(
            {
                "kind": kind,
                "check": check,
                "size": len(items),
                "task_ids": sorted({i["task_id"] for i in items}),
                "rule_id": rule,
                "sample_detail": items[0]["detail"],
            }
        )
    clusters.sort(key=lambda c: (-c["size"], KIND_ORDER.index(c["kind"]), c["check"]))
    return clusters


def propose_edit(store: TraceStore, session_id: int,
                 current_rules: list[str]) -> dict | None:
    """Top cluster whose rule is not already in the skill. None when
    every failing cluster's fix is already present, or when the top
    cluster has no known fix."""
    for cluster in diagnose(store, session_id):
        if cluster["rule_id"] is None:
            continue
        if cluster["rule_id"] in current_rules:
            continue
        return {
            "rule_id": cluster["rule_id"],
            "rule_text": RULES[cluster["rule_id"]],
            "cluster_key": f"{cluster['kind']}/{cluster['check']}",
            "cluster_size": cluster["size"],
            "cluster_kind": cluster["kind"],
            "cluster_task_ids": cluster["task_ids"],
            "sample_detail": cluster["sample_detail"],
        }
    return None


# ----------------------------------------------------------------------
# The skill agent: one deterministic policy whose behaviour is gated
# on the rules in its SKILL.md. With no rules it is the P21 sloppy
# policy; with all four real rules it is the grounded policy. Each
# rule switches exactly one kind's behaviour to the grounded version.
# ----------------------------------------------------------------------


def run_skill_agent(task: dict, bundle: dict, rules: set[str]) -> str:
    """Run one task following only the rules in `rules`."""
    from .tobench import _compute_from_text, _fmt_number

    servers = bundle["servers"]
    ctx = task["context"]
    kind = task["kind"]

    if kind == "summarize":
        if "recall-before-write" in rules and ctx["query"]:
            res = servers["memory"].call_tool("recall", {"query": ctx["query"], "k": 5})
            text = "\n".join(r["text"] for r in res["result"])
        elif ctx.get("prompt_text"):
            text = ctx["prompt_text"]
        else:
            text = "Summary. The user likes tea every morning. The old deploy date was March 3."
        servers["files"].call_tool("write_file", {"path": ctx["artifact"], "content": text + "\n"})
        return text

    if kind == "correct":
        if "supersede-on-correct" in rules:
            res = servers["memory"].call_tool("recall", {"query": ctx["query"], "k": 5})
            target = next(
                (r for r in res["result"] if r["text"] == ctx["wrong"]), None
            )
            if target is None:
                return "could not find the wrong record"
            servers["memory"].call_tool(
                "remember",
                {"text": ctx["truth"], "type": "correction", "supersedes": target["id"]},
            )
            return ctx["truth"]
        servers["memory"].call_tool("remember", {"text": ctx["truth"], "type": "fact"})
        return ctx["truth"]

    if kind == "compute":
        res = servers["files"].call_tool("read_file", {"path": ctx["source"]})
        value = _compute_from_text(res["result"], ctx, dedup="dedup-before-sum" in rules)
        servers["files"].call_tool(
            "write_file", {"path": ctx["artifact"], "content": _fmt_number(value) + "\n"}
        )
        return _fmt_number(value)

    if kind == "deliver":
        tid = bundle["tracker_ids"][ctx["title"]]
        if "complete-first" in rules:
            # Planted-negative behaviour: complete before any other work.
            servers["tracker"].call_tool("complete_task", {"id": tid})
            if "recall-deliver-in-order" in rules:
                res = servers["memory"].call_tool("recall", {"query": ctx["query"], "k": 5})
                text = "\n".join(r["text"] for r in res["result"])
            else:
                text = "Deliverable drafted."
            servers["files"].call_tool("write_file", {"path": ctx["artifact"], "content": text + "\n"})
            servers["files"].call_tool(
                "write_file",
                {"path": ctx["status_file"], "content": f"Completed: {ctx['title']}\n"},
            )
            return text
        if "recall-deliver-in-order" in rules:
            res = servers["memory"].call_tool("recall", {"query": ctx["query"], "k": 5})
            text = "\n".join(r["text"] for r in res["result"])
            servers["files"].call_tool("write_file", {"path": ctx["artifact"], "content": text + "\n"})
            servers["tracker"].call_tool("complete_task", {"id": tid})
            servers["files"].call_tool(
                "write_file",
                {"path": ctx["status_file"], "content": f"Completed: {ctx['title']}\n"},
            )
            return text
        servers["tracker"].call_tool("complete_task", {"id": tid})
        servers["files"].call_tool(
            "write_file", {"path": ctx["artifact"], "content": "Deliverable drafted.\n"}
        )
        servers["files"].call_tool(
            "write_file",
            {"path": ctx["status_file"], "content": f"Completed: {ctx['title']}\n"},
        )
        return "Deliverable drafted."

    raise ValueError(f"unknown kind: {kind}")


def run_skill_eval(rules: set[str] | list[str], n_tasks: int = 12,
                   seed: int = 7) -> dict:
    """Score the skill agent on the fixed P21 task set. Same result
    shape as one agent's entry in run_tobench_eval, plus the raw
    per-task trace events so the caller can log them."""
    rules = set(rules)
    tasks = build_tobench_tasks(seed=seed)[:n_tasks]
    per_task = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="gepa-") as root:
            bundle = _seed_env(task, root)
            try:
                run_skill_agent(task, bundle, rules)
            except Exception as exc:  # a crashed agent fails the task
                bundle["trace"].events.append(
                    {"server": "agent", "tool": "crash", "args": {"error": str(exc)}}
                )
            # Snapshot the agent's calls BEFORE verify_task's own
            # reads land in the trace — the log is the agent's work.
            n_agent_calls = len(bundle["trace"])
            verdict = verify_task(task, bundle)
            per_task.append(
                {
                    **verdict,
                    "trace_events": [dict(e) for e in bundle["trace"].events[:n_agent_calls]],
                }
            )
    passed = sum(1 for t in per_task if t["passed"])
    by_kind: dict[str, dict] = {}
    for t in per_task:
        by_kind.setdefault(t["kind"], {"passed": 0, "total": 0})
        by_kind[t["kind"]]["total"] += 1
        by_kind[t["kind"]]["passed"] += 1 if t["passed"] else 0
    return {
        "passed": passed,
        "total": len(per_task),
        "rate": passed / len(per_task) if per_task else 0.0,
        "per_task": per_task,
        "by_kind": by_kind,
    }


# ----------------------------------------------------------------------
# Phase 3 — A/B gate. Same discipline as P20's adoption gate, on the
# fixed task set instead of a leave-one-out ablation.
# ----------------------------------------------------------------------


def ab_test(control_rules: set[str] | list[str], candidate_rule: str,
            target_kind: str, n_tasks: int = 12, seed: int = 7) -> dict:
    """Control = current skill, treatment = current skill + candidate.
    Win requires ALL of: strictly more passes, no task the control
    passed now fails, and at least one task of the target kind fixed.
    """
    control = run_skill_eval(set(control_rules), n_tasks=n_tasks, seed=seed)
    treatment_rules = set(control_rules) | {candidate_rule}
    treatment = run_skill_eval(treatment_rules, n_tasks=n_tasks, seed=seed)
    control_pass = {t["task_id"] for t in control["per_task"] if t["passed"]}
    treatment_pass = {t["task_id"] for t in treatment["per_task"] if t["passed"]}
    fixed = sorted(treatment_pass - control_pass)
    regressed = sorted(control_pass - treatment_pass)
    fixed_target = [t for t in fixed
                    if next(x for x in treatment["per_task"] if x["task_id"] == t)["kind"] == target_kind]
    win = (
        treatment["passed"] > control["passed"]
        and not regressed
        and len(fixed_target) > 0
    )
    return {
        "candidate_rule": candidate_rule,
        "target_kind": target_kind,
        "control_passed": control["passed"],
        "treatment_passed": treatment["passed"],
        "control": control,
        "treatment": treatment,
        "fixed_tasks": fixed,
        "regressed_tasks": regressed,
        "fixed_target_tasks": fixed_target,
        "win": win,
    }


# ----------------------------------------------------------------------
# The nightly loop: one edit per run.
# ----------------------------------------------------------------------


def run_nightly(store: TraceStore, skill_path: str, notes: str = "",
                n_tasks: int = 12, seed: int = 7) -> dict:
    """One night: log a full eval run, diagnose the top failure
    cluster from the trace store, A/B the proposed edit, keep or
    reject it. Returns the night's report."""
    rules = load_skill_rules(skill_path)
    version = skill_version(rules)
    session_id = store.new_session("skill-agent", version, notes=notes)
    result = run_skill_eval(rules, n_tasks=n_tasks, seed=seed)
    tasks = build_tobench_tasks(seed=seed)[:n_tasks]
    for task, verdict in zip(tasks, result["per_task"]):
        store.log_run(
            session_id, task, verdict.pop("trace_events"), verdict,
            "skill-agent", version,
        )
    report: dict = {
        "session_id": session_id,
        "skill_version_before": version,
        "rules_before": sorted(rules),
        "eval_passed": result["passed"],
        "eval_total": result["total"],
        "eval_by_kind": result["by_kind"],
    }
    proposal = propose_edit(store, session_id, rules)
    if proposal is None:
        report.update({"proposal": None, "verdict": "no_proposal",
                       "reason": "no failing cluster has a new fix to try"})
        return report
    report["proposal"] = proposal
    gate = ab_test(rules, proposal["rule_id"], proposal["cluster_kind"],
                   n_tasks=n_tasks, seed=seed)
    report["ab"] = {
        "candidate_rule": gate["candidate_rule"],
        "control_passed": gate["control_passed"],
        "treatment_passed": gate["treatment_passed"],
        "fixed_tasks": gate["fixed_tasks"],
        "regressed_tasks": gate["regressed_tasks"],
        "win": gate["win"],
    }
    if gate["win"]:
        apply_rule(skill_path, proposal["rule_id"])
        store.record_edit(
            session_id, proposal["rule_id"], proposal["cluster_key"],
            proposal["cluster_size"], gate["control_passed"],
            gate["treatment_passed"], gate["fixed_tasks"],
            gate["regressed_tasks"], "kept",
        )
        report["verdict"] = "kept"
    else:
        store.record_edit(
            session_id, proposal["rule_id"], proposal["cluster_key"],
            proposal["cluster_size"], gate["control_passed"],
            gate["treatment_passed"], gate["fixed_tasks"],
            gate["regressed_tasks"], "rejected",
        )
        report["verdict"] = "rejected"
    report["rules_after"] = load_skill_rules(skill_path)
    return report


def run_gepa_demo(nights: int = 4, bad_edit_trial: bool = True,
                  n_tasks: int = 12, seed: int = 7) -> dict:
    """Full loop demo in temp dirs: fresh skill + trace store, N
    nights, then the planted bad-edit trial. Returns everything the
    README table needs."""
    tmp = tempfile.mkdtemp(prefix="gepa-demo-")
    skill_path = os.path.join(tmp, "SKILL.md")
    with open(skill_path, "w", encoding="utf-8") as fh:
        fh.write(render_skill([]))
    store = TraceStore(os.path.join(tmp, "traces.db"))
    reports = []
    for night in range(1, nights + 1):
        reports.append(run_nightly(store, skill_path, notes=f"night {night}",
                                   n_tasks=n_tasks, seed=seed))
    bad_edit = None
    if bad_edit_trial:
        rules = load_skill_rules(skill_path)
        session_id = store.new_session("skill-agent", skill_version(rules),
                                       notes="planted bad-edit trial")
        gate = ab_test(rules, "complete-first", "deliver",
                       n_tasks=n_tasks, seed=seed)
        store.record_edit(
            session_id, "complete-first", "deliver/ordering (planted)",
            3, gate["control_passed"], gate["treatment_passed"],
            gate["fixed_tasks"], gate["regressed_tasks"],
            "kept" if gate["win"] else "rejected",
        )
        bad_edit = {
            "candidate_rule": "complete-first",
            "control_passed": gate["control_passed"],
            "treatment_passed": gate["treatment_passed"],
            "fixed_tasks": gate["fixed_tasks"],
            "regressed_tasks": gate["regressed_tasks"],
            "win": gate["win"],
            "verdict": "kept" if gate["win"] else "rejected",
        }
    kept = [e for e in store.list_edits() if e["verdict"] == "kept"]
    rejected = [e for e in store.list_edits() if e["verdict"] == "rejected"]
    result = {
        "skill_path": skill_path,
        "traces_db": store.path,
        "nights": reports,
        "final_rules": load_skill_rules(skill_path),
        "final_eval": run_skill_eval(load_skill_rules(skill_path),
                                     n_tasks=n_tasks, seed=seed),
        "kept_edits": kept,
        "rejected_edits": rejected,
        "bad_edit_trial": bad_edit,
    }
    store.close()
    return result


def print_gepa_report(result: dict) -> None:
    print("P23 GEPA-style loop — one edit per night, A/B on the fixed P21 set")
    print()
    for i, night in enumerate(result["nights"], 1):
        p = night["proposal"]
        print(f"night {i}: eval {night['eval_passed']}/{night['eval_total']}"
              f" (skill {night['skill_version_before']})")
        if p is None:
            print(f"  no proposal: {night['reason']}")
            continue
        ab = night["ab"]
        print(f"  cluster {p['cluster_key']} (n={p['cluster_size']}):"
              f" propose `{p['rule_id']}`")
        print(f"  A/B control {ab['control_passed']} -> treatment"
              f" {ab['treatment_passed']}, fixed {ab['fixed_tasks']},"
              f" regressed {ab['regressed_tasks']} -> {night['verdict']}")
    final = result["final_eval"]
    print()
    print(f"final skill ({len(result['final_rules'])} rules):"
          f" {', '.join(result['final_rules'])}")
    print(f"final eval: {final['passed']}/{final['total']}"
          f" — " + ", ".join(
              f"{k} {v['passed']}/{v['total']}" for k, v in final["by_kind"].items()))
    if result["bad_edit_trial"]:
        b = result["bad_edit_trial"]
        print(f"planted bad edit `complete-first`: control {b['control_passed']}"
              f" -> treatment {b['treatment_passed']}, fixed {b['fixed_tasks']},"
              f" regressed {b['regressed_tasks']} -> {b['verdict']}")
    print(f"kept: {len(result['kept_edits'])},"
          f" rejected: {len(result['rejected_edits'])}")
