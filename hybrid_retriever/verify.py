"""P20 — Verify-by-Doing (VbD) harness.

The paper (arXiv:2611.15276) co-evolves an Actor that executes and an
Evaluator that verifies by *doing checks in the environment* — recomputing
a spreadsheet total from cell values rather than trusting the actor's
headings. This module is the deterministic adaptation of that idea to
this repo, with two substitutions stated plainly:

1. No LLM anywhere. The "actor" is a small deterministic solver that
   parses a tool-output artifact, applies whatever rule memories it
   was given, and writes an answer artifact. The "evaluator" is a
   programmatic verifier that re-parses the raw artifact itself and
   recomputes the expected answer from scratch — it never reads the
   actor's reasoning or its memories, only the artifact it produced.
   What this measures is therefore the causal chain
   retrieval -> memory use -> task success, not free-form agent skill.
2. The Actor prompt + skill files of the paper's Phase 1 become the
   rule registry below, versioned in git with this module. The
   adoption gate (Phase 2) becomes the leave-one-out ablation: a
   rule memory is adopted only if removing it from the store lowers
   task success outside noise.

Tasks reuse the P19 tool-output shapes (a JSON API response with a
duplicated item, a log tail with repeated heartbeats, a markdown
table with a duplicated row) and the P1 synthetic corpus as
background memories. The task wrappers themselves — question +
computed-answer ground truth — are new, because neither fixture
ships a verifiable *doing* task: the P1 queries have gold records
but no computed answer, and the P19 turns have gold facts but no
action to take on them.

Every task is a trap for the naive solver: the artifact contains a
duplicate / repeat that must be handled before computing, and the
handling is stated in exactly one rule memory in the store. The
no-memory arm is the strongest no-feedback baseline; the oracle arm
(the correct rule handed to the solver directly) is the ceiling.
A retrieval arm succeeds only when it surfaces the right rule
memory *and* the solver applies it — so the lift is causal, not
retrieval overlap: a distractor memory that fires first makes the
agent fail, and that failure is counted.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile

from .eval import load_synthetic
from .retrieval import retrieve
from .store import Store

# ----------------------------------------------------------------------
# Rule registry: the actor's "skills", versioned in git.
#
# A rule fires when a memory's text contains ALL of its trigger words
# (whole words, lowercased). The solver applies the FIRST firing rule
# for the task's kind, in the order memories are presented — retrieval
# rank order for the retrieval arms, insertion order for oracle.
# Distractor rules are plausible but wrong: they fire on memories
# that share more words with the question than the correct rule does,
# so a purely lexical retriever can be misled into following them.
# ----------------------------------------------------------------------

RULES = {
    "json": [
        {
            "key": "json-dedup-by-id",
            "triggers": {"deduplicate", "id"},
            "effect": "dedup_by_id",
            "correct": True,
        },
        {
            "key": "json-dedup-by-note",
            "triggers": {"deduplicate", "note"},
            "effect": "dedup_by_note",
            "correct": False,
        },
        {
            "key": "json-include-all",
            "triggers": {"include", "every"},
            "effect": "include_all",
            "correct": False,
        },
    ],
    "log": [
        {
            "key": "log-distinct-only",
            "triggers": {"distinct", "messages"},
            "effect": "distinct_only",
            "correct": True,
        },
        {
            "key": "log-count-all",
            "triggers": {"count", "all"},
            "effect": "count_all",
            "correct": False,
        },
        {
            "key": "log-heartbeats-only",
            "triggers": {"heartbeat", "only"},
            "effect": "heartbeats_only",
            "correct": False,
        },
    ],
    "table": [
        {
            "key": "table-dedup-rows",
            "triggers": {"deduplicate", "rows"},
            "effect": "dedup_rows",
            "correct": True,
        },
        {
            "key": "table-first-ten",
            "triggers": {"first", "ten"},
            "effect": "first_ten",
            "correct": False,
        },
        {
            "key": "table-exclude-even",
            "triggers": {"exclude", "even"},
            "effect": "exclude_even",
            "correct": False,
        },
    ],
}

# The rule memories that live in the store. One correct rule per kind
# plus two distractors per kind. Distractors are worded to share MORE
# vocabulary with the question than the correct rule does: the JSON
# question asks for a "sum" over "JSON" "scores", and the include-all
# distractor says exactly "sum every JSON score"; the correct rule
# says "adding up result scores" and never uses the words "sum" or
# "JSON". That is the lexical trap, by design.
RULE_MEMORIES = {
    "json-dedup-by-id": (
        "When adding up result scores, deduplicate by id first — "
        "the same entry can appear twice."
    ),
    "json-dedup-by-note": (
        "Deduplicate result entries by their note text before "
        "adding up scores."
    ),
    "json-include-all": (
        "Include every JSON item in the sum, even repeats — the "
        "total must reflect the raw file."
    ),
    "log-distinct-only": (
        "Count distinct messages only — the heartbeat lines repeat "
        "and must not be counted twice."
    ),
    "log-count-all": (
        "Count all log lines, including repeats — every line is "
        "evidence."
    ),
    "log-heartbeats-only": (
        "Only the heartbeat lines matter for the log count."
    ),
    "table-dedup-rows": (
        "Deduplicate table rows before summing — one row is "
        "printed twice."
    ),
    "table-first-ten": (
        "Sum only the first ten rows of the table."
    ),
    "table-exclude-even": (
        "Exclude rows with an even id from the table sum."
    ),
}

# The question each kind asks. Deliberately shares vocabulary with the
# distractors, not the correct rule — see RULE_MEMORIES.
QUESTIONS = {
    "json": "What is the sum of scores of unique items in the JSON results?",
    "log": "How many distinct INFO messages are in the log?",
    "table": "What is the sum of the score column over unique rows in the table?",
}

KINDS = ("json", "log", "table")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def matching_rule(kind: str, memory_text: str) -> dict | None:
    """First rule of this kind whose triggers all appear in the memory."""
    words = _words(memory_text)
    for rule in RULES[kind]:
        if rule["triggers"] <= words:
            return rule
    return None


# ----------------------------------------------------------------------
# Task generator — deterministic, seeded. Artifact shapes mirror P19
# (_compression_task_turns in eval.py): JSON list with an exact
# duplicate, log tail with repeated heartbeats, table with a
# duplicated row. Parameters rotate with the task index so the trap
# is not always in the same place.
# ----------------------------------------------------------------------

def _json_artifact(task_idx: int) -> tuple[str, float, float]:
    """(artifact_text, ground_truth, naive_answer)."""
    items = [
        {
            "id": i,
            "note": "unrelated entry %d about topic %d" % (i, i % 5),
            "score": 0.5 * i,
            "detail": "x" * 50,
        }
        for i in range(12)
    ]
    dup_idx = 1 + (task_idx % 11)  # never 0: item 0 scores 0.0, so
    # duplicating it would not change the sum and the trap would
    # be a no-op.
    dup_pos = (task_idx * 5) % 13
    dup = dict(items[dup_idx])
    # Same id, different note: a re-fetch, not a byte-identical
    # copy. Dedup-by-note (the distractor rule) must NOT catch it —
    # only dedup-by-id does. If the duplicate were an exact copy,
    # dedup-by-note would accidentally be correct.
    dup["note"] = "re-fetched entry %d" % dup_idx
    items.insert(dup_pos, dup)
    text = json.dumps(
        {"status": "ok", "count": len(items), "results": items}
    )
    truth = sum(0.5 * i for i in range(12))
    naive = truth + 0.5 * dup_idx
    return text, truth, naive


def _log_artifact(task_idx: int) -> tuple[str, int, int]:
    """(artifact_text, ground_truth, naive_answer)."""
    lines = []
    for i in range(30):
        lines.append(
            "2026-11-24T10:%02d:00Z INFO fetched record %d from shard %d"
            % (i % 60, i, i % 4)
        )
    for _ in range(5):
        lines.append("2026-11-24T10:00:00Z INFO heartbeat ok")
    lines.append("2026-11-24T10:00:00Z ERROR timeout on shard 7")
    text = "\n".join(lines)
    # Distinct INFO messages: 30 fetch messages + 1 heartbeat message.
    truth = 31
    naive = 35  # every INFO line, repeats included
    return text, truth, naive


def _table_artifact(task_idx: int) -> tuple[str, float, float]:
    """(artifact_text, ground_truth, naive_answer)."""
    rows = ["| id | note | score |", "|---|---|---|"]
    for i in range(15):
        rows.append("| %d | unrelated entry %d | %.1f |" % (i, i, 0.1 * i))
    dup_idx = 1 + (task_idx % 14)  # never 0: row 0 scores 0.0.
    rows.append(
        "| %d | unrelated entry %d | %.1f |" % (dup_idx, dup_idx, 0.1 * dup_idx)
    )
    text = "\n".join(rows)
    truth = round(sum(0.1 * i for i in range(15)), 6)
    naive = round(truth + 0.1 * dup_idx, 6)
    return text, truth, naive


_GENERATORS = {
    "json": _json_artifact,
    "log": _log_artifact,
    "table": _table_artifact,
}


def build_vbd_tasks(n_tasks: int = 24, seed: int = 7) -> list[dict]:
    """Deterministic task set: kinds rotate json/log/table.

    ``seed`` is accepted for API stability; the generation is
    already fully determined by the task index (no RNG is used),
    so the same (n_tasks, seed) always yields the same tasks.
    """
    tasks = []
    for i in range(n_tasks):
        kind = KINDS[i % len(KINDS)]
        artifact, truth, naive = _GENERATORS[kind](i)
        assert truth != naive, "trap must change the answer: %s #%d" % (
            kind,
            i,
        )
        tasks.append(
            {
                "task_id": "%s-%02d" % (kind, i),
                "kind": kind,
                "question": QUESTIONS[kind],
                "artifact": artifact,
                "ground_truth": truth,
                "naive_answer": naive,
                "rule_key": next(
                    r["key"] for r in RULES[kind] if r["correct"]
                ),
            }
        )
    return tasks


# ----------------------------------------------------------------------
# The actor: deterministic solver. Parses the artifact, applies the
# first firing rule from its memories (or the naive computation when
# no rule fires), and returns an answer artifact dict.
# ----------------------------------------------------------------------

def _solve_json(artifact: str, effect: str) -> float:
    data = json.loads(artifact)
    items = data["results"]
    if effect == "dedup_by_id":
        seen: set[int] = set()
        total = 0.0
        for it in items:
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            total += it["score"]
        return total
    if effect == "dedup_by_note":
        seen_notes: set[str] = set()
        total = 0.0
        for it in items:
            if it["note"] in seen_notes:
                continue
            seen_notes.add(it["note"])
            total += it["score"]
        return total
    # include_all / no rule: sum everything, repeats included.
    return sum(it["score"] for it in items)


def _solve_log(artifact: str, effect: str) -> int:
    info_msgs = [
        line.split(" INFO ", 1)[1]
        for line in artifact.splitlines()
        if " INFO " in line
    ]
    if effect == "distinct_only":
        return len(set(info_msgs))
    if effect == "heartbeats_only":
        return sum(1 for m in info_msgs if m == "heartbeat ok")
    # count_all / no rule: every INFO line.
    return len(info_msgs)


def _solve_table(artifact: str, effect: str) -> float:
    rows = []
    for line in artifact.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 3 or parts[0] == "id" or set(parts[0]) <= set("-"):
            continue
        try:
            rows.append((int(parts[0]), parts[1], float(parts[2])))
        except ValueError:
            continue
    if effect == "dedup_rows":
        seen: set[tuple] = set()
        total = 0.0
        for row in rows:
            if row in seen:
                continue
            seen.add(row)
            total += row[2]
        return round(total, 6)
    if effect == "first_ten":
        return round(sum(r[2] for r in rows[:10]), 6)
    if effect == "exclude_even":
        return round(sum(r[2] for r in rows if r[0] % 2 == 1), 6)
    # no rule: sum every row.
    return round(sum(r[2] for r in rows), 6)


_SOLVERS = {"json": _solve_json, "log": _solve_log, "table": _solve_table}


def solve_task(task: dict, memories: list[str]) -> dict:
    """Run the deterministic actor on one task.

    ``memories`` are the texts the actor was given, in the order it
    recalls them. Returns the answer artifact dict the verifier will
    read back from disk.
    """
    effect = None
    used_rule = None
    used_memory = None
    for mem in memories:
        rule = matching_rule(task["kind"], mem)
        if rule is not None:
            effect = rule["effect"]
            used_rule = rule["key"]
            used_memory = mem
            break
    answer = _SOLVERS[task["kind"]](task["artifact"], effect or "naive")
    return {
        "task_id": task["task_id"],
        "answer": answer,
        "used_rule": used_rule,
        "used_memory": used_memory,
        "effect": effect or "naive",
    }


# ----------------------------------------------------------------------
# The evaluator: verifies by doing. It re-parses the raw artifact
# itself and recomputes the expected answer from the task spec —
# it never trusts the generator's stored ground truth, the actor's
# reasoning, or its memories. The answer artifact is read back from
# the file the actor wrote, not from the actor's return value.
# ----------------------------------------------------------------------

def expected_answer(task: dict) -> float | int:
    """Independent recomputation from the raw artifact text."""
    kind = task["kind"]
    if kind == "json":
        items = json.loads(task["artifact"])["results"]
        seen: set[int] = set()
        total = 0.0
        for it in items:
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            total += it["score"]
        return total
    if kind == "log":
        msgs = {
            line.split(" INFO ", 1)[1]
            for line in task["artifact"].splitlines()
            if " INFO " in line
        }
        return len(msgs)
    if kind == "table":
        seen: set[tuple] = set()
        total = 0.0
        for line in task["artifact"].splitlines():
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) != 3 or parts[0] == "id" or set(parts[0]) <= set("-"):
                continue
            try:
                row = (int(parts[0]), parts[1], float(parts[2]))
            except ValueError:
                continue
            if row in seen:
                continue
            seen.add(row)
            total += row[2]
        return round(total, 6)
    raise ValueError("unknown kind %r" % kind)


def _answers_equal(got, expected) -> bool:
    if isinstance(expected, float) or isinstance(got, float):
        try:
            return abs(float(got) - float(expected)) < 1e-6
        except (TypeError, ValueError):
            return False
    return got == expected


def write_answer_artifact(run_dir: str, arm: str, result: dict) -> str:
    """Persist one actor run's answer artifact; returns its path."""
    path = os.path.join(run_dir, "%s_%s.json" % (arm, result["task_id"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return path


def verify_artifact(path: str, task: dict) -> dict:
    """Read an answer artifact file and check it against the task.

    Returns {"task_id", "passed", "expected", "got", "reason"}.
    A missing file, unparseable JSON, or missing/wrong-typed answer
    fails with a reason — the verifier reports what it found, it
    never assumes the actor ran.
    """
    expected = expected_answer(task)
    try:
        with open(path, encoding="utf-8") as f:
            artifact = json.load(f)
    except FileNotFoundError:
        return {
            "task_id": task["task_id"],
            "passed": False,
            "expected": expected,
            "got": None,
            "reason": "no answer artifact",
        }
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "task_id": task["task_id"],
            "passed": False,
            "expected": expected,
            "got": None,
            "reason": "unreadable answer artifact: %s" % exc,
        }
    if artifact.get("task_id") != task["task_id"]:
        return {
            "task_id": task["task_id"],
            "passed": False,
            "expected": expected,
            "got": None,
            "reason": "artifact is for another task",
        }
    got = artifact.get("answer", None)
    if got is None or isinstance(got, bool) or not isinstance(got, (int, float)):
        return {
            "task_id": task["task_id"],
            "passed": False,
            "expected": expected,
            "got": got,
            "reason": "answer missing or not a number",
        }
    if _answers_equal(got, expected):
        return {
            "task_id": task["task_id"],
            "passed": True,
            "expected": expected,
            "got": got,
            "reason": "ok",
        }
    return {
        "task_id": task["task_id"],
        "passed": False,
        "expected": expected,
        "got": got,
        "reason": "wrong value",
    }


# ----------------------------------------------------------------------
# Store + arms
# ----------------------------------------------------------------------

def build_vbd_store(
    corpus_path: str, embedder=None, db_path: str | None = None
) -> tuple[Store, dict[str, int]]:
    """Store with the P1 synthetic facts as background noise plus the
    9 rule memories (3 correct + 6 distractors).

    Returns (store, rule_rid) mapping rule key -> record id.
    """
    records, _ = load_synthetic(
        corpus_path,
        os.path.join(
            os.path.dirname(corpus_path), "synthetic_queries.json"
        ),
    )
    tmp = db_path or os.path.join(tempfile.mkdtemp(prefix="hr-vbd-"), "vbd.db")
    store = Store(tmp, embedder=embedder)
    store.add_texts(
        [r["text"] for r in records], type_="fact", embedder=embedder
    )
    rule_rid: dict[str, int] = {}
    for key, text in RULE_MEMORIES.items():
        rid = store.add_texts([text], type_="fact", embedder=embedder)[0]
        rule_rid[key] = rid
    return store, rule_rid


def _retrieve_memories(
    store: Store,
    embedder,
    question: str,
    mode: str,
    k: int,
    candidate_ids: list[int] | None = None,
) -> list[str]:
    if mode == "bm25":
        weights = (1.0, 0.0, 0.0)
    elif mode == "vector":
        weights = (0.0, 1.0, 0.0)
    else:  # hybrid
        weights = (0.50, 0.35, 0.15)
    res = retrieve(
        store,
        embedder,
        question,
        k=k,
        weights=weights,
        use_mmr=False,
        candidate_ids=candidate_ids,
    )
    return [r.text for r in res]


def _paired_lift(
    arm_pass: list[bool], base_pass: list[bool]
) -> dict:
    """McNemar paired lift of an arm over a baseline on the same tasks.

    b = tasks the arm passes and the baseline fails; c = the reverse.
    lift = (b - c) / n; SE = sqrt(b + c) / n. A lift is outside noise
    when |lift| > 2 * SE. With b + c == 0 the arms agree everywhere
    and the lift is exactly 0 with no measurable spread.
    """
    n = len(arm_pass)
    b = sum(1 for a, p in zip(arm_pass, base_pass) if a and not p)
    c = sum(1 for a, p in zip(arm_pass, base_pass) if p and not a)
    lift = (b - c) / n if n else 0.0
    se = math.sqrt(b + c) / n if n and (b + c) > 0 else 0.0
    return {
        "lift": lift,
        "se": se,
        "ci95": 1.96 * se,
        "outside_noise": abs(lift) > 2 * se if (b + c) > 0 else False,
        "arm_only_wins": b,
        "base_only_wins": c,
    }


def run_vbd_eval(
    corpus_path: str,
    embedder=None,
    n_tasks: int = 24,
    k: int = 5,
    seed: int = 7,
    verbose: bool = True,
) -> dict:
    """Run the VbD A/B harness.

    Arms: no-memory (baseline), bm25, vector, hybrid, oracle
    (correct rule handed directly — the ceiling). Then the
    adoption gate: for each correct rule memory, the hybrid arm's
    recalled set is re-run with that memory dropped (leave-one-out
    on the recalled set, retrieval of everything else held fixed);
    the memory's causal lift is hybrid-minus-memory vs hybrid.

    Every actor answer is written to an artifact file under a
    temp run dir and verified by reading that file back.
    """
    tasks = build_vbd_tasks(n_tasks=n_tasks, seed=seed)
    store, _ = build_vbd_store(corpus_path, embedder=embedder)
    correct_keys = sorted(
        r["key"] for rules in RULES.values() for r in rules if r["correct"]
    )
    run_dir = tempfile.mkdtemp(prefix="hr-vbd-run-")
    if verbose:
        print(
            "vbd eval: %d tasks (%s), store %d records "
            "(%d background facts + %d rule memories), k=%d, "
            "embedder=%s"
            % (
                len(tasks),
                ", ".join(
                    "%s=%d" % (kind, sum(1 for t in tasks if t["kind"] == kind))
                    for kind in KINDS
                ),
                store.count(),
                store.count() - len(RULE_MEMORIES),
                len(RULE_MEMORIES),
                k,
                embedder.name if embedder is not None else "none",
            )
        )

    def _run_arm(name: str, memory_fn) -> dict:
        passed: list[bool] = []
        per_task: dict[str, dict] = {}
        for task in tasks:
            memories = memory_fn(task)
            result = solve_task(task, memories)
            path = write_answer_artifact(run_dir, name, result)
            verdict = verify_artifact(path, task)
            passed.append(verdict["passed"])
            per_task[task["task_id"]] = {
                "passed": verdict["passed"],
                "used_rule": result["used_rule"],
                "effect": result["effect"],
                "expected": verdict["expected"],
                "got": verdict["got"],
                "memories": memories,
            }
        return {
            "n": len(tasks),
            "successes": sum(passed),
            "success_rate": sum(passed) / len(tasks) if tasks else 0.0,
            "passed": passed,
            "per_task": per_task,
        }

    arms: dict[str, dict] = {}
    arms["no-memory"] = _run_arm("no-memory", lambda task: [])
    for mode in ("bm25", "vector", "hybrid"):
        arms[mode] = _run_arm(
            mode,
            lambda task, mode=mode: _retrieve_memories(
                store, embedder, task["question"], mode, k
            ),
        )
    arms["oracle"] = _run_arm(
        "oracle",
        lambda task: [RULE_MEMORIES[task["rule_key"]]],
    )

    base = arms["no-memory"]["passed"]
    lifts = {
        name: _paired_lift(arm["passed"], base)
        for name, arm in arms.items()
        if name != "no-memory"
    }

    # --- adoption gate: leave-one-out per correct rule memory -------
    # The ablation removes the memory from the hybrid arm's *recalled*
    # set for each task (holding retrieval of everything else fixed)
    # and re-solves: it isolates the memory's contribution to the
    # agent's decisions. It deliberately does NOT re-run retrieval,
    # so it cannot measure second-order rank shifts — and it is
    # monotone by construction: dropping a memory can only change
    # which rule fires to a later one (or none), never invent a
    # correct firing that was not there before.
    ablations: dict[str, dict] = {}
    hybrid_per_task = arms["hybrid"]["per_task"]
    for key in correct_keys:
        excluded_text = RULE_MEMORIES[key]
        minus_passed: list[bool] = []
        for task in tasks:
            kept = [
                m
                for m in hybrid_per_task[task["task_id"]]["memories"]
                if m != excluded_text
            ]
            result = solve_task(task, kept)
            path = write_answer_artifact(
                run_dir, "hybrid-minus-%s" % key, result
            )
            minus_passed.append(verify_artifact(path, task)["passed"])
        ablations[key] = {
            "success_rate": sum(minus_passed) / len(minus_passed)
            if minus_passed
            else 0.0,
            "successes": sum(minus_passed),
            "causal_lift": _paired_lift(
                arms["hybrid"]["passed"], minus_passed
            ),
        }

    per_kind: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        idx = [i for i, t in enumerate(tasks) if t["kind"] == kind]
        per_kind[kind] = {
            name: (
                sum(arm["passed"][i] for i in idx) / len(idx)
                if idx
                else 0.0
            )
            for name, arm in arms.items()
        }

    store.close()
    return {
        "corpus": "synthetic VbD tasks (labelled) — P19 artifact shapes, "
        "P1 synthetic facts as background memories",
        "n_tasks": len(tasks),
        "k": k,
        "seed": seed,
        "embedder": embedder.name if embedder is not None else "none",
        "arms": {
            name: {
                "n": a["n"],
                "successes": a["successes"],
                "success_rate": a["success_rate"],
                "per_task": a["per_task"],
            }
            for name, a in arms.items()
        },
        "lifts_vs_no_memory": lifts,
        "ablations": ablations,
        "per_kind": per_kind,
        "run_dir": run_dir,
    }


def print_vbd_report(result: dict) -> None:
    print("\n== %s ==" % result["corpus"])
    print(
        "tasks: %d   k=%d   seed=%d   embedder: %s"
        % (
            result["n_tasks"],
            result["k"],
            result["seed"],
            result["embedder"],
        )
    )
    print(
        "%-10s %10s %12s %10s %8s %10s"
        % ("arm", "success", "rate", "lift", "+/- 95%", "noise?")
    )
    base_rate = result["arms"]["no-memory"]["success_rate"]
    for name, arm in result["arms"].items():
        if name == "no-memory":
            print(
                "%-10s %4d/%-4d %12.3f %10s %8s %10s"
                % (
                    name,
                    arm["successes"],
                    arm["n"],
                    arm["success_rate"],
                    "-",
                    "-",
                    "baseline",
                )
            )
            continue
        lift = result["lifts_vs_no_memory"][name]
        print(
            "%-10s %4d/%-4d %12.3f %+10.3f %8.3f %10s"
            % (
                name,
                arm["successes"],
                arm["n"],
                arm["success_rate"],
                lift["lift"],
                lift["ci95"],
                "outside" if lift["outside_noise"] else "within",
            )
        )
    print("per-kind success rate:")
    for kind, rates in result["per_kind"].items():
        print(
            "  %-6s %s"
            % (
                kind,
                "  ".join(
                    "%s=%.3f" % (name, rate)
                    for name, rate in rates.items()
                ),
            )
        )
    print("adoption gate (hybrid minus one correct rule memory):")
    for key, ab in result["ablations"].items():
        lift = ab["causal_lift"]
        print(
            "  %-18s hybrid %d/%d -> minus-memory %d/%d, "
            "causal lift %+.3f +/- %.3f (%s)"
            % (
                key,
                result["arms"]["hybrid"]["successes"],
                result["arms"]["hybrid"]["n"],
                ab["successes"],
                result["arms"]["hybrid"]["n"],
                lift["lift"],
                lift["ci95"],
                "adopt" if lift["outside_noise"] and lift["lift"] > 0
                else "reject/within noise",
            )
        )
    print(
        "baseline (no-memory) rate: %.3f — every task is a trap "
        "for the naive solver by construction" % base_rate
    )
