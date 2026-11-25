"""P5 — IBM-style lessons miner: mine lessons from trajectories,
merge them into guidelines, inject at inference, A/B the injection.

Source paper: "How Much Should an Agent Remember?"
(arXiv:2611.26361, IBM, ships in altk-evolve): mine 1-3 short lessons
per episode from the agent's own trajectories, merge them into a
guidelines file, inject the guidelines at inference, and A/B test
full-guideline injection vs retrieved-subset injection — the right
amount of memory is a property of the model, not the memory.

Substitutions, stated plainly (same stance as P21/P23):

1. No LLM anywhere, and no model API key exists on this VM. The
   paper's miner is an LLM that reads a trajectory and writes
   lessons; here lesson extraction is DETERMINISTIC and
   rule-based: a failed verifier check becomes a lesson naming
   that check and its fix (the fix text comes from a fixed table,
   ``CHECK_FIX``, keyed by check name — for checks the P23 rule
   registry covers, the lesson text IS the registry rule text);
   a passed episode yields a "what worked" lesson only when a
   verifier check that maps to a concrete, checkable rule passed.
   A lesson the table does not know how to fix can never be
   mined, which is the same limitation P23's diagnoser has.
2. Trajectories are seeded, not live: the episodes are runs of
   the deterministic P23 skill agent over the fixed P21 12-task
   set (seed 7). ``mine_from_eval`` mines from the in-memory
   verifier verdicts; ``mine_from_trace_store`` mines the same
   episodes from the P23 trace store, so the two paths can be
   checked against each other. The trace store logs only failed
   checks, so a passed episode mined from the store names its
   kind's primary verifier check as the worked lesson — that
   check must have passed for the task to pass, but the wording
   is inferred from the kind, not read from a logged record.
3. The "model" being improved is not a model: it is the P23
   skill agent, whose behaviour is gated on rule ids. A
   guideline injects by activating its ``rule_id`` in the skill
   agent, so injection is measurable on the fixed task set.
   Guideline retrieval for the retrieved-subset arm is
   deterministic token-overlap (light stemming, stopwords
   removed) between the task text and the guideline text — no
   embeddings, no LLM ranker.
4. The backfiring guideline is a PLANTED NEGATIVE (the P2/P23
   merge-audit pattern): a guideline whose rule is P23's
   ``complete-first`` (complete the tracker task before writing
   anything), which contradicts the verifier's deliver-ordering
   rule. Its support is set SYNTHETICALLY high (12, above every
   real guideline's support) precisely to mimic the paper's
   high-support guideline that backfired — support must not
   override the gate. It is flagged ``planted=True`` with
   provenance ``["planted-negative"]``, never attributed to a
   real episode.

Provenance is the load-bearing invariant (the paper's failure
modes include provenance silently dropped in consolidation):
every guideline tracks ``support`` (how many lessons it merged)
AND ``provenance`` (the sorted episode ids it came from) AND
``checks`` (the distinct verifier checks that produced those
lessons). Merging unions provenance; it never drops an episode.
``test_lessons.py::test_merge_preserves_provenance`` pins this,
and the guidelines file round-trips provenance through Markdown.
"""

from __future__ import annotations

import math
import os
import re
import tempfile

from .gepa import RULES, TraceStore, run_skill_agent, run_skill_eval
from .tobench import _seed_env, build_tobench_tasks, verify_task

# ----------------------------------------------------------------------
# Phase 1 — lesson extraction from verifier verdicts.
#
# CHECK_FIX maps a verifier check name to (rule_id, failed-lesson
# text, worked-lesson text). Checks the P23 rule registry covers
# reuse the registry rule text verbatim as the failed lesson, so
# the merged guideline's canonical text is the rule the agent
# follows. Checks with no rule (rule_id None) still mine a
# lesson — a real fix the agent cannot currently gate on — and
# merge into a guideline with rule_id None, which injects
# nothing. A check nobody wrote a fix for falls back to the
# generic template instead of being guessed.
# ----------------------------------------------------------------------

_GENERIC_FAILED = (
    "Check {check} failed on this episode; find what the verifier "
    "expected, fix the tool use that produced it, and re-run the "
    "verifier before moving on."
)
_GENERIC_WORKED = (
    "Check {check} passed; keep doing whatever produced it and "
    "change nothing about that step."
)

# (rule_id, failed text, worked text) per verifier check.
CHECK_FIX: dict[str, tuple[str | None, str, str]] = {
    "recall_before_write": (
        "recall-before-write",
        RULES["recall-before-write"],
        "What worked: " + RULES["recall-before-write"],
    ),
    "gold_facts_present": (
        "recall-before-write",
        RULES["recall-before-write"],
        "What worked: " + RULES["recall-before-write"],
    ),
    "no_forbidden_text": (
        "recall-before-write",
        RULES["recall-before-write"],
        "What worked: " + RULES["recall-before-write"],
    ),
    "old_record_superseded": (
        "supersede-on-correct",
        RULES["supersede-on-correct"],
        "What worked: " + RULES["supersede-on-correct"],
    ),
    "correction_text": (
        "supersede-on-correct",
        RULES["supersede-on-correct"],
        "What worked: " + RULES["supersede-on-correct"],
    ),
    "recall_before_remember": (
        "supersede-on-correct",
        RULES["supersede-on-correct"],
        "What worked: " + RULES["supersede-on-correct"],
    ),
    "value_matches": (
        "dedup-before-sum",
        RULES["dedup-before-sum"],
        "What worked: " + RULES["dedup-before-sum"],
    ),
    "deliverable_content": (
        "recall-deliver-in-order",
        RULES["recall-deliver-in-order"],
        "What worked: " + RULES["recall-deliver-in-order"],
    ),
    "task_completed": (
        "recall-deliver-in-order",
        RULES["recall-deliver-in-order"],
        "What worked: " + RULES["recall-deliver-in-order"],
    ),
    "status_mentions_task": (
        "recall-deliver-in-order",
        RULES["recall-deliver-in-order"],
        "What worked: " + RULES["recall-deliver-in-order"],
    ),
    "ordering_deliver_then_complete_then_status": (
        "recall-deliver-in-order",
        RULES["recall-deliver-in-order"],
        "What worked: " + RULES["recall-deliver-in-order"],
    ),
    "recall_called": (
        "recall-deliver-in-order",
        RULES["recall-deliver-in-order"],
        "What worked: " + RULES["recall-deliver-in-order"],
    ),
    "artifact_exists": (
        None,
        "Write the artifact file before finishing the task; a "
        "missing artifact fails the task outright.",
        "What worked: the artifact file was written before the task finished.",
    ),
    "source_read_before_write": (
        None,
        "Read the source file before writing the artifact; compute "
        "from the real file, never from memory of it.",
        "What worked: the source file was read before the artifact was written.",
    ),
    "call_budget": (
        None,
        "Keep the tool-call count within budget; do not re-read or "
        "re-list what was already retrieved.",
        "What worked: the task finished within the tool-call budget.",
    ),
    "deliverable_exists": (
        None,
        "Write the deliverable file before finishing; a missing "
        "deliverable fails the task outright.",
        "What worked: the deliverable file was written before the task finished.",
    ),
    "no_forget_used": (
        None,
        "Never delete a memory to fix it; store the correction with "
        "supersedes so the old fact stops being live.",
        "What worked: no memory was deleted to fix anything.",
    ),
}

# The one check per kind whose passing proves the kind's rule was
# followed — the source of a passed episode's single worked lesson.
KIND_PRIMARY_CHECK: dict[str, str] = {
    "summarize": "recall_before_write",
    "correct": "old_record_superseded",
    "compute": "value_matches",
    "deliver": "ordering_deliver_then_complete_then_status",
}

MAX_LESSONS_PER_EPISODE = 3


class Lesson:
    """One mined lesson: what (failed | worked), on which check,
    from which episode, with the verifier's own detail as evidence."""

    def __init__(self, lesson_id: str, episode_id: str, session_id: int,
                 kind: str, check: str, polarity: str, text: str,
                 rule_id: str | None, evidence: str = "") -> None:
        self.lesson_id = lesson_id
        self.episode_id = episode_id
        self.session_id = session_id
        self.kind = kind
        self.check = check
        self.polarity = polarity  # 'failed' | 'worked'
        self.text = text
        self.rule_id = rule_id
        self.evidence = evidence

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Lesson({self.lesson_id!r}, {self.polarity}, "
                f"rule={self.rule_id!r})")


def _failed_text(check: str) -> tuple[str | None, str]:
    entry = CHECK_FIX.get(check)
    if entry is None:
        return None, _GENERIC_FAILED.format(check=check)
    return entry[0], entry[1]


def _worked_text(check: str) -> tuple[str | None, str]:
    entry = CHECK_FIX.get(check)
    if entry is None:
        return None, _GENERIC_WORKED.format(check=check)
    return entry[0], entry[2]


def extract_lessons(task: dict, verdict: dict, session_id: int = 0) -> list[Lesson]:
    """Mine 0-3 lessons from one episode's verifier verdict.

    Failed checks first, in verdict order, capped at 3 — each
    names its check and its fix. A passed episode yields at most
    one worked lesson, and only when the kind's primary check
    passed (a concrete, checkable rule, not "it worked"). An
    episode with no failure and no primary check in its verdict
    (e.g. S3, prompt-contained, no recall required) yields zero
    lessons: there is nothing attributable to learn from it.
    """
    task_id = task["id"]
    kind = task["kind"]
    lessons: list[Lesson] = []
    failed = [c for c in verdict["checks"] if not c["passed"]]
    for c in failed[:MAX_LESSONS_PER_EPISODE]:
        rule_id, text = _failed_text(c["name"])
        lessons.append(Lesson(
            lesson_id=f"{task_id}:{c['name']}:failed",
            episode_id=task_id, session_id=session_id, kind=kind,
            check=c["name"], polarity="failed", text=text,
            rule_id=rule_id, evidence=c["detail"],
        ))
    if not failed:
        primary = KIND_PRIMARY_CHECK.get(kind)
        passed = {c["name"]: c for c in verdict["checks"] if c["passed"]}
        if primary is not None and primary in passed:
            rule_id, text = _worked_text(primary)
            lessons.append(Lesson(
                lesson_id=f"{task_id}:{primary}:worked",
                episode_id=task_id, session_id=session_id, kind=kind,
                check=primary, polarity="worked", text=text,
                rule_id=rule_id, evidence=passed[primary]["detail"],
            ))
    return lessons


def mine_from_eval(rules: set[str] | list[str] = (), n_tasks: int = 12,
                   seed: int = 7) -> dict:
    """Run the skill agent on the fixed set and mine every episode.
    Returns the lessons plus the cleaned eval result (trace events
    stripped — they are the agent's calls, not the lesson data)."""
    result = run_skill_eval(set(rules), n_tasks=n_tasks, seed=seed)
    tasks = build_tobench_tasks(seed=seed)[:n_tasks]
    lessons: list[Lesson] = []
    for task, verdict in zip(tasks, result["per_task"]):
        lessons.extend(extract_lessons(task, verdict))
    for verdict in result["per_task"]:
        verdict.pop("trace_events", None)
    return {"lessons": lessons, "eval": result}


def mine_from_trace_store(store: TraceStore, session_id: int) -> list[Lesson]:
    """Mine the same episodes from the P23 trace store instead of
    from in-memory verdicts. The store logs failed checks (with
    details) and the per-task outcome, so failure lessons are
    read straight from the log; a passed episode's worked lesson
    names its kind's primary check, which must have passed for
    the task to pass (see the module docstring, substitution 2).
    """
    rows = store.conn.execute(
        "SELECT task_id, kind, outcome FROM traces WHERE session_id = ?"
        " ORDER BY task_id",
        (session_id,),
    ).fetchall()
    failures: dict[str, list[dict]] = {}
    for f in store.failures(session_id):
        failures.setdefault(f["task_id"], []).append(f)
    lessons: list[Lesson] = []
    for task_id, kind, outcome in rows:
        failed = sorted(failures.get(task_id, []), key=lambda f: f["check"])
        for f in failed[:MAX_LESSONS_PER_EPISODE]:
            rule_id, text = _failed_text(f["check"])
            lessons.append(Lesson(
                lesson_id=f"{task_id}:{f['check']}:failed",
                episode_id=task_id, session_id=session_id, kind=kind,
                check=f["check"], polarity="failed", text=text,
                rule_id=rule_id, evidence=f["detail"],
            ))
        if outcome == "pass" and not failed:
            primary = KIND_PRIMARY_CHECK.get(kind)
            if primary is not None:
                rule_id, text = _worked_text(primary)
                lessons.append(Lesson(
                    lesson_id=f"{task_id}:{primary}:worked",
                    episode_id=task_id, session_id=session_id, kind=kind,
                    check=primary, polarity="worked", text=text,
                    rule_id=rule_id, evidence="task passed; primary check must hold",
                ))
    return lessons


# ----------------------------------------------------------------------
# Phase 2 — dedup + merge into guidelines. Provenance survives
# every merge: a guideline's provenance is the union of its
# lessons' episode ids, its support is the lesson count, and its
# checks list is the union of the checks that produced them.
# ----------------------------------------------------------------------


class Guideline:
    """One merged guideline: canonical text, how many lessons it
    merged (support), and exactly which episodes they came from
    (provenance). ``rule_id`` is what injection activates in the
    skill agent; None means the guideline injects nothing."""

    def __init__(self, guideline_id: str, text: str, kind: str, check: str,
                 checks: list[str], polarity: str, rule_id: str | None,
                 support: int, provenance: list[str],
                 lesson_ids: list[str], planted: bool = False) -> None:
        self.guideline_id = guideline_id
        self.text = text
        self.kind = kind
        self.check = check
        self.checks = sorted(checks)
        self.polarity = polarity
        self.rule_id = rule_id
        self.support = support
        self.provenance = sorted(provenance)
        self.lesson_ids = sorted(lesson_ids)
        self.planted = planted

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Guideline({self.guideline_id!r}, support={self.support}, "
                f"episodes={self.provenance})")


def _merge_key(lesson: Lesson) -> tuple[str, str, str]:
    """Lessons merge when they teach the same fix for the same
    kind: same rule when one exists, else same check. Polarity
    never merges — a failed lesson and a worked lesson are
    different claims even when the rule matches."""
    fix = lesson.rule_id if lesson.rule_id is not None else f"check:{lesson.check}"
    return (lesson.kind, fix, lesson.polarity)


def _canonical_text(lessons: list[Lesson], rule_id: str | None) -> str:
    """The group's canonical text: the registry rule text when
    the group has a rule (every lesson then carries it verbatim),
    else the most common lesson text (ties: lexicographically
    smallest, so the choice is deterministic)."""
    if rule_id is not None and rule_id in RULES:
        return RULES[rule_id]
    counts: dict[str, int] = {}
    for lesson in lessons:
        counts[lesson.text] = counts.get(lesson.text, 0) + 1
    best = max(counts.values())
    return sorted(t for t, n in counts.items() if n == best)[0]


def _primary_check(lessons: list[Lesson]) -> str:
    counts: dict[str, int] = {}
    for lesson in lessons:
        counts[lesson.check] = counts.get(lesson.check, 0) + 1
    best = max(counts.values())
    return sorted(c for c, n in counts.items() if n == best)[0]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t}


def merge_lessons(lessons: list[Lesson]) -> list[Guideline]:
    """Deduplicate + merge lessons into guidelines.

    Pass 1 groups by (kind, rule-or-check, polarity). Pass 2
    merges groups of the same kind + polarity whose canonical
    texts are near-duplicates (token Jaccard >= 0.8) — a
    second-chance dedup for lessons whose fixes were worded
    differently but say the same thing. Both passes UNION
    provenance, support, checks and lesson ids; nothing is
    dropped. Guidelines are numbered G1.. in (-support, kind,
    fix, polarity) order, so ids are deterministic.
    """
    groups: dict[tuple[str, str, str], list[Lesson]] = {}
    for lesson in lessons:
        groups.setdefault(_merge_key(lesson), []).append(lesson)
    merged: list[dict] = []
    for (kind, fix, polarity), group in groups.items():
        rule_id = group[0].rule_id
        merged.append({
            "kind": kind, "fix": fix, "polarity": polarity,
            "rule_id": rule_id,
            "text": _canonical_text(group, rule_id),
            "check": _primary_check(group),
            "checks": sorted({le.check for le in group}),
            "support": len(group),
            "provenance": sorted({le.episode_id for le in group}),
            "lesson_ids": sorted(le.lesson_id for le in group),
        })
    # Pass 2: near-duplicate canonical texts, same kind+polarity.
    changed = True
    while changed:
        changed = False
        merged.sort(key=lambda g: (-g["support"], g["kind"], g["fix"], g["polarity"]))
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if a["kind"] != b["kind"] or a["polarity"] != b["polarity"]:
                    continue
                if _jaccard(_tokens(a["text"]), _tokens(b["text"])) >= 0.8:
                    # Keep the higher-support (then lexicographically
                    # smaller) text; union everything else.
                    keep, drop = (a, b) if (a["support"], a["text"]) >= (b["support"], b["text"]) else (b, a)
                    keep["support"] += drop["support"]
                    keep["provenance"] = sorted(set(keep["provenance"]) | set(drop["provenance"]))
                    keep["checks"] = sorted(set(keep["checks"]) | set(drop["checks"]))
                    keep["lesson_ids"] = sorted(set(keep["lesson_ids"]) | set(drop["lesson_ids"]))
                    if keep["rule_id"] is None:
                        keep["rule_id"] = drop["rule_id"]
                    merged.remove(drop)
                    changed = True
                    break
            if changed:
                break
    merged.sort(key=lambda g: (-g["support"], g["kind"], g["fix"], g["polarity"]))
    guidelines = []
    for n, g in enumerate(merged, 1):
        guidelines.append(Guideline(
            guideline_id=f"G{n}", text=g["text"], kind=g["kind"],
            check=g["check"], checks=g["checks"], polarity=g["polarity"],
            rule_id=g["rule_id"], support=g["support"],
            provenance=g["provenance"], lesson_ids=g["lesson_ids"],
        ))
    return guidelines


# ----------------------------------------------------------------------
# Guidelines file: Markdown, round-trippable, provenance included.
# ----------------------------------------------------------------------

_GUIDELINE_RE = re.compile(
    r"^##\s+(G\d+)\s+—\s+(\S+)\s*$"
    r"(.*?)(?=^##\s+G\d+\s+—\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_FIELD_RE = re.compile(r"^-\s+(\w+):\s*(.*)$", re.MULTILINE)


def write_guidelines(path: str, guidelines: list[Guideline]) -> None:
    """Write the guidelines file. Every guideline carries its
    support, provenance, checks and rule in the header, so the
    file alone — no sidecar — reconstructs the merge."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "# Guidelines — mined lessons (P5)",
        "",
        "Mined deterministically from verifier verdicts over the "
        "fixed P21 12-task set (no LLM). Each guideline lists how "
        "many lessons it merged (`support`) and exactly which "
        "episodes they came from (`episodes`); a guideline with "
        "`planted: yes` is a deliberate negative, not mined.",
        "",
    ]
    for g in guidelines:
        lines.append(f"## {g.guideline_id} — {g.kind}/{g.rule_id or g.check}")
        lines.append(f"- support: {g.support}")
        lines.append(f"- episodes: {', '.join(g.provenance)}")
        lines.append(f"- polarity: {g.polarity}")
        lines.append(f"- rule: {g.rule_id or '-'}")
        lines.append(f"- checks: {', '.join(g.checks)}")
        lines.append(f"- planted: {'yes' if g.planted else 'no'}")
        lines.append("")
        lines.append(g.text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def read_guidelines(path: str) -> list[Guideline]:
    """Parse a guidelines file back. Round-trip preserves support,
    provenance, checks, rule and planted flag (pinned by test)."""
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    guidelines = []
    for match in _GUIDELINE_RE.finditer(content):
        gid, heading, body = match.group(1), match.group(2), match.group(3)
        fields = dict(_FIELD_RE.findall(body))
        # The text is everything after the field lines.
        lines = [ln for ln in body.strip().splitlines()
                 if not _FIELD_RE.match(ln)]
        text = "\n".join(lines).strip()
        kind = heading.split("/")[0]
        episodes = [e.strip() for e in fields.get("episodes", "").split(",") if e.strip()]
        checks = [c.strip() for c in fields.get("checks", "").split(",") if c.strip()]
        rule = fields.get("rule", "-")
        guidelines.append(Guideline(
            guideline_id=gid, text=text, kind=kind,
            check=checks[0] if checks else heading.split("/")[-1],
            checks=checks, polarity=fields.get("polarity", "failed"),
            rule_id=None if rule == "-" else rule,
            support=int(fields.get("support", "0")),
            provenance=episodes,
            lesson_ids=[],
            planted=fields.get("planted", "no") == "yes",
        ))
    return guidelines


# ----------------------------------------------------------------------
# Phase 3 — injection at inference: full file vs retrieved subset.
# Retrieval is deterministic token overlap (light stemming,
# stopwords removed) between the task text and the guideline
# text. Injection activates the guidelines' rule_ids in the P23
# skill agent; a guideline with rule_id None injects nothing.
# ----------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "with", "is", "are", "was", "were", "it", "its", "this", "that",
    "then", "before", "after", "first", "what", "when", "you", "your",
    "from", "into", "never", "only", "any", "each", "one", "two",
    "not", "but", "so", "as", "at", "by", "be", "do", "does", "did",
    "if", "no", "yes", "all", "every", "just", "than", "too",
}


def _stem(token: str) -> str:
    """Light stemmer for retrieval scoring only. It only has to
    equate word forms that differ by a suffix — 'deduplicating'
    with 'deduplicate', 'totalling' with 'total', 'tasks' with
    'task' — not be a real stemmer. Suffixes are stripped in
    order (ing/ed/es, then a trailing 'e', then a doubled final
    letter), applied identically to both sides of the match, so
    any over-stemming is at least symmetric. Measured effect on
    the fixed set: without the 'e'/double-letter steps the
    compute guideline was missed for M1/M2 (retrieved 10/12);
    with them the retrieved subset matches full injection (see
    the P5 README section)."""
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    for suffix in ("ing", "ed", "es"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    if len(token) > 3 and token[-1] == token[-2]:
        token = token[:-1]
    if token.endswith("s") and not token.endswith(("ss", "us")) and len(token) > 3:
        token = token[:-1]
    return token


def _content_stems(text: str) -> set[str]:
    return {
        _stem(t) for t in re.split(r"[^A-Za-z0-9]+", text.lower())
        if t and t not in STOPWORDS and len(t) > 1
    }


def retrieve_guidelines(guidelines: list[Guideline], task_text: str,
                        k: int = 2) -> list[Guideline]:
    """Top-k guidelines by shared content stems with the task
    text. Ties: higher support first, then guideline id. A
    guideline sharing no stem scores 0 and is never retrieved —
    the subset can be empty, and the eval reports that honestly
    instead of injecting an irrelevant guideline."""
    query = _content_stems(task_text)
    scored = []
    for g in guidelines:
        overlap = len(query & _content_stems(g.text))
        scored.append((overlap, g.support, g))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2].guideline_id))
    return [g for overlap, _support, g in scored[:k] if overlap > 0]


def rules_for_guidelines(guidelines: list[Guideline]) -> set[str]:
    return {g.rule_id for g in guidelines if g.rule_id is not None}


def _eval_with_rules(rules_by_task: list[set[str]], n_tasks: int = 12,
                     seed: int = 7) -> dict:
    """Score the skill agent on the fixed set with a possibly
    different rule set per task (the retrieved-subset arm)."""
    tasks = build_tobench_tasks(seed=seed)[:n_tasks]
    per_task = []
    for task, rules in zip(tasks, rules_by_task):
        with tempfile.TemporaryDirectory(prefix="lessons-") as root:
            bundle = _seed_env(task, root)
            try:
                run_skill_agent(task, bundle, rules)
            except Exception as exc:  # a crashed agent fails the task
                bundle["trace"].events.append(
                    {"server": "agent", "tool": "crash",
                     "args": {"error": str(exc)}}
                )
            per_task.append(verify_task(task, bundle))
    passed = sum(1 for t in per_task if t["passed"])
    by_kind: dict[str, dict] = {}
    for t in per_task:
        by_kind.setdefault(t["kind"], {"passed": 0, "total": 0})
        by_kind[t["kind"]]["total"] += 1
        by_kind[t["kind"]]["passed"] += 1 if t["passed"] else 0
    return {
        "passed": passed, "total": len(per_task),
        "rate": passed / len(per_task) if per_task else 0.0,
        "per_task": per_task, "by_kind": by_kind,
    }


def _tokens(text: str) -> int:
    """Deterministic token estimate, same convention as P18/P19
    eviction: ceil(len(text) / 4)."""
    return math.ceil(len(text) / 4)


def run_lessons_eval(guidelines: list[Guideline], mode: str,
                     k: int = 2, n_tasks: int = 12, seed: int = 7) -> dict:
    """Score guideline injection on the fixed P21 set.

    modes: 'none' (no guidelines — the sloppy baseline),
    'full' (every guideline injected on every task), 'retrieved'
    (top-k guidelines per task, by token overlap with the task
    text). Reports injected tokens alongside passes: the cost
    of the memory, not just its benefit.
    """
    if mode not in ("none", "full", "retrieved"):
        raise ValueError(f"unknown mode: {mode}")
    tasks = build_tobench_tasks(seed=seed)[:n_tasks]
    rules_by_task: list[set[str]] = []
    injected: list[list[Guideline]] = []
    for task in tasks:
        if mode == "none":
            chosen: list[Guideline] = []
        elif mode == "full":
            chosen = list(guidelines)
        else:
            text = f"{task['goal']} {task['context'].get('query', '')}"
            chosen = retrieve_guidelines(guidelines, text, k=k)
        injected.append(chosen)
        rules_by_task.append(rules_for_guidelines(chosen))
    result = _eval_with_rules(rules_by_task, n_tasks=n_tasks, seed=seed)
    result["mode"] = mode
    result["k"] = k if mode == "retrieved" else None
    result["injected_tokens"] = sum(
        _tokens(g.text) for chosen in injected for g in chosen
    )
    result["injected_per_task"] = [
        [g.guideline_id for g in chosen] for chosen in injected
    ]
    return result


# ----------------------------------------------------------------------
# The A/B gate (same discipline as P20/P23): a candidate guideline
# ships only if the treatment passes strictly more tasks than the
# control, regresses nothing the control passed, and fixes at
# least one task of the target kind. Support is not a gate input
# — a high-support guideline that regresses still loses.
# ----------------------------------------------------------------------


def ab_test_guideline(existing: list[Guideline], candidate: Guideline,
                     target_kind: str, n_tasks: int = 12,
                     seed: int = 7) -> dict:
    """Control = existing guidelines injected (full); treatment =
    existing + candidate. Win requires all three gate conditions."""
    control = run_lessons_eval(list(existing), "full",
                               n_tasks=n_tasks, seed=seed)
    treatment = run_lessons_eval(list(existing) + [candidate], "full",
                                 n_tasks=n_tasks, seed=seed)
    control_pass = {t["task_id"] for t in control["per_task"] if t["passed"]}
    treatment_pass = {t["task_id"] for t in treatment["per_task"] if t["passed"]}
    fixed = sorted(treatment_pass - control_pass)
    regressed = sorted(control_pass - treatment_pass)
    fixed_target = [
        t for t in fixed
        if next(x for x in treatment["per_task"] if x["task_id"] == t)["kind"] == target_kind
    ]
    win = (
        treatment["passed"] > control["passed"]
        and not regressed
        and len(fixed_target) > 0
    )
    return {
        "candidate_rule": candidate.rule_id,
        "candidate_guideline": candidate.guideline_id,
        "candidate_support": candidate.support,
        "candidate_planted": candidate.planted,
        "target_kind": target_kind,
        "control_passed": control["passed"],
        "treatment_passed": treatment["passed"],
        "fixed_tasks": fixed,
        "regressed_tasks": regressed,
        "fixed_target_tasks": fixed_target,
        "win": win,
        "verdict": "kept" if win else "rejected",
    }


def planted_backfire_guideline() -> Guideline:
    """The deliberate negative: completes the tracker task before
    writing anything, contradicting the verifier's deliver-ordering
    rule (P23's 'complete-first'). Support is SYNTHETICALLY 12 —
    above every real guideline — so the trial proves the gate
    judges by measured effect, not by support. Flagged planted,
    provenance is not a real episode."""
    return Guideline(
        guideline_id="GX-backfire",
        text=RULES["complete-first"],
        kind="deliver",
        check="ordering_deliver_then_complete_then_status",
        checks=["ordering_deliver_then_complete_then_status"],
        polarity="failed",
        rule_id="complete-first",
        support=12,
        provenance=["planted-negative"],
        lesson_ids=["planted:complete-first"],
        planted=True,
    )


# ----------------------------------------------------------------------
# Full demo: mine -> merge -> write file -> A/B (none / retrieved /
# full) -> backfire trial -> good-candidate trial (the gate must
# also be able to say yes).
# ----------------------------------------------------------------------


def run_lessons_demo(n_tasks: int = 12, seed: int = 7, k: int = 2) -> dict:
    """End-to-end P5 demo in temp dirs. Returns everything the
    README table and the report need."""
    tmp = tempfile.mkdtemp(prefix="lessons-demo-")
    guidelines_path = os.path.join(tmp, "guidelines.md")
    traces_path = os.path.join(tmp, "traces.db")

    # Phase 1: run the no-rule (sloppy) agent, log to the P23 trace
    # store, mine lessons from the verdicts AND from the store.
    store = TraceStore(traces_path)
    baseline = run_skill_eval(set(), n_tasks=n_tasks, seed=seed)
    from .gepa import skill_version

    session_id = store.new_session("skill-agent", skill_version([]),
                                   notes="p5 mining run (no rules)")
    tasks = build_tobench_tasks(seed=seed)[:n_tasks]
    lessons: list[Lesson] = []
    for task, verdict in zip(tasks, baseline["per_task"]):
        store.log_run(session_id, task, verdict.pop("trace_events"),
                      verdict, "skill-agent", skill_version([]))
        lessons.extend(extract_lessons(task, verdict, session_id=session_id))
    store_lessons = mine_from_trace_store(store, session_id)
    # The two mining paths must agree on every failure lesson.
    in_memory_failures = sorted(
        (le.episode_id, le.check) for le in lessons if le.polarity == "failed"
    )
    store_failures = sorted(
        (le.episode_id, le.check) for le in store_lessons if le.polarity == "failed"
    )
    mining_agrees = in_memory_failures == store_failures

    # Phase 2: merge into guidelines, write the file.
    guidelines = merge_lessons(lessons)
    write_guidelines(guidelines_path, guidelines)
    round_tripped = read_guidelines(guidelines_path)

    # Phase 3: A/B — none vs retrieved (k and k=1) vs full.
    eval_none = run_lessons_eval(guidelines, "none", n_tasks=n_tasks, seed=seed)
    eval_retrieved = run_lessons_eval(guidelines, "retrieved", k=k,
                                      n_tasks=n_tasks, seed=seed)
    eval_retrieved1 = run_lessons_eval(guidelines, "retrieved", k=1,
                                       n_tasks=n_tasks, seed=seed)
    eval_full = run_lessons_eval(guidelines, "full", n_tasks=n_tasks, seed=seed)

    # Gate trials: the planted backfire must be rejected; a real
    # guideline added to an empty set must be kept (the gate can
    # say yes, not just no).
    backfire = planted_backfire_guideline()
    backfire_trial = ab_test_guideline(guidelines, backfire, "deliver",
                                       n_tasks=n_tasks, seed=seed)
    first_real = guidelines[0]
    good_trial = ab_test_guideline([], first_real, first_real.kind,
                                   n_tasks=n_tasks, seed=seed)

    store.close()
    return {
        "guidelines_path": guidelines_path,
        "traces_db": traces_path,
        "baseline_eval": {"passed": baseline["passed"], "total": baseline["total"]},
        "n_lessons": len(lessons),
        "n_lessons_from_store": len(store_lessons),
        "mining_agrees": mining_agrees,
        "guidelines": guidelines,
        "round_tripped": round_tripped,
        "eval_none": eval_none,
        "eval_retrieved": eval_retrieved,
        "eval_retrieved_k1": eval_retrieved1,
        "eval_full": eval_full,
        "backfire_trial": backfire_trial,
        "good_trial": good_trial,
    }


def print_lessons_report(result: dict) -> None:
    print("P5 IBM-style lessons miner — mine -> merge -> inject -> A/B")
    print()
    b = result["baseline_eval"]
    print(f"mining run (no rules): {b['passed']}/{b['total']} passed, "
          f"{result['n_lessons']} lessons mined "
          f"({result['n_lessons_from_store']} from the trace store, "
          f"failure lessons agree: {result['mining_agrees']})")
    print(f"merged into {len(result['guidelines'])} guidelines "
          f"-> {result['guidelines_path']}")
    for g in result["guidelines"]:
        print(f"  {g.guideline_id} [{g.kind}] support={g.support} "
              f"episodes={','.join(g.provenance)} rule={g.rule_id}")
    print()
    print(f"{'arm':<16}{'passed':<9}{'injected tokens':<17}by kind")
    for key, label in (("eval_none", "none"),
                       ("eval_retrieved_k1", "retrieved k=1"),
                       ("eval_retrieved", "retrieved k=2"),
                       ("eval_full", "full")):
        r = result[key]
        kinds = ", ".join(
            f"{k} {v['passed']}/{v['total']}" for k, v in r["by_kind"].items()
        )
        print(f"{label:<16}{r['passed']}/{r['total']:<8}"
              f"{r['injected_tokens']:<17}{kinds}")
    print()
    bt = result["backfire_trial"]
    print(f"planted backfire `{bt['candidate_rule']}` (support "
          f"{bt['candidate_support']}, planted): control "
          f"{bt['control_passed']} -> treatment {bt['treatment_passed']}, "
          f"fixed {bt['fixed_tasks']}, regressed {bt['regressed_tasks']} "
          f"-> {bt['verdict']}")
    gt = result["good_trial"]
    print(f"good candidate `{gt['candidate_rule']}` (support "
          f"{gt['candidate_support']}): control {gt['control_passed']} "
          f"-> treatment {gt['treatment_passed']}, fixed "
          f"{gt['fixed_tasks']}, regressed {gt['regressed_tasks']} "
          f"-> {gt['verdict']}")
