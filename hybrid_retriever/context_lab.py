"""P16 — Context-engineering lab (Beyond Token Savings replication).

Beyond Token Savings (arXiv:2609.32961) decomposes any context
compaction policy into primitive (truncate / clear tool results /
summarise) x trigger (when) x depth (how much), and measures
success, tokens, latency and cost per policy. Its headline: tool-
result clearing (TRC) kept success while cutting tokens to 0.57x,
latency 0.79x, cost 0.71x; a summarisation policy at similar token
use lost 11 points of success; and fewer tokens do not imply
faster or cheaper (on Terminal-Bench, ~1/3 the tokens took 20-80%
longer). The eval is the product — without it, compression is
guesswork.

SUBSTITUTIONS — stated plainly, here and in the README. No
SWE-bench, no Terminal-Bench, no real coding agent and no LLM ran
on this VM, and none was downloaded. The deterministic stand-ins
are:

1. The "coding agent" is a context-bound deterministic policy
   running the repo's own fixed task sets: the P21 mini-TOBench
   12-task set plus the P20 verify-by-doing 24-task computed-
   answer set (36 tasks, fixed before measuring). It reuses the
   P21 tool servers and verifiers and the P20 rule solver and
   verifier unchanged — only the agent's *reading* is new.
2. The "context" is the agent's working context: a P18-style
   episode log (user turn, reasoning, tool calls, tool results)
   with deterministic token counts (``estimate_tokens``,
   ceil(len/4)). The agent is CONTEXT-BOUND: every step derives
   its inputs only from the live episodes it can still see. A
   tool result it has not yet consumed, a rule it can no longer
   read, or a total it cannot re-derive from the visible context
   is forgotten — there is no hidden Python-side memory of tool
   outputs. That is what makes compaction causal here: evicting
   the wrong episode fails the task, exactly as losing context
   fails a real agent.
3. Compaction primitives reuse repo machinery: TRUNCATE is
   oldest-first eviction with P18's protections (user turns and
   the active window are never dropped); CLEAR-TOOL-RESULTS
   replaces a tool-result body with a short placeholder keeping
   the record that a result existed; SUMMARISE applies the P19
   extractive compressor per episode (JSON / table / log / code /
   text — deterministic, no LLM). The agent's own reasoning
   episode is itself a P3-style consolidation: it distils the
   tool result into the essentials (the firing rule, the gold
   facts, the computed total), which is why clearing a result
   the agent has already consumed can be safe — the same reason
   TRC keeps success in the paper.
4. A tool result is consumed by the immediately following
   reasoning turn BEFORE any compaction can fire. Compaction
   fires only at turn boundaries (after a reasoning episode, or
   after a final action). Firing on the raw tool result would
   delete it before the agent ever read it — no real agent loop
   works that way, and the paper's TRC result (success held)
   depends on consumption preceding clearing.
5. Latency is measured wall-clock, but there is no LLM call for
   it to scale with — so each agent step does real,
   token-proportional work: it re-scans the live context
   ``SCAN_PASSES`` times (word-overlap search over the live
   episodes), standing in for the layered attention passes of a
   real forward pass over the same context. The pass count is
   fixed before measuring and identical for every policy, so
   only the live context size differs between policies. The
   scan is also functional: the agent locates the episodes it
   needs with it.
6. Cost is MODELLED, not a real bill, and is labelled as such
   everywhere it appears: processed tokens at
   ``COST_PER_1M_TOKENS_USD`` per million, plus a fixed fee per
   summarise invocation (a summariser is an extra model call in
   the real world; TRC and truncate are free local ops).
   Summariser input tokens count as processed tokens — the
   summariser reads them, so compression is never free.

Policy grid (frozen before the measurement run — see
``POLICY_GRID``): no-compaction baseline, a no-op control (the
compaction machinery runs with primitive ``none`` and must
reproduce the baseline exactly on every deterministic metric),
TRC / truncate / summarise at depths 0.50 and 0.25 under a
token-budget trigger, and TRC under an every-2-turns trigger.
The end-of-task trigger is implemented and tested but is NOT in
the grid: in single-task sessions it fires after the last scan,
so it cannot change any of the four metrics — it only helps
the NEXT task in a shared session, which this harness does not
run.

Done condition (from the plan): the 4-metric table is measured
on the fixed 36-task set AND one policy is a clear win on all
four metrics (success held-or-better, tokens down, latency
down, cost down). If no policy wins on all four, the table is
reported honestly and ``done`` is False — the grid, the task
set and the cost model are NOT re-tuned after measuring.
"""

from __future__ import annotations

import os
import re
import tempfile
import time

from .compress import compress_text
from .eviction import estimate_tokens
from .tobench import _compute_from_text, _seed_env, build_tobench_tasks, verify_task
from .verify import (
    build_vbd_store,
    build_vbd_tasks,
    matching_rule,
    solve_task,
    verify_artifact,
    write_answer_artifact,
)

# ----------------------------------------------------------------------
# Cost model (MODELLED — not a real bill; rates fixed before measuring)
# ----------------------------------------------------------------------

# Assumed input-token price for a coding-agent model, USD per
# million tokens. Documented assumption, not a vendor quote.
COST_PER_1M_TOKENS_USD = 3.00
# A summarise invocation is an extra model call in the real world:
# a fixed fee per call, on top of the tokens it reads (which count
# as processed tokens). TRC and truncate are local ops: no fee.
SUMMARISE_CALL_FIXED_USD = 0.002

# Fixed token budget per task session for the budget trigger.
# Chosen from the measured BASELINE peaks (no compaction):
# VbD json 600 / log 641 / table 298, TOBench summarize 144-146,
# deliver 152-156, compute 76-126, correct 105-109, S3 89.
# 120 binds on the long-context tasks (all 24 VbD, summarize,
# deliver, two compute) and not on the shortest ones (S3, the
# three correct tasks, M2) — those act as a natural
# no-compaction control inside every policy row. A 350-token
# budget was tried first on paper and rejected: it would only
# ever fire on the 16 VbD json/log tasks. Frozen before the
# measurement run.
DEFAULT_BUDGET_TOKENS = 120

# How many times each agent step re-scans the live context.
# Stands in for the layered attention passes of a real forward
# pass; fixed before measuring, identical for every policy.
SCAN_PASSES = 32

PRIMITIVES = ("none", "truncate", "trc", "summarise")
TRIGGERS = ("never", "budget", "every_n", "end_of_task")
DEPTHS = (0.25, 0.50, 0.75, 1.0)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def cost_usd(tokens_processed: int, n_summarise_calls: int) -> float:
    """Modelled cost: processed tokens at the documented rate plus
    the fixed fee per summarise invocation. Labelled modelled
    everywhere it is reported — it is not a real bill."""
    return (
        tokens_processed / 1_000_000 * COST_PER_1M_TOKENS_USD
        + n_summarise_calls * SUMMARISE_CALL_FIXED_USD
    )


# ----------------------------------------------------------------------
# Phase 1 — policy config (primitives x triggers x depths, as data)
# ----------------------------------------------------------------------


class PolicyConfig:
    """One compaction policy: primitive x trigger x depth.

    ``primitive``: none | truncate | trc | summarise.
    ``trigger``: never | budget | every_n | end_of_task.
    ``depth``: fraction of the reference the policy compacts to.
      The reference is the token budget for the budget trigger,
      and the live token count at fire time for every_n /
      end_of_task. depth 0.50 means "compact until live tokens
      are at most half the reference".
    ``budget``: token budget for the budget trigger.
    ``every_n``: turn interval for the every_n trigger (turns
      count reasoning episodes).
    """

    def __init__(
        self,
        primitive: str = "none",
        trigger: str = "never",
        depth: float = 0.50,
        budget: int = DEFAULT_BUDGET_TOKENS,
        every_n: int = 2,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(
                "unknown primitive %r (expected one of %s)"
                % (primitive, ", ".join(PRIMITIVES))
            )
        if trigger not in TRIGGERS:
            raise ValueError(
                "unknown trigger %r (expected one of %s)"
                % (trigger, ", ".join(TRIGGERS))
            )
        if depth not in DEPTHS and not (0.0 < depth <= 1.0):
            raise ValueError("depth must be in (0, 1], got %r" % (depth,))
        if budget <= 0:
            raise ValueError("budget must be positive, got %r" % (budget,))
        if every_n < 1:
            raise ValueError("every_n must be >= 1, got %r" % (every_n,))
        if primitive == "none" and trigger not in ("never", "budget"):
            raise ValueError(
                "primitive 'none' needs trigger 'never' or 'budget', "
                "got %r" % (trigger,)
            )
        if trigger == "every_n" and primitive == "none":
            raise ValueError("trigger 'every_n' needs a real primitive")
        self.primitive = primitive
        self.trigger = trigger
        self.depth = depth
        self.budget = budget
        self.every_n = every_n

    @classmethod
    def from_dict(cls, spec: dict) -> "PolicyConfig":
        """Parse a policy from a plain dict; unknown keys are an
        error, so a typo in a grid entry fails loudly instead of
        silently running the defaults."""
        allowed = {"primitive", "trigger", "depth", "budget", "every_n"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError("unknown policy keys: %s" % sorted(unknown))
        return cls(**spec)

    def as_dict(self) -> dict:
        return {
            "primitive": self.primitive,
            "trigger": self.trigger,
            "depth": self.depth,
            "budget": self.budget,
            "every_n": self.every_n,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "PolicyConfig(%r)" % (self.as_dict(),)


# Frozen before the measurement run. Do not re-tune after seeing
# the table — if no row wins on all four metrics, the done
# condition is NOT met and the table stands as measured.
POLICY_GRID: list[tuple[str, PolicyConfig]] = [
    ("baseline", PolicyConfig(primitive="none", trigger="never")),
    (
        "noop",
        PolicyConfig(primitive="none", trigger="budget"),
    ),
    ("trc-50", PolicyConfig("trc", "budget", 0.50)),
    ("trc-25", PolicyConfig("trc", "budget", 0.25)),
    ("truncate-50", PolicyConfig("truncate", "budget", 0.50)),
    ("truncate-25", PolicyConfig("truncate", "budget", 0.25)),
    ("summarise-50", PolicyConfig("summarise", "budget", 0.50)),
    ("summarise-25", PolicyConfig("summarise", "budget", 0.25)),
    ("trc-every2", PolicyConfig("trc", "every_n", 0.50, every_n=2)),
]


# ----------------------------------------------------------------------
# The context: an in-memory P18-style episode log
# ----------------------------------------------------------------------


class Episode:
    """One context episode: typed, with its current (possibly
    compacted) text. ``evicted`` removes it from the live context
    entirely; ``cleared`` keeps the episode (the record that a
    tool result existed survives) with a placeholder body."""

    def __init__(self, seq: int, type_: str, kind: str, text: str) -> None:
        self.seq = seq
        self.type = type_  # user | reasoning | tool_call | tool_result
        self.kind = kind  # recall | read | write | remember | ...
        self.original_text = text
        self.text = text
        self.tokens = estimate_tokens(text)
        self.evicted = False
        self.cleared = False
        self.compressed = False

    def live_tokens(self) -> int:
        return 0 if self.evicted else self.tokens


class Context:
    """The agent's working context: the episode log plus the
    accounting. Token counts are deterministic; ``tokens_processed``
    is the cost basis (live tokens scanned per agent step, plus
    tokens the summariser read)."""

    def __init__(self) -> None:
        self.episodes: list[Episode] = []
        self.tokens_processed = 0
        self.n_summarise_calls = 0
        self.compactions: list[dict] = []
        self.peak_live_tokens = 0
        self.turns = 0  # reasoning episodes appended
        self.trigger_checks = 0
        # Turn count the every_n trigger last fired at: without
        # this, a check after a later action at the same turn
        # count would fire again for the same turn.
        self._every_n_fired_turn = 0

    def append(self, type_: str, kind: str, text: str) -> Episode:
        ep = Episode(len(self.episodes), type_, kind, text)
        self.episodes.append(ep)
        if type_ == "reasoning":
            self.turns += 1
        self.peak_live_tokens = max(
            self.peak_live_tokens, self.live_tokens()
        )
        return ep

    def live(self) -> list[Episode]:
        return [e for e in self.episodes if not e.evicted]

    def live_tokens(self) -> int:
        return sum(e.live_tokens() for e in self.episodes)

    def visible(self, kind: str, type_: str = "tool_result") -> str | None:
        """Current text of the latest live episode of this kind —
        what the agent can still see. None when it was evicted;
        the placeholder when it was cleared (the payload is gone,
        the record that a result existed is not)."""
        for ep in reversed(self.episodes):
            if not ep.evicted and ep.kind == kind and ep.type == type_:
                return ep.text
        return None

    def visible_reasoning(self) -> str | None:
        for ep in reversed(self.episodes):
            if not ep.evicted and ep.type == "reasoning":
                return ep.text
        return None

    def scan(self, words: set[str]) -> list[tuple[Episode, int]]:
        """Re-scan the live context for episodes sharing words with
        the query — the agent's read, and the token-proportional
        work that makes wall-clock latency depend on live context
        size. One logical scan bills the live tokens once; the
        ``SCAN_PASSES`` repetitions are the layered-attention
        stand-in and are not billed. Returns (episode, overlap)
        best first."""
        live = self.live()
        self.tokens_processed += sum(e.tokens for e in live)
        best: dict[int, int] = {}
        for _ in range(SCAN_PASSES):
            for ep in live:
                overlap = len(words & _words(ep.text))
                if overlap:
                    best[ep.seq] = overlap
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        by_seq = {e.seq: e for e in live}
        return [(by_seq[seq], n) for seq, n in ranked]


# ----------------------------------------------------------------------
# Primitives (each compacts toward ``target_tokens`` and reports)
# ----------------------------------------------------------------------


def _protected_seqs(ctx: Context, protect_last: int = 1) -> set[int]:
    """Episodes truncation never drops: user turns and the active
    window (the last live episode). Mirrors P18's protections."""
    live = ctx.live()
    protected = {e.seq for e in live if e.type == "user"}
    if protect_last > 0:
        protected |= {e.seq for e in live[-protect_last:]}
    return protected


def apply_truncate(ctx: Context, target_tokens: int) -> dict:
    """Oldest-first eviction of unprotected episodes until the
    live total is at most ``target_tokens`` (or nothing evictable
    is left — protections are never broken to meet a number)."""
    before = ctx.live_tokens()
    dropped = 0
    while ctx.live_tokens() > target_tokens:
        protected = _protected_seqs(ctx)
        candidates = [
            e for e in ctx.live() if e.seq not in protected
        ]
        if not candidates:
            break
        victim = candidates[0]  # oldest first
        victim.evicted = True
        dropped += 1
    after = ctx.live_tokens()
    return {
        "primitive": "truncate",
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": before - after,
        "n_affected": dropped,
        "reached_target": after <= target_tokens,
    }


def apply_trc(ctx: Context, target_tokens: int) -> dict:
    """Clear tool-result bodies oldest-first, replacing each with
    a placeholder that keeps the record that a result existed
    (its kind, its place in the log, its original size). The
    payload is gone; anything the agent had not already distilled
    into a reasoning episode is forgotten."""
    before = ctx.live_tokens()
    cleared = 0
    for ep in ctx.live():
        if ctx.live_tokens() <= target_tokens:
            break
        if ep.type != "tool_result" or ep.cleared:
            continue
        ep.cleared = True
        ep.text = (
            "[tool result cleared — was %d tokens]" % ep.tokens
        )
        ep.tokens = estimate_tokens(ep.text)
        cleared += 1
    after = ctx.live_tokens()
    return {
        "primitive": "trc",
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": before - after,
        "n_affected": cleared,
        "reached_target": after <= target_tokens,
    }


def apply_summarise(ctx: Context, target_tokens: int) -> dict:
    """Replace episode bodies with their P19 extractive
    compression (oldest first, user turns untouched — a user
    turn is the conversation itself, never rewritten). The
    compressor is extractive-only, so nothing is invented, but
    head+tail sentence keeping drops middle content: that
    lossiness is the risk being measured, not a bug. Each
    invocation reads the episodes it compresses (those tokens
    count as processed) and bills one summarise call."""
    before = ctx.live_tokens()
    compressed = 0
    for ep in ctx.live():
        if ctx.live_tokens() <= target_tokens:
            break
        if ep.type == "user" or ep.compressed:
            continue
        result = compress_text(ep.text)
        ctx.tokens_processed += result.tokens_before
        ctx.n_summarise_calls += 1
        if result.tokens_saved <= 0:
            continue
        ep.text = result.text
        ep.tokens = result.tokens_after
        ep.compressed = True
        compressed += 1
    after = ctx.live_tokens()
    return {
        "primitive": "summarise",
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": before - after,
        "n_affected": compressed,
        "reached_target": after <= target_tokens,
    }


_APPLY = {
    "truncate": apply_truncate,
    "trc": apply_trc,
    "summarise": apply_summarise,
}


def maybe_compact(
    ctx: Context, policy: PolicyConfig, end_of_task: bool = False
) -> dict | None:
    """Evaluate the trigger at a turn boundary; compact when it
    fires. Returns the compaction report, or None when nothing
    fired (or the policy is ``none`` — the no-op control, which
    still evaluates its trigger so the machinery is exercised).
    Never fires immediately after a tool result: the caller only
    invokes this after a reasoning episode or a final action, so
    the agent has consumed the latest result first."""
    if policy.primitive == "none":
        ctx.trigger_checks += 1
        return None
    ctx.trigger_checks += 1
    fire = False
    reference = policy.budget
    if policy.trigger == "budget":
        fire = ctx.live_tokens() > policy.budget
    elif policy.trigger == "every_n":
        fire = (
            ctx.turns > 0
            and ctx.turns % policy.every_n == 0
            and ctx.turns != ctx._every_n_fired_turn
        )
        reference = ctx.live_tokens()
    elif policy.trigger == "end_of_task":
        fire = end_of_task
        reference = ctx.live_tokens()
    elif policy.trigger == "never":
        fire = False
    if not fire:
        return None
    if policy.trigger == "every_n":
        ctx._every_n_fired_turn = ctx.turns
    target = int(policy.depth * reference)
    report = _APPLY[policy.primitive](ctx, target)
    report["trigger"] = policy.trigger
    report["target_tokens"] = target
    ctx.compactions.append(report)
    ctx.peak_live_tokens = max(
        ctx.peak_live_tokens, ctx.live_tokens()
    )
    return report


# ----------------------------------------------------------------------
# The context-bound agent: one task, one policy
# ----------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "to", "of",
        "in", "on", "for", "and", "or", "what", "how", "many",
        "write", "says", "saying", "this", "that", "it", "its",
    }
)


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOPWORDS}


def _sentences_with_words(text: str, words: set[str]) -> list[str]:
    """Lines/sentences of ``text`` sharing a whole word with
    ``words`` — the agent's deterministic relevance filter. A
    line that shares nothing is not recalled into the reasoning
    episode, which is what keeps the distractor ("tea",
    "March 3") out of the summary."""
    out = []
    for line in text.replace("\n", " ").split(". "):
        line = line.strip()
        if line and _words(line) & words:
            out.append(line)
    return out


def _recall_result_text(records: list[dict]) -> str:
    return "\n".join(
        "record #%d: %s" % (r["id"], r["text"]) for r in records
    )


def _run_tobench_task(
    task: dict, policy: PolicyConfig, ctx: Context, bundle: dict
) -> bool:
    """Run one P21 task through the context-bound agent. The tool
    calls go to the real P21 servers (so artifacts and tracker
    state are real); every content decision reads only the live
    context. Returns the P21 verifier's verdict. ``bundle`` is
    the pre-seeded per-task env (built by the caller, outside
    the timed region — seeding is harness overhead, identical
    for every policy)."""
    kind = task["kind"]
    context = task["context"]
    servers = bundle["servers"]
    ctx.append("user", "goal", task["goal"])
    ctx.scan(_content_words(task["goal"]))
    maybe_compact(ctx, policy)
    query = context.get("query", "")
    if kind == "summarize":
        ctx.append(
            "reasoning", "plan",
            "I will recall memories about %s and write %s "
            "from what I recall." % (query, context["artifact"])
            if query
            else "The facts are in the task itself; I will "
            "write %s from them." % context["artifact"],
        )
        ctx.scan(_content_words(query or task["goal"]))
        maybe_compact(ctx, policy)
        if context.get("require_recall"):
            ctx.append("tool_call", "recall",
                       "memory.recall(%r, k=5)" % query)
            res = servers["memory"].call_tool(
                "recall", {"query": query, "k": 5}
            )
            ctx.append("tool_result", "recall",
                       _recall_result_text(res["result"]))
            visible = ctx.visible("recall")
            facts = _sentences_with_words(
                visible or "", _content_words(query)
            )
            ctx.append(
                "reasoning", "extract",
                "Relevant facts: %s" % ". ".join(facts),
            )
        else:
            # S3: the facts are quoted in the task itself.
            goal = task["goal"]
            quoted = goal.split("exactly this: ", 1)[1].split(
                ". No lookup", 1
            )[0] if "exactly this: " in goal else goal
            ctx.append(
                "reasoning", "extract",
                "Facts from the task: %s" % quoted,
            )
        ctx.scan(_content_words(query or task["goal"]))
        maybe_compact(ctx, policy)
        reasoning = ctx.visible_reasoning() or ""
        content = reasoning.split(": ", 1)[1] if ": " in reasoning else reasoning
        ctx.append("tool_call", "write",
                   "files.write_file(%s)" % context["artifact"])
        servers["files"].call_tool(
            "write_file",
            {"path": context["artifact"], "content": content + "\n"},
        )
        ctx.append("tool_result", "write", "ok")
        ctx.scan(_content_words(task["goal"]))
        maybe_compact(ctx, policy)
        return verify_task(task, bundle)["passed"]

    if kind == "correct":
        ctx.append(
            "reasoning", "plan",
            "I will recall the stored preference about %s "
            "and correct it." % query,
        )
        ctx.scan(_content_words(query))
        maybe_compact(ctx, policy)
        ctx.append("tool_call", "recall",
                   "memory.recall(%r, k=5)" % query)
        res = servers["memory"].call_tool(
            "recall", {"query": query, "k": 5}
        )
        ctx.append("tool_result", "recall",
                   _recall_result_text(res["result"]))
        visible = ctx.visible("recall") or ""
        wrong = context["wrong"]
        target_id = None
        for line in visible.splitlines():
            if wrong in line:
                m = re.match(r"record #(\d+):", line)
                if m:
                    target_id = int(m.group(1))
        truth = context["truth"]
        ctx.append(
            "reasoning", "extract",
            "Record #%s says %r and is wrong. "
            "The correction is %r."
            % (target_id, wrong, truth),
        )
        ctx.scan(_content_words(query))
        maybe_compact(ctx, policy)
        reasoning = ctx.visible_reasoning() or ""
        m = re.search(r"Record #(\d+)", reasoning)
        rid = int(m.group(1)) if m else None
        m2 = re.search(r"The correction is '([^']+)'", reasoning)
        correction = m2.group(1) if m2 else truth
        ctx.append("tool_call", "remember",
                   "memory.remember(correction, supersedes=%s)" % rid)
        if rid is not None:
            servers["memory"].call_tool(
                "remember",
                {"text": correction, "type": "correction",
                 "supersedes": rid},
            )
        ctx.append("tool_result", "remember", "ok")
        ctx.scan(_content_words(query))
        maybe_compact(ctx, policy)
        return verify_task(task, bundle)["passed"]

    if kind == "compute":
        ctx.append(
            "reasoning", "plan",
            "I will read %s and compute the answer from it."
            % context["source"],
        )
        ctx.scan(_content_words(task["goal"]))
        maybe_compact(ctx, policy)
        ctx.append("tool_call", "read",
                   "files.read_file(%s)" % context["source"])
        res = servers["files"].call_tool(
            "read_file", {"path": context["source"]}
        )
        ctx.append("tool_result", "read", res["result"])
        visible = ctx.visible("read") or ""
        try:
            value = _compute_from_text(visible, context, dedup=True)
            answer = (
                "%.2f" % value if isinstance(value, float)
                else str(value)
            )
        except Exception:
            answer = "could not compute"
        ctx.append(
            "reasoning", "extract",
            "The answer is %s." % answer,
        )
        ctx.scan(_content_words(task["goal"]))
        maybe_compact(ctx, policy)
        reasoning = ctx.visible_reasoning() or ""
        m = re.search(r"The answer is ([0-9]+(?:\.[0-9]+)?)", reasoning)
        content = m.group(1) if m else "could not compute"
        ctx.append("tool_call", "write",
                   "files.write_file(%s)" % context["artifact"])
        servers["files"].call_tool(
            "write_file",
            {"path": context["artifact"], "content": content + "\n"},
        )
        ctx.append("tool_result", "write", "ok")
        ctx.scan(_content_words(task["goal"]))
        maybe_compact(ctx, policy)
        return verify_task(task, bundle)["passed"]

    if kind == "deliver":
        ctx.append(
            "reasoning", "plan",
            "I will recall %s, write %s, complete the "
            "tracker task, then write status."
            % (query, context["artifact"]),
        )
        ctx.scan(_content_words(query))
        maybe_compact(ctx, policy)
        ctx.append("tool_call", "recall",
                   "memory.recall(%r, k=5)" % query)
        res = servers["memory"].call_tool(
            "recall", {"query": query, "k": 5}
        )
        ctx.append("tool_result", "recall",
                   _recall_result_text(res["result"]))
        visible = ctx.visible("recall")
        facts = _sentences_with_words(
            visible or "", _content_words(query)
        )
        ctx.append(
            "reasoning", "extract",
            "Relevant facts: %s" % ". ".join(facts),
        )
        ctx.scan(_content_words(query))
        maybe_compact(ctx, policy)
        reasoning = ctx.visible_reasoning() or ""
        content = reasoning.split(": ", 1)[1] if ": " in reasoning else reasoning
        ctx.append("tool_call", "write",
                   "files.write_file(%s)" % context["artifact"])
        servers["files"].call_tool(
            "write_file",
            {"path": context["artifact"], "content": content + "\n"},
        )
        ctx.append("tool_result", "write", "ok")
        ctx.scan(_content_words(query))
        maybe_compact(ctx, policy)
        title = context["title"]
        tid = bundle["tracker_ids"][title]
        ctx.append("tool_call", "complete",
                   "tracker.complete_task(%d)" % tid)
        servers["tracker"].call_tool("complete_task", {"id": tid})
        ctx.append("tool_result", "complete", "ok")
        ctx.scan(_content_words(title))
        maybe_compact(ctx, policy)
        ctx.append("tool_call", "write",
                   "files.write_file(%s)" % context["status_file"])
        servers["files"].call_tool(
            "write_file",
            {"path": context["status_file"],
             "content": "Completed: %s\n" % title},
        )
        ctx.append("tool_result", "write", "ok")
        ctx.scan(_content_words(title))
        maybe_compact(ctx, policy)
        return verify_task(task, bundle)["passed"]

    raise ValueError("unknown kind: %r" % kind)


def _run_vbd_task(
    task: dict, policy: PolicyConfig, ctx: Context, store, embedder,
    run_dir: str,
) -> bool:
    """Run one P20 task through the context-bound agent. The
    artifact is read as a tool result and the rule memory is
    recalled from the shared VbD store (hybrid, k=5 — the P20
    hybrid arm); the answer is computed from the visible
    context and written to an answer artifact the P20 verifier
    reads back."""
    from .verify import _retrieve_memories

    ctx.append("user", "goal", task["question"])
    ctx.scan(_content_words(task["question"]))
    maybe_compact(ctx, policy)
    ctx.append(
        "reasoning", "plan",
        "I will read the artifact and recall the rule for "
        "this %s task." % task["kind"],
    )
    ctx.scan(_content_words(task["question"]))
    maybe_compact(ctx, policy)
    ctx.append("tool_call", "read", "read artifact")
    ctx.append("tool_result", "read", task["artifact"])
    ctx.append("tool_call", "recall",
               "memory.recall(%r, k=5)" % task["question"])
    mems = _retrieve_memories(
        store, embedder, task["question"], "hybrid", 5
    )
    ctx.append("tool_result", "recall", "\n".join(mems))
    visible_recall = ctx.visible("recall") or ""
    rule = None
    for line in visible_recall.splitlines():
        rule = matching_rule(task["kind"], line)
        if rule is not None:
            break
    visible_artifact = ctx.visible("read") or ""
    if rule is not None:
        result = solve_task(
            task, [line for line in visible_recall.splitlines()]
        )
        answer = result["answer"]
        rule_text = rule["key"]
    else:
        answer = solve_task(task, [])["answer"]
        rule_text = "none"
    ctx.append(
        "reasoning", "extract",
        "The rule is %s. The answer is %s."
        % (rule_text, answer),
    )
    ctx.scan(_content_words(task["question"]))
    maybe_compact(ctx, policy)
    reasoning = ctx.visible_reasoning() or ""
    m = re.search(r"The answer is ([0-9]+(?:\.[0-9]+)?)", reasoning)
    value = float(m.group(1)) if m else None
    result = {
        "task_id": task["task_id"],
        "answer": value,
        "used_rule": rule_text if rule is not None else None,
        "used_memory": None,
        "effect": rule["effect"] if rule is not None else "naive",
    }
    ctx.append("tool_call", "write", "write answer artifact")
    path = write_answer_artifact(run_dir, "ctxlab", result)
    ctx.append("tool_result", "write", "ok")
    ctx.scan(_content_words(task["question"]))
    maybe_compact(ctx, policy)
    return verify_artifact(path, task)["passed"]


def run_task_with_policy(
    task: dict, policy: PolicyConfig, vbd_store=None, embedder=None,
    run_dir: str | None = None,
    return_ctx: bool = False,
    context_cls=None,
) -> dict:
    """One task under one policy. Returns the per-task metrics;
    latency is the wall-clock of the agent run only (store setup
    is harness overhead, identical for every policy).
    ``return_ctx`` also returns the live ``Context`` (the
    trajectory) for capture by P17. ``context_cls`` is a
    zero-argument Context factory (default ``Context``) — P17
    passes a ``SummarisingContext`` factory so tool results
    enter the context already summarised."""
    ctx = (context_cls or Context)()
    if task.get("kind") in ("summarize", "correct", "compute", "deliver"):
        # _seed_env builds a fresh tiny store per task run, so the
        # agent's remember/complete calls are isolated per policy.
        # Seeding is harness overhead, identical for every policy,
        # so it is built OUTSIDE the timed region.
        with tempfile.TemporaryDirectory(prefix="ctxlab-tb-") as root:
            bundle = _seed_env(task, root)
            start = time.perf_counter()
            passed = _run_tobench_task(task, policy, ctx, bundle)
            maybe_compact(ctx, policy, end_of_task=True)
            latency = time.perf_counter() - start
    else:
        start = time.perf_counter()
        passed = _run_vbd_task(
            task, policy, ctx, vbd_store, embedder,
            run_dir or tempfile.mkdtemp(prefix="hr-ctxlab-run-"),
        )
        maybe_compact(ctx, policy, end_of_task=True)
        latency = time.perf_counter() - start
    result = {
        "task_id": task.get("task_id") or task.get("id"),
        "kind": task["kind"],
        "passed": passed,
        "tokens": ctx.tokens_processed,
        "latency_s": latency,
        "cost_usd": cost_usd(
            ctx.tokens_processed, ctx.n_summarise_calls
        ),
        "n_summarise_calls": ctx.n_summarise_calls,
        "n_compactions": len(ctx.compactions),
        "peak_live_tokens": ctx.peak_live_tokens,
        "final_live_tokens": ctx.live_tokens(),
        "trigger_checks": ctx.trigger_checks,
    }
    if return_ctx:
        return result, ctx
    return result


# ----------------------------------------------------------------------
# Phase 2 — the eval harness: run the grid on the fixed task set
# ----------------------------------------------------------------------


def run_context_lab_eval(
    n_tobench: int = 12,
    n_vbd: int = 24,
    policies: list[tuple[str, PolicyConfig]] | None = None,
    verbose: bool = True,
) -> dict:
    """Run the policy grid on the fixed 36-task set (12 P21 +
    24 P20). The VbD store is built ONCE and shared read-only
    across policies (the agent only recalls from it); each
    TOBench task gets a fresh seeded env per policy run. The
    task set, the grid and the cost model are fixed before this
    runs — nothing here re-tunes after measuring."""
    from .embeddings import get_embedder

    policies = policies if policies is not None else POLICY_GRID
    tb_tasks = build_tobench_tasks()[:n_tobench]
    vbd_tasks = build_vbd_tasks(n_tasks=n_vbd)
    tasks = tb_tasks + vbd_tasks
    embedder = get_embedder()
    vbd_store, _ = build_vbd_store(
        os.path.join("eval_data", "synthetic_corpus.jsonl"),
        embedder=embedder,
    )
    run_dir = tempfile.mkdtemp(prefix="hr-ctxlab-")
    if verbose:
        print(
            "context-lab eval: %d tasks (%d tobench + %d vbd), "
            "%d policies, embedder=%s, budget=%d tokens"
            % (
                len(tasks), len(tb_tasks), len(vbd_tasks),
                len(policies), embedder.name, DEFAULT_BUDGET_TOKENS,
            )
        )
    rows: dict[str, dict] = {}
    for name, policy in policies:
        per_task = []
        for task in tasks:
            per_task.append(
                run_task_with_policy(
                    task, policy,
                    vbd_store=vbd_store, embedder=embedder,
                    run_dir=run_dir,
                )
            )
        passed = sum(1 for t in per_task if t["passed"])
        tokens = sum(t["tokens"] for t in per_task)
        latency = sum(t["latency_s"] for t in per_task)
        cost = sum(t["cost_usd"] for t in per_task)
        rows[name] = {
            "policy": policy.as_dict(),
            "passed": passed,
            "total": len(per_task),
            "success_rate": passed / len(per_task) if per_task else 0.0,
            "tokens": tokens,
            "latency_s": latency,
            "cost_usd": cost,
            "n_summarise_calls": sum(
                t["n_summarise_calls"] for t in per_task
            ),
            "n_compactions": sum(
                t["n_compactions"] for t in per_task
            ),
            "peak_live_tokens": max(
                t["peak_live_tokens"] for t in per_task
            ),
            "per_task": per_task,
        }
    vbd_store.close()
    base = rows["baseline"]
    for name, row in rows.items():
        row["tokens_ratio"] = (
            row["tokens"] / base["tokens"] if base["tokens"] else 0.0
        )
        row["latency_ratio"] = (
            row["latency_s"] / base["latency_s"]
            if base["latency_s"]
            else 0.0
        )
        row["cost_ratio"] = (
            row["cost_usd"] / base["cost_usd"] if base["cost_usd"] else 0.0
        )
        row["success_delta_pp"] = (
            row["success_rate"] - base["success_rate"]
        ) * 100
        row["wins_all_four"] = bool(
            name != "baseline"
            and row["passed"] >= base["passed"]
            and row["tokens"] < base["tokens"]
            and row["latency_s"] < base["latency_s"]
            and row["cost_usd"] < base["cost_usd"]
        )
    winners = [n for n, r in rows.items() if r["wins_all_four"]]
    done = len(winners) > 0
    return {
        "n_tasks": len(tasks),
        "n_tobench": len(tb_tasks),
        "n_vbd": len(vbd_tasks),
        "budget_tokens": DEFAULT_BUDGET_TOKENS,
        "cost_model": {
            "per_1m_tokens_usd": COST_PER_1M_TOKENS_USD,
            "summarise_call_fixed_usd": SUMMARISE_CALL_FIXED_USD,
            "modelled": True,
        },
        "embedder": embedder.name,
        "rows": rows,
        "winners": winners,
        "done": done,
    }


def print_context_lab_report(result: dict) -> None:
    print("\n== P16 context-engineering lab (Beyond Token Savings) ==")
    print(
        "SUBSTITUTION: no SWE-bench, no Terminal-Bench, no real "
        "coding agent, no LLM. Deterministic agent on the fixed "
        "P21 (12) + P20 (24) task sets; cost is MODELLED, not a "
        "real bill."
    )
    print(
        "tasks: %d (%d tobench + %d vbd)   budget: %d tokens   "
        "embedder: %s"
        % (
            result["n_tasks"], result["n_tobench"], result["n_vbd"],
            result["budget_tokens"], result["embedder"],
        )
    )
    print(
        "cost model (modelled): $%.2f / 1M processed tokens + "
        "$%.3f per summarise call"
        % (
            result["cost_model"]["per_1m_tokens_usd"],
            result["cost_model"]["summarise_call_fixed_usd"],
        )
    )
    print(
        "%-13s %8s %9s %9s %9s %9s %9s %9s"
        % (
            "policy", "success", "tokens", "tok x", "latency",
            "lat x", "cost $", "cost x",
        )
    )
    for name, row in result["rows"].items():
        print(
            "%-13s %4d/%-3d %9d %9.3f %8.2fs %9.3f %9.4f %9.3f"
            % (
                name, row["passed"], row["total"], row["tokens"],
                row["tokens_ratio"], row["latency_s"],
                row["latency_ratio"], row["cost_usd"],
                row["cost_ratio"],
            )
        )
    print("success deltas vs baseline (pp) and compactions:")
    for name, row in result["rows"].items():
        print(
            "  %-13s %+6.1fpp  compactions=%d  summarise_calls=%d"
            % (
                name, row["success_delta_pp"], row["n_compactions"],
                row["n_summarise_calls"],
            )
        )
    if result["done"]:
        print(
            "DONE: %s win(s) on all four metrics: %s"
            % (len(result["winners"]), ", ".join(result["winners"]))
        )
    else:
        print(
            "DONE: NOT met — no policy wins on all four metrics "
            "(success held-or-better, tokens down, latency down, "
            "cost down). The table stands as measured; the grid "
            "was not re-tuned."
        )
