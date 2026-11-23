"""P18 — CWL structured context eviction.

Sources: Beyond Compaction / CWL (arXiv:2606.11213) — instead of
LLM summarisation ("compaction"), the agent annotates its
trajectory as typed, dependency-linked episodes as it works, and a
deterministic, LLM-free policy evicts in priority order when a
token budget is exceeded.

Model
-----
Episodes are the *context*: typed, dependency-linked units of one
session's trajectory, each linked to its turn record in the store.
Records are the *memory*. Eviction removes an episode from the
context; it never deletes a record. That is the whole point of the
paper's design: an episode whose effects are already persisted in
the environment (consolidated into a segment record by P3, or
written out by a tool) can be shed without loss, because its
content lives on in the record that persists it.

Eviction tiers (lower number evicted first)
--------------------------------------------
0  tool_result whose effect is persisted (safest — nothing is lost)
1  chatter (filler; evicting it loses nothing by definition)
2  tool_call whose result is persisted
3  any persisted episode (fact / preference / decision /
   error_fix / reasoning whose content lives in a merged record)
4  unpersisted reasoning (old, superseded reasoning)
5  unpersisted decision / error_fix (valuable; kept longer)
6  unpersisted tool_call / tool_result (evicting loses
   information — last resort)

Hard protections (never evicted, whatever the budget):
- user turns — the conversation itself;
- the active window: the last ``protect_last`` episodes;
- any episode a still-live episode depends on (dependency links
  are causal structure; evicting the dependency first would
  strand the dependent).

If the protections make the budget unreachable, the policy evicts
everything evictable and reports ``under_budget=False`` with the
overage — it never breaks a protection to meet a number.

Tokens are estimated deterministically as ``ceil(len(text) / 4)``
(one token per four characters, the standard rough estimate).
No tokenizer, no model, no LLM call anywhere in this module.

The naive control (``policy="naive"``) is oldest-first truncation
with no tiers, no user-turn protection and no dependency
protection — only the same active window. It exists so the demo
can show the policy matters, not just that eviction happened.
"""

from __future__ import annotations

import math

from .segments import classify_segment, turn_text
from .store import Store

EPISODE_TYPES = (
    "user",
    "reasoning",
    "tool_call",
    "tool_result",
    "decision",
    "error_fix",
    "preference",
    "fact",
    "chatter",
)

# Episode types that are never evicted, whatever the budget.
NEVER_EVICT_TYPES = frozenset({"user"})

DEFAULT_PROTECT_LAST = 3


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: 1 token per 4 characters."""
    return max(1, math.ceil(len(text) / 4))


# ----------------------------------------------------------------------
# Phase 1 — annotation (the delimiter tool)
# ----------------------------------------------------------------------
def annotate_episode(
    store: Store,
    session_id: str,
    record_id: int,
    type_: str,
    depends_on: list[int] | None = None,
    persisted_by: int | None = None,
) -> int:
    """Tag one turn record as a typed, dependency-linked episode.

    This is the delimiter tool: the agent (or the transcript
    ingester) calls it as work proceeds. Validates everything it
    is given — a bad type, an unknown record, a dependency outside
    the session, or a persisted_by that names no record are all
    errors, because an episode log that lies is worse than none.
    """
    if type_ not in EPISODE_TYPES:
        raise ValueError(
            "unknown episode type %r (expected one of %s)"
            % (type_, ", ".join(EPISODE_TYPES))
        )
    record = store.get_record(record_id)
    if record is None:
        raise ValueError("unknown record id %d" % record_id)
    deps = list(depends_on or [])
    for dep in deps:
        ep = store.get_episode(dep)
        if ep is None:
            raise ValueError("unknown dependency episode %d" % dep)
        if ep["session_id"] != session_id:
            raise ValueError(
                "dependency episode %d belongs to session %r, not %r"
                % (dep, ep["session_id"], session_id)
            )
    if persisted_by is not None and store.get_record(persisted_by) is None:
        raise ValueError("unknown persisted_by record %d" % persisted_by)
    text = record["text"]
    return store.add_episode(
        session_id,
        record_id,
        type_,
        estimate_tokens(text),
        depends_on=deps,
        persisted_by=persisted_by,
    )


def _default_type(turn: dict, text: str) -> str:
    """Role-based default typing, with cue-based upgrades.

    user -> user, tool -> tool_result, assistant -> reasoning
    unless the text carries decision / error-fix / preference
    cues (``classify_segment``), in which case it takes that
    label. An explicit ``episode_type`` on the turn always wins.
    """
    role = turn.get("role") if isinstance(turn, dict) else None
    if isinstance(turn, dict) and turn.get("episode_type"):
        return turn["episode_type"]
    if role == "user":
        return "user"
    if role == "tool":
        return "tool_result"
    label = classify_segment(text)
    if label == "decision":
        return "decision"
    if label == "error-fix":
        return "error_fix"
    if label == "preference":
        return "preference"
    return "reasoning"


def ingest_session(
    store: Store, session_id: str, turns: list
) -> list[int]:
    """Ingest turns as episode records + episode rows, in order.

    Each turn is a dict with ``text``/``content`` and an optional
    ``role`` (``turn_text`` rules), plus optional ``episode_type``,
    ``depends_on`` (turn indices within this call, resolved to
    episode ids) and ``persisted_by`` (a record id). A
    ``tool_result`` turn with no explicit dependency is linked to
    the latest preceding ``tool_call`` episode — a result without
    its call is meaningless, so the link is structural, not
    optional. Returns the episode ids in turn order.
    """
    episode_ids: list[int] = []
    last_tool_call: int | None = None
    for i, turn in enumerate(turns):
        text = turn_text(turn)
        role = turn.get("role") if isinstance(turn, dict) else None
        rid = store.add_record(
            text,
            type_="episode",
            meta={
                "turn": i,
                "role": role,
                "session": session_id,
            },
        )
        type_ = _default_type(turn, text)
        deps: list[int] = []
        if isinstance(turn, dict) and turn.get("depends_on"):
            for idx in turn["depends_on"]:
                if not 0 <= idx < len(episode_ids):
                    raise ValueError(
                        "depends_on index %r out of range in turn %d"
                        % (idx, i)
                    )
                deps.append(episode_ids[idx])
        if type_ == "tool_result" and not deps and last_tool_call is not None:
            deps.append(last_tool_call)
        persisted_by = (
            turn.get("persisted_by")
            if isinstance(turn, dict)
            else None
        )
        eid = annotate_episode(
            store,
            session_id,
            rid,
            type_,
            depends_on=deps,
            persisted_by=persisted_by,
        )
        if type_ == "tool_call":
            last_tool_call = eid
        episode_ids.append(eid)
    return episode_ids


def mark_persisted(store: Store, episode_id: int, record_id: int) -> None:
    """Record that an episode's effects live on in ``record_id``."""
    if store.get_episode(episode_id) is None:
        raise ValueError("unknown episode id %d" % episode_id)
    if store.get_record(record_id) is None:
        raise ValueError("unknown record id %d" % record_id)
    store.set_episode_persisted(episode_id, record_id)


def mark_session_persisted_by_merges(
    store: Store, session_id: str
) -> int:
    """Mark every episode whose turn a committed merge absorbed.

    This wires P3 consolidation into eviction: once a segment
    record holds a turn's content, the turn's episode is
    sheddable (tier 0–3 instead of 4–6). Returns how many
    episodes were newly marked.
    """
    seg_of_turn: dict[int, int] = {}
    for m in store.list_merges(status="committed"):
        if m["merged_record_id"] is None:
            continue
        for src in m["merged_ids"]:
            seg_of_turn[src] = m["merged_record_id"]
    marked = 0
    for ep in store.list_episodes(session_id, status="live"):
        target = seg_of_turn.get(ep["record_id"])
        if target is not None and ep["persisted_by"] != target:
            store.set_episode_persisted(ep["id"], target)
            marked += 1
    return marked


# ----------------------------------------------------------------------
# Phase 2 — the deterministic eviction policy
# ----------------------------------------------------------------------
def eviction_tier(episode: dict) -> int:
    """Which tier an episode evicts in (lower evicted first)."""
    type_ = episode["type"]
    persisted = episode["persisted_by"] is not None
    if type_ == "tool_result" and persisted:
        return 0
    if type_ == "chatter":
        return 1
    if type_ == "tool_call" and persisted:
        return 2
    if persisted:
        return 3
    if type_ == "reasoning":
        return 4
    if type_ in ("decision", "error_fix"):
        return 5
    return 6


def eviction_reason(episode: dict, importance: float) -> str:
    """The human-readable reason this episode is the next to go."""
    tier = eviction_tier(episode)
    persisted = episode["persisted_by"]
    base = "%s, %d tokens, importance %.2f, uses %d" % (
        episode["type"],
        episode["tokens"],
        importance,
        episode.get("uses", 0),
    )
    if tier == 0:
        return "%s; effect persisted by record #%d" % (base, persisted)
    if tier == 1:
        return "%s; filler, nothing is lost" % base
    if tier == 2:
        return "%s; its result persisted by record #%d" % (
            base,
            persisted,
        )
    if tier == 3:
        return "%s; content preserved in merged record #%d" % (
            base,
            persisted,
        )
    if tier == 4:
        return "%s; old reasoning, superseded by later turns" % base
    if tier == 5:
        return "%s; unpersisted, kept longest among evictable" % base
    return "%s; unpersisted — evicting loses information" % base


def _protected_ids(
    episodes: list[dict], protect_last: int
) -> set[int]:
    """Episodes the policy may never evict: user turns + the
    active window (the last ``protect_last`` episodes)."""
    protected = {
        e["id"] for e in episodes if e["type"] in NEVER_EVICT_TYPES
    }
    if protect_last > 0:
        protected |= {e["id"] for e in episodes[-protect_last:]}
    return protected


def evict_to_budget(
    store: Store,
    session_id: str,
    budget_tokens: int,
    protect_last: int = DEFAULT_PROTECT_LAST,
    dry_run: bool = False,
    policy: str = "cwl",
) -> dict:
    """Evict episodes until the session fits ``budget_tokens``.

    ``policy="cwl"`` is the tiered policy with all protections.
    ``policy="naive"`` is oldest-first truncation: same active
    window, but no tiers, no user-turn protection and no
    dependency protection. Returns a summary dict; every eviction
    is written to the ``evictions`` audit table unless
    ``dry_run`` (a dry run then reports ``merges_would_commit``-
    style counts as ``would_evict`` so it can never be misread
    as real evictions).
    """
    if policy not in ("cwl", "naive"):
        raise ValueError("unknown policy %r" % policy)
    episodes = store.list_episodes(session_id, status="live")
    total_before = sum(e["tokens"] for e in episodes)
    total = total_before
    # P2 importance of the linked records, for the within-tier order.
    record_ids = [e["record_id"] for e in episodes]
    importances = store.get_importances(record_ids)
    uses = {
        r["id"]: r.get("uses", 0)
        for r in store.fetch(record_ids)
        if r is not None
    }
    protected = (
        _protected_ids(episodes, protect_last)
        if policy == "cwl"
        else {e["id"] for e in episodes[-protect_last:]}
        if protect_last > 0
        else set()
    )
    evicted: list[dict] = []
    evicted_ids: set[int] = set()

    def _depended_on(ep_id: int) -> bool:
        # A live episode depends on this one — evicting it would
        # strand the dependent. Checked against the currently
        # live set, so evicting the dependent first frees it.
        live = [
            e
            for e in episodes
            if e["id"] not in evicted_ids
        ]
        return any(ep_id in e["depends_on"] for e in live)

    while total > budget_tokens:
        candidates = [
            e
            for e in episodes
            if e["id"] not in evicted_ids and e["id"] not in protected
        ]
        if policy == "cwl":
            candidates = [
                e for e in candidates if not _depended_on(e["id"])
            ]
        if not candidates:
            break
        if policy == "cwl":
            candidates.sort(
                key=lambda e: (
                    eviction_tier(e),
                    e["seq"],
                    importances.get(e["record_id"], 0.5),
                    uses.get(e["record_id"], 0),
                )
            )
        else:
            candidates.sort(key=lambda e: e["seq"])
        victim = candidates[0]
        tier = eviction_tier(victim) if policy == "cwl" else -1
        reason = (
            eviction_reason(
                victim, importances.get(victim["record_id"], 0.5)
            )
            if policy == "cwl"
            else "naive oldest-first truncation"
        )
        if not dry_run:
            store.set_episode_status(victim["id"], "evicted")
            store.add_eviction(
                victim["id"],
                session_id,
                reason,
                tier,
                victim["tokens"],
                budget_tokens,
                total,
            )
        evicted.append(
            {
                "episode_id": victim["id"],
                "type": victim["type"],
                "tier": tier,
                "tokens": victim["tokens"],
                "reason": reason,
            }
        )
        evicted_ids.add(victim["id"])
        total -= victim["tokens"]

    by_tier: dict[int, int] = {}
    for ev in evicted:
        by_tier[ev["tier"]] = by_tier.get(ev["tier"], 0) + 1
    live_now = store.list_episodes(session_id, status="live")
    user_total = sum(1 for e in episodes if e["type"] == "user")
    user_live = sum(1 for e in live_now if e["type"] == "user")
    return {
        "session_id": session_id,
        "policy": policy,
        "budget_tokens": budget_tokens,
        "total_before": total_before,
        "total_after": total,
        "evicted": evicted,
        "n_evicted": len(evicted),
        "evicted_by_tier": by_tier,
        "would_evict": len(evicted) if dry_run else 0,
        "under_budget": total <= budget_tokens,
        "overage": max(0, total - budget_tokens),
        "live_episodes": len(live_now),
        "user_turns_live": user_live,
        "user_turns_total": user_total,
        "dry_run": dry_run,
    }


def dependency_violations(store: Store, session_id: str) -> list[dict]:
    """Evicted episodes a live episode still depends on.

    CWL must report zero: the policy never evicts a dependency
    of a live episode. The naive control is expected to report
    some — that is the point of the comparison.
    """
    live = store.list_episodes(session_id, status="live")
    evicted = store.list_episodes(session_id, status="evicted")
    evicted_ids = {e["id"] for e in evicted}
    violations = []
    for e in live:
        for dep in e["depends_on"]:
            if dep in evicted_ids:
                violations.append(
                    {"episode_id": e["id"], "depends_on": dep}
                )
    return violations


def explain_episode(
    store: Store, episode_id: int, protect_last: int = DEFAULT_PROTECT_LAST
) -> str:
    """Why this episode is gone, or why it stays.

    Every eviction is explainable from the audit log; a live
    episode explains itself from its type, tier and protections.
    ``protect_last`` must match the window the eviction call
    used, or the active-window answer will disagree with it.
    """
    ep = store.get_episode(episode_id)
    if ep is None:
        raise ValueError("unknown episode id %d" % episode_id)
    rows = store.list_evictions()
    for row in rows:
        if row["episode_id"] == episode_id:
            return (
                "episode #%d (%s, %d tokens) evicted: %s "
                "(budget %d, session total was %d)"
                % (
                    episode_id,
                    ep["type"],
                    ep["tokens"],
                    row["reason"],
                    row["budget_tokens"],
                    row["total_tokens_before"],
                )
            )
    if ep["type"] in NEVER_EVICT_TYPES:
        return (
            "episode #%d (%s) is protected: user turns are never "
            "evicted" % (episode_id, ep["type"])
        )
    live = store.list_episodes(ep["session_id"], status="live")
    seqs = [e["seq"] for e in live]
    if live and protect_last > 0 and ep["seq"] >= max(seqs) - protect_last + 1:
        return (
            "episode #%d (%s) is protected: inside the active "
            "window (last %d episodes)"
            % (episode_id, ep["type"], protect_last)
        )
    if store.episodes_depending_on(episode_id, ep["session_id"]):
        return (
            "episode #%d (%s) is protected: a live episode "
            "depends on it" % (episode_id, ep["type"])
        )
    tier = eviction_tier(ep)
    record = store.get_record(ep["record_id"])
    importance = record["importance"] if record else 0.5
    return (
        "episode #%d (%s) is live and evictable at tier %d: %s"
        % (
            episode_id,
            ep["type"],
            tier,
            eviction_reason(ep, importance),
        )
    )
