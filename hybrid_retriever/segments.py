"""P3 — Lychee-style lazy segment consolidator.

Sources: LycheeMemory V2 (arXiv:2608.12990) — segment-level
consolidation with embedding-based boundary detection, typed
context-independent records, and query-planned multi-route
retrieval. The paper's numbers are with an LLM summariser; this
module has no LLM, so the "summary" of a closed segment is the
sentence union of its turns (``draft_merge_text`` from P2) and the
measured saving is in consolidation *operations* and records
created, not LLM tokens. The README says so instead of pretending
the token numbers transfer.

Design rules
------------
- Consolidation is an offline job (``hr consolidate``), never on
  the online write path. Only *closed* segments are consolidated;
  the last (still open) segment keeps buffering.
- Boundary detection is embedding drift: cosine between adjacent
  turns drops past a threshold. The threshold is tuned on the
  synthetic transcripts (see ``eval.run_segment_eval``) and the
  fixed-window control is measured alongside — if semantic
  boundaries do not beat fixed-window batching, the eval says so.
- Segment typing is rule-based cues, no model: decision /
  preference / error-fix / fact. The merged record takes the
  segment's type (``propose_merge``'s ``type_override``).
- The merge machinery is reused, not reimplemented:
  ``propose_merge`` drafts (sentence union), judges (sufficiency
  certificate) and audits (``merges`` table). One certificate
  parameter differs from duplicate-merging: ``max_length_ratio``
  is 1.0 for segments, because a segment merge compresses
  *operations* (N turns -> 1 record) rather than words — the
  sentence union of distinct turns cannot be shorter than its
  inputs, and rejecting it for that would reject every segment.
  Faithfulness and coverage still apply, and the audit row still
  records the full certificate.
- The query planner picks one route per query — temporal, typed,
  keyword, vector, or hybrid — from text cues plus cheap corpus
  probes (FTS5 doc frequencies, a 5-hit BM25/vector probe). The
  choice is returned in the ``Plan`` so it can be logged and
  evaluated; ``retrieve_planned`` runs the chosen route.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .lifecycle import (
    content_tokens,
    draft_merge_text,
    propose_merge,
    word_tokens,
)
from .retrieval import DEFAULT_WEIGHTS, Retrieved, retrieve
from .store import Store

DEFAULT_THRESHOLD = 0.30
DEFAULT_MIN_TURNS = 2
DEFAULT_MAX_TURNS = 12
DEFAULT_WINDOW = 3
DEFAULT_RECENT_DAYS = 7.0
DEFAULT_RECENT_LIMIT = 200

# A turn whose embedding is all zeros (no indexable tokens) never
# forces a boundary: empty content is not a topic change.
_EMPTY_SIM = 1.0

_TEMPORAL_CUES = frozenset(
    """
    recent recently latest last yesterday today newest lately
    """.split()
)
_TEMPORAL_PHRASES = ("this week", "this month", "this year", "when did")

# Word -> record type. Checked against the query's content tokens.
# Only decision / preference are here: "fix" / "bug" / "error" look
# like type cues but are topic words in queries ("how does Dev fix
# a wobbly wheel" is about the repair topic, not error-fix records),
# so routing on them restricts candidates away from the gold record.
# The classifier still types records as error-fix; the planner just
# does not route on it.
_TYPE_CUES = {
    "decision": "decision",
    "decisions": "decision",
    "decide": "decision",
    "decided": "decision",
    "preference": "preference",
    "preferences": "preference",
    "prefer": "preference",
    "prefers": "preference",
}

# Segment typing cues, checked against the segment text.
_DECISION_CUES = (
    "decided", "decide", "we will", "let's", "lets", "chose", "choose",
    "going with", "agreed", "settled on",
)
_PREFERENCE_CUES = (
    "prefer", "prefers", "always", "never", "favourite", "favorite",
    "like best", "rather",
)
_ERROR_CUES = (
    "error", "failed", "failure", "bug", "broke", "broken", "crash",
    "crashed", "fix", "fixed", "wrong", "leak", "cracked",
)

SEGMENT_TYPES = ("fact", "decision", "error-fix", "preference")


# ----------------------------------------------------------------------
# Turns
# ----------------------------------------------------------------------
def turn_text(turn) -> str:
    """One turn as text: "role: content" when a role is given."""
    if isinstance(turn, str):
        return turn
    if isinstance(turn, dict):
        content = turn.get("content", turn.get("text", ""))
        role = turn.get("role")
        return "%s: %s" % (role, content) if role else content
    raise TypeError("turn must be str or dict, got %r" % type(turn))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return _EMPTY_SIM
    return float(a @ b / (na * nb))


# ----------------------------------------------------------------------
# Boundary detection
# ----------------------------------------------------------------------
def detect_boundaries(
    texts: list[str],
    embedder,
    threshold: float = DEFAULT_THRESHOLD,
    min_turns: int = DEFAULT_MIN_TURNS,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> list[list[int]]:
    """Split turn indices into segments by embedding drift.

    Cut before turn i+1 when cosine(turn[i], turn[i+1]) < threshold,
    unless the current segment is shorter than ``min_turns`` (a lone
    turn never splits a segment on its own) — and force a cut at
    ``max_turns`` so a long run of similar turns still consolidates.
    Returns a list of segments, each a list of turn indices.
    """
    n = len(texts)
    if n == 0:
        return []
    vecs = embedder.embed(texts)
    segments: list[list[int]] = []
    current: list[int] = [0]
    for i in range(n - 1):
        if len(current) >= max_turns:
            segments.append(current)
            current = [i + 1]
            continue
        sim = _cosine(vecs[i], vecs[i + 1])
        if sim < threshold and len(current) >= min_turns:
            segments.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    segments.append(current)
    return segments


def detect_boundaries_fixed_window(
    n_turns: int, window: int = DEFAULT_WINDOW
) -> list[list[int]]:
    """The control: cut every ``window`` turns, topic-blind."""
    if n_turns <= 0:
        return []
    return [
        list(range(i, min(i + window, n_turns)))
        for i in range(0, n_turns, window)
    ]


def boundary_positions(segments: list[list[int]]) -> set[int]:
    """Indices where a new segment starts (excluding 0)."""
    return {s[0] for s in segments if s and s[0] != 0}


def boundary_f1(
    predicted: set[int], truth: set[int]
) -> dict[str, float]:
    """Precision / recall / F1 on boundary positions (exact match)."""
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ----------------------------------------------------------------------
# Segment typing (rule-based, no model)
# ----------------------------------------------------------------------
def classify_segment(text: str) -> str:
    """fact / decision / error-fix / preference from cue words.

    Decision and error-fix cues win over preference (a decision that
    mentions a preference is still a decision); preference wins over
    the fact default. Ties inside one class are irrelevant — the
    label is the class, not the cue.
    """
    lowered = " " + text.lower() + " "
    if any(c in lowered for c in _DECISION_CUES):
        return "decision"
    if any(c in lowered for c in _ERROR_CUES):
        return "error-fix"
    if any(c in lowered for c in _PREFERENCE_CUES):
        return "preference"
    return "fact"


# ----------------------------------------------------------------------
# Consolidation (offline job)
# ----------------------------------------------------------------------
def ingest_transcript(store: Store, turns: list) -> list[int]:
    """Store each turn as a live 'episode' record, in order.

    Meta keeps the turn index and role so the transcript can be
    reconstructed; the text is what boundary detection embeds.
    """
    ids = []
    for i, turn in enumerate(turns):
        text = turn_text(turn)
        role = turn.get("role") if isinstance(turn, dict) else None
        ids.append(
            store.add_record(
                text,
                type_="episode",
                meta={"turn": i, "role": role},
            )
        )
    return ids


def consolidate_records(
    store: Store,
    embedder,
    record_ids: list[int],
    threshold: float = DEFAULT_THRESHOLD,
    min_turns: int = DEFAULT_MIN_TURNS,
    max_turns: int = DEFAULT_MAX_TURNS,
    fixed_window: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Consolidate the closed segments of an ordered record list.

    The last segment is still open (buffering) and is left alone;
    single-turn segments have nothing to merge and are skipped.
    Every closed segment goes through ``propose_merge`` with the
    segment profile (``max_length_ratio=1.0``, type override from
    ``classify_segment``), so the judge and the ``merges`` audit
    table apply exactly as in P2. Returns a summary dict.
    """
    summary = {
        "segments": 0,
        "closed_segments": 0,
        "consolidated": 0,
        "rejected": 0,
        "skipped_single_turn": 0,
        "left_open": 0,
        "merged_record_ids": [],
        "dry_run": dry_run,
    }
    records = store.fetch(record_ids)
    if len(records) < 2:
        summary["left_open"] = len(records)
        return summary
    texts = [r["text"] for r in records]
    if fixed_window is not None:
        segments = detect_boundaries_fixed_window(len(texts), fixed_window)
    else:
        segments = detect_boundaries(
            texts, embedder, threshold, min_turns, max_turns
        )
    summary["segments"] = len(segments)
    for seg in segments[:-1]:
        summary["closed_segments"] += 1
        if len(seg) < 2:
            summary["skipped_single_turn"] += 1
            continue
        ids = [records[i]["id"] for i in seg]
        text = draft_merge_text([texts[i] for i in seg])
        label = classify_segment(text)
        res = propose_merge(
            store,
            embedder,
            ids,
            merged_text=text,
            dry_run=dry_run,
            type_override=label,
            max_length_ratio=1.0,
        )
        if res.status in ("committed", "dry-run-pass"):
            summary["consolidated"] += 1
            if res.merged_record_id is not None:
                summary["merged_record_ids"].append(res.merged_record_id)
        else:
            summary["rejected"] += 1
    summary["left_open"] = len(segments[-1]) if segments else 0
    return summary


def consolidate_transcript(
    store: Store,
    embedder,
    turns: list,
    threshold: float = DEFAULT_THRESHOLD,
    min_turns: int = DEFAULT_MIN_TURNS,
    max_turns: int = DEFAULT_MAX_TURNS,
    fixed_window: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Ingest a transcript, then consolidate its closed segments.

    Returns the consolidation summary plus ``turns_ingested``.
    """
    ids = ingest_transcript(store, turns)
    summary = consolidate_records(
        store,
        embedder,
        ids,
        threshold=threshold,
        min_turns=min_turns,
        max_turns=max_turns,
        fixed_window=fixed_window,
        dry_run=dry_run,
    )
    summary["turns_ingested"] = len(ids)
    return summary


# ----------------------------------------------------------------------
# Query planner
# ----------------------------------------------------------------------
@dataclass
class Plan:
    route: str  # temporal | typed | keyword | vector | hybrid
    weights: tuple[float, float, float]
    candidate_ids: list[int] | None
    importance_weight: float
    reason: str


def _importance_weight(
    store: Store, query: str, candidate_ids: list[int] | None
) -> float:
    """0.5 when the BM25 probe's hits are high-importance, else 0.0.

    The P2 prior: per-record importance/credit is the planner's
    signal that some candidates are worth boosting. The probe is
    one cheap FTS5 call, not a full retrieval.
    """
    probe = store.bm25_search(query, 10, candidate_ids)
    if not probe:
        return 0.0
    imps = store.get_importances([rid for rid, _ in probe])
    if not imps:
        return 0.0
    mean_imp = sum(imps.values()) / len(imps)
    return 0.5 if mean_imp > 0.6 else 0.0


def plan_query(
    store: Store, query: str, embedder=None
) -> Plan:
    """Pick one retrieval route for a query, with the reason logged.

    Signals, in order:
    1. recency — a temporal cue ("recent", "latest", "this week")
       restricts candidates to the last 7 days;
    2. record type — a type cue ("decision", "preference", "fix")
       restricts candidates to that type;
    3. quoted phrase — exact-phrase matching is BM25's job;
    4. no indexable tokens at all — only the vector leg can answer;
    5. a 5-hit BM25/vector probe: no hits on one leg -> the other;
       legs that disagree (no overlap) -> hybrid, don't trust
       either alone; legs that agree -> keyword when *every*
       content term appears in the corpus, vector otherwise
       (paraphrase queries use synonyms, so at least one term
       misses);
    6. default hybrid, with the importance prior when the probe's
       hits are high-importance.
    Every route falls back to hybrid when its candidate set is
    empty — a route that can return nothing is not a route.
    """
    lowered = " " + query.lower() + " "
    tokens = word_tokens(query)
    content = content_tokens(query)

    # 1. Temporal: recency is the signal, not the words.
    if any(c in tokens for c in _TEMPORAL_CUES) or any(
        p in lowered for p in _TEMPORAL_PHRASES
    ):
        cand = store.recent_live_ids(
            days=DEFAULT_RECENT_DAYS, limit=DEFAULT_RECENT_LIMIT
        )
        if cand:
            return Plan(
                "temporal",
                DEFAULT_WEIGHTS,
                cand,
                _importance_weight(store, query, cand),
                "temporal cue; %d records from the last %.0f days"
                % (len(cand), DEFAULT_RECENT_DAYS),
            )

    # 2. Typed: the query names a record type.
    for token in tokens:
        label = _TYPE_CUES.get(token)
        if label is None:
            continue
        cand = store.live_ids_by_type(label)
        if cand:
            return Plan(
                "typed",
                DEFAULT_WEIGHTS,
                cand,
                _importance_weight(store, query, cand),
                "type cue %r -> %s; %d live records"
                % (token, label, len(cand)),
            )
        break  # a type cue with no records falls through to hybrid

    # 3. Quoted phrase: BM25's exact-match strength.
    if '"' in query:
        return Plan(
            "keyword",
            (1.0, 0.0, 0.0),
            None,
            _importance_weight(store, query, None),
            "quoted phrase",
        )

    # 4. Nothing indexable: only the vector leg can answer.
    if not tokens:
        return Plan(
            "vector", (0.0, 1.0, 0.0), None, 0.0, "no keyword tokens"
        )

    # 5. Cheap two-leg probe (5 hits each).
    bm25_ids = [
        rid for rid, _ in store.bm25_search(query, 5)
    ]
    vec_ids: list[int] = []
    if embedder is not None:
        vec = embedder.embed([query])[0]
        if float(np.linalg.norm(vec)) > 0:
            vec_ids = [
                rid for rid, _ in store.vector_search(vec, 5)
            ]
    if not bm25_ids and not vec_ids:
        return Plan(
            "hybrid",
            DEFAULT_WEIGHTS,
            None,
            0.0,
            "no probe hits on either leg; default hybrid",
        )
    if not bm25_ids:
        return Plan(
            "vector",
            (0.0, 1.0, 0.0),
            None,
            _importance_weight(store, query, None),
            "no BM25 hits",
        )
    if not vec_ids:
        return Plan(
            "keyword",
            (1.0, 0.0, 0.0),
            None,
            _importance_weight(store, query, None),
            "no vector hits",
        )
    overlap = len(set(bm25_ids) & set(vec_ids))
    if overlap == 0:
        iw = _importance_weight(store, query, None)
        return Plan(
            "hybrid",
            DEFAULT_WEIGHTS,
            None,
            iw,
            "legs disagree (no overlap in top 5); default hybrid",
        )
    live = max(1, store.count_live())
    hit_frac = (
        sum(1 for t in content if store.doc_freq(t) > 0) / len(content)
        if content
        else 0.0
    )
    # Keyword only when *every* content term appears in the corpus.
    # Keyword queries are built from the records' own words, so they
    # clear this bar (32/48 on the synthetic set); paraphrase queries
    # use synonyms, so they mostly don't (2/48). A 0.6 bar misroutes
    # paraphrase queries to BM25, which cannot match synonyms.
    if hit_frac >= 1.0:
        return Plan(
            "keyword",
            (1.0, 0.0, 0.0),
            None,
            _importance_weight(store, query, None),
            "legs agree (%d/5 overlap); %.0f%% of content terms "
            "appear in the corpus" % (overlap, hit_frac * 100),
        )
    return Plan(
        "vector",
        (0.0, 1.0, 0.0),
        None,
        _importance_weight(store, query, None),
        "legs agree (%d/5 overlap); only %.0f%% of content terms "
        "appear in the corpus (paraphrase)" % (overlap, hit_frac * 100),
    )


def retrieve_planned(
    store: Store,
    embedder,
    query: str,
    k: int = 10,
    use_mmr: bool = False,
    reranker=None,
) -> tuple[list[Retrieved], Plan]:
    """Plan the route, then run ``retrieve`` with it.

    Returns (results, plan) so the caller can log the choice. The
    plan's importance weight rides the P2 hook in ``retrieve``.
    """
    plan = plan_query(store, query, embedder)
    results = retrieve(
        store,
        embedder,
        query,
        k=k,
        weights=plan.weights,
        use_mmr=use_mmr,
        reranker=reranker,
        candidate_ids=plan.candidate_ids,
        importance_weight=plan.importance_weight,
    )
    return results, plan
