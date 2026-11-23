"""P2 — MEMOIR-style memory lifecycle: TRACE-UCB credit, judge-gated
merge, and the offline sleep job.

Sources: MEMOIR (arXiv:2611.07631) for TRACE-UCB + the sufficiency
certificate, and "Do agents need to sleep?" (arXiv:2611.02216) for the
offline consolidation ("sleep") framing.

Design rules
------------
- No LLM call anywhere on the write path, and no network service.
  The judge is rules + embedding checks (the same ONNX MiniLM the
  store already uses). A local model is only added if the certificate
  fails too often — it does not (see tests/test_lifecycle.py).
- Work is at record level, not whole sessions. P1's honest finding
  (vector/rerank legs suffer when whole sessions are truncated to
  256 tokens) applies here too: on LongMemEval-S the records *are*
  whole sessions, so TRACE's credit signal is only as good as the
  truncated embeddings that produced the retrieval. The README says
  so instead of hiding it.
- Merges never delete. Sources are marked 'superseded' and stay in
  the file; the merged record points back at them; the `merges`
  table is the audit trail; `rollback_merge` restores the sources.

TRACE-UCB
---------
After each answer, `record_use` attributes credit to the records
that were retrieved for it:

- overlap: how much of a record's content tokens the answer actually
  used (recall of the record's tokens in the answer);
- uniqueness: the leave-one-out proxy — the share of that overlap no
  *other* retrieved record also explains;
- legs: records the legs agree on get a small multiplier (1.0 + 0.1
  per extra leg), bounded at 1.2x.

``credit = recall * (0.5 + 0.5 * uniqueness) * leg_factor``.

Importance is a UCB1-style prior: ``mean_credit + c * sqrt(ln(N+1) /
uses)`` with a neutral prior of 0.5 for records never retrieved, so
rarely-tried memories stay explorable; a record's importance only
sinks below the neutral prior after enough fruitless uses. (The
exploration bonus dominates after a single use — that is the bandit
working as intended, not a bug: one trial is weak evidence.)
`retrieve(..., importance_weight=w)` multiplies the fused RRF score
by ``1 + w * normalised_importance``; w = 0 is exactly P1 behaviour.

Judge-gated merge
-----------------
`propose_merge` drafts a merged text (sentence union of the sources
— no LLM, so faithfulness holds by construction) and runs the
sufficiency certificate. All four checks must pass:

- faithfulness: >= 0.95 of the merged text's content tokens appear
  in the sources (catches invented content);
- coverage: >= 0.45 of the sources are represented in the merged
  text (per-source token recall >= 0.5 AND cosine >= 0.30 — catches
  dropped facts);
- length: merged token count <= 0.90 of the sources' total (a merge
  must actually compress);
- source similarity: mean cosine(merged, source) >= 0.30.

On pass the merged record goes live and the sources are superseded;
on fail nothing is written except the rejected audit row.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .retrieval import Retrieved
from .store import Store

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Small closed-class set: credit is about content, not glue words.
STOPWORDS = frozenset(
    """
    a an the and or but of to in on at for with by from as is are was were
    be been being it its this that these those there here he she they we you
    i his her their our your my me him them us what which who whom when
    where why how do does did can could will would should may might have
    has had not no yes if then than so such very just also more most other
    some any all each every own same too only over under into out up down
    about after before between through during per via
    """.split()
)

DEFAULT_UCB_C = 1.0
DEFAULT_MIN_FAITHFULNESS = 0.95
DEFAULT_MIN_COVERAGE = 0.45
DEFAULT_MAX_LENGTH_RATIO = 0.90
DEFAULT_MIN_SOURCE_SIM = 0.30
DEFAULT_COSINE_THRESHOLD = 0.90
DEFAULT_JACCARD_THRESHOLD = 0.9


# ----------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------
def word_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in word_tokens(text) if t not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------
# TRACE-UCB
# ----------------------------------------------------------------------
def attribute_credit(
    retrieved: list[Retrieved], answer_text: str
) -> dict[int, float]:
    """Credit per retrieved record id, from lexical overlap with the answer.

    ``credit = recall * (0.5 + 0.5 * uniqueness) * leg_factor`` where
    recall is the share of the record's content tokens present in the
    answer, uniqueness is the leave-one-out proxy (the share of that
    overlap no other retrieved record also explains), and leg_factor
    is 1.0 + 0.1 per extra leg that surfaced the record (max 1.2).
    Records with no overlap get 0.0 — retrieved but not used.
    """
    ans_tokens = content_tokens(answer_text)
    if not ans_tokens or not retrieved:
        return {r.id: 0.0 for r in retrieved}
    per_rec = {r.id: content_tokens(r.text) & ans_tokens for r in retrieved}
    credits: dict[int, float] = {}
    for r in retrieved:
        shared = per_rec[r.id]
        rec_tokens = content_tokens(r.text)
        if not shared or not rec_tokens:
            credits[r.id] = 0.0
            continue
        others: set[str] = set()
        for o in retrieved:
            if o.id != r.id:
                others |= per_rec[o.id]
        unique = shared - others
        recall = len(shared) / len(rec_tokens)
        uniqueness = len(unique) / len(shared)
        leg_factor = 1.0 + 0.1 * max(0, len(r.legs) - 1)
        credits[r.id] = recall * (0.5 + 0.5 * uniqueness) * leg_factor
    return credits


def record_use(
    store: Store,
    retrieved: list[Retrieved],
    answer_text: str,
    ucb_c: float = DEFAULT_UCB_C,
) -> dict[int, float]:
    """Attribute credit after an answer and update importance (TRACE-UCB).

    No LLM call: the answer text is the only new evidence. Records
    with no answer overlap still count a use (they were retrieved)
    but earn no credit, so their mean credit — and their importance —
    decays. Co-recall pairs (two credited records used for one
    answer) are logged for the sleep job; the online path never
    touches the graph edges directly.
    Returns {record_id: credit} for the caller to log or test.
    """
    if not retrieved or not answer_text or not answer_text.strip():
        return {}
    credits = attribute_credit(retrieved, answer_text)
    total = store.total_uses() + len(retrieved)
    for r in retrieved:
        rec = store.get_record(r.id)
        if rec is None:
            continue
        uses = rec["uses"] + 1
        credit = rec["credit"] + credits.get(r.id, 0.0)
        mean = credit / uses
        bonus = ucb_c * math.sqrt(math.log(total + 1) / uses)
        importance = min(2.0, max(0.0, mean + bonus))
        store.record_credit_update(r.id, uses, credit, importance)
    used = [r.id for r in retrieved if credits.get(r.id, 0.0) > 0]
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            store.log_co_recall(used[i], used[j])
    return credits


# ----------------------------------------------------------------------
# Merge drafting (no LLM)
# ----------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _norm_sentence(sentence: str) -> str:
    # Trailing sentence punctuation is not content: "morning." and
    # "morning!" are the same sentence for dedup purposes.
    s = re.sub(r"\s+", " ", sentence.strip().lower())
    return s.rstrip(".!?").strip()


def draft_merge_text(texts: list[str]) -> str:
    """Sentence union: every distinct sentence from the sources, once.

    Faithfulness holds by construction (no new words are invented).
    The length check in the certificate decides whether the union is
    actually a compression — for near-duplicates it is, for distinct
    facts it is not, and the merge is then correctly rejected.
    """
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for sentence in split_sentences(text):
            key = _norm_sentence(sentence)
            if key not in seen:
                seen.add(key)
                out.append(sentence)
    return " ".join(out)


# ----------------------------------------------------------------------
# Sufficiency certificate (the judge)
# ----------------------------------------------------------------------
def certificate_for_merge(
    store: Store,
    embedder,
    source_ids: list[int],
    merged_text: str,
    min_faithfulness: float = DEFAULT_MIN_FAITHFULNESS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    min_source_sim: float = DEFAULT_MIN_SOURCE_SIM,
) -> dict:
    """Run the four certificate checks. Returns the certificate dict.

    The certificate is pure rules + embeddings: faithfulness and
    length are token arithmetic, coverage and source similarity use
    the same MiniLM vectors the store already holds (falling back to
    word-set Jaccard when no embedder is available).
    """
    sources = [
        r
        for r in store.fetch(source_ids)
        if r is not None and r["status"] == "live"
    ]
    checks: dict[str, dict] = {}

    def _fail(reason: str) -> dict:
        return {
            "passed": False,
            "checks": checks,
            "reasons": [reason],
            "n_sources": len(sources),
        }

    if len(sources) < 2:
        return _fail("fewer than 2 live sources")
    if len({s["type"] for s in sources}) > 1:
        return _fail("mixed record types")

    merged_tokens = content_tokens(merged_text)
    merged_words = word_tokens(merged_text)
    if not merged_tokens:
        return _fail("empty merged text")
    source_union: set[str] = set()
    source_words_total = 0
    for s in sources:
        source_union |= content_tokens(s["text"])
        source_words_total += len(word_tokens(s["text"]))

    faithfulness = len(merged_tokens & source_union) / len(merged_tokens)
    checks["faithfulness"] = {
        "value": round(faithfulness, 4),
        "threshold": min_faithfulness,
        "passed": faithfulness >= min_faithfulness,
    }
    length_ratio = len(merged_words) / max(1, source_words_total)
    checks["length_ratio"] = {
        "value": round(length_ratio, 4),
        "threshold": max_length_ratio,
        "passed": length_ratio <= max_length_ratio,
    }

    merged_vec = None
    if embedder is not None:
        merged_vec = embedder.embed([merged_text])[0]
    covered = 0
    sims: list[float] = []
    for s in sources:
        s_tokens = content_tokens(s["text"])
        recall = (
            len(s_tokens & merged_tokens) / len(s_tokens) if s_tokens else 0.0
        )
        if merged_vec is not None:
            s_vec = store.get_vector(s["id"])
            if s_vec is None:
                s_vec = embedder.embed([s["text"]])[0]
            sim = float(merged_vec @ s_vec)
        else:
            sim = jaccard(merged_tokens, s_tokens)
        sims.append(sim)
        if recall >= 0.5 and sim >= min_source_sim:
            covered += 1
    coverage = covered / len(sources)
    checks["coverage"] = {
        "value": round(coverage, 4),
        "threshold": min_coverage,
        "passed": coverage >= min_coverage,
        "covered_sources": covered,
        "n_sources": len(sources),
    }
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    checks["source_similarity"] = {
        "value": round(mean_sim, 4),
        "threshold": min_source_sim,
        "passed": mean_sim >= min_source_sim,
    }

    reasons = [name for name, c in checks.items() if not c["passed"]]
    return {
        "passed": not reasons,
        "checks": checks,
        "reasons": reasons,
        "n_sources": len(sources),
    }


@dataclass
class MergeResult:
    status: str  # committed | rejected | dry-run-pass | dry-run-fail
    merge_id: int | None
    merged_record_id: int | None
    certificate: dict
    merged_text: str = ""


def propose_merge(
    store: Store,
    embedder,
    source_ids: list[int],
    merged_text: str | None = None,
    dry_run: bool = False,
    type_override: str | None = None,
    **cert_kwargs,
) -> MergeResult:
    """Draft (if needed), judge, and commit or reject a merge.

    On commit the merged record goes live, the sources are marked
    'superseded' (never deleted), and the audit row is written. On
    reject only the audit row is written. Dry-run judges without
    writing anything. ``type_override`` (P3) labels the merged
    record with the segment's classified type instead of the
    sources' type.
    """
    sources = [
        r
        for r in store.fetch(source_ids)
        if r is not None and r["status"] == "live"
    ]
    if len(sources) < 2:
        cert = {
            "passed": False,
            "checks": {},
            "reasons": ["fewer than 2 live sources"],
            "n_sources": len(sources),
        }
        if dry_run:
            return MergeResult("dry-run-fail", None, None, cert, merged_text or "")
        merge_id = store.add_merge(source_ids, None, merged_text or "", cert, "rejected")
        return MergeResult("rejected", merge_id, None, cert, merged_text or "")

    text = merged_text if merged_text is not None else draft_merge_text(
        [s["text"] for s in sources]
    )
    cert = certificate_for_merge(
        store, embedder, [s["id"] for s in sources], text, **cert_kwargs
    )
    if dry_run:
        return MergeResult(
            "dry-run-pass" if cert["passed"] else "dry-run-fail",
            None,
            None,
            cert,
            text,
        )
    if not cert["passed"]:
        merge_id = store.add_merge(
            [s["id"] for s in sources], None, text, cert, "rejected"
        )
        return MergeResult("rejected", merge_id, None, cert, text)

    tags = sorted({t for s in sources for t in s["tags"]})
    merged_id = store.add_record(
        text,
        type_=type_override or sources[0]["type"],
        tags=tags,
        meta={"merged_from": [s["id"] for s in sources]},
        embedding=(
            embedder.embed([text])[0] if embedder is not None else None
        ),
    )
    store.set_status_many(
        [s["id"] for s in sources], "superseded", superseded_by=merged_id
    )
    merge_id = store.add_merge(
        [s["id"] for s in sources], merged_id, text, cert, "committed"
    )
    return MergeResult("committed", merge_id, merged_id, cert, text)


def rollback_merge(store: Store, merge_id: int) -> bool:
    """Undo a committed merge: archive the merged record, restore sources.

    Raises ValueError for unknown ids and for merges that are not
    committed (rejected merges have nothing to undo).
    """
    m = store.get_merge(merge_id)
    if m is None:
        raise ValueError("unknown merge id %d" % merge_id)
    if m["status"] != "committed":
        raise ValueError(
            "merge %d is %s, not committed" % (merge_id, m["status"])
        )
    if m["merged_record_id"] is not None:
        store.set_status(m["merged_record_id"], "archived")
    store.set_status_many(m["merged_ids"], "live", superseded_by=None)
    store.set_merge_status(merge_id, "rolled-back")
    return True


# ----------------------------------------------------------------------
# Merge candidates (near-duplicates)
# ----------------------------------------------------------------------
def find_merge_candidates(
    store: Store,
    embedder=None,
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    max_pairs: int = 20,
) -> list[tuple[int, int, float]]:
    """Live same-type record pairs that look like the same memory twice.

    A pair qualifies on word-set Jaccard >= 0.9 (the .larp dedup rule)
    or, for pairs sharing at least half their tokens, cosine >= 0.90
    on the stored vectors. Each record appears in at most one pair
    (greedy by similarity, best first) so one sleep run never merges
    the same record twice. Returns [(id_a, id_b, similarity)].
    """
    live = store.list_live_records()
    by_type: dict[str, list[dict]] = {}
    for r in live:
        by_type.setdefault(r["type"], []).append(r)
    scored: list[tuple[float, int, int]] = []
    for group in by_type.values():
        tokens = {r["id"]: content_tokens(r["text"]) for r in group}
        vecs: dict[int, np.ndarray] = {}
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                j = jaccard(tokens[a["id"]], tokens[b["id"]])
                if j >= jaccard_threshold:
                    scored.append((j, a["id"], b["id"]))
                    continue
                if embedder is None or j < 0.5:
                    continue
                if a["id"] not in vecs:
                    v = store.get_vector(a["id"])
                    vecs[a["id"]] = (
                        v
                        if v is not None
                        else embedder.embed([a["text"]])[0]
                    )
                if b["id"] not in vecs:
                    v = store.get_vector(b["id"])
                    vecs[b["id"]] = (
                        v
                        if v is not None
                        else embedder.embed([b["text"]])[0]
                    )
                cos = float(vecs[a["id"]] @ vecs[b["id"]])
                if cos >= cosine_threshold:
                    scored.append((cos, a["id"], b["id"]))
    scored.sort(key=lambda t: -t[0])
    used: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for sim, a, b in scored:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        out.append((a, b, sim))
        if len(out) >= max_pairs:
            break
    return out


# ----------------------------------------------------------------------
# Sleep job
# ----------------------------------------------------------------------
def sleep(
    store: Store,
    embedder=None,
    max_merges: int = 20,
    prune_days: float = 90.0,
    min_co_recall: int = 2,
    dry_run: bool = False,
) -> dict:
    """The offline consolidation job ("Do agents need to sleep?").

    1. Propose merges for near-duplicate clusters; the judge decides.
    2. Prune dead memories: live records never used, with zero credit,
       older than ``prune_days``. Pruned records are marked, not
       deleted — the audit trail survives.
    3. Promote co-recall pairs (count >= ``min_co_recall``) to
       'co-recall' graph edges, both directions, weight = count.

    Dry-run reports what would happen and writes nothing.
    """
    summary = {
        "merges_proposed": 0,
        "merges_committed": 0,
        "merges_would_commit": 0,
        "merges_rejected": 0,
        "pruned": 0,
        "co_recall_pairs_promoted": 0,
        "dry_run": dry_run,
    }
    for a, b, _sim in find_merge_candidates(
        store, embedder, max_pairs=max_merges
    ):
        summary["merges_proposed"] += 1
        res = propose_merge(store, embedder, [a, b], dry_run=dry_run)
        if res.status == "committed":
            summary["merges_committed"] += 1
        elif res.status == "dry-run-pass":
            summary["merges_would_commit"] += 1
        else:
            summary["merges_rejected"] += 1

    old = store.list_live_records(older_than_days=prune_days)
    to_prune = [
        r for r in old if r["uses"] == 0 and r["credit"] == 0.0
    ]
    summary["pruned"] = len(to_prune)
    if to_prune and not dry_run:
        store.set_status_many([r["id"] for r in to_prune], "pruned")

    promoted = 0
    for a, b, count in store.co_recall_pairs(min_count=min_co_recall):
        ra = store.get_record(a)
        rb = store.get_record(b)
        if ra is None or rb is None:
            continue
        if ra["status"] != "live" or rb["status"] != "live":
            continue
        promoted += 1
        if not dry_run:
            store.add_edge(a, b, weight=float(count), kind="co-recall")
            store.add_edge(b, a, weight=float(count), kind="co-recall")
    summary["co_recall_pairs_promoted"] = promoted
    return summary
