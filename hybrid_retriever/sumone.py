"""P17 — SUMONE-style learned summariser (distilled version).

SUMONE (arXiv:2611.06445) trains a small standalone summariser
that plugs into any agent: GRPO fine-tuning of a 0.5-2B model
(Qwen) where the reward is masked reconstruction — a frozen
reconstructor must recover masked spans of the original
trajectory from the summary, plus a length penalty — learned
self-supervised from raw trajectories only, no reference
summaries. Its done condition: -75% inference cost with ~no
performance loss on a small agent benchmark, using the same
checkpoint across two harnesses.

SUBSTITUTIONS — stated plainly, here and in the README. No LLM,
no Qwen model, no GRPO training and no torch ran on this VM,
and none was downloaded or installed. The deterministic
stand-ins are:

1. Phase 1 (prompt-based compact-or with a reconstruction
   self-check) becomes an EXTRACTIVE summariser: sentence
   selection over trajectory turns, scored by tunable weights
   (goal-word overlap, numbers, quoted values, tool-result vs
   reasoning type, first-in-segment position), reusing P3's
   segment framing (each episode is a segment; a minimum
   number of sentences is kept per segment) and P19's
   extractive-only discipline (every summary sentence is
   verbatim from the original; nothing is invented). The
   reconstruction self-check is the quality gate, exactly as
   in the paper's Phase 1: mask spans of the original
   trajectory, attempt to recover each masked span from the
   summary alone with deterministic retrieval (BM25-style
   word-overlap over the summary sentences — the P6-style
   prototype-answering idea, no LLM), and REJECT the summary
   when its masked-span recovery rate falls below the fixed
   threshold ``RECOVERY_THRESHOLD`` (0.75). A rejected summary
   is re-summarised at a looser budget (x1.5, then x2.0) and,
   if it still fails, returned UNCOMPRESSED with a flag —
   the gate protects correctness instead of silently
   degrading it.
2. Phase 2 (GRPO fine-tuning) becomes a REWARD-DRIVEN
   PARAMETER SEARCH: the summariser exposes tunable
   parameters (sentence-selection weights, compression
   budget, minimum sentences per segment); a fixed grid
   search over that parameter space optimises EXACTLY the
   paper's reward — masked-span recovery rate minus a length
   penalty — on a training set of trajectories (seeded from
   the P21/P20 task trajectories and P3 segment data). The
   best config is kept as the "checkpoint" and evaluated
   held-out. That is parameter search, NOT GRPO — no weights
   were updated, no model was trained, and nothing here
   should be read as claiming otherwise.
3. The agent benchmark is the repo's own deterministic
   agent: the P21 grounded policy + P20 rule solver from
   P16, run on the fixed P21 12-task + P20 24-task sets
   TWICE — once with full trajectories as context, once
   with summaries from the tuned summariser (tool-result
   episodes enter the context already summarised, via
   ``SummarisingContext``). The two task sets are the two
   harnesses; the same checkpoint serves both. Inference
   cost is total context tokens fed to the agent (P16's
   ``tokens_processed`` accounting, deterministic
   ``ceil(len/4)``). Summariser input tokens are reported
   separately as overhead — they are NOT folded into the
   agent's cost, because in the paper the summariser is a
   small cheap model and the agent is the expensive one;
   the split is labelled wherever it appears.

Masking scheme (frozen before the search — see
``MASK_*`` constants). Candidate spans of the original
trajectory are (a) numeric literals (``12``, ``33.00`` —
including record ids and computed totals), (b) quoted
fact values (``'...'`` / ``"..."``), and (c) goal key
tokens (content words, length >= 3, from the task goal
that appear in the trajectory). From the candidates, a
FIXED seeded sample of at most ``MASKS_PER_TRAJECTORY``
spans is masked (``random.Random(MASK_SEED)`` over a
deterministically sorted candidate list) — a sample, not
all spans, because masking every number would make the
recovery rate equal the kept fraction by construction and
the reward would then favour no compression at all. The
sample is fixed before the search, identical for every
config, so configs are compared on the same masks.
``coverage`` (fraction of ALL candidate spans present in
the summary) is reported as a diagnostic, never as the
gate.

Done condition (frozen before the measurement run): agent
inference cost down >= 75% AND success within <= 1 task
lost per set, on BOTH harnesses (P21 12-task and P20
24-task), with the single tuned checkpoint. If either
harness misses, the table is reported honestly and
``done`` is False — the grid, the masking scheme, the
reward weights and the threshold are NOT re-tuned after
the held-out measurement.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import re
import tempfile

from .context_lab import Context, PolicyConfig, run_task_with_policy
from .eviction import estimate_tokens

# ----------------------------------------------------------------------
# Frozen constants (fixed before the search / eval — do not re-tune
# after measuring)
# ----------------------------------------------------------------------

# A summary is REJECTED when its masked-span recovery rate is
# below this. Documented, fixed, and applied identically in
# Phase 1 and in the parameter search.
RECOVERY_THRESHOLD = 0.75
# The paper's reward: recovery rate minus LENGTH_PENALTY times
# the summary/original token ratio.
LENGTH_PENALTY = 0.5
# Masked-span sample size: at most this many spans per
# trajectory (search) / per episode (agent arm). Seeded, so
# every config is judged on the same masks.
MASKS_PER_TRAJECTORY = 8
MASKS_PER_EPISODE = 6
MASK_SEED = 7
# Retry budgets on rejection: re-summarise at these multiples
# of the configured budget (capped at 1.0) before falling back
# to the uncompressed original.
RETRY_MULTIPLIERS = (1.5, 2.0)
# Done condition: agent cost down >= this fraction, success
# within this many tasks lost per harness set.
DONE_MIN_REDUCTION = 0.75
DONE_MAX_LOST_PER_SET = 1

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Quoted fact values: the quote must not sit between word
# characters, so a possessive apostrophe ("Maya's") is never
# mistaken for the opening quote of a fact value.
_QUOTED_RE = re.compile(
    r"(?<![A-Za-z0-9])'([^']{2,})'(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])\"([^\"]{2,})\"(?![A-Za-z0-9])"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "to", "of",
        "in", "on", "for", "and", "or", "what", "how", "many",
        "write", "says", "saying", "this", "that", "it", "its",
        "with", "from", "will", "then", "when", "does", "do",
        "i", "you", "we", "our", "your", "his", "her", "their",
        "about", "into", "after", "before", "which", "there",
        "have", "has", "had", "been", "being", "not", "but",
        "all", "each", "more", "most", "some", "such", "than",
        "them", "they", "these", "those", "who", "whom",
    }
)


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOPWORDS}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


# ----------------------------------------------------------------------
# Phase 1 — extractive summariser
# ----------------------------------------------------------------------


class SummariserConfig:
    """Tunable parameters of the extractive summariser — the
    "policy" the parameter search optimises (Phase 2 analogue).

    ``budget_frac``: fraction of the original tokens the
      summary may use (before the per-segment minimum).
    ``w_number``: score bonus for a sentence containing a
      number (record ids, totals, scores — the facts the
      verifiers check).
    ``w_goal``: score per goal-word overlapping the sentence
      (the task's own words — what the agent will look for).
    ``w_tool``: score bonus for tool-result sentences.
    ``w_reason``: score bonus for reasoning sentences.
    ``w_first``: score bonus for the first sentence of a
      segment (segment openers carry the topic).
    ``min_per_segment``: minimum sentences kept per segment
      (P3 framing: each episode is a segment); may push the
      summary over budget — the minimum wins, and the
      overrun is visible in the reported token ratio.
    """

    def __init__(
        self,
        budget_frac: float = 0.25,
        w_number: float = 2.0,
        w_goal: float = 1.0,
        w_tool: float = 0.5,
        w_reason: float = 0.0,
        w_first: float = 0.5,
        min_per_segment: int = 1,
    ) -> None:
        if not (0.0 < budget_frac <= 1.0):
            raise ValueError(
                "budget_frac must be in (0, 1], got %r" % (budget_frac,)
            )
        for name, value in (
            ("w_number", w_number),
            ("w_goal", w_goal),
            ("w_tool", w_tool),
            ("w_reason", w_reason),
            ("w_first", w_first),
        ):
            if value < 0:
                raise ValueError(
                    "%s must be >= 0, got %r" % (name, value)
                )
        if min_per_segment < 0:
            raise ValueError(
                "min_per_segment must be >= 0, got %r"
                % (min_per_segment,)
            )
        self.budget_frac = budget_frac
        self.w_number = w_number
        self.w_goal = w_goal
        self.w_tool = w_tool
        self.w_reason = w_reason
        self.w_first = w_first
        self.min_per_segment = min_per_segment

    @classmethod
    def from_dict(cls, spec: dict) -> "SummariserConfig":
        allowed = {
            "budget_frac", "w_number", "w_goal", "w_tool",
            "w_reason", "w_first", "min_per_segment",
        }
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(
                "unknown summariser keys: %s" % sorted(unknown)
            )
        return cls(**spec)

    def as_dict(self) -> dict:
        return {
            "budget_frac": self.budget_frac,
            "w_number": self.w_number,
            "w_goal": self.w_goal,
            "w_tool": self.w_tool,
            "w_reason": self.w_reason,
            "w_first": self.w_first,
            "min_per_segment": self.min_per_segment,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SummariserConfig(%r)" % (self.as_dict(),)


DEFAULT_CONFIG = SummariserConfig()


def _split_units(text: str) -> list[str]:
    """Split text into selectable units: one per non-empty line,
    with long prose lines further split on sentence boundaries.
    JSON / table / log lines stay whole — splitting them on
    ". " would shred numbers like ``0.5`` and record ids."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > 200 and ". " in line:
            units.extend(
                p.strip()
                for p in _SENTENCE_SPLIT_RE.split(line)
                if p.strip()
            )
        else:
            units.append(line)
    return units


class _Unit:
    """One selectable sentence/line of a trajectory."""

    __slots__ = ("segment", "index", "text", "type", "is_first")

    def __init__(
        self, segment: int, index: int, text: str, type_: str,
        is_first: bool,
    ) -> None:
        self.segment = segment
        self.index = index
        self.text = text
        self.type = type_
        self.is_first = is_first


def _score_unit(
    unit: _Unit, goal_words: set[str], config: SummariserConfig,
) -> float:
    words = _words(unit.text)
    score = config.w_goal * len(words & goal_words)
    if _NUMBER_RE.search(unit.text):
        score += config.w_number
    if _QUOTED_RE.search(unit.text):
        score += 1.0  # quoted fact values are mask candidates too
    if unit.type == "tool_result":
        score += config.w_tool
    elif unit.type == "reasoning":
        score += config.w_reason
    if unit.is_first:
        score += config.w_first
    return score


def _select_units(
    units: list[_Unit],
    goal_words: set[str],
    config: SummariserConfig,
    budget_frac: float,
) -> list[_Unit]:
    """Greedy selection by score (ties: earliest first) until
    the token budget is reached, then enforce the per-segment
    minimum. Always keeps at least one unit when any exist.
    Returns the kept units in original order."""
    if not units:
        return []
    original_tokens = sum(
        estimate_tokens(u.text) for u in units
    )
    budget_tokens = max(1, int(budget_frac * original_tokens))
    scored = sorted(
        ((_score_unit(u, goal_words, config), u) for u in units),
        key=lambda kv: (-kv[0], kv[1].index),
    )
    kept: list[_Unit] = []
    kept_tokens = 0
    for _, unit in scored:
        if kept and kept_tokens >= budget_tokens:
            break
        kept.append(unit)
        kept_tokens += estimate_tokens(unit.text)
    if config.min_per_segment > 0:
        kept_idx = {u.index for u in kept}
        by_segment: dict[int, list[_Unit]] = {}
        for unit in units:
            by_segment.setdefault(unit.segment, []).append(unit)
        for seg_units in by_segment.values():
            have = sum(1 for u in seg_units if u.index in kept_idx)
            if have >= config.min_per_segment:
                continue
            ranked = sorted(
                seg_units,
                key=lambda u: (
                    -_score_unit(u, goal_words, config), u.index
                ),
            )
            for unit in ranked:
                if have >= config.min_per_segment:
                    break
                if unit.index not in kept_idx:
                    kept.append(unit)
                    kept_idx.add(unit.index)
                    have += 1
    kept.sort(key=lambda u: u.index)
    return kept


class Mask:
    """One masked span: its text, the unit it came from, and
    the context words the reconstructor may use (content words
    of its unit with the span's own tokens removed)."""

    __slots__ = ("span", "unit_index", "context")

    def __init__(
        self, span: str, unit_index: int, context: set[str],
    ) -> None:
        self.span = span
        self.unit_index = unit_index
        self.context = context


def _candidate_spans(
    units: list[_Unit], goal_words: set[str],
) -> list[Mask]:
    """All maskable spans, deterministically ordered by
    (unit, start offset, span text): numbers, quoted values,
    goal key tokens. Deduplicated per (span, unit)."""
    seen: set[tuple[str, int]] = set()
    out: list[Mask] = []
    for unit in units:
        words = _words(unit.text)
        spans: list[tuple[int, str]] = []
        for m in _NUMBER_RE.finditer(unit.text):
            spans.append((m.start(), m.group(0)))
        for m in _QUOTED_RE.finditer(unit.text):
            inner = m.group(1) or m.group(2) or ""
            spans.append((m.start(), inner))
        for w in sorted(words & goal_words):
            if len(w) >= 3:
                for m in re.finditer(
                    r"\b%s\b" % re.escape(w), unit.text.lower()
                ):
                    spans.append((m.start(), w))
        spans.sort()
        for _, span in spans:
            key = (span, unit.index)
            if key in seen:
                continue
            seen.add(key)
            context = _content_words(unit.text) - _words(span)
            out.append(Mask(span, unit.index, context))
    return out


def sample_masks(
    units: list[_Unit],
    goal_words: set[str],
    max_masks: int,
    seed: int = MASK_SEED,
) -> list[Mask]:
    """The FIXED masked-span sample for a trajectory: at most
    ``max_masks`` candidates, drawn with ``random.Random(seed)``
    over the deterministically sorted candidate list. The same
    (units, goal, seed) always yields the same masks, so every
    config in the search is judged on identical masks."""
    candidates = _candidate_spans(units, goal_words)
    if len(candidates) <= max_masks:
        return candidates
    rng = random.Random(seed)
    return sorted(
        rng.sample(candidates, max_masks),
        key=lambda m: (m.unit_index, m.span),
    )


def _retrieve_best(
    summary_units: list[_Unit], context: set[str],
) -> _Unit | None:
    """Deterministic retrieval over the summary sentences:
    BM25-style word overlap with the mask's context words,
    ties broken by earliest sentence — the frozen
    reconstructor (no LLM, no vectors; the P6-style
    prototype-answering idea applied to the summary)."""
    best: _Unit | None = None
    best_score = -1
    for unit in summary_units:
        score = len(context & _words(unit.text))
        if score > best_score:
            best_score = score
            best = unit
    return best


def _span_recovered(mask: Mask, summary_units: list[_Unit]) -> bool:
    """Recovered when the masked span's text (or all of its
    tokens) appears in the single best-retrieved summary
    sentence. A span is NOT recovered merely by appearing
    somewhere unretrieved — the reconstructor must find it
    from the mask's context."""
    if not summary_units:
        return False
    best = _retrieve_best(summary_units, mask.context)
    if best is None:
        return False
    span_norm = _norm(mask.span)
    best_norm = _norm(best.text)
    if span_norm and span_norm in best_norm:
        return True
    tokens = _words(mask.span)
    return bool(tokens) and tokens <= _words(best.text)


def recovery_rate(
    masks: list[Mask], summary_units: list[_Unit],
) -> float:
    """Fraction of masked spans recovered from the summary
    alone. No masks means nothing to recover: 1.0 (vacuous)."""
    if not masks:
        return 1.0
    recovered = sum(
        1 for m in masks if _span_recovered(m, summary_units)
    )
    return recovered / len(masks)


def coverage_rate(
    units: list[_Unit],
    goal_words: set[str],
    summary_units: list[_Unit],
) -> float:
    """Diagnostic only (never the gate): fraction of ALL
    candidate spans whose text appears anywhere in the
    summary. Where recovery is the sampled, retrieval-gated
    measure, coverage is the unsampled presence measure."""
    candidates = _candidate_spans(units, goal_words)
    if not candidates:
        return 1.0
    summary_norm = _norm("\n".join(u.text for u in summary_units))
    present = 0
    for mask in candidates:
        span_norm = _norm(mask.span)
        tokens = _words(mask.span)
        if (span_norm and span_norm in summary_norm) or (
            tokens and tokens <= _words(summary_norm)
        ):
            present += 1
    return present / len(candidates)


def reward_fn(recovery: float, token_ratio: float) -> float:
    """The paper's reward: masked-span recovery rate minus a
    length penalty times the summary/original token ratio."""
    return recovery - LENGTH_PENALTY * token_ratio


class SummaryResult:
    """One summarisation, with its gate verdict."""

    def __init__(
        self,
        text: str,
        accepted: bool,
        fell_back: bool,
        recovery_rate: float,
        coverage: float,
        tokens_before: int,
        tokens_after: int,
        reward: float,
        attempts: list[dict],
        n_masks: int,
        n_recovered: int,
        config: SummariserConfig,
    ) -> None:
        self.text = text
        self.accepted = accepted
        self.fell_back = fell_back
        self.recovery_rate = recovery_rate
        self.coverage = coverage
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.reward = reward
        self.attempts = attempts
        self.n_masks = n_masks
        self.n_recovered = n_recovered
        self.config = config

    @property
    def token_ratio(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_after / self.tokens_before

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "accepted": self.accepted,
            "fell_back": self.fell_back,
            "recovery_rate": self.recovery_rate,
            "coverage": self.coverage,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "token_ratio": self.token_ratio,
            "reward": self.reward,
            "attempts": self.attempts,
            "n_masks": self.n_masks,
            "n_recovered": self.n_recovered,
            "config": self.config.as_dict(),
        }


def _units_from_episodes(
    episodes: list[dict],
) -> tuple[list[_Unit], str]:
    """Flatten episodes (each a P3-style segment) into scored
    units; returns the units and the full original text."""
    units: list[_Unit] = []
    texts: list[str] = []
    for seg_id, ep in enumerate(episodes):
        text = ep["text"]
        texts.append(text)
        seg_units = _split_units(text)
        for j, sent in enumerate(seg_units):
            units.append(
                _Unit(
                    seg_id, len(units), sent,
                    ep.get("type", "tool_result"),
                    j == 0,
                )
            )
    return units, "\n".join(texts)


def summarise_units(
    units: list[_Unit],
    goal: str,
    config: SummariserConfig,
    max_masks: int = MASKS_PER_TRAJECTORY,
) -> SummaryResult:
    """Summarise a unit list with the reconstruction gate:
    select, check recovery, retry at looser budgets, fall back
    to the uncompressed original when every attempt fails the
    threshold. The fallback keeps the agent correct at the
    price of no saving on that trajectory — the gate's job."""
    goal_words = _content_words(goal)
    original_text = "\n".join(u.text for u in units)
    tokens_before = estimate_tokens(original_text)
    masks = sample_masks(units, goal_words, max_masks)
    attempts: list[dict] = []
    budgets = [config.budget_frac] + [
        min(1.0, config.budget_frac * m) for m in RETRY_MULTIPLIERS
    ]
    seen_budgets: list[float] = []
    for budget in budgets:
        if budget in seen_budgets:
            continue
        seen_budgets.append(budget)
        kept = _select_units(units, goal_words, config, budget)
        rate = recovery_rate(masks, kept)
        cov = coverage_rate(units, goal_words, kept)
        kept_tokens = estimate_tokens(
            "\n".join(u.text for u in kept)
        )
        attempts.append(
            {
                "budget_frac": budget,
                "recovery_rate": rate,
                "coverage": cov,
                "tokens_after": kept_tokens,
                "accepted": rate >= RECOVERY_THRESHOLD,
            }
        )
        if rate >= RECOVERY_THRESHOLD:
            text = "\n".join(u.text for u in kept)
            return SummaryResult(
                text=text,
                accepted=True,
                fell_back=False,
                recovery_rate=rate,
                coverage=cov,
                tokens_before=tokens_before,
                tokens_after=kept_tokens,
                reward=reward_fn(rate, kept_tokens / tokens_before)
                if tokens_before
                else rate,
                attempts=attempts,
                n_masks=len(masks),
                n_recovered=sum(
                    1 for m in masks if _span_recovered(m, kept)
                ),
                config=config,
            )
    # Every attempt failed the gate: return the original
    # uncompressed, flagged — never a failing summary.
    full_units = list(units)
    full_rate = recovery_rate(masks, full_units)
    full_cov = coverage_rate(units, goal_words, full_units)
    return SummaryResult(
        text=original_text,
        accepted=False,
        fell_back=True,
        recovery_rate=full_rate,
        coverage=full_cov,
        tokens_before=tokens_before,
        tokens_after=tokens_before,
        reward=reward_fn(full_rate, 1.0) if tokens_before else full_rate,
        attempts=attempts,
        n_masks=len(masks),
        n_recovered=sum(
            1 for m in masks if _span_recovered(m, full_units)
        ),
        config=config,
    )


def summarise_trajectory(
    episodes: list[dict],
    goal: str,
    config: SummariserConfig,
    max_masks: int = MASKS_PER_TRAJECTORY,
) -> SummaryResult:
    """Summarise a whole trajectory (list of episode dicts
    with ``type`` / ``kind`` / ``text`` — one P3-style segment
    per episode) under the reconstruction gate."""
    units, _ = _units_from_episodes(episodes)
    return summarise_units(units, goal, config, max_masks)


def summarise_text(
    text: str,
    goal: str,
    config: SummariserConfig,
    type_: str = "tool_result",
) -> SummaryResult:
    """Summarise one episode's text (a single segment)."""
    units = [
        _Unit(0, i, sent, type_, i == 0)
        for i, sent in enumerate(_split_units(text))
    ]
    return summarise_units(
        units, goal, config, max_masks=MASKS_PER_EPISODE
    )


def verify_extractive(original: str, summary: str) -> list[str]:
    """Summary lines that are not verbatim from the original
    (whitespace-normalised). Extractive-only summarisation
    must return an empty list — except for a fell-back
    summary, which IS the original."""
    orig_lines = {_norm(ln) for ln in original.splitlines() if ln.strip()}
    orig_norm = _norm(original)
    bad: list[str] = []
    for line in summary.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _norm(stripped) in orig_lines or _norm(stripped) in orig_norm:
            continue
        bad.append(line)
    return bad


# ----------------------------------------------------------------------
# The agent arm: tool results enter the context already summarised
# ----------------------------------------------------------------------


class SummarisingContext(Context):
    """A P16 Context whose tool-result episodes are summarised
    (tuned config + reconstruction gate) the moment they are
    appended — so the agent's live context is the trajectory
    summary, never the full trajectory. User turns, reasoning
    and tool calls are untouched: they are the agent's own
    words, not trajectory history. ``summariser_tokens_read``
    is the overhead accounting (original tokens the
    summariser read); it is reported separately from the
    agent's ``tokens_processed`` and is NOT folded into the
    agent's inference cost."""

    def __init__(self, config: SummariserConfig) -> None:
        super().__init__()
        self.config = config
        self.goal = ""
        self.summariser_tokens_read = 0
        self.n_summarised = 0
        self.n_rejected = 0
        self.n_fell_back = 0
        self.recovery_rates: list[float] = []

    def append(self, type_: str, kind: str, text: str):
        if type_ == "user" and not self.goal:
            self.goal = text
        if type_ == "tool_result":
            result = summarise_text(text, self.goal, self.config)
            self.summariser_tokens_read += result.tokens_before
            self.n_summarised += 1
            self.recovery_rates.append(result.recovery_rate)
            if not result.accepted:
                self.n_rejected += 1
            if result.fell_back:
                self.n_fell_back += 1
            text = result.text
        return super().append(type_, kind, text)


# ----------------------------------------------------------------------
# Phase 2 — reward-driven parameter search (NOT GRPO)
# ----------------------------------------------------------------------

# Fixed grid, frozen before the measurement run. Grid order is
# the tie-break order: the first config to reach the best mean
# reward wins, so the search is fully deterministic.
GRID_BUDGETS = (0.15, 0.25, 0.40)
GRID_W_NUMBER = (1.0, 3.0)
GRID_W_GOAL = (0.5, 1.5)
GRID_W_TOOL = (0.0, 0.5)
GRID_MIN_PER_SEGMENT = (1, 2)


def build_grid() -> list[SummariserConfig]:
    """All 48 grid configs in fixed order (budget x w_number x
    w_goal x w_tool x min_per_segment); the remaining weights
    stay at their defaults."""
    grid: list[SummariserConfig] = []
    for budget, w_num, w_goal, w_tool, min_seg in itertools.product(
        GRID_BUDGETS,
        GRID_W_NUMBER,
        GRID_W_GOAL,
        GRID_W_TOOL,
        GRID_MIN_PER_SEGMENT,
    ):
        grid.append(
            SummariserConfig(
                budget_frac=budget,
                w_number=w_num,
                w_goal=w_goal,
                w_tool=w_tool,
                min_per_segment=min_seg,
            )
        )
    return grid


def mean_reward(
    trajectories: list[dict], config: SummariserConfig
) -> dict:
    """Mean paper reward of one config over trajectories
    (each a ``{"goal", "episodes"}`` dict)."""
    rewards: list[float] = []
    recoveries: list[float] = []
    ratios: list[float] = []
    fallbacks = 0
    for traj in trajectories:
        result = summarise_trajectory(
            traj["episodes"], traj["goal"], config
        )
        rewards.append(result.reward)
        recoveries.append(result.recovery_rate)
        ratios.append(result.token_ratio)
        if result.fell_back:
            fallbacks += 1
    n = len(rewards)
    return {
        "config": config.as_dict(),
        "mean_reward": sum(rewards) / n if n else 0.0,
        "mean_recovery": sum(recoveries) / n if n else 0.0,
        "mean_token_ratio": sum(ratios) / n if n else 0.0,
        "n_fell_back": fallbacks,
        "n_trajectories": n,
    }


def parameter_search(
    trajectories: list[dict],
    grid: list[SummariserConfig] | None = None,
) -> dict:
    """Grid search over the parameter space, optimising exactly
    the paper's reward (masked-span recovery minus length
    penalty) on the training trajectories. Deterministic: fixed
    grid order, strict improvement to replace the incumbent.
    Returns the best config ("checkpoint") plus the default
    config's score for comparison. This is parameter search,
    not GRPO — no weights were updated and no model trained."""
    grid = grid if grid is not None else build_grid()
    default = mean_reward(trajectories, DEFAULT_CONFIG)
    best: dict | None = None
    for config in grid:
        scored = mean_reward(trajectories, config)
        if best is None or scored["mean_reward"] > best["mean_reward"]:
            best = scored
    assert best is not None
    return {
        "best": best,
        "default": default,
        "n_configs": len(grid),
        "n_trajectories": len(trajectories),
        "reward_improvement": (
            best["mean_reward"] - default["mean_reward"]
        ),
    }


# ----------------------------------------------------------------------
# Training trajectories: P21/P20 teacher runs + P3 segment data
# ----------------------------------------------------------------------


def _ctx_to_trajectory(task: dict, ctx: Context) -> dict:
    goal = task.get("goal") or task.get("question") or ""
    return {
        "task_id": task.get("task_id") or task.get("id"),
        "goal": goal,
        "episodes": [
            {"type": e.type, "kind": e.kind, "text": e.original_text}
            for e in ctx.episodes
        ],
    }


def capture_trajectories(
    tasks: list[dict],
    vbd_store=None,
    embedder=None,
    run_dir: str | None = None,
) -> list[dict]:
    """Teacher runs: each task through the P16 baseline agent
    (no compaction) with the context captured — the raw
    trajectories the search learns from. Self-supervised: no
    reference summaries, only the trajectories themselves."""
    from .context_lab import PolicyConfig as _Policy

    baseline = _Policy(primitive="none", trigger="never")
    trajectories: list[dict] = []
    for task in tasks:
        _, ctx = run_task_with_policy(
            task, baseline,
            vbd_store=vbd_store, embedder=embedder,
            run_dir=run_dir, return_ctx=True,
        )
        trajectories.append(_ctx_to_trajectory(task, ctx))
    return trajectories


def build_p3_segment_trajectories(
    n_trajectories: int = 4,
) -> list[dict]:
    """Trajectories seeded from P3 segment data: the P1
    synthetic corpus is segmented (P3 semantic boundaries
    with the repo embedder, fixed-window fallback), and each
    trajectory is a run of consecutive segments — one episode
    per segment — with a synthetic-corpus query as its goal.
    These carry no agent task, so they train the selector
    only (recovery reward), never the agent eval."""
    from .eval import load_synthetic
    from .segments import (
        detect_boundaries,
        detect_boundaries_fixed_window,
    )

    corpus_path = os.path.join(
        "eval_data", "synthetic_corpus.jsonl"
    )
    queries_path = os.path.join(
        "eval_data", "synthetic_queries.json"
    )
    records, queries = load_synthetic(corpus_path, queries_path)
    texts = [r["text"] for r in records[:40]]
    try:
        from .embeddings import get_embedder

        segments = detect_boundaries(
            texts, get_embedder(), min_turns=2, max_turns=8
        )
    except Exception:
        segments = detect_boundaries_fixed_window(len(texts), 5)
    trajectories: list[dict] = []
    # Group consecutive segments into trajectories of 3.
    groups = [
        segments[i: i + 3] for i in range(0, len(segments), 3)
    ]
    for gi, group in enumerate(groups[:n_trajectories]):
        episodes = [
            {
                "type": "tool_result",
                "kind": "segment",
                "text": "\n".join(texts[i] for i in seg),
            }
            for seg in group
            if len(seg) >= 2
        ]
        if len(episodes) < 2:
            continue
        q = queries[gi % len(queries)]
        goal = q.get("question", "") if isinstance(q, dict) else str(q)
        trajectories.append(
            {
                "task_id": "p3-seg-%d" % gi,
                "goal": goal,
                "episodes": episodes,
            }
        )
    return trajectories


# ----------------------------------------------------------------------
# Phase 2 — the eval harness: same checkpoint, two harnesses
# ----------------------------------------------------------------------


def run_sumone_eval(
    n_tobench: int = 12,
    n_vbd: int = 24,
    verbose: bool = True,
) -> dict:
    """Full P17 eval. (1) Capture teacher trajectories on the
    train split (first half of each set) + P3 segment
    trajectories; grid-search the reward; the best config is
    the checkpoint. (2) Score default vs checkpoint on the
    held-out split (second half of each set). (3) Run the
    deterministic agent on the FIXED full sets twice — full
    trajectories vs summaries from the checkpoint — per
    harness (P21 12-task, P20 24-task). Nothing is re-tuned
    after the held-out measurement: the grid, the masking
    scheme, the reward weights and the threshold were frozen
    before this ran."""
    import functools

    from .embeddings import get_embedder
    from .tobench import build_tobench_tasks
    from .verify import build_vbd_store, build_vbd_tasks

    tb_tasks = build_tobench_tasks()[:n_tobench]
    vbd_tasks = build_vbd_tasks(n_tasks=n_vbd)
    embedder = get_embedder()
    vbd_store, _ = build_vbd_store(
        os.path.join("eval_data", "synthetic_corpus.jsonl"),
        embedder=embedder,
    )
    run_dir = tempfile.mkdtemp(prefix="hr-sumone-")
    if verbose:
        print(
            "sumone eval: %d tasks (%d tobench + %d vbd); "
            "train split %d + %d, held-out %d + %d; "
            "grid %d configs"
            % (
                len(tb_tasks) + len(vbd_tasks),
                len(tb_tasks), len(vbd_tasks),
                len(tb_tasks) // 2, len(vbd_tasks) // 2,
                len(tb_tasks) - len(tb_tasks) // 2,
                len(vbd_tasks) - len(vbd_tasks) // 2,
                len(build_grid()),
            )
        )
    # (1) Train trajectories + search.
    train_tasks = (
        tb_tasks[: len(tb_tasks) // 2]
        + vbd_tasks[: len(vbd_tasks) // 2]
    )
    heldout_tasks = (
        tb_tasks[len(tb_tasks) // 2:]
        + vbd_tasks[len(vbd_tasks) // 2:]
    )
    train_trajs = capture_trajectories(
        train_tasks, vbd_store=vbd_store, embedder=embedder,
        run_dir=run_dir,
    )
    p3_trajs = build_p3_segment_trajectories()
    train_all = train_trajs + p3_trajs
    search = parameter_search(train_all)
    checkpoint = SummariserConfig.from_dict(search["best"]["config"])
    if verbose:
        print(
            "search: best mean reward %.3f (default %.3f, "
            "improvement %+.3f) over %d train trajectories "
            "(%d agent + %d P3-segment); checkpoint=%s"
            % (
                search["best"]["mean_reward"],
                search["default"]["mean_reward"],
                search["reward_improvement"],
                len(train_all), len(train_trajs), len(p3_trajs),
                search["best"]["config"],
            )
        )
    # (2) Held-out reward: default vs checkpoint.
    heldout_trajs = capture_trajectories(
        heldout_tasks, vbd_store=vbd_store, embedder=embedder,
        run_dir=run_dir,
    )
    heldout_default = mean_reward(heldout_trajs, DEFAULT_CONFIG)
    heldout_best = mean_reward(heldout_trajs, checkpoint)
    if verbose:
        print(
            "held-out reward (%d trajectories): default %.3f "
            "(recovery %.3f, ratio %.3f) vs checkpoint %.3f "
            "(recovery %.3f, ratio %.3f)"
            % (
                len(heldout_trajs),
                heldout_default["mean_reward"],
                heldout_default["mean_recovery"],
                heldout_default["mean_token_ratio"],
                heldout_best["mean_reward"],
                heldout_best["mean_recovery"],
                heldout_best["mean_token_ratio"],
            )
        )
    # (3) Agent eval, full vs summarised, same checkpoint.
    baseline = PolicyConfig(primitive="none", trigger="never")
    summarising_cls = functools.partial(
        SummarisingContext, config=checkpoint
    )
    harnesses: dict[str, dict] = {}
    all_tasks = [("tobench", tb_tasks), ("vbd", vbd_tasks)]
    for name, tasks in all_tasks:
        full_passed = 0
        sum_passed = 0
        full_tokens = 0
        sum_tokens = 0
        summariser_read = 0
        n_fell_back = 0
        n_summarised = 0
        recovery_all: list[float] = []
        per_task: list[dict] = []
        for task in tasks:
            full = run_task_with_policy(
                task, baseline,
                vbd_store=vbd_store, embedder=embedder,
                run_dir=run_dir,
            )
            summarised, sctx = run_task_with_policy(
                task, baseline,
                vbd_store=vbd_store, embedder=embedder,
                run_dir=run_dir, return_ctx=True,
                context_cls=summarising_cls,
            )
            full_passed += 1 if full["passed"] else 0
            sum_passed += 1 if summarised["passed"] else 0
            full_tokens += full["tokens"]
            sum_tokens += summarised["tokens"]
            summariser_read += sctx.summariser_tokens_read
            n_fell_back += sctx.n_fell_back
            n_summarised += sctx.n_summarised
            recovery_all.extend(sctx.recovery_rates)
            per_task.append(
                {
                    "task_id": full["task_id"],
                    "full_passed": full["passed"],
                    "sum_passed": summarised["passed"],
                    "full_tokens": full["tokens"],
                    "sum_tokens": summarised["tokens"],
                }
            )
        reduction = (
            1.0 - sum_tokens / full_tokens if full_tokens else 0.0
        )
        lost = full_passed - sum_passed
        harnesses[name] = {
            "n_tasks": len(tasks),
            "full_passed": full_passed,
            "sum_passed": sum_passed,
            "tasks_lost": lost,
            "full_tokens": full_tokens,
            "sum_tokens": sum_tokens,
            "cost_reduction": reduction,
            "token_ratio": (
                sum_tokens / full_tokens if full_tokens else 0.0
            ),
            "summariser_tokens_read": summariser_read,
            "n_tool_results_summarised": n_summarised,
            "n_fell_back": n_fell_back,
            "mean_episode_recovery": (
                sum(recovery_all) / len(recovery_all)
                if recovery_all
                else 0.0
            ),
            "done": bool(
                reduction >= DONE_MIN_REDUCTION
                and lost <= DONE_MAX_LOST_PER_SET
            ),
            "per_task": per_task,
        }
    vbd_store.close()
    done = all(h["done"] for h in harnesses.values())
    return {
        "n_tobench": len(tb_tasks),
        "n_vbd": len(vbd_tasks),
        "n_train_trajectories": len(train_all),
        "n_train_agent": len(train_trajs),
        "n_train_p3": len(p3_trajs),
        "n_heldout_trajectories": len(heldout_trajs),
        "search": search,
        "checkpoint": checkpoint.as_dict(),
        "heldout_default": heldout_default,
        "heldout_best": heldout_best,
        "heldout_reward_improvement": (
            heldout_best["mean_reward"]
            - heldout_default["mean_reward"]
        ),
        "harnesses": harnesses,
        "done": done,
        "done_condition": {
            "min_cost_reduction": DONE_MIN_REDUCTION,
            "max_tasks_lost_per_set": DONE_MAX_LOST_PER_SET,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "length_penalty": LENGTH_PENALTY,
        },
    }


def print_sumone_report(result: dict) -> None:
    print("\n== P17 SUMONE-style learned summariser (distilled) ==")
    print(
        "SUBSTITUTION: no LLM, no Qwen model, no GRPO training, "
        "no torch. Deterministic extractive summariser + "
        "reconstruction self-check; Phase 2 is a reward-driven "
        "PARAMETER SEARCH (grid over selection weights / "
        "budget / min-per-segment), not fine-tuning. Agent cost "
        "is context tokens fed to the agent (P16 accounting); "
        "summariser input tokens are reported separately as "
        "overhead, not folded into agent cost."
    )
    print(
        "checkpoint (best of %d configs on %d train "
        "trajectories: %d agent + %d P3-segment): %s"
        % (
            result["search"]["n_configs"],
            result["n_train_trajectories"],
            result["n_train_agent"],
            result["n_train_p3"],
            result["checkpoint"],
        )
    )
    print(
        "search mean reward: best %.3f vs default %.3f "
        "(improvement %+.3f); held-out (%d trajectories): "
        "checkpoint %.3f vs default %.3f (improvement %+.3f)"
        % (
            result["search"]["best"]["mean_reward"],
            result["search"]["default"]["mean_reward"],
            result["search"]["reward_improvement"],
            result["n_heldout_trajectories"],
            result["heldout_best"]["mean_reward"],
            result["heldout_default"]["mean_reward"],
            result["heldout_reward_improvement"],
        )
    )
    print(
        "%-8s %9s %9s %9s %9s %9s %9s"
        % (
            "harness", "full", "summed", "lost",
            "full tok", "sum tok", "reduction",
        )
    )
    for name, row in result["harnesses"].items():
        print(
            "%-8s %4d/%-4d %4d/%-4d %9d %9d %9d %8.1f%%"
            % (
                name, row["full_passed"], row["n_tasks"],
                row["sum_passed"], row["n_tasks"],
                row["tasks_lost"], row["full_tokens"],
                row["sum_tokens"], row["cost_reduction"] * 100,
            )
        )
    for name, row in result["harnesses"].items():
        print(
            "  %-8s summariser read %d tokens; %d tool results "
            "summarised, %d fell back uncompressed; mean "
            "episode recovery %.3f"
            % (
                name, row["summariser_tokens_read"],
                row["n_tool_results_summarised"],
                row["n_fell_back"], row["mean_episode_recovery"],
            )
        )
    if result["done"]:
        print(
            "DONE: MET — >= %.0f%% agent cost reduction with "
            "<= %d task lost per set, on BOTH harnesses, same "
            "checkpoint."
            % (
                DONE_MIN_REDUCTION * 100,
                DONE_MAX_LOST_PER_SET,
            )
        )
    else:
        print(
            "DONE: NOT met — at least one harness misses "
            ">= %.0f%% cost reduction or <= %d task lost. The "
            "table stands as measured; the grid, masking "
            "scheme, reward weights and threshold were not "
            "re-tuned."
            % (DONE_MIN_REDUCTION * 100, DONE_MAX_LOST_PER_SET)
        )
