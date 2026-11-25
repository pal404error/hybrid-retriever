"""P12 — Retrieval-based drafter for small local models.

"An Empirical Study of Speculative Decoding for Small Language
Models" (EACL 2026): for 1-2B targets, drafting OVERHEAD — not
draft quality — is the bottleneck; retrieval-based drafting
(near-zero overhead) wins consistently. The plan's done
condition: retrieval drafting beats model-based drafting on
wall-clock for a 1-2B target. Independent drafting fails
outright for small targets — it is not built here; the
retrieval drafter is the whole project.

SUBSTITUTIONS, stated plainly (this VM has no llama.cpp, no
MLX, no torch, no GPU, no real 1-2B model and no model
download; do NOT read this module as a run on a real model):
- No llama.cpp / MLX. Everything is a deterministic toy
  decoder in pure NumPy, reusing P10's toy target distribution
  (a seeded trigram target + arithmetic head over the same
  64-token vocab, trained on the same fixed synthetic text)
  as the "small target", and P11's from-scratch LOSSLESS
  verification loop (``speculative_round``) unchanged — both
  drafters feed the SAME loop, so any wall-clock difference
  is the drafter, not the verifier. The model-based drafter
  is P10's bigram draft model, reused as-is.
- The retrieval drafter is a prompt-lookup / n-gram drafter
  (a la Prompt Lookup Decoding): given the current context,
  find the LONGEST matching suffix in the prompt + generated
  history and in a fixed lookup corpus built from the same
  synthetic text the toy LM was trained on, and propose its
  most frequent continuation token. Near-zero overhead: a
  dict lookup, no model forward. Its proposal is a
  DETERMINISTIC token, so the draft distribution it hands to
  the lossless trial is one-hot on that token — the standard
  treatment of a deterministic proposal (accept with
  probability p(token); on rejection repair from
  norm(max(0, p - q))). If no suffix matches at any length,
  it falls back to the uniform distribution.
- The "small target" regime is modelled by the COST MODEL,
  not by a real small model: the target forward is given a
  SMALL simulated cost (``SMALL_TARGET_WORK_ROUNDS = 1``,
  one 384x384 elementwise round per batched forward — 1/8 of
  P10/P11's default size), cheap enough that drafting
  overhead dominates, which is the paper's whole point. The
  model-based drafter pays a simulated draft-model forward
  per draft token (``DRAFT_COST_ROUNDS = 2`` rounds of
  128x128 work — a small draft model is cheaper per forward
  than the target, 128^2/384^2 = 1/9 of the work per round,
  but K=6 draft steps are charged per verification round, so
  drafting costs roughly one extra small-target forward per
  round). The retrieval drafter pays ZERO simulated draft
  work (``draft_cost_rounds=0``): a dict lookup has no model
  forward to simulate. Its real lookup cost still shows up
  in wall-clock, and is measured separately by the
  microbenchmark below.
- The same comparison is also run at a LARGE-target cost
  setting (``LARGE_TARGET_WORK_ROUNDS = 32``, P11's config
  B), where the target forward dominates and the drafter
  with the higher acceptance rate should win or tie — the
  paper's claim is specific to small targets, so the
  crossover (or its absence) is reported honestly.

FIXED BEFORE MEASURING (do not tune to hit the target):
``SMALL_TARGET_WORK_ROUNDS = 1``,
``LARGE_TARGET_WORK_ROUNDS = 32``, ``DRAFT_COST_ROUNDS = 2``,
``DRAFT_K = 6``, ``MAX_NGRAM = 5``, ``MAX_NEW = 24``, the
40-prompt set (P10's ``PROMPTS``), ``REPEATS = 3``. The eval
is run once with these and reported as measured — if the
retrieval drafter does not win on wall-clock at the
small-target setting, the report says so and nothing is
re-tuned: not the toy LM, not the lookup corpus, not the
cost model.

Overhead and acceptance are DIFFERENT axes and are reported
SEPARATELY: acceptance rate (accepted / examined draft
tokens), tokens per target forward, draft steps + simulated
draft work (the overhead account), draft microseconds per
token (the microbenchmark), and wall-clock tokens/sec (the
outcome). A drafter can have lower acceptance and still win
on wall-clock — that is the paper's point, and the table
shows both columns so the reader can see it.
"""

import time

import numpy as np

# Distributions + the lossless loop only, from P10/P11.
# (generate_speculative / verify_tree / the margin rule /
# TreePolicy are NOT imported — verification here is P11's
# lossless loop, reused unchanged.)
from .specdec import (
    PROMPT_IDS,
    PROMPTS,
    TOK,
    V,
    VOCAB,
    DraftModel,
    TargetModel,
    build_corpus,
)
from .specdec_scratch import (
    MAX_NEW,
    CostModel,
    sample_target_only,
    speculative_round,
)

# ---------------------------------------------------------------------------
# Fixed parameters (fixed before measuring; see module docstring).
# ---------------------------------------------------------------------------

SMALL_TARGET_WORK_ROUNDS = 1
LARGE_TARGET_WORK_ROUNDS = 32
DRAFT_COST_ROUNDS = 2
DRAFT_K = 6
MAX_NGRAM = 5
N_PROMPTS = 40
REPEATS = 3
EVAL_SEED = 20261128
MICROBENCH_CALLS = 2000

NOTE = (
    "retrieval-based drafting for a small target: prompt-lookup "
    "/ n-gram drafter vs the P10 bigram draft model, both feeding "
    "P11's lossless verification loop on the seeded trigram "
    "target (64-token vocab, toy decoder, NumPy, no real model, "
    "no llama.cpp/MLX); small-target cost = 1 simulated work "
    "round per target forward, large-target cost = 32"
)

COST_SETTINGS = [
    {"name": "small-target (1-2B analogue)",
     "work_rounds": SMALL_TARGET_WORK_ROUNDS},
    {"name": "large-target analogue",
     "work_rounds": LARGE_TARGET_WORK_ROUNDS},
]


# ---------------------------------------------------------------------------
# Prompt-lookup / n-gram drafter
# ---------------------------------------------------------------------------

def build_lookup_index(sentences, max_ngram=MAX_NGRAM):
    """N-gram lookup index over the corpus sentences.

    Maps every context tuple of length 1..max_ngram that
    occurs in a sentence to its MOST FREQUENT continuation
    token (ties broken by lowest token id, so the build is
    deterministic). Only WITHIN-sentence continuations are
    indexed: the sentences are independent synthetic texts,
    so a sentence-final token does not "continue" into the
    next sentence. Returns {context_tuple: token_id}.
    """
    counts = {}
    for s in sentences:
        ids = [TOK[t] for t in s]
        for i in range(len(ids)):
            for n in range(1, max_ngram + 1):
                if i - n < 0:
                    break
                ctx = tuple(ids[i - n:i])
                nxt = ids[i]
                counts.setdefault(ctx, {})
                counts[ctx][nxt] = counts[ctx].get(nxt, 0) + 1
    index = {}
    for ctx, nxt_counts in counts.items():
        # Most frequent; lowest token id wins ties.
        best = min(nxt_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        index[ctx] = best[0]
    return index


class PromptLookupDrafter:
    """Retrieval drafter: longest-suffix prompt lookup.

    ``lookup(context)`` tries suffix lengths from
    ``max_ngram`` down to 1 and, at each length, checks the
    running context ITSELF first (the prompt + generated
    history so far, a la Prompt Lookup Decoding: the most
    recent earlier occurrence of the suffix, and the token
    that followed it there), then the fixed corpus index.
    The longest match wins regardless of source. Returns
    ``(token_id, n_matched, source)`` with source in
    ``{"history", "corpus"}``, or ``(None, 0, None)`` when
    nothing matches at any length (only possible for a
    near-empty context: every single vocab token occurs in
    the corpus, so a length-1 suffix almost always hits).

    ``dist(context)`` is the draft distribution the lossless
    trial consumes: one-hot on the looked-up token (the
    proposal is deterministic), uniform when there is no
    match. Drafting cost is a dict lookup — no model forward
    — so this class charges NO simulated work; the caller
    accounts for that by running it under a CostModel with
    ``draft_cost_rounds=0``.
    """

    def __init__(self, sentences, max_ngram=MAX_NGRAM):
        self.max_ngram = max_ngram
        self.index = build_lookup_index(sentences, max_ngram)
        self.lookups = 0
        self.history_hits = 0
        self.corpus_hits = 0
        self.misses = 0

    def reset_stats(self):
        self.lookups = 0
        self.history_hits = 0
        self.corpus_hits = 0
        self.misses = 0

    def _history_lookup(self, context, n):
        """Most recent earlier occurrence of the length-``n``
        suffix of ``context`` and the token after it."""
        suffix = tuple(context[-n:])
        # Occurrences must end strictly before the suffix
        # starts, and must have a following token.
        for j in range(len(context) - n - 1, -1, -1):
            if tuple(context[j:j + n]) == suffix:
                return context[j + n]
        return None

    def lookup(self, context):
        """(token_id, n_matched, source) or (None, 0, None)."""
        context = tuple(int(t) for t in context)
        self.lookups += 1
        if not context:
            self.misses += 1
            return None, 0, None
        top = min(self.max_ngram, len(context))
        for n in range(top, 0, -1):
            tok = self._history_lookup(context, n)
            if tok is not None:
                self.history_hits += 1
                return tok, n, "history"
            tok = self.index.get(tuple(context[-n:]))
            if tok is not None:
                self.corpus_hits += 1
                return tok, n, "corpus"
        self.misses += 1
        return None, 0, None

    def dist(self, context):
        """One-hot on the looked-up token; uniform on no match."""
        tok, _, _ = self.lookup(context)
        p = np.zeros(V, dtype=np.float64)
        if tok is None:
            p[:] = 1.0 / V
        else:
            p[tok] = 1.0
        return p


# ---------------------------------------------------------------------------
# Generation: thin wrapper over P11's lossless round
# ---------------------------------------------------------------------------

def generate_with_drafter(target, draft, prompt, rng, cost,
                          k=DRAFT_K, max_new=MAX_NEW):
    """Lossless speculative generation with an arbitrary draft
    object exposing ``dist(context)``.

    This is a THIN WRAPPER over ``specdec_scratch.
    speculative_round`` — the verification (rejection trial +
    adjusted resample + bonus) is P11's, reused unchanged;
    the wrapper only loops rounds to ``max_new`` tokens and
    counts examined draft tokens exactly, which
    ``generate_lossless`` does not return. Per round the
    examined count is ``len(out)`` when a rejection cut the
    round short (accepted + repair) and ``k`` when all ``k``
    draft tokens were accepted (``len(out) == k + 1`` with
    the bonus). Returns (generated, n_rounds, n_accepted,
    n_examined, target_forwards_delta).
    """
    ctx = list(prompt)
    generated = []
    rounds = 0
    n_accepted = 0
    n_examined = 0
    fwd0 = cost.forwards
    while len(generated) < max_new:
        out, acc = speculative_round(target, draft, ctx, rng, cost, k=k)
        n_accepted += acc
        n_examined += len(out) if len(out) <= k else k
        take = out[:max_new - len(generated)]
        generated.extend(take)
        ctx.extend(take)
        rounds += 1
    return generated, rounds, n_accepted, n_examined, \
        cost.forwards - fwd0


def _run_spec_arm(target, draft, prompts, cost, seed,
                  max_new=MAX_NEW):
    """One speculative arm run: returns tokens, rounds,
    accepted/examined, acceptance rate, target forwards,
    draft steps, tokens per target forward, wall seconds,
    tokens/sec, generations."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    gens = []
    rounds = 0
    accepted = 0
    examined = 0
    forwards = 0
    draft_steps0 = cost.draft_steps
    for prompt in prompts:
        gen, r, acc, ex, fwd = generate_with_drafter(
            target, draft, prompt, rng, cost, max_new=max_new)
        gens.append(gen)
        rounds += r
        accepted += acc
        examined += ex
        forwards += fwd
    wall = time.perf_counter() - t0
    n_tokens = sum(len(g) for g in gens)
    return {
        "generations": gens,
        "tokens": n_tokens,
        "rounds": rounds,
        "accepted_draft": accepted,
        "examined_draft": examined,
        "acceptance_rate": accepted / max(1, examined),
        "target_forwards": forwards,
        "draft_steps": cost.draft_steps - draft_steps0,
        "tokens_per_target_forward": n_tokens / max(1, forwards),
        "tokens_per_sec": n_tokens / max(wall, 1e-9),
        "wall_sec": wall,
    }


def _run_target_only_arm(target, prompts, cost, seed,
                         max_new=MAX_NEW):
    """Target-only sampling baseline (one forward per token)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    tokens = 0
    forwards = 0
    for prompt in prompts:
        _, fwd = sample_target_only(target, prompt, rng, cost,
                                    max_new=max_new)
        tokens += max_new
        forwards += fwd
    wall = time.perf_counter() - t0
    return {
        "tokens": tokens,
        "target_forwards": forwards,
        "tokens_per_target_forward": tokens / max(1, forwards),
        "tokens_per_sec": tokens / max(wall, 1e-9),
        "wall_sec": wall,
    }


def _draft_overhead_us(draft, contexts, n_calls=MICROBENCH_CALLS):
    """Microbenchmark: mean microseconds per ``dist`` call
    (one draft step) over fixed contexts. This is the
    OVERHEAD axis measured on its own: the retrieval
    drafter's dict lookup vs the bigram draft model's
    distribution computation, before any target forward is
    charged."""
    t0 = time.perf_counter()
    for i in range(n_calls):
        draft.dist(contexts[i % len(contexts)])
    wall = time.perf_counter() - t0
    return wall / n_calls * 1e6


def _microbench_contexts(sentences, n=40, seed=EVAL_SEED):
    """Fixed contexts for the overhead microbenchmark: the 40
    prompts plus 40 corpus sentences cut at random positions."""
    rng = np.random.default_rng(seed)
    contexts = [list(p) for p in PROMPT_IDS]
    for _ in range(n):
        s = sentences[int(rng.integers(0, len(sentences)))]
        cut = int(rng.integers(1, len(s)))
        contexts.append([TOK[t] for t in s[:cut]])
    return contexts


# ---------------------------------------------------------------------------
# Full eval
# ---------------------------------------------------------------------------

def run_retrieval_eval(repeats=REPEATS, n_prompts=None,
                       max_new=MAX_NEW):
    """Full P12 eval: at each cost setting (small-target and
    large-target), target-only vs model-based drafting vs
    retrieval drafting over the same prompts, same K, same
    lossless loop. Wall-clock is the mean over ``repeats``
    (different fixed seeds per repeat); acceptance rate is
    the mean over repeats; deterministic columns (tokens,
    forwards) come from repeat 0. ``done`` is the plan's
    done condition: retrieval beats model-based drafting on
    wall-clock at the SMALL-target setting.
    """
    prompts = PROMPT_IDS[:n_prompts] if n_prompts else PROMPT_IDS
    sentences = build_corpus()
    target = TargetModel(sentences)
    draft_model = DraftModel(sentences)

    # Overhead measured separately from acceptance/wall-clock.
    contexts = _microbench_contexts(sentences)
    retrieval_probe = PromptLookupDrafter(sentences)
    overhead = {
        "retrieval_us_per_draft_token": _draft_overhead_us(
            retrieval_probe, contexts),
        "model_us_per_draft_token": _draft_overhead_us(
            draft_model, contexts),
        "n_calls": MICROBENCH_CALLS,
    }

    settings = []
    for setting in COST_SETTINGS:
        arms = {}
        # Target-only baseline: same simulated target cost,
        # no draft at all.
        base_runs = []
        for i in range(repeats):
            cost = CostModel(work_rounds=setting["work_rounds"],
                             draft_cost_rounds=0)
            base_runs.append(_run_target_only_arm(
                target, prompts, cost, EVAL_SEED + 100 + i,
                max_new=max_new))
        base = dict(base_runs[0])
        base["tokens_per_sec"] = float(np.mean(
            [r["tokens_per_sec"] for r in base_runs]))
        base["wall_sec"] = float(np.mean(
            [r["wall_sec"] for r in base_runs]))
        arms["target-only"] = base
        # The two drafters. The ONLY difference between them
        # is the drafter and its simulated draft cost: the
        # model drafter pays DRAFT_COST_ROUNDS of simulated
        # work per draft step; the retrieval drafter pays
        # zero (dict lookup, no model forward).
        for arm, draft, draft_cost in (
                ("model-based", draft_model, DRAFT_COST_ROUNDS),
                ("retrieval", PromptLookupDrafter(sentences), 0)):
            runs = []
            for i in range(repeats):
                cost = CostModel(
                    work_rounds=setting["work_rounds"],
                    draft_cost_rounds=draft_cost)
                runs.append(_run_spec_arm(
                    target, draft, prompts, cost,
                    EVAL_SEED + 200 + i, max_new=max_new))
            agg = dict(runs[0])
            agg["tokens_per_sec"] = float(np.mean(
                [r["tokens_per_sec"] for r in runs]))
            agg["wall_sec"] = float(np.mean(
                [r["wall_sec"] for r in runs]))
            agg["acceptance_rate"] = float(np.mean(
                [r["acceptance_rate"] for r in runs]))
            agg["draft_cost_rounds"] = draft_cost
            arms[arm] = agg
        small = setting["work_rounds"] == SMALL_TARGET_WORK_ROUNDS
        settings.append({
            "name": setting["name"],
            "work_rounds": setting["work_rounds"],
            "arms": arms,
            "retrieval_speedup_vs_model": (
                arms["retrieval"]["tokens_per_sec"]
                / max(arms["model-based"]["tokens_per_sec"], 1e-9)),
            "retrieval_speedup_vs_target_only": (
                arms["retrieval"]["tokens_per_sec"]
                / max(arms["target-only"]["tokens_per_sec"], 1e-9)),
            "model_speedup_vs_target_only": (
                arms["model-based"]["tokens_per_sec"]
                / max(arms["target-only"]["tokens_per_sec"], 1e-9)),
        })
    small_setting = settings[0]
    done = bool(
        small_setting["retrieval_speedup_vs_model"] > 1.0)
    return {
        "note": NOTE,
        "k": DRAFT_K,
        "max_new": max_new,
        "n_prompts": len(prompts),
        "repeats": repeats,
        "small_target_work_rounds": SMALL_TARGET_WORK_ROUNDS,
        "large_target_work_rounds": LARGE_TARGET_WORK_ROUNDS,
        "draft_cost_rounds": DRAFT_COST_ROUNDS,
        "max_ngram": MAX_NGRAM,
        "overhead": overhead,
        "settings": settings,
        "done_retrieval_beats_model_small_target": done,
    }


def print_retrieval_report(result):
    print("P12 — Retrieval-based drafter for small local models "
          "(toy decoder, NumPy)")
    print("NOTE: %s" % result["note"])
    print("K=%d, max_new=%d, prompts=%d, repeats=%d, "
          "max_ngram=%d, draft cost=%d simulated rounds/step"
          % (result["k"], result["max_new"],
             result["n_prompts"], result["repeats"],
             result["max_ngram"], result["draft_cost_rounds"]))
    ov = result["overhead"]
    print("\nOverhead, measured separately (mean over %d "
          "dist calls): retrieval drafter %.2f us/draft-token, "
          "model drafter %.2f us/draft-token (%.1fx)"
          % (ov["n_calls"], ov["retrieval_us_per_draft_token"],
             ov["model_us_per_draft_token"],
             ov["model_us_per_draft_token"]
             / max(ov["retrieval_us_per_draft_token"], 1e-9)))
    print("Simulated draft work per step: retrieval 0 rounds "
          "(dict lookup, no model forward), model-based %d "
          "rounds" % result["draft_cost_rounds"])
    for setting in result["settings"]:
        print("\n%s — target forward cost %d simulated work "
              "round(s)" % (setting["name"],
                            setting["work_rounds"]))
        print("%-12s %10s %10s %10s %10s %10s %10s" % (
            "arm", "tok/sec", "tok/fwd", "accept", "tgt fwd",
            "draft steps", "wall s"))
        for label in ("target-only", "model-based", "retrieval"):
            a = setting["arms"][label]
            accept = ("%.3f" % a["acceptance_rate"]) \
                if "acceptance_rate" in a else "-"
            steps = str(a["draft_steps"]) \
                if "draft_steps" in a else "-"
            print("%-12s %10.1f %10.2f %10s %10d %10s %10.2f" % (
                label, a["tokens_per_sec"],
                a["tokens_per_target_forward"], accept,
                a["target_forwards"], steps, a["wall_sec"]))
        print("  retrieval vs model-based wall-clock: %.2fx; "
              "retrieval vs target-only: %.2fx; model-based vs "
              "target-only: %.2fx"
              % (setting["retrieval_speedup_vs_model"],
                 setting["retrieval_speedup_vs_target_only"],
                 setting["model_speedup_vs_target_only"]))
    print("\nDone condition (retrieval beats model-based "
          "drafting on wall-clock at the SMALL-target cost "
          "setting): %s"
          % result["done_retrieval_beats_model_small_target"])
