"""P11 — From-scratch speculative decoding (device-agnostic).

"Lossless but Not Free" (arXiv:2607.17283): learn speculative
decoding by building it from scratch — CUDA/MPS/CPU in the
paper — with distribution-equivalence verified by a chi^2 test.
The plan's done condition: 1.6x wall-clock on at least one
config, a passing chi^2 test, and a written note on when it
does *not* pay.

SUBSTITUTIONS, stated plainly (this VM has no torch, no GPU,
no CUDA/MPS, no real LM, no model download; do NOT read this
module as a run on a real model or a real device):
- CPU-only, pure NumPy, deterministic toy distributions.
  The target/draft *distributions* are P10's (a seeded
  trigram target + arithmetic head, and a bigram draft, over
  the same 64-token vocab, trained on the same fixed synthetic
  text) — imported from ``specdec`` as distributions only.
  The speculative LOOP in this module is written FROM SCRATCH:
  it does not import ``generate_speculative``, ``verify_tree``,
  ``build_draft_tree``, ``margin_accept`` or either policy
  from ``specdec`` (pinned by test), and unlike P10 it uses
  NO margin rule at all.
- The algorithm is the standard LOSSLESS speculative sampling
  algorithm (Leviathan et al. / Chen et al. 2023), not P10's
  lossy margin rule. Per round the draft proposes K tokens
  sampled from its distribution q; the target verifies them
  in ONE batched forward; each draft token x is accepted with
  probability min(1, p(x)/q(x)) where p is the target
  distribution; the first rejection is repaired by resampling
  from the ADJUSTED distribution norm(max(0, p - q)); if all
  K are accepted a bonus token is sampled from p. The output
  token at every position is therefore distributed EXACTLY as
  the target's — that is the "lossless" part, and it is what
  the chi^2 test in Phase 2 checks. (P10's margin rule cannot
  make that claim: it accepts a token whenever p(x) is within
  a margin of the top-1 probability, which shifts the output
  distribution toward the draft's mistakes.)
- There is no batched GPU forward. The "one batched target
  forward per round" is the SAME simulated-cost model as P10
  (one fixed block of elementwise NumPy work per forward,
  independent of batch size, plus a tiny per-node op), re-
  implemented here from scratch in ``CostModel`` so its size
  can be varied per benchmark config. A second, RAW wall-clock
  column is also reported with the simulated cost switched
  off (work_rounds=0): there each node's distribution is a
  separate Python n-gram computation, i.e. verification is
  NOT batch-parallel at all — which is exactly the config
  where speculative decoding stops paying, and the numbers
  show it.
- No GSM8K / MATH / HumanEval and no laptop GPU: the
  benchmark is wall-clock tok/s at K=6 on this VM's CPU
  across configs that vary draft cost, tree width and the
  target-forward cost.

FIXED BEFORE MEASURING (do not tune to hit the target):
``K = 6``, ``CHI2_N = 9200``, ``CHI2_SEED = 20261127``,
``ALPHA = 0.05``, the P10 corpus seed / vocab / prompts,
``MAX_NEW = 24`` for the benchmark, and the seven benchmark
configs in ``BENCH_CONFIGS``. The eval is run once with
these and reported as measured — if no config reaches 1.6x,
the report says so and nothing is re-tuned.

Tree verification, and its honest limit (read this before
assuming width helps): a draft tree is built by sampling
``width`` children per node and verified in one batched
forward over all its nodes, but the walk follows ONE child
per position (the first sampled child) and applies the
lossless rejection trial to it; on rejection it resamples
from the adjusted distribution and stops, discarding the
subtree. That single-trial-per-position rule is what keeps
the output distribution exactly equal to the target's.
Trying a second sibling after a rejection would NOT be
lossless: with i.i.d. proposals the first-accepted token
would be distributed as min(p, q) / sum(min(p, q)), not p
— a second trial from the same q biases the output toward
tokens where the draft overestimates the target. So under
the lossless rule extra tree width buys no accepted length:
it only costs draft sampling plus a larger batched forward.
The benchmark measures that cost (configs E/F) instead of
assuming the tree helps.
"""

import math
import time

import numpy as np

# Distributions only from P10 — NOT its speculative loop.
# (generate_speculative / verify_tree / build_draft_tree /
# margin_accept / TreePolicy / FixedPolicy are NOT imported.)
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

# ---------------------------------------------------------------------------
# Fixed parameters (fixed before measuring; see module docstring).
# ---------------------------------------------------------------------------

K = 6
CHI2_N = 9200
CHI2_SEED = 20261127
ALPHA = 0.05
MAX_NEW = 24
N_PROMPTS = 40

# Simulated target-forward cost, same size as P10's default:
# 8 rounds of a 384x384 float32 elementwise reduction.
WORK_ROUNDS = 8
WORK_DIM = 384
DRAFT_WORK_DIM = 128

NOTE = (
    "from-scratch lossless speculative decoding: seeded trigram "
    "target + bigram draft over a 64-token vocab (P10's "
    "distributions only, loop written from scratch); no real "
    "model, no GPU/CUDA/MPS, no torch; chi^2 over 9200 tokens; "
    "wall-clock at K=6 under a simulated batched-forward cost "
    "model (P10's size) plus a raw-CPU column with the "
    "simulated cost off"
)


# ---------------------------------------------------------------------------
# chi^2 machinery (no scipy on this VM — implemented by hand
# from the regularised incomplete gamma function, stdlib only).
# ---------------------------------------------------------------------------

def _gammaincc(a, x):
    """Regularised upper incomplete gamma Q(a, x).

    Numerical Recipes series (x < a+1) / continued fraction
    (x >= a+1), to ~1e-14. Q(a, 0) = 1, Q(a, inf) = 0.
    """
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        # Series for P(a, x); return 1 - P.
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return min(1.0, max(0.0, 1.0 - p))
    # Continued fraction for Q(a, x).
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return min(1.0, max(0.0, q))


def chi2_sf(stat, df):
    """Survival function P(Chi^2_df > stat) = Q(df/2, stat/2)."""
    if df <= 0:
        raise ValueError("df must be positive")
    if stat <= 0.0:
        return 1.0
    return _gammaincc(0.5 * df, 0.5 * stat)


def chi2_homogeneity(counts_a, counts_b):
    """Chi^2 test of homogeneity for two count vectors.

    H0: both samples come from the same distribution over the
    V categories. Returns (statistic, df, p_value). Columns
    with zero total in both samples carry no information and
    are dropped (df = n_kept_columns - 1); cells with expected
    count 0 in one arm but not the other contribute via the
    usual (O - E)^2 / E term only when E > 0.
    """
    a = np.asarray(counts_a, dtype=np.float64)
    b = np.asarray(counts_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("count vectors must have the same length")
    keep = (a + b) > 0
    a, b = a[keep], b[keep]
    n_a, n_b = float(a.sum()), float(b.sum())
    n = n_a + n_b
    col = a + b
    exp_a = n_a * col / n
    exp_b = n_b * col / n
    stat = float(np.sum((a - exp_a) ** 2 / exp_a)
                 + np.sum((b - exp_b) ** 2 / exp_b))
    df = int(len(a) - 1)
    return stat, df, chi2_sf(stat, df)


# ---------------------------------------------------------------------------
# Simulated cost model (from scratch; same default size as P10).
# ---------------------------------------------------------------------------

class CostModel:
    """Accounts target/draft cost for one arm.

    ``target_batch`` charges ONE fixed block of elementwise
    work (``work_rounds`` x WORK_DIM x WORK_DIM) per call —
    the weight-load-dominated cost analogue, independent of
    batch size — plus a tiny per-node op, exactly P10's
    shape, then returns the real distributions. With
    ``work_rounds=0`` no simulated work is done at all: the
    wall-clock is then the RAW CPU cost of the Python n-gram
    computations. ``draft_work`` charges the simulated draft
    cost per draft step. ``forwards`` counts target forwards;
    ``work_checksum`` keeps the simulated work live so it
    cannot be optimised away.
    """

    def __init__(self, work_rounds=WORK_ROUNDS,
                 draft_cost_rounds=0, seed=1234):
        self.work_rounds = work_rounds
        self.draft_cost_rounds = draft_cost_rounds
        rng = np.random.default_rng(seed)
        self._work = rng.standard_normal(
            (WORK_DIM, WORK_DIM)).astype(np.float32)
        self._vec = rng.standard_normal(WORK_DIM).astype(np.float32)
        self._dwork = rng.standard_normal(
            (DRAFT_WORK_DIM, DRAFT_WORK_DIM)).astype(np.float32)
        self._dvec = rng.standard_normal(
            DRAFT_WORK_DIM).astype(np.float32)
        self.forwards = 0
        self.draft_steps = 0
        self.work_checksum = 0.0

    def _target_work(self, n_nodes):
        if self.work_rounds == 0:
            # Raw mode: no simulated work at all — the
            # wall-clock is the real Python cost of the
            # n-gram distribution computations.
            return
        acc = np.float32(0.0)
        for _ in range(self.work_rounds):
            acc += np.sum(self._work * self._vec)
        for _ in range(n_nodes):
            acc += np.sum(self._vec * self._work[0])
        self.work_checksum += float(acc)

    def target_batch(self, target, contexts):
        """One batched target forward over ``contexts``."""
        self._target_work(len(contexts))
        self.forwards += 1
        return [target.dist(c) for c in contexts]

    def target_single(self, target, context):
        """One target forward for a single context."""
        self._target_work(1)
        self.forwards += 1
        return target.dist(context)

    def draft_work(self):
        """Simulated draft cost for one draft step."""
        self.draft_steps += 1
        acc = np.float32(0.0)
        for _ in range(self.draft_cost_rounds):
            acc += np.sum(self._dwork * self._dvec)
        self.work_checksum += float(acc)


# ---------------------------------------------------------------------------
# Lossless speculative sampling (from scratch).
# ---------------------------------------------------------------------------

def adjusted_resample(p, q, rng):
    """Sample from norm(max(0, p - q)) — the repair after a
    rejection. Only tokens the target rates above the draft
    can come out (support check pinned by test)."""
    r = np.maximum(np.asarray(p, dtype=np.float64)
                   - np.asarray(q, dtype=np.float64), 0.0)
    total = float(r.sum())
    if total <= 0.0:
        # p == q (rejection then cannot happen); fall back to
        # the target argmax rather than crashing.
        return int(np.argmax(p))
    return int(rng.choice(len(r), p=r / total))


def rejection_sample(p, q, rng):
    """One lossless speculative trial.

    Propose x ~ q; accept with probability min(1, p(x)/q(x));
    on rejection return a sample from norm(max(0, p - q)).
    The returned token is distributed EXACTLY as p (the
    textbook proof: P(return x) = q(x)*min(1, p(x)/q(x))
    + P(reject)*max(0, p(x)-q(x))/sum(max(0, p-q)) = p(x)).
    Returns (token, accepted_bool).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    x = int(rng.choice(len(q), p=q))
    if float(q[x]) <= 0.0:
        return adjusted_resample(p, q, rng), False
    if float(rng.random()) < min(1.0, float(p[x]) / float(q[x])):
        return x, True
    return adjusted_resample(p, q, rng), False


def speculative_round(target, draft, context, rng, cost, k=K):
    """One lossless chain round, from scratch.

    Draft K tokens sequentially (each sampled from the draft
    distribution at the running context), then ONE batched
    target forward over the K+1 contexts (prefix + each draft
    prefix), then sequential rejection sampling over the
    precomputed target distributions. Returns
    (tokens_out, n_accepted_draft, n_rounds=1). ``tokens_out``
    has at least 1 token (the repair/bonus) and at most K+1.
    """
    ctx = list(context)
    draft_tokens = []
    qs = []
    for _ in range(k):
        q = draft.dist(ctx)
        cost.draft_work()
        x = int(rng.choice(V, p=q))
        draft_tokens.append(x)
        qs.append(q)
        ctx.append(x)
    contexts = [tuple(context) + tuple(draft_tokens[:i])
                for i in range(k + 1)]
    dists = cost.target_batch(target, contexts)
    out = []
    n_accepted = 0
    for i, x in enumerate(draft_tokens):
        # Trial THIS draft token (not a fresh proposal — the
        # draft token is the proposal; rejection_sample is the
        # same trial as a standalone helper, used by the tests).
        p, q = dists[i], qs[i]
        if float(q[x]) <= 0.0:
            out.append(adjusted_resample(p, q, rng))
            break
        if float(rng.random()) < min(1.0, float(p[x]) / float(q[x])):
            out.append(x)
            n_accepted += 1
        else:
            out.append(adjusted_resample(p, q, rng))
            break
    else:
        out.append(int(rng.choice(V, p=dists[k])))
    return out, n_accepted


def build_sampled_tree(draft, context, rng, cost, depth=K, width=2,
                       max_nodes=127):
    """Draft tree, sampled (not top-k) from scratch.

    Node = (parent_idx, token_or_None, context_tuple,
    q_of_token, draft_dist_at_node). Children are drawn
    WITHOUT replacement from the draft distribution at the
    parent, so siblings never duplicate; the walked child
    (first drawn) still has marginal q. Node 0 is the root.
    """
    q_root = draft.dist(list(context))
    nodes = [(-1, None, tuple(context), None, q_root)]
    frontier = [0]
    for _ in range(depth):
        nxt = []
        for idx in frontier:
            q = nodes[idx][4]
            cost.draft_work()
            w = min(width, V)
            for tok in rng.choice(V, size=w, replace=False, p=q):
                if len(nodes) >= max_nodes:
                    break
                tok = int(tok)
                nodes.append((idx, tok, nodes[idx][2] + (tok,),
                              float(q[tok]), draft.dist(
                                  list(nodes[idx][2]) + [tok])))
                nxt.append(len(nodes) - 1)
            if len(nodes) >= max_nodes:
                break
        frontier = nxt
        if not frontier:
            break
    return nodes


def verify_tree_lossless(nodes, target_dists, rng):
    """Verify a sampled draft tree under the lossless rule.

    Walk from the root following the FIRST sampled child at
    each node (see module docstring for why only one child
    per position may be tried). At each node apply one
    rejection trial with the node's draft distribution; on
    acceptance descend, on rejection resample from the
    adjusted distribution and stop. If the walk reaches a
    leaf, sample the bonus token from the leaf's target
    distribution. Returns (tokens_out, n_accepted_draft,
    n_examined).
    """
    children = {}
    for idx, (parent, _tok, _ctx, _qt, _q) in enumerate(nodes):
        if parent >= 0:
            children.setdefault(parent, []).append(idx)
    out = []
    n_accepted = 0
    n_examined = 0
    node = 0
    while True:
        kids = children.get(node, [])
        if not kids:
            out.append(int(rng.choice(V, p=target_dists[node])))
            break
        child = kids[0]
        tok = nodes[child][1]
        n_examined += 1
        p = target_dists[node]
        q = nodes[node][4]
        # One rejection trial on THIS child token (not a fresh
        # proposal from q — the child is the proposal).
        if float(q[tok]) <= 0.0:
            out.append(adjusted_resample(p, q, rng))
            break
        if float(rng.random()) < min(1.0, float(p[tok]) / float(q[tok])):
            out.append(tok)
            n_accepted += 1
            node = child
        else:
            out.append(adjusted_resample(p, q, rng))
            break
    return out, n_accepted, n_examined


def speculative_tree_round(target, draft, context, rng, cost,
                           depth=K, width=2, max_nodes=127):
    """One lossless tree round: sample the tree, verify it in
    ONE batched target forward over all nodes, walk the
    accepted path. Returns (tokens_out, n_accepted_draft)."""
    nodes = build_sampled_tree(draft, context, rng, cost,
                               depth=depth, width=width,
                               max_nodes=max_nodes)
    dists = cost.target_batch(target, [n[2] for n in nodes])
    out, n_acc, _ = verify_tree_lossless(nodes, dists, rng)
    return out, n_acc


def generate_lossless(target, draft, prompt, rng, cost, k=K,
                      max_new=MAX_NEW, tree=None):
    """Lossless speculative generation to ``max_new`` tokens.

    ``tree=None``: chain rounds. ``tree=(depth, width,
    max_nodes)``: tree rounds. Returns (tokens, n_rounds,
    n_accepted_draft, target_forwards_at_start_delta).
    """
    ctx = list(prompt)
    generated = []
    rounds = 0
    n_accepted = 0
    fwd0 = cost.forwards
    while len(generated) < max_new:
        if tree is None:
            out, acc = speculative_round(
                target, draft, ctx, rng, cost, k=k)
        else:
            depth, width, max_nodes = tree
            out, acc = speculative_tree_round(
                target, draft, ctx, rng, cost,
                depth=depth, width=width, max_nodes=max_nodes)
        n_accepted += acc
        take = out[:max_new - len(generated)]
        generated.extend(take)
        ctx.extend(take)
        rounds += 1
    return generated, rounds, n_accepted, cost.forwards - fwd0


def sample_target_only(target, prompt, rng, cost, max_new=MAX_NEW):
    """Ancestral (sampling, not greedy) target-only decoding:
    one target forward per token, token sampled from p."""
    ctx = list(prompt)
    out = []
    fwd0 = cost.forwards
    for _ in range(max_new):
        p = cost.target_single(target, ctx)
        tok = int(rng.choice(V, p=p))
        out.append(tok)
        ctx.append(tok)
    return out, cost.forwards - fwd0


# ---------------------------------------------------------------------------
# Phase 2 — chi^2 verification.
# ---------------------------------------------------------------------------

def _random_contexts(sentences, n, seed):
    """n contexts drawn independently: a random corpus
    sentence cut at a random position (context = tokens
    before the cut, at least 1 token)."""
    rng = np.random.default_rng(seed)
    contexts = []
    for _ in range(n):
        s = sentences[int(rng.integers(0, len(sentences)))]
        cut = int(rng.integers(1, len(s)))
        contexts.append([TOK[t] for t in s[:cut]])
    return contexts


def run_chi2_eval(n_tokens=CHI2_N, seed=CHI2_SEED, k=K):
    """Primary chi^2: n_tokens INDEPENDENT contexts per arm,
    one token per context.

    The two arms use INDEPENDENT context sets drawn from the
    same distribution (different seeds). That independence is
    what the homogeneity test assumes: the two count vectors
    are then independent multinomial mixtures over contexts.
    (Sharing one context set would make the test
    conservative — the target's conditional is often peaked,
    so the two arms' counts would track each other and the
    statistic would run small. A target-vs-target control
    with the same design lands in the nominal range, which
    is what validates the setup; see the module tests.)
    Each speculative token here is the FIRST token of a fresh
    K=6 round from its context, so under the lossless theorem
    it is distributed exactly as the target's token from the
    same context. Cost model off (work_rounds=0):
    distribution equivalence is cost-independent, and the
    simulated work would only slow the 9,200 rounds down.
    Categories with expected count < 5 under H0 are pooled
    into one "other" bucket (Cochran's rule) and the degrees
    of freedom count the pooled categories.
    Returns the result dict (counts, statistic, df, p,
    pass/fail at ALPHA, min expected count, pooled count).
    """
    sentences = build_corpus()
    target, draft = TargetModel(sentences), DraftModel(sentences)
    contexts_t = _random_contexts(sentences, n_tokens, seed)
    contexts_s = _random_contexts(sentences, n_tokens, seed + 7)
    cost = CostModel(work_rounds=0)
    rng_t = np.random.default_rng(seed + 1)
    rng_s = np.random.default_rng(seed + 2)
    counts_target = np.zeros(V, dtype=np.int64)
    counts_spec = np.zeros(V, dtype=np.int64)
    for ctx in contexts_t:
        p = target.dist(ctx)
        counts_target[int(rng_t.choice(V, p=p))] += 1
    for ctx in contexts_s:
        out, _ = speculative_round(target, draft, ctx, rng_s, cost, k=k)
        counts_spec[out[0]] += 1
    # Pool rare categories (expected < 5 under H0) into one
    # "other" bucket before testing.
    n_a = float(counts_target.sum())
    n_b = float(counts_spec.sum())
    col = counts_target + counts_spec
    exp = np.where(col > 0, (n_a * col / (n_a + n_b)), 0.0)
    keep = (col == 0) | (exp >= 5.0)
    n_pooled = int(np.sum(~keep & (col > 0)))
    if n_pooled:
        pooled_a = np.append(counts_target[keep],
                             counts_target[~keep].sum())
        pooled_b = np.append(counts_spec[keep],
                             counts_spec[~keep].sum())
    else:
        pooled_a, pooled_b = counts_target, counts_spec
    stat, df, p_value = chi2_homogeneity(pooled_a, pooled_b)
    exp_kept = exp[keep & (col > 0)]
    exp_min = float(np.min(exp_kept)) if len(exp_kept) else 0.0
    return {
        "n_tokens_per_arm": n_tokens,
        "k": k,
        "seed": seed,
        "alpha": ALPHA,
        "counts_target": counts_target,
        "counts_spec": counts_spec,
        "pooled_categories": n_pooled,
        "statistic": stat,
        "df": df,
        "p_value": p_value,
        "pass": bool(p_value >= ALPHA),
        "min_expected_count": exp_min,
    }


def run_longgen_comparison(n_per_prompt=230, seed=CHI2_SEED, k=K):
    """Secondary, descriptive only: long generations (40
    prompts x n_per_prompt tokens per arm), unigram counts
    and total-variation distance. NO pass/fail: tokens
    inside a generation are autocorrelated, so a chi^2 on
    them would not have its nominal null distribution. The
    statistic is still reported for transparency, labelled
    as not a test."""
    sentences = build_corpus()
    target, draft = TargetModel(sentences), DraftModel(sentences)
    cost = CostModel(work_rounds=0)
    counts_target = np.zeros(V, dtype=np.int64)
    counts_spec = np.zeros(V, dtype=np.int64)
    for prompt in PROMPT_IDS:
        rng_t = np.random.default_rng(seed + 11)
        rng_s = np.random.default_rng(seed + 12)
        out_t, _ = sample_target_only(target, prompt, rng_t, cost,
                                      max_new=n_per_prompt)
        out_s, _, _, _ = generate_lossless(
            target, draft, prompt, rng_s, cost, k=k,
            max_new=n_per_prompt)
        for t in out_t:
            counts_target[t] += 1
        for t in out_s:
            counts_spec[t] += 1
    stat, df, p_value = chi2_homogeneity(counts_target, counts_spec)
    tv = 0.5 * float(np.sum(np.abs(
        counts_target / counts_target.sum()
        - counts_spec / counts_spec.sum())))
    return {
        "n_tokens_per_arm": int(counts_target.sum()),
        "counts_target": counts_target,
        "counts_spec": counts_spec,
        "chi2_statistic_descriptive_only": stat,
        "chi2_df": df,
        "chi2_p_descriptive_only": p_value,
        "total_variation_distance": tv,
    }


# ---------------------------------------------------------------------------
# Phase 3 — wall-clock benchmark at K=6.
# ---------------------------------------------------------------------------

# Configs fixed before measuring. work_rounds is the simulated
# target-forward size (8 = P10's size); draft_cost_rounds is
# simulated draft work per draft step (0 = P10's lookup draft);
# tree=None is the chain, otherwise (depth, width, max_nodes).
BENCH_CONFIGS = [
    {"name": "A target x1, cheap draft, chain",
     "work_rounds": 8, "draft_cost_rounds": 0, "tree": None},
    {"name": "B target x4, cheap draft, chain",
     "work_rounds": 32, "draft_cost_rounds": 0, "tree": None},
    {"name": "C target x0.25, cheap draft, chain",
     "work_rounds": 2, "draft_cost_rounds": 0, "tree": None},
    {"name": "D target x1, expensive draft, chain",
     "work_rounds": 8, "draft_cost_rounds": 6, "tree": None},
    {"name": "E target x1, cheap draft, tree w=2",
     "work_rounds": 8, "draft_cost_rounds": 0,
     "tree": (6, 2, 127)},
    {"name": "F target x4, cheap draft, tree w=2",
     "work_rounds": 32, "draft_cost_rounds": 0,
     "tree": (6, 2, 127)},
    {"name": "G RAW cpu, no simulated cost, chain",
     "work_rounds": 0, "draft_cost_rounds": 0, "tree": None},
]


def _run_bench_arm(target, draft, prompts, arm, cfg, seed):
    """One benchmark arm run: returns tok/s, tok/fwd, tokens,
    target forwards, wall seconds. Deterministic columns
    (tokens, forwards) are identical every repeat; only
    wall-clock varies."""
    cost = CostModel(work_rounds=cfg["work_rounds"],
                     draft_cost_rounds=cfg["draft_cost_rounds"])
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    tokens = 0
    forwards = 0
    rounds = 0
    accepted = 0
    for prompt in prompts:
        if arm == "target-only":
            _, fwd = sample_target_only(target, prompt, rng, cost,
                                        max_new=MAX_NEW)
            tokens += MAX_NEW
            forwards += fwd
        else:
            _, r, acc, fwd = generate_lossless(
                target, draft, prompt, rng, cost, k=K,
                max_new=MAX_NEW, tree=cfg["tree"])
            tokens += MAX_NEW
            forwards += fwd
            rounds += r
            accepted += acc
    wall = time.perf_counter() - t0
    return {
        "tokens": tokens,
        "target_forwards": forwards,
        "rounds": rounds,
        "accepted_draft": accepted,
        "tokens_per_target_forward": tokens / max(1, forwards),
        "tokens_per_sec": tokens / max(wall, 1e-9),
        "wall_sec": wall,
    }


def run_benchmark(repeats=3, seed=CHI2_SEED):
    """Phase 3: wall-clock at K=6 for every BENCH_CONFIG.

    Each config runs target-only sampling and lossless
    speculative generation over the same 40 prompts x 24
    tokens, ``repeats`` times; tok/s is the mean over
    repeats, speedup is spec tok/s / target-only tok/s
    WITHIN the config (same simulated cost on both sides).
    """
    sentences = build_corpus()
    target, draft = TargetModel(sentences), DraftModel(sentences)
    results = []
    for cfg in BENCH_CONFIGS:
        base_runs = [_run_bench_arm(target, draft, PROMPT_IDS,
                                    "target-only", cfg, seed + 100 + i)
                     for i in range(repeats)]
        spec_runs = [_run_bench_arm(target, draft, PROMPT_IDS,
                                    "spec", cfg, seed + 200 + i)
                     for i in range(repeats)]
        base = dict(base_runs[0])
        base["tokens_per_sec"] = float(np.mean(
            [r["tokens_per_sec"] for r in base_runs]))
        base["wall_sec"] = float(np.mean(
            [r["wall_sec"] for r in base_runs]))
        spec = dict(spec_runs[0])
        spec["tokens_per_sec"] = float(np.mean(
            [r["tokens_per_sec"] for r in spec_runs]))
        spec["wall_sec"] = float(np.mean(
            [r["wall_sec"] for r in spec_runs]))
        spec["acceptance_rate"] = (
            spec["accepted_draft"]
            / max(1, spec["accepted_draft"] + spec["rounds"]))
        results.append({
            "config": cfg["name"],
            "work_rounds": cfg["work_rounds"],
            "draft_cost_rounds": cfg["draft_cost_rounds"],
            "tree": cfg["tree"],
            "target_only": base,
            "spec": spec,
            "speedup": spec["tokens_per_sec"]
            / max(base["tokens_per_sec"], 1e-9),
        })
    return results


def run_scratch_eval(repeats=3):
    """Full P11 eval: chi^2 (Phase 2) + long-gen descriptive
    comparison + wall-clock benchmark at K=6 (Phase 3)."""
    chi2 = run_chi2_eval()
    longgen = run_longgen_comparison()
    bench = run_benchmark(repeats=repeats)
    best = max(bench, key=lambda r: r["speedup"])
    done = bool(chi2["pass"] and best["speedup"] >= 1.6)
    return {
        "note": NOTE,
        "k": K,
        "alpha": ALPHA,
        "chi2": chi2,
        "longgen": longgen,
        "benchmark": bench,
        "best_config": best["config"],
        "best_speedup": best["speedup"],
        "done_chi2_pass_and_1_6x": done,
    }


def print_scratch_report(result):
    print("P11 — From-scratch speculative decoding (device-agnostic)")
    print("NOTE: %s" % result["note"])
    chi2 = result["chi2"]
    print("\nPhase 2 — chi^2 distribution equivalence "
          "(%d independent contexts, 1 token each per arm, K=%d)"
          % (chi2["n_tokens_per_arm"], chi2["k"]))
    print("  statistic %.2f, df %d, p-value %.4f, alpha %.2f, "
          "min expected count %.1f, %d rare categor%s pooled "
          "-> %s"
          % (chi2["statistic"], chi2["df"], chi2["p_value"],
             chi2["alpha"], chi2["min_expected_count"],
             chi2["pooled_categories"],
             "y" if chi2["pooled_categories"] == 1 else "ies",
             "PASS" if chi2["pass"] else "FAIL"))
    lg = result["longgen"]
    print("  long-gen descriptive (%d tokens/arm): TV distance "
          "%.4f; chi^2 %.2f (df %d, p %.4f) reported only — "
          "tokens in one generation are autocorrelated, so "
          "this is not a test"
          % (lg["n_tokens_per_arm"],
             lg["total_variation_distance"],
             lg["chi2_statistic_descriptive_only"], lg["chi2_df"],
             lg["chi2_p_descriptive_only"]))
    print("\nPhase 3 — wall-clock at K=%d (40 prompts x %d tokens, "
          "mean over repeats)" % (result["k"], MAX_NEW))
    print("%-38s %10s %10s %10s %10s %8s" % (
        "config", "spec t/s", "base t/s", "tok/fwd",
        "accept", "speedup"))
    for row in result["benchmark"]:
        print("%-38s %10.1f %10.1f %10.2f %10.3f %8.2fx" % (
            row["config"], row["spec"]["tokens_per_sec"],
            row["target_only"]["tokens_per_sec"],
            row["spec"]["tokens_per_target_forward"],
            row["spec"]["acceptance_rate"], row["speedup"]))
    print("\nBest config: %s at %.2fx"
          % (result["best_config"], result["best_speedup"]))
    print("Done condition (chi^2 pass AND >= 1.6x on at least "
          "one config): %s" % result["done_chi2_pass_and_1_6x"])
