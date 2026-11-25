"""P10 — AdaptiveSpec: adaptive lossy speculative decoding.

AdaptiveSpec (arXiv:2609.02897): training-free speculative
decoding that adapts per step — a margin rule promotes
mismatched draft tokens, and a tree policy reshapes draft
depth/width from draft confidence + rolling acceptance.
The plan's done condition is a throughput gain over EAGLE-3
with 94-96% downstream accuracy retained.

SUBSTITUTIONS, stated plainly (this VM has no torch, no GPU,
no vLLM/SGLang, no EAGLE-3, no real LM, no GSM8K/MATH-500/
HumanEval; do NOT read this module as a run on a real model):
- The "target" and "draft" are deterministic SEEDED N-GRAM
  LMs over a 64-token vocab, trained on fixed synthetic text
  (see ``build_corpus``). Target = trigram + a deterministic
  arithmetic head (see ``TargetModel``); draft = bigram with
  no arithmetic head. No neural network anywhere.
- There is no EAGLE-3. The EAGLE-like baseline is FIXED-TREE
  speculative decoding (depth 3, width 2) on the SAME
  draft/target and the SAME margin rule, so the adaptive
  policy's effect is measured in isolation — a fixed tree
  is just EAGLE, per the plan's key decision.
- No GSM8K / MATH-500 / HumanEval. Accuracy is AGREEMENT
  with target-only greedy decoding on a fixed prompt set:
  token-level agreement + sequence exact-match rate, plus a
  checkable arithmetic task (the decoder must emit the
  answer to ``A op B`` prompts; the evaluated answer must
  match the target-only answer) as the GSM8K-style analogue
  with verifiable answers.
- Throughput is measured two ways: (1) TOKENS PER TARGET
  FORWARD — the deterministic cost model: one batched target
  pass per verification round, however many tree nodes it
  covers; target-only greedy scores exactly 1.0. (2)
  WALL-CLOCK tokens/sec on CPU. The target forward's cost
  is SIMULATED, not a real model forward: each target
  forward does one fixed block of elementwise NumPy work
  (the weight-load-dominated cost analogue, independent of
  batch size) plus a tiny per-node vector op, so a batched
  verification round costs ~one forward, not one forward
  per node. The simulated work is real compute whose result
  is folded into a checksum, so it cannot be optimised
  away; its size (``_TARGET_WORK_ROUNDS`` x 384x384
  float32) is fixed BEFORE measuring and is identical for
  every arm. Draft forwards are cheap bigram lookups with
  precomputed rankings (the draft is a lookup table, as a
  small draft model would be orders of magnitude cheaper
  than the target). Wall-clock is therefore a property of
  this cost model, not of a real LM — the tokens-per-forward
  column is the portable number.

FIXED BEFORE MEASURING (do not tune to hit the target):
``MARGIN = 0.05``, ``CORPUS_SEED = 20261126``,
``HEAD_MASS = 0.90``, ``MAX_NEW = 24``, the 40-prompt set
(``PROMPTS``), the policy thresholds (``TreePolicy``) and
the simulated-cost size. The eval is run once with these
and reported as measured — if the adaptive arm does not
beat the fixed tree, or agreement falls below 94%, the
report says so and nothing is re-tuned.

Margin rule (the lossy acceptance): a mismatched draft
token ``t`` is still accepted when the target probability
of ``t`` is within ``margin`` of the target's top choice,
i.e. ``p_target(t) >= max(p_target) - margin`` (boundary
inclusive). At margin 0 only a target top-1 token is
accepted, so speculative output equals target-only greedy
output exactly (pinned by test).

Adaptive tree policy (the contribution; exact rule):
draft confidence ``c`` = draft top-1 probability at the
round's root context; rolling acceptance ``a`` = EMA
(alpha 0.25, init 0.5) of the per-round acceptance rate
(accepted draft tokens / draft tokens examined on the
accepted path, bonus token excluded). Shape chosen per
round BEFORE drafting:
- ``a >= 0.7 and c >= 0.5``  -> depth 5, width 3 (widen+deepen)
- ``a >= 0.5 and c >= 0.35`` -> depth 4, width 2
- ``a < 0.3 or c < 0.2``     -> depth 2, width 1 (narrow)
- otherwise                  -> depth 3, width 2 (base)
Total tree nodes are capped at ``MAX_TREE_NODES = 40``.
"""

import time

import numpy as np

# ---------------------------------------------------------------------------
# Fixed parameters (fixed before measuring; see module docstring).
# ---------------------------------------------------------------------------

CORPUS_SEED = 20261126
MARGIN = 0.05
HEAD_MASS = 0.90
SMOOTHING = 0.1
MAX_NEW = 24
MAX_TREE_NODES = 40
FIXED_DEPTH = 3
FIXED_WIDTH = 2
BASE_DEPTH = 3
BASE_WIDTH = 2
N_PROMPTS = 40
N_ARITHMETIC_PROMPTS = 24

# Simulated target-forward cost: fixed elementwise work per
# forward (weight-load analogue) + a tiny per-node op.
_TARGET_WORK_ROUNDS = 8
_TARGET_WORK_DIM = 384

VOCAB = (
    # digits
    [str(d) for d in range(10)]
    # operators / punctuation
    + ["+", "-", "*", "=", "?", ".", ","]
    # arithmetic / sentence words
    + [
        "what", "is", "the", "sum", "difference", "product", "of",
        "and", "answer", "compute", "find", "total", "result",
        "plus", "minus", "times", "equals", "first", "then", "add",
        "subtract", "multiply", "check", "given", "number",
        "numbers", "small", "large", "by", "work", "step",
    ]
    # filler words
    + [
        "cat", "dog", "sun", "rain", "book", "tree", "house", "red",
        "blue", "big", "runs", "sits", "on", "in", "under", "near",
    ]
)
assert len(VOCAB) == 64, len(VOCAB)
TOK = {t: i for i, t in enumerate(VOCAB)}
V = len(VOCAB)
DIGITS = set(str(d) for d in range(10))
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
       "*": lambda a, b: a * b}
OP_TOKENS = {"+": "+", "plus": "+", "-": "-", "minus": "-",
             "*": "*", "times": "*"}

NOTE = (
    "toy decoder: seeded trigram target + bigram draft over a "
    "64-token vocab on fixed synthetic text; no real model, no "
    "EAGLE-3, no GSM8K/MATH/HumanEval; throughput is "
    "tokens-per-target-forward (cost model) + CPU wall-clock "
    "under a simulated batched target-forward cost"
)


# ---------------------------------------------------------------------------
# Synthetic corpus + n-gram models
# ---------------------------------------------------------------------------

def _arith_expr(rng):
    """One (a, op, b, answer-string) with a, b single digits."""
    op = rng.choice(["+", "-", "*"])
    if op == "-":
        a = int(rng.integers(0, 10))
        b = int(rng.integers(0, a + 1))
    else:
        a = int(rng.integers(0, 10))
        b = int(rng.integers(0, 10))
    return a, op, b, str(OPS[op](a, b))


def build_corpus(seed=CORPUS_SEED, n_arith=3000, n_filler=3000):
    """Fixed synthetic text: arithmetic sentences + filler sentences.

    Arithmetic templates put the answer digits where the
    trigram target (with its arithmetic head) can find them;
    the bigram draft sees only the previous token, so it
    cannot compute the answer — that gap is what the margin
    rule and the tree policy have to work with.
    """
    rng = np.random.default_rng(seed)
    sentences = []
    for _ in range(n_arith):
        a, op, b, ans = _arith_expr(rng)
        word = {"+": "plus", "-": "minus", "*": "times"}[op]
        template = int(rng.integers(0, 4))
        if template == 0:
            s = ["what", "is", str(a), op, str(b), "?",
                 "the", "answer", "is"] + list(ans) + ["."]
        elif template == 1:
            s = ["compute", str(a), word, str(b), ".",
                 "the", "total", "is"] + list(ans) + ["."]
        elif template == 2 and op == "*":
            s = ["find", "the", "product", "of", str(a), "and",
                 str(b), ".", "the", "answer", "is"] + list(ans) + ["."]
        else:
            s = [str(a), op, str(b), "="] + list(ans) + ["."]
        sentences.append(s)
    nouns = ["cat", "dog", "sun", "rain", "book", "tree", "house"]
    adjs = ["red", "blue", "big", "small"]
    verbs = ["runs", "sits"]
    preps = ["on", "in", "under", "near"]
    # Coverage sentences: every vocab word must occur in the
    # corpus, or prompts using it fall back to the bigram.
    coverage = [
        ["the", "sum", "of", "the", "numbers", "is", "small", "."],
        ["find", "the", "difference", "of", "the", "numbers", "."],
        ["the", "result", "equals", "the", "total", "."],
        ["the", "first", "step", "is", "small", "."],
        ["subtract", "the", "small", "number", "then", "check",
         "the", "work", "."],
        ["multiply", "the", "given", "number", "by", "2", "."],
        ["the", "rain", "is", "small", "."],
        ["the", "large", "book", "is", "red", "."],
        ["the", "small", "cat", ",", "the", "big", "dog", "."],
    ]
    for _ in range(n_filler):
        template = int(rng.integers(0, 4))
        if template == 0:
            s = ["the", rng.choice(adjs), rng.choice(nouns),
                 rng.choice(verbs), rng.choice(preps), "the",
                 rng.choice(nouns), "."]
        elif template == 1:
            s = ["the", rng.choice(nouns), "is",
                 rng.choice(adjs), "."]
        elif template == 2:
            s = ["first", rng.choice(["add", "check"]), "the",
                 rng.choice(adjs), rng.choice(["number", "book"]),
                 "then", "check", "the", "number", "."]
        else:
            s = list(coverage[int(rng.integers(0, len(coverage)))])
        sentences.append([str(t) for t in s])
    return sentences


def _counts(sentences, order):
    """Count dict: context tuple -> counts over next token."""
    counts = {}
    for s in sentences:
        ids = [TOK[t] for t in s]
        for i in range(len(ids) - 1):
            # Context = the (order-1) tokens before ids[i+1].
            ctx = tuple(ids[max(0, i - order + 2):i + 1])
            counts.setdefault(ctx, np.zeros(V, dtype=np.float64))
            counts[ctx][ids[i + 1]] += 1.0
    return counts


def _dist_from_counts(counts, ctx, smoothing=SMOOTHING):
    vec = counts.get(ctx)
    if vec is None:
        vec = np.zeros(V, dtype=np.float64)
    p = (vec + smoothing) / (vec.sum() + smoothing * V)
    return p


def _break_ties(p):
    """Add a 1e-12 per-index tilt so exact ties resolve to the
    lowest-index token (the argmax tie-break). At margin 0
    this is what makes speculative output equal target-only
    output exactly: a tied non-argmax draft token is then
    strictly below the top-1 probability and is rejected.
    The tilt is far below any margin used here."""
    p = p + 1e-12 * (V - np.arange(V, dtype=np.float64))
    return p / p.sum()


class NGramModel:
    """Order-n n-gram LM with add-k smoothing over VOCAB.

    Backoff: if the full (order-1)-token context was never
    seen, fall back to the shorter context, down to the
    unigram. Without this, any unseen context (e.g. right
    after a sentence-final ".") would collapse to near-
    uniform and greedy decoding would emit the tie-break
    token forever.
    """

    def __init__(self, sentences, order):
        self.order = order
        self.counts = _counts(sentences, order)
        self.counts_by_len = {
            L: _counts(sentences, L + 1) for L in range(order)
        }
        # Precomputed draft rankings: for a bigram model the
        # top-k order per previous token never changes.
        self._rankings = {}
        if order == 2:
            for prev in range(V):
                p = _dist_from_counts(self.counts_by_len[1], (prev,))
                self._rankings[prev] = list(np.argsort(-p))

    def dist(self, context):
        for L in range(self.order - 1, -1, -1):
            ctx = tuple(context[-L:]) if L > 0 else tuple()
            d = self.counts_by_len[L]
            if ctx in d:
                return _break_ties(_dist_from_counts(d, ctx))
        return _break_ties(np.full(V, 1.0 / V))

    def top_k(self, context, k):
        if self.order == 2 and context:
            return self._rankings[context[-1]][:k]
        p = self.dist(context)
        return list(np.argsort(-p)[:k])

    def top1(self, context):
        return int(np.argmax(self.dist(context)))

    def confidence(self, context):
        return float(np.max(self.dist(context)))


def _pending_answer(context):
    """Arithmetic head: if ``context`` ends in a pending
    ``A op B`` answer position, return the next expected
    answer digit (token id); else None.

    Triggers: a trailing ``=`` after an expression, or a
    trailing ``answer is`` / ``total is`` after an
    expression. Digits already emitted after the trigger
    count toward the answer string; once the full answer
    has been emitted the head switches off.
    """
    ctx = [VOCAB[i] for i in context]

    def parse_expr(upto):
        # Closest op before `upto` with digit runs each side.
        # Ops may be symbols (+ - *), words (plus minus times),
        # or implied by "product of A and B".
        for j in range(upto - 1, -1, -1):
            if ctx[j] in OP_TOKENS:
                op = OP_TOKENS[ctx[j]]
                left = []
                k = j - 1
                while k >= 0 and ctx[k] in DIGITS:
                    left.append(ctx[k])
                    k -= 1
                right = []
                k = j + 1
                while k < upto and ctx[k] in DIGITS:
                    right.append(ctx[k])
                    k += 1
                if left and right:
                    a = int("".join(reversed(left)))
                    b = int("".join(right))
                    if op == "-" and a < b:
                        return None
                    return str(OPS[op](a, b))
            if ctx[j] == "and" and "product" in ctx[max(0, j - 8):j]:
                left = []
                k = j - 1
                while k >= 0 and ctx[k] in DIGITS:
                    left.append(ctx[k])
                    k -= 1
                right = []
                k = j + 1
                while k < upto and ctx[k] in DIGITS:
                    right.append(ctx[k])
                    k += 1
                if left and right:
                    a = int("".join(reversed(left)))
                    b = int("".join(right))
                    return str(a * b)
        return None

    # Trigger 1: trailing "=".
    if ctx and ctx[-1] == "=":
        ans = parse_expr(len(ctx) - 1)
        return TOK[ans[0]] if ans else None
    # Digits already emitted after "=".
    if "=" in ctx:
        i = len(ctx) - 1 - ctx[::-1].index("=")
        tail = ctx[i + 1:]
        if tail and all(t in DIGITS for t in tail):
            ans = parse_expr(i)
            if ans and len(tail) < len(ans):
                return TOK[ans[len(tail)]]
            return None
    # Trigger 2: "answer is" / "total is" (+ digits so far).
    for j in range(len(ctx) - 1, 0, -1):
        if ctx[j] == "is" and ctx[j - 1] in ("answer", "total"):
            tail = ctx[j + 1:]
            if tail and not all(t in DIGITS for t in tail):
                return None
            ans = parse_expr(j - 1)
            if ans is None:
                return None
            if len(tail) < len(ans):
                return TOK[ans[len(tail)]]
            return None
    return None


class TargetModel:
    """Trigram target + arithmetic head (HEAD_MASS on the
    expected answer digit when a pending expression exists,
    else the pure trigram distribution).

    ``forward`` / ``forward_batch`` charge the SIMULATED
    cost: one fixed block of elementwise work per call
    (weight-load analogue, independent of batch size) plus
    a tiny per-node op. ``forwards`` counts target forwards;
    ``work_checksum`` keeps the simulated work live.
    """

    def __init__(self, sentences):
        self.ngram = NGramModel(sentences, order=3)
        rng = np.random.default_rng(1234)
        self._work = rng.standard_normal(
            (_TARGET_WORK_DIM, _TARGET_WORK_DIM)).astype(np.float32)
        self._vec = rng.standard_normal(_TARGET_WORK_DIM).astype(np.float32)
        self.forwards = 0
        self.work_checksum = 0.0

    def _simulated_cost(self, n_nodes):
        acc = np.float32(0.0)
        for _ in range(_TARGET_WORK_ROUNDS):
            acc += np.sum(self._work * self._vec)
        for _ in range(n_nodes):
            acc += np.sum(self._vec * self._work[0])
        self.work_checksum += float(acc)

    def dist(self, context):
        p = self.ngram.dist(context)
        nxt = _pending_answer(context)
        if nxt is not None:
            onehot = np.zeros(V, dtype=np.float64)
            onehot[nxt] = 1.0
            p = HEAD_MASS * onehot + (1.0 - HEAD_MASS) * p
            p = _break_ties(p)
        return p

    def forward(self, context):
        self._simulated_cost(1)
        self.forwards += 1
        return self.dist(context)

    def forward_batch(self, contexts):
        self._simulated_cost(len(contexts))
        self.forwards += 1
        return [self.dist(c) for c in contexts]

    def greedy(self, context):
        return int(np.argmax(self.forward(context)))


class DraftModel:
    """Bigram draft, no arithmetic head. Cheap by design:
    rankings are precomputed, so a draft step is a lookup."""

    def __init__(self, sentences):
        self.ngram = NGramModel(sentences, order=2)
        self.forwards = 0

    def dist(self, context):
        return self.ngram.dist(context)

    def top_k(self, context, k):
        self.forwards += 1
        return self.ngram.top_k(context, k)

    def confidence(self, context):
        return self.ngram.confidence(context)


# ---------------------------------------------------------------------------
# Margin rule + draft tree + verification
# ---------------------------------------------------------------------------

def margin_accept(p_target, token, margin):
    """Lossy acceptance: accept ``token`` when its target
    probability is within ``margin`` of the target top-1
    (boundary inclusive)."""
    return float(p_target[token]) >= float(np.max(p_target)) - margin


def build_draft_tree(draft, context, depth, width,
                     max_nodes=MAX_TREE_NODES):
    """Draft candidate tree, BFS from ``context``.

    Node = (parent_idx, token_or_None, context_tuple).
    Node 0 is the root (current context). At each level every
    frontier node expands to its top-``width`` draft tokens,
    until ``depth`` levels or ``max_nodes`` nodes.
    """
    nodes = [(-1, None, tuple(context))]
    frontier = [0]
    for _ in range(depth):
        nxt = []
        for idx in frontier:
            for tok in draft.top_k(nodes[idx][2], width):
                if len(nodes) >= max_nodes:
                    break
                nodes.append((idx, tok, nodes[idx][2] + (tok,)))
                nxt.append(len(nodes) - 1)
            if len(nodes) >= max_nodes:
                break
        frontier = nxt
        if not frontier:
            break
    return nodes


def verify_tree(nodes, target_dists, margin):
    """Verify a draft tree against precomputed target dists
    (ONE batched target pass produced them all).

    Greedy DFS from the root: at each node take the first
    child (draft order) whose token the margin rule accepts;
    on the first node with no acceptable child, emit the
    target greedy token at that node as the bonus and stop.
    Returns (accepted_tokens_incl_bonus, n_accepted_draft,
    n_examined_draft).
    """
    children = {}
    for idx, (parent, tok, _ctx) in enumerate(nodes):
        if parent >= 0:
            children.setdefault(parent, []).append(idx)
    accepted = []
    n_accepted = 0
    n_examined = 0
    node = 0
    while True:
        progressed = False
        for child in children.get(node, []):
            tok = nodes[child][1]
            n_examined += 1
            if margin_accept(target_dists[node], tok, margin):
                accepted.append(tok)
                n_accepted += 1
                node = child
                progressed = True
                break
        if not progressed:
            accepted.append(int(np.argmax(target_dists[node])))
            break
    return accepted, n_accepted, n_examined


# ---------------------------------------------------------------------------
# Adaptive tree policy
# ---------------------------------------------------------------------------

class TreePolicy:
    """AdaptiveSpec tree policy (exact rule, see module docstring).

    ``choose`` is called per round BEFORE drafting, from the
    rolling acceptance EMA and the draft confidence at the
    root. ``update`` feeds the round's acceptance rate back
    into the EMA.
    """

    def __init__(self, alpha=0.25, init_accept=0.5):
        self.alpha = alpha
        self.ema_accept = init_accept
        self.history = []

    def choose(self, draft_confidence):
        a, c = self.ema_accept, draft_confidence
        if a >= 0.7 and c >= 0.5:
            shape = (5, 3)
        elif a >= 0.5 and c >= 0.35:
            shape = (4, 2)
        elif a < 0.3 or c < 0.2:
            shape = (2, 1)
        else:
            shape = (BASE_DEPTH, BASE_WIDTH)
        self.history.append(shape)
        return shape

    def update(self, n_accepted, n_examined):
        rate = n_accepted / max(1, n_examined)
        self.ema_accept = (1.0 - self.alpha) * self.ema_accept \
            + self.alpha * rate
        return rate


class FixedPolicy:
    """EAGLE-like baseline: depth 3, width 2, every round."""

    def __init__(self):
        self.history = []

    def choose(self, draft_confidence):
        shape = (FIXED_DEPTH, FIXED_WIDTH)
        self.history.append(shape)
        return shape

    def update(self, n_accepted, n_examined):
        return n_accepted / max(1, n_examined)


# ---------------------------------------------------------------------------
# Generation arms
# ---------------------------------------------------------------------------

def generate_target_only(target, prompt, max_new=MAX_NEW):
    """Greedy target-only decoding: one target forward/token."""
    ctx = list(prompt)
    for _ in range(max_new):
        ctx.append(target.greedy(ctx))
    return ctx[len(prompt):]


def generate_speculative(target, draft, prompt, policy,
                         margin=MARGIN, max_new=MAX_NEW):
    """Speculative decoding with tree verification.

    Per round: policy picks (depth, width), the draft builds
    a tree, the target verifies it in ONE batched forward,
    accepted tokens (+ bonus) are appended. Returns
    (generated_tokens, n_rounds).
    """
    ctx = list(prompt)
    generated = []
    rounds = 0
    while len(generated) < max_new:
        conf = draft.confidence(ctx)
        depth, width = policy.choose(conf)
        nodes = build_draft_tree(draft, ctx, depth, width)
        dists = target.forward_batch([n[2] for n in nodes])
        accepted, n_acc, n_ex = verify_tree(nodes, dists, margin)
        policy.update(n_acc, n_ex)
        take = accepted[:max_new - len(generated)]
        generated.extend(take)
        ctx.extend(take)
        rounds += 1
    return generated, rounds


# ---------------------------------------------------------------------------
# Prompt set (fixed before measuring)
# ---------------------------------------------------------------------------

PROMPTS = [
    # 24 arithmetic prompts (answers verified vs target-only).
    ["what", "is", "3", "+", "4", "?"],
    ["what", "is", "7", "+", "8", "?"],
    ["what", "is", "9", "-", "4", "?"],
    ["what", "is", "2", "*", "6", "?"],
    ["what", "is", "9", "*", "9", "?"],
    ["what", "is", "5", "+", "5", "?"],
    ["what", "is", "8", "-", "8", "?"],
    ["what", "is", "6", "*", "3", "?"],
    ["what", "is", "1", "+", "9", "?"],
    ["what", "is", "4", "*", "7", "?"],
    ["what", "is", "0", "+", "6", "?"],
    ["what", "is", "9", "-", "1", "?"],
    ["compute", "3", "plus", "5", "."],
    ["compute", "8", "minus", "2", "."],
    ["compute", "4", "times", "6", "."],
    ["compute", "7", "plus", "7", "."],
    ["find", "the", "product", "of", "3", "and", "8", "."],
    ["find", "the", "product", "of", "5", "and", "5", "."],
    ["3", "+", "4", "="],
    ["9", "*", "8", "="],
    ["7", "-", "3", "="],
    ["6", "+", "6", "="],
    ["2", "*", "9", "="],
    ["8", "+", "1", "="],
    # 16 filler prompts.
    ["the", "cat", "sits", "on", "the"],
    ["the", "big", "dog", "runs"],
    ["the", "sun", "is"],
    ["the", "red", "book", "is"],
    ["first", "add", "the", "small"],
    ["check", "the", "total"],
    ["the", "tree", "is", "big", ".", "the"],
    ["the", "house", "is", "red", "."],
    ["the", "dog", "sits", "under", "the"],
    ["first", "check", "the", "small", "number"],
    ["the", "rain", "is"],
    ["the", "blue", "cat", "runs", "on"],
    ["the", "book", "is", "small", "."],
    ["the", "sun", "is", "red", ".", "the"],
    ["first", "add", "the", "large", "number"],
    ["the", "cat", "is"],
]
assert len(PROMPTS) == N_PROMPTS, len(PROMPTS)
PROMPT_IDS = [[TOK[t] for t in p] for p in PROMPTS]


# ---------------------------------------------------------------------------
# Agreement + arithmetic-answer metrics
# ---------------------------------------------------------------------------

def token_agreement(a, b):
    """Fraction of positions equal (equal-length lists)."""
    return sum(1 for x, y in zip(a, b) if x == y) / max(1, len(a))


def extract_answer(prompt_ids, generated_ids):
    """Answer digit string emitted for an arithmetic prompt,
    or None if no answer was emitted. Looks for the first
    ``=`` or ``answer is`` / ``total is`` trigger in
    prompt + generation and collects the digit run after it.
    """
    toks = [VOCAB[i] for i in list(prompt_ids) + list(generated_ids)]
    for j, t in enumerate(toks):
        if t == "=":
            tail = toks[j + 1:]
            digits = []
            for x in tail:
                if x in DIGITS and len(digits) < 2:
                    digits.append(x)
                else:
                    break
            return "".join(digits) if digits else None
        if t == "is" and j > 0 and toks[j - 1] in ("answer", "total"):
            tail = toks[j + 1:]
            digits = []
            for x in tail:
                if x in DIGITS and len(digits) < 2:
                    digits.append(x)
                else:
                    break
            return "".join(digits) if digits else None
    return None


def build_models():
    sentences = build_corpus()
    return TargetModel(sentences), DraftModel(sentences)


def run_arm(target, draft, prompts, arm, margin=MARGIN,
            max_new=MAX_NEW):
    """Run one arm over all prompts. Returns a result dict
    (tokens, target forwards, draft forwards, rounds, wall
    seconds, per-prompt generations)."""
    target.forwards = 0
    draft.forwards = 0
    policy = TreePolicy() if arm == "adaptive" else \
        FixedPolicy() if arm == "fixed" else None
    gens = []
    t0 = time.perf_counter()
    for prompt in prompts:
        if arm == "target-only":
            gens.append(generate_target_only(target, prompt, max_new))
        else:
            gen, _ = generate_speculative(
                target, draft, prompt, policy, margin, max_new)
            gens.append(gen)
    wall = time.perf_counter() - t0
    n_tokens = sum(len(g) for g in gens)
    return {
        "arm": arm,
        "generations": gens,
        "tokens": n_tokens,
        "target_forwards": target.forwards,
        "draft_forwards": draft.forwards,
        "tokens_per_target_forward": n_tokens / max(1, target.forwards),
        "tokens_per_sec": n_tokens / max(wall, 1e-9),
        "wall_sec": wall,
        "policy_shapes": list(policy.history) if policy else [],
    }


def run_spec_eval(repeats=3, margin=MARGIN, max_new=MAX_NEW,
                  n_prompts=None):
    """Full P10 eval: three arms, same draft/target, same
    margin — the policy effect is isolated. Wall-clock is
    the mean over ``repeats``; tokens-per-forward and
    agreement are deterministic (identical every repeat).
    """
    prompts = PROMPT_IDS[:n_prompts] if n_prompts else PROMPT_IDS
    n_arith = sum(
        1 for p in PROMPTS[:len(prompts)]
        if p[0] in ("compute", "find") or "?" in p or "=" in p)
    target, draft = build_models()
    ref = run_arm(target, draft, prompts, "target-only",
                  margin, max_new)
    arms = {"target-only": ref}
    for arm in ("fixed", "adaptive"):
        runs = [run_arm(target, draft, prompts, arm, margin, max_new)
                for _ in range(repeats)]
        agg = dict(runs[0])
        agg["tokens_per_sec"] = float(
            np.mean([r["tokens_per_sec"] for r in runs]))
        agg["wall_sec"] = float(np.mean([r["wall_sec"] for r in runs]))
        arms[arm] = agg

    def agreement(arm):
        tok = float(np.mean([
            token_agreement(g, r)
            for g, r in zip(arms[arm]["generations"],
                            ref["generations"])]))
        seq = float(np.mean([
            1.0 if g == r else 0.0
            for g, r in zip(arms[arm]["generations"],
                            ref["generations"])]))
        return tok, seq

    def arith_match(arm):
        ok, total = 0, 0
        for i in range(n_arith):
            a = extract_answer(prompts[i], arms[arm]["generations"][i])
            b = extract_answer(prompts[i], ref["generations"][i])
            if b is not None:
                total += 1
                if a == b:
                    ok += 1
        return ok, total

    for arm in ("fixed", "adaptive"):
        tok, seq = agreement(arm)
        arms[arm]["token_agreement"] = tok
        arms[arm]["sequence_exact_match"] = seq
        ok, total = arith_match(arm)
        arms[arm]["arith_match"] = ok
        arms[arm]["arith_total"] = total
        arms[arm]["arith_match_rate"] = ok / max(1, total)
    arms["target-only"]["token_agreement"] = 1.0
    arms["target-only"]["sequence_exact_match"] = 1.0
    ok, total = arith_match("target-only")
    arms["target-only"]["arith_match"] = ok
    arms["target-only"]["arith_total"] = total
    arms["target-only"]["arith_match_rate"] = ok / max(1, total)

    fixed, adap = arms["fixed"], arms["adaptive"]
    done = (
        adap["tokens_per_sec"] > fixed["tokens_per_sec"]
        and adap["tokens_per_target_forward"]
        > fixed["tokens_per_target_forward"]
        and adap["token_agreement"] >= 0.94
    )
    return {
        "note": NOTE,
        "margin": margin,
        "max_new": max_new,
        "n_prompts": len(prompts),
        "n_arithmetic_prompts": n_arith,
        "repeats": repeats,
        "arms": arms,
        "done_adaptive_beats_fixed_with_agreement": done,
    }


def print_spec_report(result):
    print("P10 — AdaptiveSpec: adaptive lossy speculative decoding "
          "(toy decoder, NumPy)")
    print("NOTE: %s" % result["note"])
    print("margin %.2f, max_new %d, prompts %d (%d arithmetic), "
          "repeats %d"
          % (result["margin"], result["max_new"],
             result["n_prompts"], result["n_arithmetic_prompts"],
             result["repeats"]))
    print("\n%-12s %10s %12s %12s %10s %10s %10s" % (
        "arm", "tok/sec", "tok/fwd", "tgt fwd", "draft fwd",
        "tok agree", "seq exact"))
    for label in ("target-only", "fixed", "adaptive"):
        a = result["arms"][label]
        print("%-12s %10.1f %12.2f %12d %10d %10.3f %10.3f" % (
            label, a["tokens_per_sec"],
            a["tokens_per_target_forward"], a["target_forwards"],
            a["draft_forwards"], a["token_agreement"],
            a["sequence_exact_match"]))
    for label in ("target-only", "fixed", "adaptive"):
        a = result["arms"][label]
        print("  %-12s arithmetic-answer match vs target-only: "
              "%d/%d (%.3f)"
              % (label, a["arith_match"], a["arith_total"],
                 a["arith_match_rate"]))
    adap = result["arms"]["adaptive"]
    fixed = result["arms"]["fixed"]
    from collections import Counter
    print("  adaptive tree shapes used (depth,width): %s"
          % dict(Counter(adap["policy_shapes"])))
    print("  fixed tree shapes used (depth,width): %s"
          % dict(Counter(fixed["policy_shapes"])))
    print("\nDone condition (adaptive beats fixed on tok/sec AND "
          "tok/target-forward, token agreement >= 0.94): %s"
          % result["done_adaptive_beats_fixed_with_agreement"])
