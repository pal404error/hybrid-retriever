"""P13 — DISCO: distributed speculative decoding.

DISCO (arXiv:2610.35957): split the draft model (laptop) from
the target model (server) WITHOUT shipping vocab-scale
distributions. The server keeps a replica of the small draft
model; only token indices + one scalar probability per
proposed token cross the wire. The server recomputes the
draft probabilities locally from its replica — that is the
DISCO trick: the full distribution never crosses the wire —
and runs the same lossless verification as P11. The naive
baseline ships the full draft distribution per token.

SUBSTITUTIONS, stated plainly (this VM has one machine, no
laptop + server split, no second host, no real network;
do NOT read this module as a run across two physical
machines):
- The split is an IN-PROCESS client/server simulation.
  ``DiscoClient`` holds the draft model (proposer);
  ``DiscoServer`` holds the target model AND a replica of
  the draft model (a second, separately built ``DraftModel``
  from the same sentences — a genuine replica, not the same
  object; pinned by test). No sockets, no real network:
  the two ends exchange real serialised byte payloads
  through an explicit bandwidth-limited ``Link`` object
  that counts exact bytes and simulates latency from a
  configured bandwidth (bytes/sec) plus a fixed
  per-message overhead. Link time is SIMULATED (added to
  the measured compute wall-clock), never slept, so no
  result depends on network timing.
- The toy decoder is P10/P11's: a seeded trigram target +
  arithmetic head and a bigram draft over the same
  64-token vocab, trained on the same fixed synthetic
  text. Verification is P11's ``speculative_round``,
  reused unchanged through the thin wrappers
  ``generate_split`` / ``generate_local`` — the wrappers
  only do the protocol (propose / encode / send / verify /
  reply) and loop rounds to ``max_new``; they do not
  re-implement the rejection trial, the adjusted
  resample, or the bonus token.
- The wire format is float32, and sampling on BOTH ends
  consumes the float32-rounded distribution, so the
  client's proposal and the server's verification consume
  bit-identical distributions. This matters: had the
  client sampled from the float64 distribution while the
  server re-sampled from the float32-rounded one (naive
  arm) or the replica's float64 one (DISCO arm), a
  borderline ``rng.choice`` draw could flip a token and
  break output identity for a rounding reason, not a
  protocol reason. The canonical distribution is the
  float32 array; the float64 sampling distribution is
  derived from it deterministically (``_canonical_p``:
  astype float64, renormalise) identically on both ends.
  The lossless theorem still holds exactly — q is
  whatever distribution both ends agree on; the trial
  uses that q consistently.
- The per-round RNG is reseeded deterministically
  (``EVAL_SEED + 7919 * prompt_index + round_index``)
  so client and server derive the SAME round RNG
  independently — the simulation device that stands in
  for "the client sampled these tokens and sent them".
  The server additionally CHECKS the received tokens and
  scalars against its replica (DISCO) — if the replica
  ever diverged from the client's draft, the check fails
  loudly instead of silently verifying different tokens.
  The local arm (``generate_local``) is the same P11
  ``speculative_round`` chained with the same seed
  schedule and no link; it differs from P11's
  ``generate_lossless`` only in RNG scheduling (per-round
  reseeding vs one continuous stream).

WIRE FORMAT (fixed before measuring):
- Token index: uint16, 2 bytes (vocab 64 fits).
- Probability: float32, 4 bytes, little-endian.
- DISCO request (client -> server), K tokens: K token
  indices + K scalar probabilities = 6K bytes (K=6: 36).
- NAIVE request (client -> server), K tokens: K token
  indices + K full V-float distributions = 2K + 4KV
  bytes (K=6, V=64: 1548).
- Reply (server -> client), either protocol: accepted
  count uint8 (1 byte) + repair/bonus token uint16
  (2 bytes) = 3 bytes. The client already knows its
  proposed tokens, so the count + final token
  reconstruct the round's output exactly.
- Per-round totals: DISCO 39 bytes, NAIVE 1551 bytes —
  a 39.77x bandwidth ratio before measuring anything.

FIXED BEFORE MEASURING (do not tune to hit the target):
``DISCO_K = 6``, ``MAX_NEW = 24`` (P11's), the 40-prompt
set (P10's ``PROMPTS``), ``EVAL_SEED = 20261129``,
``TARGET_WORK_ROUNDS = 8`` (P11 config A's simulated
target-forward size, same for every arm and link),
``REPEATS = 3``, and the two link settings in
``LINK_SETTINGS``. The eval is run once with these and
reported as measured — if the >= 10x bandwidth saving or
the slow-link wall-clock win does not hold, the report
says so and nothing is re-tuned: not the vocab, not the
wire format, not the link parameters.

What the numbers can and cannot say: the byte counts
and the bandwidth ratio are exact (real serialised
payloads, counted). The tokens/sec columns are
compute wall-clock (real NumPy work under P11's
simulated target-forward cost) PLUS simulated link time
— the link dominates on the slow setting and is
negligible on the fast one, which is the crossover the
eval is built to show. Nothing here is a claim about a
real laptop, a real server, or a real network.
"""

import struct
import time

import numpy as np

# Distributions + the lossless loop only, from P10/P11.
# (generate_speculative / verify_tree / the margin rule /
# TreePolicy are NOT imported — verification here is P11's
# lossless loop, reused unchanged.)
from .specdec import (
    PROMPT_IDS,
    PROMPTS,
    V,
    DraftModel,
    TargetModel,
    build_corpus,
)
from .specdec_scratch import (
    MAX_NEW,
    CostModel,
    speculative_round,
)

# ---------------------------------------------------------------------------
# Fixed parameters (fixed before measuring; see module docstring).
# ---------------------------------------------------------------------------

DISCO_K = 6
EVAL_SEED = 20261129
REPEATS = 3
N_PROMPTS = 40
TARGET_WORK_ROUNDS = 8
# Per-round seed stride: distinct seeds per (prompt, round).
SEED_STRIDE = 7919

# Link settings, fixed before measuring. Slow = a throttled /
# mobile uplink for the laptop end; fast = a 10 MB/s link
# where bandwidth stops mattering and compute dominates.
LINK_SETTINGS = [
    {"name": "slow link (10 KB/s, 1 ms/msg)",
     "bandwidth_bytes_per_sec": 10_000,
     "per_message_overhead_sec": 0.001},
    {"name": "fast link (10 MB/s, 0.1 ms/msg)",
     "bandwidth_bytes_per_sec": 10_000_000,
     "per_message_overhead_sec": 0.0001},
]

NOTE = (
    "DISCO distributed speculative decoding: in-process "
    "client/server simulation (no two physical machines, no "
    "real network) — client holds the bigram draft, server "
    "holds the trigram target AND a replica of the draft; "
    "DISCO ships token indices + one float32 scalar per "
    "token, naive ships the full 64-float distribution per "
    "token; both arms verify with P11's lossless "
    "speculative_round on the seeded toy decoder (64-token "
    "vocab, NumPy); link time is simulated from bandwidth "
    "+ per-message overhead, never slept"
)


# ---------------------------------------------------------------------------
# Wire format: exact byte counts, real serialisation.
# ---------------------------------------------------------------------------

def disco_request_bytes(k=DISCO_K):
    """DISCO request size: K uint16 indices + K float32 scalars."""
    return k * 2 + k * 4


def naive_request_bytes(k=DISCO_K):
    """NAIVE request size: K uint16 indices + K full V-float
    float32 distributions."""
    return k * 2 + k * V * 4


def reply_bytes():
    """Reply size, either protocol: uint8 accepted count +
    uint16 repair/bonus token."""
    return 1 + 2


def encode_disco_request(tokens, scalars):
    """Serialise a DISCO request: uint16 indices then float32
    scalars, little-endian."""
    tokens = list(tokens)
    scalars = [float(s) for s in scalars]
    if len(tokens) != len(scalars):
        raise ValueError("tokens and scalars must have equal length")
    return struct.pack("<%dH" % len(tokens), *tokens) \
        + struct.pack("<%df" % len(scalars), *scalars)


def decode_disco_request(payload, k=DISCO_K):
    """Decode a DISCO request. Returns (token ids, float32
    scalars). Raises on a short / over-long payload."""
    if len(payload) != disco_request_bytes(k):
        raise ValueError(
            "DISCO request must be %d bytes, got %d"
            % (disco_request_bytes(k), len(payload)))
    tokens = list(struct.unpack("<%dH" % k, payload[:2 * k]))
    scalars = [np.float32(s) for s in struct.unpack(
        "<%df" % k, payload[2 * k:])]
    return tokens, scalars


def encode_naive_request(tokens, dists):
    """Serialise a NAIVE request: uint16 indices then K full
    V-float float32 distributions, little-endian."""
    tokens = list(tokens)
    if len(dists) != len(tokens):
        raise ValueError("one distribution per token")
    out = struct.pack("<%dH" % len(tokens), *tokens)
    for d in dists:
        arr = np.asarray(d, dtype=np.float32)
        if len(arr) != V:
            raise ValueError("distribution must have %d entries" % V)
        out += struct.pack("<%df" % V, *[float(x) for x in arr])
    return out


def decode_naive_request(payload, k=DISCO_K):
    """Decode a NAIVE request. Returns (token ids, list of
    float32 distribution arrays). Raises on a short /
    over-long payload."""
    if len(payload) != naive_request_bytes(k):
        raise ValueError(
            "NAIVE request must be %d bytes, got %d"
            % (naive_request_bytes(k), len(payload)))
    tokens = list(struct.unpack("<%dH" % k, payload[:2 * k]))
    dists = []
    off = 2 * k
    for _ in range(k):
        dists.append(np.array(
            struct.unpack("<%df" % V,
                          payload[off:off + 4 * V]),
            dtype=np.float32))
        off += 4 * V
    return tokens, dists


def encode_reply(n_accepted, token):
    """Serialise a reply: uint8 accepted count + uint16
    repair/bonus token."""
    if not 0 <= n_accepted <= 255:
        raise ValueError("accepted count must fit in uint8")
    return struct.pack("<BH", n_accepted, token)


def decode_reply(payload):
    """Decode a reply. Returns (n_accepted, token)."""
    if len(payload) != reply_bytes():
        raise ValueError(
            "reply must be %d bytes, got %d"
            % (reply_bytes(), len(payload)))
    n_accepted, token = struct.unpack("<BH", payload)
    return n_accepted, token


# ---------------------------------------------------------------------------
# Canonical float32 draft distribution.
# ---------------------------------------------------------------------------

def _canonical_p(q):
    """The sampling distribution both ends agree on: the
    float32 wire representation of ``q``, as float64,
    renormalised so it sums to 1 within float64 eps.
    Identical input array -> identical output, on either
    end; that bit-identity is what keeps the client's
    proposal and the server's verification in lockstep."""
    q32 = np.asarray(q, dtype=np.float32)
    p = q32.astype(np.float64)
    return p / p.sum()


class _WireDraft:
    """Draft object whose ``dist`` returns the canonical
    float32-derived distribution. ``source`` maps a context
    (list of token ids) to a distribution array — a live
    model (DISCO: the server's replica) or a table of
    received distributions (NAIVE)."""

    def __init__(self, source):
        self._source = source

    def dist(self, context):
        return _canonical_p(self._source(context))


def sample_draft_tokens(draft, context, rng, k=DISCO_K):
    """Sample K draft tokens sequentially from ``draft`` at
    the running context, consuming ``rng`` exactly as
    P11's ``speculative_round`` does (one ``rng.choice``
    per token, in order). Returns (tokens, scalars, dists):
    the token ids, the float32 draft probability of each
    sampled token, and the full float32 distribution at
    each step (what the NAIVE protocol ships). The sampled
    distribution is the canonical float32 one — see the
    module docstring for why."""
    ctx = list(context)
    tokens = []
    scalars = []
    dists = []
    for _ in range(k):
        q32 = np.asarray(draft.dist(ctx), dtype=np.float32)
        x = int(rng.choice(V, p=_canonical_p(q32)))
        tokens.append(x)
        scalars.append(q32[x])
        dists.append(q32)
        ctx.append(x)
    return tokens, scalars, dists


# ---------------------------------------------------------------------------
# The bandwidth-limited link (simulated latency, exact bytes).
# ---------------------------------------------------------------------------

class Link:
    """A slow (or fast) link, simulated in-process.

    ``send(n_bytes)`` counts the bytes, counts one
    message, and adds ``n_bytes / bandwidth`` plus the
    fixed per-message overhead to ``simulated_seconds``.
    Nothing is slept: the simulated seconds are added to
    the measured compute wall-clock by the eval, so no
    result depends on network timing. Two messages cross
    per verification round (request + reply) under either
    protocol — DISCO saves bytes, not round trips, and the
    overhead term keeps that honest.
    """

    def __init__(self, bandwidth_bytes_per_sec,
                 per_message_overhead_sec=0.0, name="link"):
        if bandwidth_bytes_per_sec <= 0:
            raise ValueError("bandwidth must be positive")
        if per_message_overhead_sec < 0:
            raise ValueError("per-message overhead must be >= 0")
        self.bandwidth_bytes_per_sec = bandwidth_bytes_per_sec
        self.per_message_overhead_sec = per_message_overhead_sec
        self.name = name
        self.bytes_sent = 0
        self.messages = 0
        self.simulated_seconds = 0.0

    def send(self, n_bytes):
        """Account one message of ``n_bytes``. Returns the
        byte count."""
        n_bytes = int(n_bytes)
        if n_bytes < 0:
            raise ValueError("cannot send a negative byte count")
        self.bytes_sent += n_bytes
        self.messages += 1
        self.simulated_seconds += (
            n_bytes / self.bandwidth_bytes_per_sec
            + self.per_message_overhead_sec)
        return n_bytes

    def reset(self):
        self.bytes_sent = 0
        self.messages = 0
        self.simulated_seconds = 0.0


# ---------------------------------------------------------------------------
# Client (draft model, proposer) and server (target + draft
# replica, verifier).
# ---------------------------------------------------------------------------

class DiscoClient:
    """The laptop end: holds the draft model, proposes K
    tokens per round, serialises the request for whichever
    protocol is in force. It never sees the target model."""

    def __init__(self, draft):
        self.draft = draft
        self.proposals = 0

    def propose(self, context, rng, k=DISCO_K):
        """(tokens, scalars, dists) for one round. The RNG
        is the round RNG — the same seed the server
        derives independently."""
        self.proposals += 1
        return sample_draft_tokens(self.draft, context, rng, k)

    def encode(self, tokens, scalars, dists, protocol):
        """Serialise one request under ``protocol``."""
        if protocol == "disco":
            return encode_disco_request(tokens, scalars)
        if protocol == "naive":
            return encode_naive_request(tokens, dists)
        raise ValueError("unknown protocol %r" % (protocol,))


class DiscoServer:
    """The server end: holds the target model AND a replica
    of the draft model. Verification is P11's
    ``speculative_round``, reused unchanged — the server
    only decodes the request, checks it against the
    replica (DISCO) or builds a draft from the received
    distributions (NAIVE), and hands the resulting draft
    object to the loop."""

    def __init__(self, target, draft_replica, cost):
        self.target = target
        self.draft_replica = draft_replica
        self.cost = cost
        self.replica_checks = 0
        self.replica_mismatches = 0

    def _check_replica(self, context, tokens, scalars):
        """DISCO's replica check: the draft probability the
        replica assigns to each proposed token, at its
        prefix context, must equal the scalar the client
        sent — exactly, after the float32 wire cast. A
        mismatch means the replica has diverged from the
        client's draft and the verification that follows
        would be verifying different tokens."""
        ctx = list(context)
        for i, tok in enumerate(tokens):
            q32 = np.asarray(
                self.draft_replica.dist(ctx), dtype=np.float32)
            self.replica_checks += 1
            if q32[tok] != scalars[i]:
                self.replica_mismatches += 1
                raise AssertionError(
                    "draft replica diverged from client at "
                    "position %d: replica q=%r, client sent %r"
                    % (i, q32[tok], scalars[i]))
            ctx.append(tok)

    def verify(self, payload, context, seed, k=DISCO_K,
               protocol="disco"):
        """Verify one round. ``seed`` is the round seed —
        the server derives the same round RNG the client
        used, so ``speculative_round`` re-samples the
        client's exact draft tokens from the replica (DISCO)
        or the received distributions (NAIVE) and runs the
        lossless trial on them. Returns (tokens_out,
        n_accepted_draft)."""
        rng = np.random.default_rng(seed)
        if protocol == "disco":
            tokens, scalars = decode_disco_request(payload, k)
            self._check_replica(context, tokens, scalars)
            draft = _WireDraft(
                lambda ctx: self.draft_replica.dist(ctx))
        elif protocol == "naive":
            tokens, dists = decode_naive_request(payload, k)
            # No replica in the naive design: the server
            # verifies against the distributions it
            # received, looked up by prefix context.
            table = {}
            ctx = list(context)
            for i in range(k):
                table[tuple(ctx)] = dists[i]
                ctx.append(tokens[i])
            draft = _WireDraft(lambda ctx: table[tuple(ctx)])
        else:
            raise ValueError("unknown protocol %r" % (protocol,))
        out, acc = speculative_round(
            self.target, draft, list(context), rng, self.cost, k=k)
        return out, acc


# ---------------------------------------------------------------------------
# Generation: thin wrappers over P11's lossless round.
# ---------------------------------------------------------------------------

def _round_seed(prompt_index, round_index):
    """Deterministic per-round seed, derived independently
    by both ends from (prompt, round)."""
    return EVAL_SEED + SEED_STRIDE * prompt_index + round_index


def generate_split(client, server, link, prompt, prompt_index,
                   protocol, k=DISCO_K, max_new=MAX_NEW):
    """Lossless speculative generation across the simulated
    split, under ``protocol`` ("disco" or "naive").

    Per round: the client proposes K tokens with the round
    RNG, serialises the request, and sends it over the
    link; the server verifies with P11's loop and the
    client applies the 3-byte reply (accepted count +
    repair/bonus token) to reconstruct the round's
    output. The reconstruction is asserted equal to the
    server's output — the protocol's correctness
    condition. Returns (generated, n_rounds, n_accepted,
    n_examined, target_forwards_delta).
    """
    ctx = list(prompt)
    generated = []
    rounds = 0
    n_accepted = 0
    n_examined = 0
    fwd0 = server.cost.forwards
    while len(generated) < max_new:
        seed = _round_seed(prompt_index, rounds)
        rng = np.random.default_rng(seed)
        tokens, scalars, dists = client.propose(ctx, rng, k)
        payload = client.encode(tokens, scalars, dists, protocol)
        link.send(len(payload))
        out, acc = server.verify(payload, ctx, seed, k, protocol)
        reply = encode_reply(acc, out[-1])
        link.send(len(reply))
        n_acc, bonus = decode_reply(reply)
        take = tokens[:n_acc] + [bonus]
        if take != out:
            raise AssertionError(
                "client reconstruction diverged from server "
                "verification: %r != %r" % (take, out))
        take = take[:max_new - len(generated)]
        generated.extend(take)
        ctx.extend(take)
        rounds += 1
        n_accepted += acc
        # Examined per round: len(out) when a rejection cut
        # the round short (accepted + repair), k when all k
        # draft tokens were accepted (len(out) == k + 1 with
        # the bonus) — P12's exact count, not the P11
        # accepted+rounds approximation.
        n_examined += len(out) if len(out) <= k else k
    return generated, rounds, n_accepted, n_examined, \
        server.cost.forwards - fwd0


def generate_local(target, draft, prompt, prompt_index, cost,
                   k=DISCO_K, max_new=MAX_NEW):
    """The same P11 lossless loop with NO split and NO link:
    ``speculative_round`` chained with the same per-round
    seed schedule. The reference arm the split arms must
    match token-for-token. Returns (generated, n_rounds,
    n_accepted, n_examined, target_forwards_delta)."""
    ctx = list(prompt)
    generated = []
    rounds = 0
    n_accepted = 0
    n_examined = 0
    fwd0 = cost.forwards
    while len(generated) < max_new:
        seed = _round_seed(prompt_index, rounds)
        rng = np.random.default_rng(seed)
        out, acc = speculative_round(
            target, _WireDraft(lambda c: draft.dist(c)), ctx,
            rng, cost, k=k)
        take = out[:max_new - len(generated)]
        generated.extend(take)
        ctx.extend(take)
        rounds += 1
        n_accepted += acc
        n_examined += len(out) if len(out) <= k else k
    return generated, rounds, n_accepted, n_examined, \
        cost.forwards - fwd0


# ---------------------------------------------------------------------------
# Full eval
# ---------------------------------------------------------------------------

def _run_local_arm(target, draft, prompts, repeats,
                   max_new=MAX_NEW):
    """Local (no-split) arm: wall-clock is the mean over
    repeats; generations / forwards / rounds come from
    repeat 0 (identical every repeat — fixed seeds)."""
    runs = []
    for _ in range(repeats):
        cost = CostModel(work_rounds=TARGET_WORK_ROUNDS,
                         draft_cost_rounds=0)
        t0 = time.perf_counter()
        gens = []
        rounds = 0
        accepted = 0
        examined = 0
        forwards = 0
        for pi, prompt in enumerate(prompts):
            gen, r, acc, ex, fwd = generate_local(
                target, draft, prompt, pi, cost, max_new=max_new)
            gens.append(gen)
            rounds += r
            accepted += acc
            examined += ex
            forwards += fwd
        wall = time.perf_counter() - t0
        n_tokens = sum(len(g) for g in gens)
        runs.append({
            "generations": gens,
            "tokens": n_tokens,
            "rounds": rounds,
            "accepted_draft": accepted,
            "examined_draft": examined,
            "target_forwards": forwards,
            "tokens_per_target_forward": n_tokens / max(1, forwards),
            "acceptance_rate": accepted / max(1, examined),
            "compute_wall_sec": wall,
            "link_seconds": 0.0,
            "bytes_sent": 0,
            "messages": 0,
            "total_sec": wall,
            "tokens_per_sec": n_tokens / max(wall, 1e-9),
        })
    agg = dict(runs[0])
    agg["tokens_per_sec"] = float(np.mean(
        [r["tokens_per_sec"] for r in runs]))
    agg["compute_wall_sec"] = float(np.mean(
        [r["compute_wall_sec"] for r in runs]))
    agg["total_sec"] = agg["compute_wall_sec"]
    return agg


def _run_split_arm(target, draft_client, draft_replica, prompts,
                   protocol, link_cfg, repeats, max_new=MAX_NEW):
    """One split arm (one protocol, one link): wall-clock is
    the mean over repeats; the deterministic columns
    (generations, forwards, bytes, messages, link seconds)
    come from repeat 0 (identical every repeat — fixed
    seeds, simulated link). End-to-end time is measured
    compute wall-clock PLUS simulated link seconds."""
    runs = []
    for _ in range(repeats):
        cost = CostModel(work_rounds=TARGET_WORK_ROUNDS,
                         draft_cost_rounds=0)
        link = Link(link_cfg["bandwidth_bytes_per_sec"],
                    link_cfg["per_message_overhead_sec"],
                    name=link_cfg["name"])
        client = DiscoClient(draft_client)
        server = DiscoServer(target, draft_replica, cost)
        t0 = time.perf_counter()
        gens = []
        rounds = 0
        accepted = 0
        examined = 0
        forwards = 0
        for pi, prompt in enumerate(prompts):
            gen, r, acc, ex, fwd = generate_split(
                client, server, link, prompt, pi, protocol,
                max_new=max_new)
            gens.append(gen)
            rounds += r
            accepted += acc
            examined += ex
            forwards += fwd
        wall = time.perf_counter() - t0
        n_tokens = sum(len(g) for g in gens)
        total = wall + link.simulated_seconds
        runs.append({
            "generations": gens,
            "tokens": n_tokens,
            "rounds": rounds,
            "accepted_draft": accepted,
            "examined_draft": examined,
            "target_forwards": forwards,
            "tokens_per_target_forward": n_tokens / max(1, forwards),
            "acceptance_rate": accepted / max(1, examined),
            "compute_wall_sec": wall,
            "link_seconds": link.simulated_seconds,
            "bytes_sent": link.bytes_sent,
            "messages": link.messages,
            "replica_checks": server.replica_checks,
            "replica_mismatches": server.replica_mismatches,
            "total_sec": total,
            "tokens_per_sec": n_tokens / max(total, 1e-9),
        })
    agg = dict(runs[0])
    agg["tokens_per_sec"] = float(np.mean(
        [r["tokens_per_sec"] for r in runs]))
    agg["compute_wall_sec"] = float(np.mean(
        [r["compute_wall_sec"] for r in runs]))
    agg["total_sec"] = agg["compute_wall_sec"] + agg["link_seconds"]
    return agg


def run_disco_eval(repeats=REPEATS, n_prompts=None,
                   max_new=MAX_NEW):
    """Full P13 eval: local (no-split) reference + DISCO vs
    NAIVE on the slow link and on the fast link, same
    prompts, same K, same lossless loop, same target cost.

    ``done`` is the plan's done condition: the DISCO arm's
    output is IDENTICAL to the naive arm's and to the local
    loop's on every prompt, DISCO ships >= 10x fewer bytes
    than naive, and DISCO wins on end-to-end tokens/sec on
    the slow link. The fast link is reported for the
    crossover: there the bandwidth saving should not
    matter, and the table shows whether it does.
    """
    prompts = PROMPT_IDS[:n_prompts] if n_prompts else PROMPT_IDS
    sentences = build_corpus()
    target = TargetModel(sentences)
    draft_client = DraftModel(sentences)
    draft_replica = DraftModel(sentences)

    local = _run_local_arm(target, draft_client, prompts, repeats,
                           max_new=max_new)
    links = []
    for link_cfg in LINK_SETTINGS:
        arms = {}
        for protocol in ("disco", "naive"):
            arms[protocol] = _run_split_arm(
                target, draft_client, draft_replica, prompts,
                protocol, link_cfg, repeats, max_new=max_new)
        disco, naive = arms["disco"], arms["naive"]
        identical = bool(
            disco["generations"] == naive["generations"]
            == local["generations"])
        ratio = naive["bytes_sent"] / max(1, disco["bytes_sent"])
        links.append({
            "name": link_cfg["name"],
            "bandwidth_bytes_per_sec":
                link_cfg["bandwidth_bytes_per_sec"],
            "per_message_overhead_sec":
                link_cfg["per_message_overhead_sec"],
            "arms": arms,
            "outputs_identical": identical,
            "bandwidth_ratio_naive_over_disco": ratio,
            "disco_speedup_vs_naive": (
                disco["tokens_per_sec"]
                / max(naive["tokens_per_sec"], 1e-9)),
            "disco_speedup_vs_local": (
                disco["tokens_per_sec"]
                / max(local["tokens_per_sec"], 1e-9)),
            "naive_speedup_vs_local": (
                naive["tokens_per_sec"]
                / max(local["tokens_per_sec"], 1e-9)),
        })
    slow = links[0]
    done = bool(
        slow["outputs_identical"]
        and slow["bandwidth_ratio_naive_over_disco"] >= 10.0
        and slow["disco_speedup_vs_naive"] > 1.0)
    return {
        "note": NOTE,
        "k": DISCO_K,
        "max_new": max_new,
        "n_prompts": len(prompts),
        "repeats": repeats,
        "target_work_rounds": TARGET_WORK_ROUNDS,
        "eval_seed": EVAL_SEED,
        "disco_request_bytes": disco_request_bytes(),
        "naive_request_bytes": naive_request_bytes(),
        "reply_bytes": reply_bytes(),
        "local": local,
        "links": links,
        "done_identical_and_10x_and_slow_win": done,
    }


def print_disco_report(result):
    print("P13 — DISCO: distributed speculative decoding "
          "(in-process client/server simulation, toy decoder, "
          "NumPy)")
    print("NOTE: %s" % result["note"])
    print("K=%d, max_new=%d, prompts=%d, repeats=%d, target "
          "forward cost=%d simulated work rounds"
          % (result["k"], result["max_new"],
             result["n_prompts"], result["repeats"],
             result["target_work_rounds"]))
    print("Wire format: token index uint16 (2 B), probability "
          "float32 (4 B); DISCO request %d B/round, NAIVE "
          "request %d B/round, reply %d B/round"
          % (result["disco_request_bytes"],
             result["naive_request_bytes"],
             result["reply_bytes"]))
    loc = result["local"]
    print("\nLocal (no split, no link) reference: %.1f tok/s, "
          "%.2f tok/fwd, acceptance %.3f, %d target forwards"
          % (loc["tokens_per_sec"],
             loc["tokens_per_target_forward"],
             loc["acceptance_rate"], loc["target_forwards"]))
    for link in result["links"]:
        print("\n%s — bandwidth %d B/s, overhead %.4f s/msg"
              % (link["name"], link["bandwidth_bytes_per_sec"],
                 link["per_message_overhead_sec"]))
        print("%-12s %10s %10s %10s %12s %10s %10s" % (
            "arm", "tok/sec", "tok/fwd", "accept",
            "bytes", "link s", "compute s"))
        for label in ("disco", "naive"):
            a = link["arms"][label]
            print("%-12s %10.1f %10.2f %10.3f %12d %10.2f %10.2f"
                  % (label, a["tokens_per_sec"],
                     a["tokens_per_target_forward"],
                     a["acceptance_rate"], a["bytes_sent"],
                     a["link_seconds"], a["compute_wall_sec"]))
        print("  bandwidth ratio (naive / DISCO): %.2fx; "
              "DISCO vs naive wall-clock: %.2fx; DISCO vs "
              "local: %.2fx; naive vs local: %.2fx"
              % (link["bandwidth_ratio_naive_over_disco"],
                 link["disco_speedup_vs_naive"],
                 link["disco_speedup_vs_local"],
                 link["naive_speedup_vs_local"]))
        print("  outputs identical (DISCO == naive == local): "
              "%s; replica checks %d, mismatches %d"
              % (link["outputs_identical"],
                 link["arms"]["disco"]["replica_checks"],
                 link["arms"]["disco"]["replica_mismatches"]))
    print("\nDone condition (identical outputs AND >= 10x fewer "
          "bytes AND DISCO wins tok/s on the slow link): %s"
          % result["done_identical_and_10x_and_slow_win"])
