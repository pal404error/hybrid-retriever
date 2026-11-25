"""P13 — DISCO: distributed speculative decoding tests.

Covers: the wire-format byte counts for both protocols
(DISCO 6K bytes/request, NAIVE 2K + 4KV, reply 3 bytes)
and encode/decode round-trips (exact, including the
float32 scalars and distributions); the Link's latency
accounting (bytes / bandwidth + per-message overhead,
exact counters, invalid inputs rejected); the server-side
draft-replica check (replica is a separate object from
the client's draft, its recomputed scalars match the
client-sent ones exactly, a tampered scalar fails
loudly); output identity DISCO == naive == local P11
loop on the same prompts/seeds; the slow-link vs
fast-link tokens/sec comparison (DISCO wins on the slow
link, the bandwidth saving stops mattering on the fast
one); eval determinism; and the reuse pin (verification
is P11's speculative_round, not P10's lossy loop).
"""

import numpy as np
import pytest

import hybrid_retriever.disco as disco
from hybrid_retriever.disco import (
    DISCO_K,
    DiscoClient,
    DiscoServer,
    Link,
    decode_disco_request,
    decode_naive_request,
    decode_reply,
    disco_request_bytes,
    encode_disco_request,
    encode_naive_request,
    encode_reply,
    generate_local,
    generate_split,
    naive_request_bytes,
    reply_bytes,
    run_disco_eval,
    sample_draft_tokens,
)
from hybrid_retriever.specdec import (
    PROMPT_IDS,
    V,
    DraftModel,
    TargetModel,
    build_corpus,
)
from hybrid_retriever.specdec_scratch import (
    CostModel,
    speculative_round,
)


def _models():
    sents = build_corpus()
    return TargetModel(sents), DraftModel(sents), DraftModel(sents)


def _ends():
    target, draft_client, draft_replica = _models()
    cost = CostModel(work_rounds=8, draft_cost_rounds=0)
    client = DiscoClient(draft_client)
    server = DiscoServer(target, draft_replica, cost)
    return client, server, cost


# ---------------------------------------------------------------------------
# Wire format: byte counts + round-trips
# ---------------------------------------------------------------------------

def test_wire_format_byte_counts():
    # K=6, V=64: DISCO 6*2 + 6*4 = 36, NAIVE 6*2 + 6*64*4
    # = 1548, reply 1 + 2 = 3. Per round: 39 vs 1551.
    assert disco_request_bytes(6) == 36
    assert naive_request_bytes(6) == 1548
    assert reply_bytes() == 3
    assert disco_request_bytes(1) == 6
    assert naive_request_bytes(1) == 2 + 4 * V
    assert disco_request_bytes() == disco_request_bytes(DISCO_K)
    assert naive_request_bytes() == naive_request_bytes(DISCO_K)


def test_disco_request_round_trip_exact():
    tokens = [0, 63, 5, 12, 12, 40]
    scalars = [np.float32(0.5), np.float32(0.001),
               np.float32(0.999), np.float32(0.0),
               np.float32(1.0), np.float32(0.123456)]
    payload = encode_disco_request(tokens, scalars)
    assert len(payload) == disco_request_bytes(6)
    rx_tokens, rx_scalars = decode_disco_request(payload, 6)
    assert rx_tokens == tokens
    assert rx_scalars == scalars
    assert all(isinstance(s, np.float32) for s in rx_scalars)


def test_naive_request_round_trip_exact():
    rng = np.random.default_rng(0)
    tokens = [3, 7, 7, 60, 1, 22]
    dists = []
    for _ in range(6):
        d = rng.random(V).astype(np.float32)
        d = d / d.sum()
        dists.append(d)
    payload = encode_naive_request(tokens, dists)
    assert len(payload) == naive_request_bytes(6)
    rx_tokens, rx_dists = decode_naive_request(payload, 6)
    assert rx_tokens == tokens
    for got, want in zip(rx_dists, dists):
        assert got.dtype == np.float32
        assert np.array_equal(got, want)


def test_reply_round_trip_and_bounds():
    for n_acc, tok in ((0, 0), (6, 63), (3, 17)):
        payload = encode_reply(n_acc, tok)
        assert len(payload) == 3
        assert decode_reply(payload) == (n_acc, tok)
    with pytest.raises(ValueError):
        encode_reply(256, 0)


def test_malformed_payloads_rejected():
    with pytest.raises(ValueError):
        decode_disco_request(b"\x00" * 35, 6)
    with pytest.raises(ValueError):
        decode_disco_request(b"\x00" * 37, 6)
    with pytest.raises(ValueError):
        decode_naive_request(b"\x00" * 1547, 6)
    with pytest.raises(ValueError):
        decode_naive_request(b"\x00" * 1549, 6)
    with pytest.raises(ValueError):
        decode_reply(b"\x00\x00")
    with pytest.raises(ValueError):
        decode_reply(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError):
        encode_disco_request([1, 2], [np.float32(0.5)])


# ---------------------------------------------------------------------------
# Link: latency accounting
# ---------------------------------------------------------------------------

def test_link_latency_accounting():
    link = Link(1000, per_message_overhead_sec=0.01, name="t")
    assert link.send(500) == 500
    # 500/1000 + 0.01 = 0.51 s simulated.
    assert link.simulated_seconds == pytest.approx(0.51)
    assert link.bytes_sent == 500
    assert link.messages == 1
    # A zero-byte message still costs the overhead.
    link.send(0)
    assert link.simulated_seconds == pytest.approx(0.52)
    assert link.messages == 2
    link.send(1000)
    assert link.simulated_seconds == pytest.approx(0.52 + 1.01)
    assert link.bytes_sent == 1500
    assert link.messages == 3


def test_link_rejects_bad_inputs():
    with pytest.raises(ValueError):
        Link(0)
    with pytest.raises(ValueError):
        Link(-5)
    with pytest.raises(ValueError):
        Link(1000, per_message_overhead_sec=-0.1)
    link = Link(1000)
    with pytest.raises(ValueError):
        link.send(-1)


def test_link_reset():
    link = Link(1000, per_message_overhead_sec=0.01)
    link.send(100)
    link.reset()
    assert (link.bytes_sent, link.messages,
            link.simulated_seconds) == (0, 0, 0.0)


# ---------------------------------------------------------------------------
# Draft sampling: canonical float32 distributions
# ---------------------------------------------------------------------------

def test_sample_draft_tokens_canonical():
    _, draft, _ = _models()
    rng = np.random.default_rng(1234)
    ctx = PROMPT_IDS[0]
    tokens, scalars, dists = sample_draft_tokens(draft, ctx, rng)
    assert len(tokens) == DISCO_K
    assert len(scalars) == DISCO_K
    assert len(dists) == DISCO_K
    for tok, scal, dist in zip(tokens, scalars, dists):
        assert isinstance(scal, np.float32)
        assert dist.dtype == np.float32
        assert len(dist) == V
        # The scalar is the float32 probability of the
        # sampled token under the shipped distribution.
        assert scal == dist[tok]
    # Deterministic: same seed, same tokens and scalars.
    rng2 = np.random.default_rng(1234)
    tokens2, scalars2, _ = sample_draft_tokens(
        draft, ctx, rng2)
    assert tokens2 == tokens
    assert scalars2 == scalars


# ---------------------------------------------------------------------------
# Replica: separate object, identical recomputation
# ---------------------------------------------------------------------------

def test_replica_is_separate_object_with_identical_dists():
    _, draft_client, draft_replica = _models()
    assert draft_client is not draft_replica
    rng = np.random.default_rng(7)
    for _ in range(20):
        ctx = [int(t) for t in
               rng.integers(0, V, size=int(rng.integers(1, 8)))]
        assert np.array_equal(draft_client.dist(ctx),
                              draft_replica.dist(ctx))


def test_replica_check_passes_and_counts():
    client, server, _cost = _ends()
    prompt = PROMPT_IDS[0]
    link = Link(10_000, 0.001)
    gen, rounds, acc, ex, fwd = generate_split(
        client, server, link, prompt, 0, "disco")
    assert len(gen) == 24
    assert server.replica_mismatches == 0
    # One replica check per proposed token per round.
    assert server.replica_checks == rounds * DISCO_K
    # And the link saw exactly 2 messages per round.
    assert link.messages == 2 * rounds
    assert link.bytes_sent == rounds * (disco_request_bytes()
                                        + reply_bytes())


def test_tampered_scalar_fails_replica_check():
    client, server, _cost = _ends()
    prompt = PROMPT_IDS[0]
    rng = np.random.default_rng(999)
    tokens, scalars, dists = client.propose(prompt, rng)
    scalars = list(scalars)
    scalars[2] = np.float32(float(scalars[2]) + 0.01)
    payload = encode_disco_request(tokens, scalars)
    with pytest.raises(AssertionError):
        server.verify(payload, prompt, 999, protocol="disco")


def test_naive_arm_needs_no_replica_check():
    # The naive server verifies against the received
    # distributions, so it runs with zero replica checks —
    # and still produces output.
    client, server, _cost = _ends()
    prompt = PROMPT_IDS[0]
    link = Link(10_000, 0.001)
    gen, rounds, acc, ex, fwd = generate_split(
        client, server, link, prompt, 0, "naive")
    assert len(gen) == 24
    assert server.replica_checks == 0
    assert link.bytes_sent == rounds * (naive_request_bytes()
                                        + reply_bytes())


# ---------------------------------------------------------------------------
# Output identity: DISCO == naive == local P11 loop
# ---------------------------------------------------------------------------

def test_output_identity_across_protocols_and_local():
    target, draft_client, draft_replica = _models()
    prompts = PROMPT_IDS
    # Local reference.
    cost = CostModel(work_rounds=8, draft_cost_rounds=0)
    local_gens = []
    for pi, prompt in enumerate(prompts):
        gen, _, _, _, _ = generate_local(
            target, draft_client, prompt, pi, cost)
        local_gens.append(gen)
    for protocol in ("disco", "naive"):
        cost = CostModel(work_rounds=8, draft_cost_rounds=0)
        link = Link(10_000, 0.001)
        client = DiscoClient(draft_client)
        server = DiscoServer(target, draft_replica, cost)
        gens = []
        for pi, prompt in enumerate(prompts):
            gen, _, _, _, _ = generate_split(
                client, server, link, prompt, pi, protocol)
            gens.append(gen)
        assert gens == local_gens, protocol
    # The two split arms also agree with each other
    # (implied by both equalling local, asserted directly
    # in the eval test below).


def test_eval_outputs_identical_and_ratio():
    result = run_disco_eval(repeats=1, n_prompts=8)
    for link in result["links"]:
        assert link["outputs_identical"] is True
        assert link["arms"]["disco"]["generations"] \
            == link["arms"]["naive"]["generations"] \
            == result["local"]["generations"]
        # Same rounds -> same byte ratio on both links.
        assert link["bandwidth_ratio_naive_over_disco"] \
            == pytest.approx(1551 / 39)
        assert link["bandwidth_ratio_naive_over_disco"] >= 10.0
        # Same rounds and forwards in every arm.
        assert link["arms"]["disco"]["rounds"] \
            == link["arms"]["naive"]["rounds"] \
            == result["local"]["rounds"]
        assert link["arms"]["disco"]["target_forwards"] \
            == result["local"]["target_forwards"]
    assert result["links"][0]["bandwidth_ratio_naive_over_disco"] \
        == result["links"][1]["bandwidth_ratio_naive_over_disco"]


# ---------------------------------------------------------------------------
# Slow link vs fast link: tokens/sec + crossover
# ---------------------------------------------------------------------------

def test_slow_link_disco_wins_fast_link_crossover():
    result = run_disco_eval(repeats=1)
    slow, fast = result["links"]
    # Bytes: the saving is link-independent.
    assert slow["arms"]["disco"]["bytes_sent"] \
        == fast["arms"]["disco"]["bytes_sent"]
    assert slow["arms"]["naive"]["bytes_sent"] \
        == fast["arms"]["naive"]["bytes_sent"]
    # Slow link: DISCO wins end-to-end by a wide margin —
    # the link time dominates and DISCO ships ~40x fewer
    # bytes.
    assert slow["disco_speedup_vs_naive"] > 1.0
    assert slow["arms"]["naive"]["link_seconds"] \
        > 10 * slow["arms"]["disco"]["link_seconds"]
    # Fast link: the bandwidth saving stops mattering —
    # the speedup collapses toward 1 and both arms sit
    # close to the local (no-link) rate, because compute
    # dominates.
    assert fast["disco_speedup_vs_naive"] \
        < slow["disco_speedup_vs_naive"]
    assert fast["disco_speedup_vs_naive"] < 1.5
    assert fast["arms"]["disco"]["tokens_per_sec"] \
        >= 0.5 * result["local"]["tokens_per_sec"]
    assert fast["arms"]["naive"]["tokens_per_sec"] \
        >= 0.5 * result["local"]["tokens_per_sec"]
    # The done flag is exactly the conjunction it claims.
    assert result["done_identical_and_10x_and_slow_win"] == bool(
        slow["outputs_identical"]
        and slow["bandwidth_ratio_naive_over_disco"] >= 10.0
        and slow["disco_speedup_vs_naive"] > 1.0)
    assert result["done_identical_and_10x_and_slow_win"] is True


def test_eval_deterministic_columns_stable():
    r1 = run_disco_eval(repeats=1, n_prompts=8)
    r2 = run_disco_eval(repeats=1, n_prompts=8)
    assert r1["local"]["generations"] == r2["local"]["generations"]
    for l1, l2 in zip(r1["links"], r2["links"]):
        for label in ("disco", "naive"):
            assert l1["arms"][label]["tokens"] \
                == l2["arms"][label]["tokens"]
            assert l1["arms"][label]["target_forwards"] \
                == l2["arms"][label]["target_forwards"]
            assert l1["arms"][label]["bytes_sent"] \
                == l2["arms"][label]["bytes_sent"]
            assert l1["arms"][label]["link_seconds"] \
                == l2["arms"][label]["link_seconds"]


# ---------------------------------------------------------------------------
# Reuse pin: verification is P11's loop, not P10's
# ---------------------------------------------------------------------------

def test_verification_is_p11_lossless_round():
    assert disco.speculative_round is speculative_round
    # P10's lossy speculative loop is NOT imported.
    assert not hasattr(disco, "generate_speculative")
    assert not hasattr(disco, "verify_tree")
    assert not hasattr(disco, "margin_accept")
    assert not hasattr(disco, "TreePolicy")
    # And the loop is actually called during a split round:
    # wrap it and count.
    calls = {"n": 0}
    orig = disco.speculative_round

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    disco.speculative_round = counting
    try:
        client, server, _cost = _ends()
        link = Link(10_000, 0.001)
        generate_split(client, server, link, PROMPT_IDS[0], 0,
                       "disco")
    finally:
        disco.speculative_round = orig
    assert calls["n"] > 0
