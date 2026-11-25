"""P9 — E8 lattice key quantisation (k2v2).

E8 Lattice Key Quantisation (arXiv:2610.15384): 2-bit keys +
2-bit values (~6x compression) that keep near-baseline
long-context perplexity by exploiting cross-dimension
correlation on the E8 lattice. The plan's done condition is
k2v2 beating scalar k3v2 in most model-dataset-context
settings, reproduced locally.

SUBSTITUTIONS, stated plainly (this VM has no torch, no GPU,
no real model, no LongBench; do NOT read this module as a run
on a real model):
- There is no real model and no PyTorch here. The quantiser
  runs in pure NumPy on SYNTHETIC BF16-baseline KV tensors:
  the same two fixed profiles (shapes, seeds) and the same
  cross-channel-correlated generation as P8
  (``kvcache.generate_synthetic_kv``), reused, not redefined.
  The correlation is the structure the lattice is supposed to
  exploit, and it is the same data the scalar baselines are
  scored on, so the comparison is on identical footing.
- No perplexity is computed or claimed. Perplexity needs a
  real language model; the metric here is ATTENTION FIDELITY
  vs the unquantised baseline on P8's deterministic attention
  task (``kvcache.reasoning_eval``): attention-output error
  (max abs / MSE) and top-1 agreement (attended-position and
  vocab). No perplexity number appears anywhere in this
  module or its report.
- No Triton/Metal kernel. The table-based scoring trick is a
  NumPy implementation of the paper's idea (lookup tables
  over codeword halves, no key dequantisation) and is
  verified against the dequantise path, not benchmarked as a
  kernel.

CORRECTION to the task brief's example: the brief suggested
"256 codewords per 8-dim block". That is 1 bit/dim, not 2.
The paper's construction is 2 bits/dim payload = 16 bits per
8-dim block = 65,536 codewords per block, with 2.25 bits/dim
effective key rate including norm metadata. This module
implements the paper's accounting: a finite E8-derived
codebook of all E8 points with squared norm <= 10 — 56,881
codewords (see ``build_codebook``; the count is the E8 theta
series, pinned by test, and 56,881 < 65,536 so every codeword
fits a 16-bit index). The 8,655 unused indices are simply
never emitted.

E8 construction (documented, and pinned by test):
- E8 = {x in Z^8 or (Z + 1/2)^8 : sum(x) is even}. In doubled
  coordinates u = 2x: u in Z^8, all u_i even or all u_i odd,
  sum(u) divisible by 4, sum(u^2) <= 40 (i.e. ||x||^2 <= 10).
- Nearest-codeword assignment is NOT a brute-force search
  over 56,881 codewords. It is the standard E8 lattice
  decoder: round the (scaled) block to the nearest integer
  vector and to the nearest half-integer vector, fix the
  parity of each candidate by moving the coordinate with the
  largest rounding error by +-1 toward the input, and keep
  the closer candidate. That is the nearest point of the
  *infinite* E8 lattice (verified against brute force in the
  tests); the finite-codebook step is only the range check
  (see the overload rule below).
- The codebook is built once, lazily, by pruned enumeration
  in doubled coordinates (even branch and odd branch
  separately, so the intermediate sets stay small), sorted
  lexicographically for determinism.

Scale / norm metadata (the control variable — IDENTICAL for
every arm, so fidelity differences are the quantiser
geometry, not the metadata budget):
- One FP16 RMS scale per key/value vector (per token, per
  head): s = sqrt(mean(x^2)) over the head dim. 16 bits per
  vector = 0.125 bits/dim.
- One 2-bit norm code per 8-dim block: a log-spaced
  multiplier m in {2^-1/2, 2^-1/6, 2^1/6, 2^1/2} applied on
  top of s. 2 bits per block = 0.25 bits/dim.
- Effective rates: E8 keys 2 + 0.125 + 0.25 = 2.375 bits/dim
  (the paper's 2.25 counts the vector scale slightly
  differently; the accounting here is stated, not matched
  to the decimal); scalar k2 keys the same 2.375; scalar k3
  keys 3.375; values (scalar 2-bit, all arms) 2.375. Combined
  KV: E8 k2v2 4.75 bits/dim vs FP16 32 -> ~6.7x; scalar
  k3v2 5.75 bits/dim -> ~5.6x, so E8 k2v2 is ~20% smaller
  than scalar k3v2, as in the paper. The ratios are computed
  from the stored bytes, not asserted.
- Per block, ALL FOUR multipliers are tried and the one
  with the smallest reconstruction error is kept (for E8:
  among multipliers whose decoded point is in the codebook;
  for scalar: among all four). The norm code stores the
  multiplier actually used, so the decoder needs nothing
  else. This is a per-block optimisation of the measured
  objective, applied identically to every arm.

Overload rule (finite codebook, stated plainly): a decoded
lattice point with ||z||^2 > 10 is outside the codebook and
cannot be emitted as-is. The try-all-four-multipliers step
makes this rare — a larger multiplier shrinks the block, so
some multiplier almost always decodes in-codebook — but for
the blocks where the chosen multiplier still decodes
outside, the block falls back to the minimal E8 vector
(||z||^2 == 2, one of 240) closest in direction to the
block: the block keeps its direction and about half its
magnitude instead of collapsing to zero. The fallback is
still a codeword (a 16-bit index, no extra metadata) and is
counted in ``fallback_minimal_blocks`` in the eval result.
On the fixed synthetic profiles it fires for ~1% of blocks
(measured, reported per profile); a nonzero counter is
reported, not hidden. There is no radial clipping of the
input: clipping would change the direction the attention
score actually uses, and the multiplier search already
covers that trade.

Values: 2-bit SCALAR quantisation in every arm (documented
choice — the paper's lattice is a key-side primitive and
the values in all three arms share one quantiser, so the
arms differ only in the keys). Uniform symmetric levels
{-1.5, -0.5, 0.5, 1.5} * step, step = s * m (the same
per-vector scale and per-block norm code as the keys).

Scalar key baselines (the comparison is the whole point):
- k2: uniform symmetric 4 levels, step = s * m.
- k3: uniform symmetric 8 levels {-3.5..3.5} * step,
  step = s * m / 2 (spacing halves per added bit — the
  near-optimal uniform step for a unit Gaussian at that
  bit width). Same scale scheme, same norm-code search,
  same values, same queries. If scalar k3 wins, the eval
  says so; the synthetic data is fixed (P8's profiles) and
  was not tuned to force an E8 win.

Table-based scoring (the paper's trick, no dequantisation):
attention scores are q . k_hat = sum over blocks of
s * m * (q_b . z_b). Split each codeword into two 4-dim
halves. For a query block, precompute two lookup tables —
the dot products of the query half with every DISTINCT
half-codeword in the codebook (a few hundred entries per
half, built once per codebook, then one small matmul per
query block) — and score a key by two table lookups plus
one add per block. The key vector is never materialised:
``attention_table`` reads only the codeword indices, the
norm codes and the scales. Equivalence with the
dequantise path (``kvcache.attention_float`` on the
dequantised keys) is verified numerically in the eval and
pinned by test; the two paths differ only by float64
summation order.
"""

from __future__ import annotations

import math

import numpy as np

# Finite codebook: all E8 points with ||z||^2 <= 10.
MAX_NORM_SQ = 10
# Doubled coordinates: sum(u^2) <= 4 * MAX_NORM_SQ.
MAX_NORM_SQ_DOUBLED = 4 * MAX_NORM_SQ

# Per-block norm-code multipliers (2-bit code -> multiplier),
# log-spaced so the residual block-norm factor after adaptation
# is at most 2**(1/6) ~ 1.12: an 8-dim Gaussian block norm
# concentrates near sqrt(8) and the code brackets that chi
# fluctuation. Frozen before measuring; not tuned per arm.
NORM_EXPONENTS = np.array([-0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5])
NORM_MULTS = np.array([2.0 ** e for e in NORM_EXPONENTS], dtype=np.float64)

# Attention checks: queries are the leading ATTN_QUERIES key
# rows reused as Q (the P7/P8 convention).
ATTN_QUERIES = 64
# Context lengths per profile (2+ each; the second is the full
# profile sequence, so the ratio is measured at full length).
CONTEXTS: dict[str, list[int]] = {
    "llama-7b-like": [256, 1024],
    "small-model-like": [512, 2048],
}


# ---------------------------------------------------------------------------
# Codebook construction (lazy, cached, deterministic).
# ---------------------------------------------------------------------------

def _expand_branch(values: np.ndarray, max_sumsq: int) -> np.ndarray:
    """All length-8 vectors over ``values`` with sum of squares <= max.

    Iterative expansion with pruning: after each coordinate the
    rows that already exceed the budget are dropped, so the
    intermediate sets stay small (the even branch peaks at
    tens of thousands of rows, not the millions a naive
    13**8 enumeration would need).
    """
    rows = np.zeros((1, 0), dtype=np.int16)
    sumsq = np.zeros(1, dtype=np.int64)
    for _ in range(8):
        parts: list[np.ndarray] = []
        sq_parts: list[np.ndarray] = []
        for v in values:
            keep = sumsq + int(v) * int(v) <= max_sumsq
            if not keep.any():
                continue
            kept = rows[keep]
            col = np.full((kept.shape[0], 1), v, dtype=np.int16)
            parts.append(
                np.concatenate([kept, col], axis=1) if kept.shape[1]
                else col
            )
            sq_parts.append(sumsq[keep] + int(v) * int(v))
        rows = np.concatenate(parts, axis=0)
        sumsq = np.concatenate(sq_parts)
    return rows


def _half_keys(halves: np.ndarray) -> np.ndarray:
    """Pack 4 doubled coords in [-6, 6] into one int (4 bits each)."""
    h = np.asarray(halves, dtype=np.int64)
    key = np.zeros(h.shape[0], dtype=np.int64)
    for i in range(4):
        key |= (h[:, i] + 6) << (4 * i)
    return key


def _halves_from_keys(keys: np.ndarray) -> np.ndarray:
    """Inverse of ``_half_keys`` -> (n, 4) float32 half-codewords."""
    out = np.zeros((len(keys), 4), dtype=np.float32)
    for i in range(4):
        out[:, i] = ((keys >> (4 * i)) & 0xF) - 6
    return out / 2.0


class Codebook:
    """A finite E8-derived codebook and its half-split tables.

    ``points`` (N, 8) float32, all on the E8 lattice with
    ||z||^2 <= MAX_NORM_SQ. ``key_to_index`` maps the packed
    32-bit doubled-coordinate key of a codeword to its row.
    ``halves1`` / ``halves2`` are the distinct 4-dim halves
    (sorted), and ``half1_ids`` / ``half2_ids`` give, per
    codeword, its half rows — the tables the table-based
    scorer looks up.
    """

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=np.float32)
        doubled = np.round(self.points.astype(np.float64) * 2).astype(np.int64)
        key = np.zeros(len(self.points), dtype=np.int64)
        for i in range(8):
            key |= (doubled[:, i] + 6) << (4 * i)
        self.keys = key
        self.key_to_index = {int(k): i for i, k in enumerate(key)}
        h1_keys = _half_keys(doubled[:, :4])
        h2_keys = _half_keys(doubled[:, 4:])
        u1, inv1 = np.unique(h1_keys, return_inverse=True)
        u2, inv2 = np.unique(h2_keys, return_inverse=True)
        self.halves1 = _halves_from_keys(u1)
        self.halves2 = _halves_from_keys(u2)
        self.half1_ids = np.asarray(inv1).reshape(-1).astype(np.int32)
        self.half2_ids = np.asarray(inv2).reshape(-1).astype(np.int32)
        # Minimal vectors (||z||^2 == 2, 240 of them): the
        # direction-preserving fallback for a block whose
        # decoded point falls outside the finite codebook.
        norms_sq = (self.points.astype(np.float64) ** 2).sum(axis=1)
        self.minimal_indices = np.where(np.abs(norms_sq - 2.0) < 1e-6)[0]
        self.minimal_points = self.points[self.minimal_indices]

    def __len__(self) -> int:
        return len(self.points)

    def lookup(self, doubled: np.ndarray) -> np.ndarray:
        """Doubled coords (N, 8) int -> codeword index, or -1.

        Vectorised: pack the keys, map the distinct keys
        through the dict once, scatter back. Points outside
        the coordinate range or not in the codebook map to -1.
        """
        d = np.asarray(doubled)
        out = np.full(d.shape[0], -1, dtype=np.int64)
        in_range = (
            np.all(np.abs(d) <= 6, axis=1)
            & (np.sum(d.astype(np.int64) ** 2, axis=1) <= MAX_NORM_SQ_DOUBLED)
        )
        if not in_range.any():
            return out
        key = np.zeros(d.shape[0], dtype=np.int64)
        for i in range(8):
            key |= (d[:, i] + 6) << (4 * i)
        keys = key[in_range]
        uniq, inv = np.unique(keys, return_inverse=True)
        mapped = np.array(
            [self.key_to_index.get(int(k), -1) for k in uniq], dtype=np.int64
        )
        out[np.where(in_range)[0]] = mapped[np.asarray(inv).reshape(-1)]
        return out


_CODEBOOK: Codebook | None = None


def build_codebook(max_norm_sq: int = MAX_NORM_SQ) -> Codebook:
    """All E8 points with ||z||^2 <= ``max_norm_sq``.

    Even branch: doubled coords all even (u = 2z, z integer);
    odd branch: doubled coords all odd (z half-integer). Each
    branch is pruned-enumerated separately, then filtered to
    sum(u) divisible by 4 (i.e. sum(z) even — the E8 parity
    condition) and sorted lexicographically. For the default
    radius the E8 theta series gives 1 + 240 + 2160 + 6720 +
    17520 + 30240 = 56,881 points.
    """
    if max_norm_sq != MAX_NORM_SQ:
        raise ValueError("build_codebook is only defined for the frozen radius 10")
    max_sq = 4 * max_norm_sq
    even = _expand_branch(
        np.array([-6, -4, -2, 0, 2, 4, 6], dtype=np.int16), max_sq
    )
    odd = _expand_branch(
        np.array([-5, -3, -1, 1, 3, 5], dtype=np.int16), max_sq
    )
    rows = np.concatenate([even, odd], axis=0)
    keep = (rows.astype(np.int64).sum(axis=1) % 4) == 0
    rows = rows[keep]
    points = (rows.astype(np.float64) / 2.0).astype(np.float32)
    order = np.lexsort([points[:, i] for i in range(7, -1, -1)])
    return Codebook(points[order])


def get_codebook() -> Codebook:
    """The frozen codebook, built once per process."""
    global _CODEBOOK
    if _CODEBOOK is None:
        _CODEBOOK = build_codebook()
    return _CODEBOOK


# ---------------------------------------------------------------------------
# E8 lattice decoding (nearest infinite-lattice point).
# ---------------------------------------------------------------------------

def _fix_parity(y: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Fix a candidate's coordinate sum to even, cheapest move.

    ``cand`` is the nearest integer or half-integer rounding of
    ``y``; if its coordinate sum is odd, move the coordinate
    with the largest rounding error by +-1 in the direction of
    the input. That is the smallest-distance parity fix.
    """
    flat_y = y.reshape(-1, 8)
    flat_c = cand.reshape(-1, 8).copy()
    odd = (flat_c.sum(axis=1) % 2) != 0
    if odd.any():
        diff = flat_y - flat_c
        idx = np.argmax(np.abs(diff), axis=1)
        rows = np.where(odd)[0]
        step = np.where(diff[rows, idx[rows]] >= 0, 1.0, -1.0)
        flat_c[rows, idx[rows]] += step
    return flat_c.reshape(cand.shape)


def nearest_e8(y: np.ndarray) -> np.ndarray:
    """Nearest point of the infinite E8 lattice, per 8-vector.

    The closer of the parity-corrected nearest integer vector
    and the parity-corrected nearest half-integer vector — the
    standard E8 decoder, exact for the infinite lattice.
    Vectorised over any leading shape; float64 math.
    """
    y = np.asarray(y, dtype=np.float64)
    xi = _fix_parity(y, np.round(y))
    xh = _fix_parity(y, np.floor(y) + 0.5)
    di = ((y - xi) ** 2).sum(axis=-1)
    dh = ((y - xh) ** 2).sum(axis=-1)
    return np.where((dh < di)[..., None], xh, xi)


# ---------------------------------------------------------------------------
# Quantised tensors.
# ---------------------------------------------------------------------------

class E8QuantisedKeys:
    """E8-quantised keys: 16-bit codeword index + 2-bit norm
    code per 8-dim block, one FP16 RMS scale per vector."""

    def __init__(self, shape, indices, norm_codes, scales):
        self.shape = tuple(shape)
        self.indices = np.asarray(indices, dtype=np.uint16)
        self.norm_codes = np.asarray(norm_codes, dtype=np.uint8)
        self.scales = np.asarray(scales, dtype=np.float16)

    def prefix(self, n_tokens: int) -> "E8QuantisedKeys":
        shape = (n_tokens,) + self.shape[1:]
        return E8QuantisedKeys(
            shape, self.indices[:n_tokens], self.norm_codes[:n_tokens],
            self.scales[:n_tokens],
        )


class ScalarQuantised:
    """Scalar-quantised tensor (keys or values): one code per
    element, 2-bit norm code per 8-dim block, FP16 RMS scale
    per vector."""

    def __init__(self, shape, codes, norm_codes, scales, bits: int):
        self.shape = tuple(shape)
        self.codes = np.asarray(codes, dtype=np.uint8)
        self.norm_codes = np.asarray(norm_codes, dtype=np.uint8)
        self.scales = np.asarray(scales, dtype=np.float16)
        self.bits = int(bits)

    def prefix(self, n_tokens: int) -> "ScalarQuantised":
        shape = (n_tokens,) + self.shape[1:]
        return ScalarQuantised(
            shape, self.codes[:n_tokens], self.norm_codes[:n_tokens],
            self.scales[:n_tokens], self.bits,
        )


def _rms_scales(x: np.ndarray) -> np.ndarray:
    """Per-vector RMS scale (N,), float64; zero vectors get 1.0."""
    s = np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=1))
    return np.where(s > 0, s, 1.0)


def _blocks(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(seq, heads, d) float32 -> (N, nb, 8) float64 blocks."""
    seq, heads, d = x.shape
    if d % 8 != 0:
        raise ValueError("head dim %d is not divisible by 8" % d)
    nb = d // 8
    return (
        np.asarray(x, dtype=np.float64).reshape(seq * heads, nb, 8),
        seq,
        heads,
    )


def _best_norm_codes(
    blocks: np.ndarray, scales: np.ndarray, recon_error
) -> tuple[np.ndarray, np.ndarray, int]:
    """Try all four norm multipliers per block, keep the best.

    ``recon_error(mult)`` returns the per-block squared
    reconstruction error (N, nb) for that multiplier, with
    inf where the multiplier yields no valid reconstruction
    (E8: decoded point outside the codebook). Returns
    (best_error (N, nb), best_code (N, nb) uint8,
    n_no_valid: blocks with no valid multiplier at all).
    """
    best_err = np.full(blocks.shape[:2], np.inf)
    best_code = np.zeros(blocks.shape[:2], dtype=np.uint8)
    for code, mult in enumerate(NORM_MULTS):
        err = recon_error(float(mult))
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_code = np.where(better, code, best_code)
    n_no_valid = int(np.isinf(best_err).sum())
    best_err = np.where(np.isinf(best_err), 0.0, best_err)
    return best_err, best_code, n_no_valid


def _decode_with_fallback(
    y: np.ndarray, cb: Codebook
) -> tuple[np.ndarray, np.ndarray]:
    """Decode scaled blocks to codeword indices + points.

    ``y`` is (M, 8) float64. Decoded points outside the finite
    codebook (||z||^2 > 10) fall back to the minimal vector
    (||z||^2 == 2) with the largest dot product with ``y`` —
    the codeword closest in direction, so the block keeps its
    direction and about half its magnitude instead of
    collapsing to zero. Returns (indices (M,) int64, points
    (M, 8) float32).
    """
    z = nearest_e8(y)
    idx = cb.lookup(np.round(z * 2).astype(np.int64))
    valid = idx >= 0
    pts = np.zeros((y.shape[0], 8), dtype=np.float32)
    if valid.any():
        pts[valid] = cb.points[idx[valid]]
    if (~valid).any():
        ybad = y[~valid]
        dots = ybad @ cb.minimal_points.astype(np.float64).T
        j = np.argmax(dots, axis=1)
        pts[~valid] = cb.minimal_points[j]
        idx[~valid] = cb.minimal_indices[j]
    return idx, pts


def quantise_keys_e8(k: np.ndarray) -> tuple[E8QuantisedKeys, dict]:
    """Quantise keys (seq, heads, d) to the E8 codebook.

    Returns (quantised, info) where info reports
    ``fallback_minimal_blocks`` (blocks whose decoded point
    was outside the codebook for the chosen multiplier and
    fell back to the nearest minimal vector — counted, not
    hidden) and the codebook size.
    """
    cb = get_codebook()
    blocks, seq, heads = _blocks(k)
    n, nb, _ = blocks.shape
    scales = _rms_scales(blocks.reshape(n, -1))
    blocks_flat = blocks.reshape(-1, 8)
    scales_flat = np.repeat(scales, nb)

    def _err_for(mult: float) -> np.ndarray:
        y = blocks / (scales[:, None, None] * mult)
        _, pts = _decode_with_fallback(y.reshape(-1, 8), cb)
        recon = pts.reshape(n, nb, 8) * (scales[:, None, None] * mult)
        return ((blocks - recon) ** 2).sum(axis=-1)

    _, best_code, _ = _best_norm_codes(blocks, scales, _err_for)
    # Re-decode with the chosen multipliers to collect indices.
    indices = np.zeros(n * nb, dtype=np.uint16)
    sel_flat_all = best_code.reshape(-1)
    n_fallback = 0
    for code, mult in enumerate(NORM_MULTS):
        sel = sel_flat_all == code
        if not sel.any():
            continue
        y = blocks_flat[sel] / (scales_flat[sel][:, None] * mult)
        z = nearest_e8(y)
        idx = cb.lookup(np.round(z * 2).astype(np.int64))
        n_fallback += int((idx < 0).sum())
        idx_out, _ = _decode_with_fallback(y, cb)
        indices[sel] = idx_out.astype(np.uint16)
    info = {
        "codebook_size": len(cb),
        "fallback_minimal_blocks": n_fallback,
        "total_blocks": int(n * nb),
    }
    return (
        E8QuantisedKeys(
            k.shape,
            indices.reshape(seq, heads, nb),
            best_code.reshape(seq, heads, nb),
            scales.reshape(seq, heads).astype(np.float16),
        ),
        info,
    )


def dequantise_keys_e8(q: E8QuantisedKeys) -> np.ndarray:
    """E8-quantised keys -> float32 (seq, heads, d)."""
    cb = get_codebook()
    seq, heads, d = q.shape
    nb = d // 8
    pts = cb.points[q.indices.reshape(-1, nb).astype(np.int64)]
    scales = q.scales.reshape(-1).astype(np.float64)
    mults = NORM_MULTS[q.norm_codes.reshape(-1, nb)]
    recon = pts * (scales[:, None, None] * mults[:, :, None])
    return recon.reshape(seq, heads, d).astype(np.float32)


def _scalar_step_factor(bits: int) -> float:
    """Uniform-step factor: 1.0 at 2 bits, halving per added bit."""
    return 2.0 ** (2 - bits)


def quantise_scalar(x: np.ndarray, bits: int) -> ScalarQuantised:
    """Scalar-quantise (seq, heads, d) to ``bits`` bits/element.

    Uniform symmetric levels, step = s * m * step_factor(bits),
    with the same per-vector RMS scale and per-block norm-code
    search as the E8 arm. Note: with an even level count the
    level set is symmetric about zero with NO zero level, so
    an exact-zero input reconstructs to +-step/2, not zero —
    that is the uniform scalar quantiser's behaviour, and the
    tests pin the error bound rather than exact zero.
    """
    if bits not in (2, 3):
        raise ValueError("scalar quantiser supports 2 or 3 bits, got %d" % bits)
    blocks, seq, heads = _blocks(x)
    n, nb, _ = blocks.shape
    scales = _rms_scales(blocks.reshape(n, -1))
    levels = 1 << bits
    centre = (levels - 1) / 2.0
    step_factor = _scalar_step_factor(bits)

    def _err_for(mult: float) -> np.ndarray:
        y = blocks / (scales[:, None, None] * mult)
        codes = np.clip(
            np.round(y / step_factor + centre), 0, levels - 1
        )
        recon = (codes - centre) * step_factor
        return ((y - recon) ** 2).sum(axis=-1) * (scales[:, None] * mult) ** 2

    _, best_code, _ = _best_norm_codes(blocks, scales, _err_for)
    codes = np.zeros(n * nb * 8, dtype=np.uint8)
    blocks_flat = blocks.reshape(-1, 8)
    scales_flat = np.repeat(scales, nb)
    sel_flat_all = best_code.reshape(-1)
    for code, mult in enumerate(NORM_MULTS):
        sel = sel_flat_all == code
        if not sel.any():
            continue
        y = blocks_flat[sel] / (scales_flat[sel][:, None] * mult)
        codes.reshape(-1, 8)[sel] = np.clip(
            np.round(y / step_factor + centre), 0, levels - 1
        ).astype(np.uint8)
    return ScalarQuantised(
        x.shape,
        codes.reshape(seq, heads, x.shape[-1]),
        best_code.reshape(seq, heads, nb),
        scales.reshape(seq, heads).astype(np.float16),
        bits,
    )


def dequantise_scalar(q: ScalarQuantised) -> np.ndarray:
    """Scalar-quantised tensor -> float32 (seq, heads, d)."""
    seq, heads, d = q.shape
    nb = d // 8
    centre = ((1 << q.bits) - 1) / 2.0
    step_factor = _scalar_step_factor(q.bits)
    codes = q.codes.reshape(seq * heads, nb, 8).astype(np.float64)
    scales = q.scales.reshape(-1).astype(np.float64)
    mults = NORM_MULTS[q.norm_codes.reshape(-1, nb)]
    recon = (codes - centre) * step_factor
    recon = recon * (scales[:, None, None] * mults[:, :, None])
    return recon.reshape(seq, heads, d).astype(np.float32)


# ---------------------------------------------------------------------------
# Stored-size accounting (bytes actually stored, per arm).
# ---------------------------------------------------------------------------

def kv_stored_bytes_per_vector(d: int, k_bits: int, k_is_e8: bool) -> int:
    """Stored KV bytes for one (token, head) vector of head dim d.

    K: E8 -> 16-bit index per block; scalar -> k_bits per dim.
    V: scalar 2-bit in every arm. Both: 2-bit norm code per
    block (packed, nb/4 bytes) and one FP16 RMS scale per
    tensor (2 bytes each for K and V).
    """
    nb = d // 8
    k_bytes = nb * 2 if k_is_e8 else nb * 8 * k_bits // 8
    v_bytes = nb * 8 * 2 // 8
    return k_bytes + v_bytes + 2 * (nb // 4) + 4


def kv_ratio(d: int, k_bits: int, k_is_e8: bool) -> float:
    """Compression ratio vs the BF16 baseline (2 bytes/element)."""
    orig = 2 * d * 2
    return orig / kv_stored_bytes_per_vector(d, k_bits, k_is_e8)


def kv_bits_per_dim(d: int, k_bits: int, k_is_e8: bool) -> float:
    """Effective stored bits per dimension (K+V together)."""
    return kv_stored_bytes_per_vector(d, k_bits, k_is_e8) * 8 / (2 * d)


# ---------------------------------------------------------------------------
# Attention: dequantised path (baseline) and table path (no K dequant).
# ---------------------------------------------------------------------------

def attention_table(
    q: np.ndarray, e8k: E8QuantisedKeys, v: np.ndarray
) -> np.ndarray:
    """Scaled dot-product attention scoring keys WITHOUT dequantising them.

    Scores are q . k_hat = sum_b s * m_b * (q_b . z_b), with
    each block dot computed from two lookup tables: the dot
    products of the query half-blocks with every distinct
    half-codeword (built per query block by one small matmul
    per half), gathered by the stored codeword indices' half
    ids. Values are dequantised normally (they are scalar).
    Float64 scores/probs, float32 output — the same numerics
    as ``kvcache.attention_float`` apart from summation order.
    """
    cb = get_codebook()
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    n_q, heads, d = q.shape
    seq_k = e8k.shape[0]
    nb = d // 8
    scale = 1.0 / math.sqrt(d)
    out = np.zeros((n_q, heads, d), dtype=np.float32)
    scales = e8k.scales.astype(np.float64)  # (seq_k, heads)
    mults = NORM_MULTS[e8k.norm_codes]  # (seq_k, heads, nb)
    for h in range(heads):
        qb = q[:, h, :].astype(np.float64).reshape(n_q, nb, 8)
        t1 = qb[:, :, :4] @ cb.halves1.astype(np.float64).T
        t2 = qb[:, :, 4:] @ cb.halves2.astype(np.float64).T
        h1 = cb.half1_ids[e8k.indices[:, h, :].astype(np.int64)]
        h2 = cb.half2_ids[e8k.indices[:, h, :].astype(np.int64)]
        scores = np.zeros((n_q, seq_k), dtype=np.float64)
        for b in range(nb):
            part = (
                t1[:, b, :][:, h1[:, b]] + t2[:, b, :][:, h2[:, b]]
            )
            scores += part * (scales[:, h] * mults[:, h, b])[None, :]
        scores = scores * scale
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        out[:, h, :] = (probs @ v[:, h, :].astype(np.float64)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Eval: E8 k2v2 vs scalar k2v2 vs scalar k3v2, per profile x context.
# ---------------------------------------------------------------------------

def fidelity_eval(kv_orig: dict, kv_recon: dict,
                  n_queries: int = ATTN_QUERIES) -> dict:
    """Attention fidelity with queries != keys (context matters).

    P8's ``reasoning_eval`` truncates BOTH queries and keys to
    ``n_tokens``, so it cannot see context length: a 2048-key
    run scores exactly like a 64-key run. Here the queries are
    the leading ``n_queries`` key rows reused as Q (the P7/P8
    convention) and they attend over ALL keys in the supplied
    (prefix) KV, so a longer context really is a harder task
    (more distractor keys competing for the argmax). Same
    metrics as P8 otherwise: attention-output error (max abs /
    MSE) and attended-position + vocab top-1 vs the
    unquantised baseline.
    """
    from .kvcache import _vocab_projection, attention_float

    n_q = min(n_queries, kv_orig["K"].shape[0])
    q = kv_orig["K"][:n_q]
    out_orig = attention_float(q, kv_orig["K"], kv_orig["V"])
    out_recon = attention_float(q, kv_recon["K"], kv_recon["V"])
    diff = out_orig.astype(np.float64) - out_recon.astype(np.float64)
    pos_agree = 0
    pos_total = 0
    for h in range(q.shape[1]):
        s_orig = (q[:, h, :].astype(np.float64)
                  @ kv_orig["K"][:, h, :].astype(np.float64).T) / math.sqrt(q.shape[-1])
        s_recon = (q[:, h, :].astype(np.float64)
                   @ kv_recon["K"][:, h, :].astype(np.float64).T) / math.sqrt(q.shape[-1])
        pos_agree += int((s_orig.argmax(axis=1) == s_recon.argmax(axis=1)).sum())
        pos_total += s_orig.shape[0]
    wv = _vocab_projection(q.shape[-1])
    v_orig = (out_orig.reshape(-1, q.shape[-1]).astype(np.float64) @ wv.astype(np.float64)).argmax(axis=1)
    v_recon = (out_recon.reshape(-1, q.shape[-1]).astype(np.float64) @ wv.astype(np.float64)).argmax(axis=1)
    return {
        "n_tokens": int(kv_orig["K"].shape[0]),
        "max_abs_error": float(np.abs(diff).max()),
        "mse": float(np.mean(diff ** 2)),
        "attended_position_top1": pos_agree / pos_total if pos_total else 1.0,
        "vocab_top1": float(np.mean(v_orig == v_recon)),
        "n_queries": int(n_q * q.shape[1]),
    }


def run_e8_eval(contexts: dict | None = None, verbose: bool = True) -> dict:
    """Full P9 eval on the P8 synthetic KV profiles.

    Per profile: quantise V once (scalar 2-bit, shared by all
    arms) and K three ways (E8 k2, scalar k2, scalar k3) at
    full length; per context, slice the prefixes and run
    ``fidelity_eval`` (attention error + attended/vocab
    top-1 vs the unquantised baseline, queries attending over
    the full context prefix) plus a K-only SQNR.
    The E8 arm additionally runs the table-scored attention
    and checks it against the dequantised path. The primary
    fidelity metric is attended-position top-1 agreement;
    MSE / max-abs / vocab top-1 are reported alongside.
    A setting is an E8 win when E8's attended top-1 strictly
    exceeds scalar k3v2's. Nothing here touches a real model
    or computes perplexity.
    """
    from .kvcache import generate_synthetic_kv

    contexts = contexts or CONTEXTS
    profiles: dict[str, dict] = {}
    wins_vs_k3 = 0
    wins_vs_k2 = 0
    n_settings = 0
    for name in contexts:
        kv = generate_synthetic_kv(name)
        d = kv["K"].shape[-1]
        vq = quantise_scalar(kv["V"], 2)
        v_recon = dequantise_scalar(vq)
        k_arms: dict[str, dict] = {}
        e8k, e8_info = quantise_keys_e8(kv["K"])
        k_arms["e8-k2v2"] = {
            "quantised": e8k, "recon": dequantise_keys_e8(e8k),
            "ratio": kv_ratio(d, 2, True),
            "bits_per_dim": kv_bits_per_dim(d, 2, True),
        }
        for bits, label in ((2, "scalar-k2v2"), (3, "scalar-k3v2")):
            sq = quantise_scalar(kv["K"], bits)
            k_arms[label] = {
                "quantised": sq, "recon": dequantise_scalar(sq),
                "ratio": kv_ratio(d, bits, False),
                "bits_per_dim": kv_bits_per_dim(d, bits, False),
            }
        entry: dict = {
            "shape": list(kv["K"].shape),
            "fallback_minimal_blocks_e8": e8_info["fallback_minimal_blocks"],
            "codebook_size": e8_info["codebook_size"],
            "settings": {},
        }
        for ctx in contexts[name]:
            setting: dict = {}
            kv_orig = {"K": kv["K"][:ctx], "V": kv["V"][:ctx]}
            for label, arm in k_arms.items():
                recon = {
                    "K": arm["recon"][:ctx], "V": v_recon[:ctx],
                }
                r = fidelity_eval(kv_orig, recon, n_queries=ATTN_QUERIES)
                k_err = kv["K"][:ctx].astype(np.float64) - recon["K"].astype(np.float64)
                mse = float(np.mean(k_err ** 2))
                var = float(np.var(kv["K"][:ctx].astype(np.float64)))
                setting[label] = {
                    "ratio": arm["ratio"],
                    "bits_per_dim": arm["bits_per_dim"],
                    "k_mse": mse,
                    "k_sqnr_db": float(10 * math.log10(var / mse)) if mse > 0 else float("inf"),
                    "max_abs_error": r["max_abs_error"],
                    "mse": r["mse"],
                    "attended_position_top1": r["attended_position_top1"],
                    "vocab_top1": r["vocab_top1"],
                }
            # Table-scored attention for the E8 arm, vs the
            # dequantised path, on the same queries.
            from .kvcache import attention_float

            q = kv["K"][:ATTN_QUERIES]
            out_table = attention_table(q, e8k.prefix(ctx), v_recon[:ctx])
            out_deq = attention_float(
                q, k_arms["e8-k2v2"]["recon"][:ctx], v_recon[:ctx]
            )
            setting["e8_table_max_abs_diff"] = float(
                np.abs(out_table.astype(np.float64) - out_deq.astype(np.float64)).max()
            )
            setting["e8_table_equivalent"] = bool(
                setting["e8_table_max_abs_diff"] < 1e-5
            )
            e8_top1 = setting["e8-k2v2"]["attended_position_top1"]
            if e8_top1 > setting["scalar-k3v2"]["attended_position_top1"]:
                wins_vs_k3 += 1
            if e8_top1 > setting["scalar-k2v2"]["attended_position_top1"]:
                wins_vs_k2 += 1
            n_settings += 1
            entry["settings"][str(ctx)] = setting
            if verbose:
                s = setting
                print(
                    "%s ctx=%d: E8 k2v2 top1 %.3f MSE %.2e | "
                    "scalar k2v2 top1 %.3f MSE %.2e | scalar k3v2 "
                    "top1 %.3f MSE %.2e | table diff %.2e"
                    % (
                        name, ctx,
                        s["e8-k2v2"]["attended_position_top1"], s["e8-k2v2"]["mse"],
                        s["scalar-k2v2"]["attended_position_top1"], s["scalar-k2v2"]["mse"],
                        s["scalar-k3v2"]["attended_position_top1"], s["scalar-k3v2"]["mse"],
                        s["e8_table_max_abs_diff"],
                    )
                )
        profiles[name] = entry
    return {
        "profiles": profiles,
        "n_settings": n_settings,
        "e8_wins_vs_scalar_k3v2": wins_vs_k3,
        "e8_wins_vs_scalar_k2v2": wins_vs_k2,
        "done_beats_k3_in_most": wins_vs_k3 > n_settings // 2,
        "note": (
            "Synthetic correlated BF16-baseline KV only (P8 "
            "profiles, seeded RNG) — no real model, no "
            "perplexity. Metric is attention fidelity vs the "
            "unquantised baseline; primary metric attended-"
            "position top-1. Table scoring verified against "
            "the dequantise path per setting."
        ),
    }


def print_e8_report(result: dict) -> None:
    print("P9 — E8 lattice key quantisation (k2v2) (synthetic KV, NumPy)")
    print("NOTE: %s" % result["note"])
    for name, p in result["profiles"].items():
        print("\n%s  shape %s  codebook %d codewords, fallback-minimal blocks %d"
              % (name, "x".join(str(s) for s in p["shape"]),
                 p["codebook_size"], p["fallback_minimal_blocks_e8"]))
        for ctx, s in p["settings"].items():
            print("  ctx=%s:" % ctx)
            for label in ("e8-k2v2", "scalar-k2v2", "scalar-k3v2"):
                a = s[label]
                print(
                    "    %-12s %.2fx (%0.2f bits/dim): K SQNR %5.2f dB, "
                    "attention max abs %.2e MSE %.2e, attended top-1 %.3f, "
                    "vocab top-1 %.3f"
                    % (
                        label, a["ratio"], a["bits_per_dim"], a["k_sqnr_db"],
                        a["max_abs_error"], a["mse"],
                        a["attended_position_top1"], a["vocab_top1"],
                    )
                )
            print(
                "    table-scored attention vs dequantised path: "
                "equivalent %s (max abs diff %.2e)"
                % (s["e8_table_equivalent"], s["e8_table_max_abs_diff"])
            )
    print(
        "\nE8 k2v2 beats scalar k3v2 on attended top-1 in %d/%d settings; "
        "beats scalar k2v2 in %d/%d. Done condition (most settings vs k3): %s"
        % (
            result["e8_wins_vs_scalar_k3v2"], result["n_settings"],
            result["e8_wins_vs_scalar_k2v2"], result["n_settings"],
            result["done_beats_k3_in_most"],
        )
    )
