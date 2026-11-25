"""P8 — KV-cache snapshot store with transform coding.

KV Cache Transform Coding (arXiv:2511.01815): resume a long agent
session without re-prefilling — transform-code the KV cache to disk
after each turn (PCA decorrelation + adaptive quantisation + entropy
coding), reload it next turn. The plan's done condition is up to
~20x compression with reasoning accuracy held on a small benchmark,
and a restarted session that resumes with identical continuation.

SUBSTITUTIONS, stated plainly (this VM has no torch, no GPU, no
real model checkpoints; do NOT read this module as a run on a real
model):
- There is no real model here. The pipeline runs on SYNTHETIC
  BF16-baseline KV-cache tensors with realistic structure: the
  same two profiles (shapes, seeds) as P7, extended with
  CROSS-CHANNEL CORRELATION that PCA can actually exploit — each
  tensor is a low-rank latent signal (rank 16 for the 128-dim
  profile, rank 8 for the 64-dim one; latent scales linearly
  decaying 1.0 -> 0.35) plus small independent per-element noise
  (std 0.05) plus per-channel scale variation (N(1, 0.25) clipped
  to [0.5, 1.8]). That correlation is the legitimate structure
  transform coding exploits in real caches (real KV channels are
  strongly correlated across dimensions); the parameters above
  were FIXED BEFORE the final ratio was measured and are not
  tuned to hit a number. The ratios below are synthetic-KV
  ratios vs a BF16 baseline (2 bytes/element, the format real
  caches store), not real-checkpoint ratios.
- No LLM, no reasoning model: the "reasoning task" is a
  deterministic NumPy attention check in the style of P7's —
  attention-output error (max abs / MSE) and top-1 agreement
  vs the FP16 uncompressed baseline on a fixed set of queries
  over the KV: (a) attended-position top-1 agreement (which key
  position each query attends to most — a value-retrieval task),
  and (b) next-token argmax agreement through a fixed seeded
  random vocab projection. Neither is a real benchmark score.
- No real agent process: the Phase 3 "restart" is a fresh
  ``KVSnapshotStore`` object (empty in-memory basis cache)
  reloading the snapshot file from disk — the
  fresh-interpreter equivalent for a pure-Python store — not a
  separate OS process. The continuation comparison is exact
  (identical argmax), not assumed — and it is reported as
  measured: a lossy codec guarantees bounded attention-output
  error, not argmax identity, so when the uninterrupted
  run's top-2 logit gap is smaller than the codec's logit
  perturbation the argmax can flip. The demo records the gap
  and the perturbation per restarted turn so a flip is
  diagnosed, not hidden.

Pipeline (per tensor, K and V separately; each has its own PCA
basis fitted on a brief calibration slice):
1. PCA decorrelation across channels: basis fitted by SVD on
   the first ``CALIBRATION_TOKENS`` tokens (all heads), keeping
   the smallest k whose cumulative explained variance reaches
   ``VARIANCE_THRESHOLD`` (0.98). The basis is cached per
   profile ("model") on disk under the store root, so a resumed
   session reuses it instead of refitting.
2. Adaptive per-component quantisation: each retained
   component gets its own bit width from its calibration
   standard deviation — ``bits = clip(round(3 +
   log2(sigma_c / median sigma)), 2, 8)`` — so high-variance
   (early) components get up to 8 bits and low-variance (tail)
   components get 2. The quantiser range is +/-4 sigma of the
   calibration slice; values outside it clip. The per-coefficient
   error is therefore bounded by half the step,
   ``step_c / 2 = scale_c / (2**bits_c - 1)``, which the tests
   pin. This is the P7 group-adaptive-width idea, applied to
   PCA components instead of exponent groups.
3. Entropy coding: a deterministic canonical Huffman coder in
   pure Python (stdlib + NumPy only, no third-party entropy
   library) over the quantised codes, with ONE codebook PER
   PCA component. A shared alphabet was tried first and
   measured: it costs ~1.3 bits/code on these profiles
   (unified entropy 4.42 vs mean per-component entropy 3.13),
   because early components spread over 256 values while tail
   components use only 4 — mixing them wastes the codebook.
   One codebook per component recovers it, at k x 256
   code-length bytes of header. The codebook is stored as
   code lengths per component; a degenerate column (one
   distinct symbol) stores an all-zero length row, writes no
   bits, and records its symbol separately.

Format (``TransformCodedTensor.to_bytes``): magic ``KVTC``,
version 1, shape, d, k, code count, bitstream length, then the
per-channel mean (float32), the basis (float32, d x k), the
per-component scales (float32) and bit widths (uint8), the
per-component Huffman code lengths (k x 256 bytes) plus the
degenerate-column symbols, then the packed bitstream. A snapshot
file (``Snapshot.to_bytes``) is magic ``KVSN`` + a JSON header
(profile, session, turn, token count, shape) + the K and V
``KVTC`` blobs, each length-prefixed.

Operating point, chosen from the accuracy/ratio tradeoff BEFORE
the final eval and then frozen (not re-tuned after measuring):
variance threshold 0.98 with bit centre 3. The aggressive
operating point (threshold 0.95, centre 4) reaches ~46x on the
same synthetic data but drops attended-position top-1
agreement to 0.84 on the small profile — ratio bought by
truncating signal the attention actually uses. The frozen point
keeps top-1 agreement >= 0.95 on both profiles at ~14x, which
is BELOW the plan's ~20x target; that miss is reported as
measured in the README, not tuned away.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import struct
import tempfile
import time

import numpy as np

MAGIC_TENSOR = b"KVTC"
MAGIC_SNAPSHOT = b"KVSN"
VERSION = 1

# Frozen operating point (see module docstring): brief calibration,
# 98% explained variance, adaptive bits centred on 3.
CALIBRATION_TOKENS = 128
VARIANCE_THRESHOLD = 0.98
BIT_CENTRE = 3
BIT_MIN = 2
BIT_MAX = 8
SCALE_SIGMAS = 4.0

ATTN_CHECK_TOKENS = 64
VOCAB_SIZE = 128
VOCAB_SEED = 123
HUFFMAN_LOOKUP_BITS = 12

# ---------------------------------------------------------------------------
# Synthetic KV profiles with cross-channel correlation.
#
# FIXED BEFORE MEASURING: shapes and seeds match P7 so the two
# plans are comparable; the correlation parameters (latent rank,
# latent scale decay, noise std, per-channel scale spread) are
# the new P8 structure and were frozen with the operating point.
# K and V share the profile statistics but draw independently
# (different RNG streams), as in a real cache.
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "llama-7b-like": {
        "seq": 1024,
        "heads": 32,
        "d_head": 128,
        "seed": 7,
        "latent_rank": 16,
        "noise_std": 0.05,
    },
    "small-model-like": {
        "seq": 2048,
        "heads": 16,
        "d_head": 64,
        "seed": 11,
        "latent_rank": 8,
        "noise_std": 0.05,
    },
}

# Latent scales decay linearly 1.0 -> 0.35 across the rank, and
# per-channel scales are N(1, 0.25) clipped to [0.5, 1.8].
LATENT_SCALE_LO = 0.35
CHANNEL_SCALE_STD = 0.25
CHANNEL_SCALE_LO = 0.5
CHANNEL_SCALE_HI = 1.8


def generate_synthetic_kv(profile_name: str) -> dict[str, np.ndarray]:
    """Generate a synthetic (K, V) float32 KV pair with correlated channels.

    Per tensor: a latent factor draw ``z`` (rank ``r``, scales
    decaying 1.0 -> 0.35) mixed through a fixed random
    orthonormal matrix (d x r), plus independent noise
    (std ``noise_std``), times a per-channel scale. The result
    has a low-rank-plus-noise covariance that PCA can exploit:
    the top few components carry most of the variance and the
    tail is near-isotropic noise. Returns float32 arrays of
    shape (seq, heads, d_head).
    """
    profile = PROFILES[profile_name]
    seq, heads, d = profile["seq"], profile["heads"], profile["d_head"]
    rank = profile["latent_rank"]

    def _one_tensor(stream_seed: int) -> np.ndarray:
        r = np.random.default_rng(stream_seed)
        mix = r.normal(size=(d, rank))
        mix, _ = np.linalg.qr(mix)  # orthonormal columns (d, rank)
        scales = np.linspace(1.0, LATENT_SCALE_LO, rank)
        n_rows = seq * heads
        z = r.normal(size=(n_rows, rank)) * scales[None, :]
        x = z @ mix.T
        x = x + r.normal(scale=profile["noise_std"], size=(n_rows, d))
        ch = np.clip(
            r.normal(1.0, CHANNEL_SCALE_STD, size=d),
            CHANNEL_SCALE_LO,
            CHANNEL_SCALE_HI,
        )
        return (x * ch[None, :]).reshape(seq, heads, d).astype(np.float32)

    return {
        "K": _one_tensor(profile["seed"] * 1000 + 1),
        "V": _one_tensor(profile["seed"] * 1000 + 2),
    }


# ---------------------------------------------------------------------------
# PCA basis + adaptive quantiser (the "coder", fitted on calibration).
# ---------------------------------------------------------------------------

class Coder:
    """PCA basis + per-component quantiser, fitted on a calibration slice.

    ``mean`` (d,), ``basis`` (d, k) float32, ``bits`` (k,) uint8,
    ``scales`` (k,) float32. Fitted once per profile ("model")
    and cached — a resumed session reloads the coder instead of
    refitting, which is the whole point of the per-model basis
    cache in the snapshot store.
    """

    def __init__(self, mean, basis, bits, scales):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.basis = np.asarray(basis, dtype=np.float32)
        self.bits = np.asarray(bits, dtype=np.uint8)
        self.scales = np.asarray(scales, dtype=np.float32)

    @property
    def d(self) -> int:
        return int(self.mean.shape[0])

    @property
    def k(self) -> int:
        return int(self.basis.shape[1])

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Coder)
            and np.array_equal(self.mean, other.mean)
            and np.array_equal(self.basis, other.basis)
            and np.array_equal(self.bits, other.bits)
            and np.array_equal(self.scales, other.scales)
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Float32 (N, d) -> PCA coefficients (N, k), float64 math."""
        flat = np.asarray(x, dtype=np.float64).reshape(-1, self.d)
        return (flat - self.mean.astype(np.float64)) @ self.basis.astype(np.float64)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """PCA coefficients (N, k) -> float32 (N, d)."""
        rec = np.asarray(coeffs, dtype=np.float64) @ self.basis.astype(np.float64).T
        return (rec + self.mean.astype(np.float64)).astype(np.float32)


def fit_coder(calibration_rows: np.ndarray) -> Coder:
    """Fit a Coder on calibration rows (N_cal, d), float32/float64 in.

    SVD on the centred calibration slice (float64); keep the
    smallest k with cumulative explained variance >=
    ``VARIANCE_THRESHOLD``. Bit widths from the calibration
    component stds (centre ``BIT_CENTRE`` on the median std,
    one bit per doubling, clipped to [2, 8]); quantiser scale
    is ``SCALE_SIGMAS`` * sigma per component (a zero-variance
    component gets scale 1.0 and its coefficients are all zero,
    so it costs nothing beyond its 2-bit codes).
    """
    cal = np.asarray(calibration_rows, dtype=np.float64)
    if cal.ndim != 2 or cal.shape[0] == 0:
        raise ValueError("fit_coder needs a non-empty (N, d) calibration slice")
    mean = cal.mean(axis=0)
    centred = cal - mean
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = s ** 2
    total = float(var.sum())
    if total <= 0:
        # Constant calibration slice: no variance to explain.
        k = 1
        basis = np.zeros((cal.shape[1], 1), dtype=np.float64)
        basis[0, 0] = 1.0
        sig = np.zeros(1)
    else:
        cum = np.cumsum(var) / total
        k = int(np.searchsorted(cum, VARIANCE_THRESHOLD) + 1)
        k = max(1, min(k, cal.shape[1]))
        basis = vt[:k].T
        sig = (centred @ basis).std(axis=0)
    med = float(np.median(sig)) if sig.size else 0.0
    if med <= 0:
        bits = np.full(k, BIT_MIN, dtype=np.uint8)
    else:
        bits = np.clip(
            np.round(BIT_CENTRE + np.log2(np.maximum(sig, 1e-12) / med)).astype(int),
            BIT_MIN,
            BIT_MAX,
        ).astype(np.uint8)
    scales = np.where(sig > 0, SCALE_SIGMAS * sig, 1.0).astype(np.float32)
    return Coder(mean.astype(np.float32), basis.astype(np.float32), bits, scales)


def quantise(coeffs: np.ndarray, coder: Coder) -> np.ndarray:
    """PCA coefficients -> uint8 quantised codes (0..255).

    Uniform mid-tread quantiser per component over
    [-scale, +scale] with an ODD number of levels
    (``2**bits - 1``), so the middle level represents exactly
    zero — a zero coefficient quantises to zero, not to half
    a step. Values outside the range clip to the end levels
    (that clipping is the only unbounded error source besides
    truncation — both are measured, not assumed).
    """
    c = np.asarray(coeffs, dtype=np.float64)
    levels = (1 << coder.bits.astype(np.int64)) - 1  # (k,), odd
    scaled = np.clip(c / coder.scales.astype(np.float64), -1.0, 1.0)
    codes = np.round((scaled + 1.0) / 2.0 * (levels - 1)).astype(np.uint8)
    return codes


def dequantise(codes: np.ndarray, coder: Coder) -> np.ndarray:
    """Quantised codes -> PCA coefficients (float64)."""
    levels = (1 << coder.bits.astype(np.int64)) - 1
    frac = np.asarray(codes, dtype=np.float64) / np.maximum(levels - 1, 1)
    return (frac * 2.0 - 1.0) * coder.scales.astype(np.float64)


def quant_step(coder: Coder) -> np.ndarray:
    """Per-component quantiser step (2*scale / (levels-1))."""
    levels = (1 << coder.bits.astype(np.int64)) - 1
    return 2.0 * coder.scales.astype(np.float64) / np.maximum(levels - 1, 1)


# ---------------------------------------------------------------------------
# Deterministic canonical Huffman coder (pure Python, no third-party lib).
# ---------------------------------------------------------------------------

def _huffman_lengths(freqs: np.ndarray) -> np.ndarray:
    """Code lengths from symbol frequencies (canonical Huffman).

    Deterministic: heap entries are (freq, tie-breaker, node)
    with a monotonically increasing tie-breaker, so equal
    frequencies always resolve the same way. A single distinct
    symbol gets length 0 (empty bitstream — the decoder fills
    it); zero-frequency symbols get length 0.
    """
    freqs = np.asarray(freqs, dtype=np.int64)
    lengths = np.zeros(256, dtype=np.uint8)
    present = [int(s) for s in np.nonzero(freqs)[0]]
    if len(present) <= 1:
        return lengths
    counter = [0]

    def _next_id():
        counter[0] += 1
        return counter[0]

    heap: list = []
    for s in present:
        heapq.heappush(heap, (int(freqs[s]), _next_id(), ("leaf", s)))
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, _next_id(), ("node", n1, n2)))

    def _walk(node, depth: int) -> None:
        if node[0] == "leaf":
            lengths[node[1]] = depth
        else:
            _walk(node[1], depth + 1)
            _walk(node[2], depth + 1)

    _walk(heap[0][2], 0)
    return lengths


def _canonical_codes(lengths: np.ndarray) -> dict[int, tuple[int, int]]:
    """Canonical Huffman codes: {symbol: (code, length)}.

    Symbols sorted by (length, symbol); codes assigned in
    counting order per length, as in DEFLATE. Length-0 symbols
    get no code.
    """
    symbols = sorted(
        (int(s) for s in range(256) if lengths[s] > 0),
        key=lambda s: (int(lengths[s]), s),
    )
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = 0
    for s in symbols:
        length = int(lengths[s])
        code <<= length - prev_len
        codes[s] = (code, length)
        code += 1
        prev_len = length
    return codes


class _BitWriter:
    """MSB-first bit accumulator with a bounded integer state."""

    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.acc_bits = 0
        self.n_bits = 0

    def write(self, code: int, length: int) -> None:
        self.acc = (self.acc << length) | code
        self.acc_bits += length
        self.n_bits += length
        while self.acc_bits >= 8:
            self.acc_bits -= 8
            self.out.append((self.acc >> self.acc_bits) & 0xFF)
        # Keep only the unflushed tail bits: without this mask
        # ``acc`` would grow to one bit per encoded symbol.
        self.acc &= (1 << self.acc_bits) - 1 if self.acc_bits else 0

    def finish(self) -> bytes:
        if self.acc_bits:
            self.out.append((self.acc << (8 - self.acc_bits)) & 0xFF)
            self.acc = 0
            self.acc_bits = 0
        return bytes(self.out)


class _BitReader:
    """MSB-first bit reader shared across Huffman columns."""

    def __init__(self, data: bytes):
        self.data = bytes(data)
        self.bitpos = 0
        self.total_bits = len(self.data) * 8

    def read_symbol(self, lookup, lookup_bits, by_len_code, max_len) -> int:
        """Decode one symbol: 12-bit lookup fast path, else bit walk."""
        if self.total_bits - self.bitpos >= lookup_bits:
            byte_idx = self.bitpos // 8
            bit_off = self.bitpos % 8
            n_bytes = (lookup_bits + bit_off + 7) // 8
            chunk = 0
            for i in range(n_bytes):
                chunk = (chunk << 8) | (
                    self.data[byte_idx + i] if byte_idx + i < len(self.data) else 0
                )
            chunk >>= 8 * n_bytes - lookup_bits - bit_off
            chunk &= (1 << lookup_bits) - 1
            hit = lookup.get(chunk)
            if hit is not None:
                sym, length = hit
                self.bitpos += length
                return sym
        code = 0
        length = 0
        while True:
            if self.bitpos >= self.total_bits:
                raise ValueError("huffman_decode: bitstream truncated")
            byte_idx = self.bitpos // 8
            bit = (self.data[byte_idx] >> (7 - (self.bitpos % 8))) & 1
            code = (code << 1) | bit
            length += 1
            self.bitpos += 1
            sym = by_len_code.get((length, code))
            if sym is not None:
                return sym
            if length > max_len:
                raise ValueError("huffman_decode: invalid code in bitstream")


def _lookup_table(codes, max_len):
    """12-bit lookup table for codes up to 12 bits: {peek: (sym, len)}."""
    lookup_bits = min(HUFFMAN_LOOKUP_BITS, max_len)
    table: dict[int, tuple[int, int]] = {}
    for sym, (code, length) in codes.items():
        if length <= lookup_bits:
            prefix = code << (lookup_bits - length)
            for suffix in range(1 << (lookup_bits - length)):
                table[prefix | suffix] = (sym, length)
    return table, lookup_bits


def _decode_column(reader: _BitReader, n_rows: int, lengths: np.ndarray) -> np.ndarray:
    """Decode one Huffman column (one PCA component) from ``reader``."""
    codes = _canonical_codes(lengths)
    max_len = int(lengths.max())
    lookup, lookup_bits = _lookup_table(codes, max_len)
    by_len_code = {(length, code): sym for sym, (code, length) in codes.items()}
    out = np.zeros(n_rows, dtype=np.uint8)
    for i in range(n_rows):
        out[i] = reader.read_symbol(lookup, lookup_bits, by_len_code, max_len)
    return out


def huffman_encode(codes_flat: np.ndarray) -> tuple[bytes, int, np.ndarray]:
    """Encode uint8 symbols -> (bitstream bytes, n_bits, lengths).

    Single-alphabet primitive, used by the tests and by the
    per-column coder below. Deterministic: same input gives
    the same bytes.
    """
    flat = np.asarray(codes_flat, dtype=np.uint8).reshape(-1)
    freqs = np.bincount(flat, minlength=256).astype(np.int64)
    lengths = _huffman_lengths(freqs)
    if flat.size == 0 or int(lengths.max()) == 0:
        return b"", 0, lengths
    table = _canonical_codes(lengths)
    writer = _BitWriter()
    for sym in flat.tolist():
        code, length = table[sym]
        writer.write(code, length)
    return writer.finish(), writer.n_bits, lengths


def huffman_decode(
    data: bytes, n_symbols: int, lengths: np.ndarray
) -> np.ndarray:
    """Decode a Huffman bitstream back to uint8 symbols."""
    lengths = np.asarray(lengths, dtype=np.uint8)
    if n_symbols == 0:
        return np.zeros(0, dtype=np.uint8)
    if int(lengths.max()) == 0:
        raise ValueError("huffman_decode: no code lengths (empty codebook)")
    reader = _BitReader(data)
    return _decode_column(reader, n_symbols, lengths)


def huffman_encode_columns(codes: np.ndarray):
    """Encode a (rows, k) code matrix with one codebook per column.

    Per-component codebooks matter: the quantised codes of
    different PCA components have very different distributions
    (early components spread over 256 values, tail components
    over 4), and a single shared alphabet mixes them — on the
    synthetic profiles the shared alphabet costs ~1.3
    bits/code (unified entropy 4.42 vs mean per-component
    entropy 3.13). One codebook per component recovers it;
    the cost is k x 256 code-length bytes in the header.
    Returns (bitstream, n_bits, lengths (k, 256),
    single_symbols (k,) int16, -1 = not degenerate).
    A column with a single distinct symbol gets an all-zero
    length row and writes no bits; its symbol is recorded in
    ``single_symbols``.
    """
    codes = np.asarray(codes, dtype=np.uint8)
    n_rows, k = codes.shape
    lengths = np.zeros((k, 256), dtype=np.uint8)
    single = np.full(k, -1, dtype=np.int16)
    writer = _BitWriter()
    for j in range(k):
        col = codes[:, j]
        uniq = np.unique(col)
        if uniq.size <= 1:
            single[j] = int(uniq[0]) if uniq.size == 1 else 0
            continue
        freqs = np.bincount(col, minlength=256).astype(np.int64)
        lengths[j] = _huffman_lengths(freqs)
        table = _canonical_codes(lengths[j])
        for sym in col.tolist():
            code, length = table[sym]
            writer.write(code, length)
    return writer.finish(), writer.n_bits, lengths, single


def huffman_decode_columns(data: bytes, n_rows: int, lengths: np.ndarray,
                           single: np.ndarray) -> np.ndarray:
    """Inverse of ``huffman_encode_columns`` -> (rows, k) uint8 codes."""
    lengths = np.asarray(lengths, dtype=np.uint8)
    single = np.asarray(single)
    k = lengths.shape[0]
    out = np.zeros((n_rows, k), dtype=np.uint8)
    reader = _BitReader(data)
    for j in range(k):
        if int(lengths[j].max()) == 0:
            out[:, j] = int(single[j]) if single[j] >= 0 else 0
        else:
            out[:, j] = _decode_column(reader, n_rows, lengths[j])
    return out


# ---------------------------------------------------------------------------
# Transform-coded tensor (one K or V tensor, with its coder).
# ---------------------------------------------------------------------------

class TransformCodedTensor:
    """One transform-coded KV tensor.

    Carries its ``Coder`` (basis fitted on the calibration
    slice) plus the Huffman-coded quantised coefficients, so a
    snapshot file is self-contained: a resumed session needs
    nothing but the file. ``original_n_bytes`` is the BF16
    baseline (2 bytes/element), matching P7's ratio definition.
    """

    def __init__(self, shape, coder: Coder, lengths, bitstream: bytes,
                 n_codes: int, n_bits: int, single=None):
        self.shape = tuple(shape)
        self.coder = coder
        # Per-component Huffman code lengths, shape (k, 256);
        # ``single`` (k,) int16 records a degenerate column's
        # only symbol (-1 = not degenerate).
        self.lengths = np.asarray(lengths, dtype=np.uint8).reshape(
            coder.k, 256)
        self.bitstream = bytes(bitstream)
        self.n_codes = int(n_codes)
        self.n_bits = int(n_bits)
        self.single = (np.full(coder.k, -1, dtype=np.int16) if single is None
                       else np.asarray(single, dtype=np.int16).reshape(coder.k))

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 0

    @property
    def original_n_bytes(self) -> int:
        return self.n_elements * 2

    @property
    def n_bytes(self) -> int:
        header = 4 + 1 + 1 + 4 * len(self.shape) + 4 + 4 + 4 + 4 + 4
        return (
            header
            + self.coder.mean.nbytes
            + self.coder.basis.nbytes
            + self.coder.scales.nbytes
            + len(self.coder.bits)
            + self.single.nbytes
            + self.lengths.nbytes
            + len(self.bitstream)
        )

    @property
    def ratio(self) -> float:
        return self.original_n_bytes / self.n_bytes if self.n_bytes else float("inf")

    @property
    def bits_per_code(self) -> float:
        return self.n_bits / self.n_codes if self.n_codes else 0.0

    @classmethod
    def encode(cls, x: np.ndarray, coder: Coder) -> "TransformCodedTensor":
        """Transform-code a float32 tensor (last axis = channels)."""
        x = np.asarray(x, dtype=np.float32)
        flat = x.reshape(-1, coder.d)
        coeffs = coder.transform(flat)
        codes = quantise(coeffs, coder)
        if codes.size == 0:
            lengths = np.zeros((coder.k, 256), dtype=np.uint8)
            return cls(x.shape, coder, lengths, b"", 0, 0)
        data, n_bits, lengths, single = huffman_encode_columns(codes)
        return cls(x.shape, coder, lengths, data, codes.size, n_bits,
                   single)

    def decode(self) -> np.ndarray:
        """Reconstruct the float32 tensor (quantised + truncated)."""
        if self.n_codes == 0:
            return np.zeros(self.shape, dtype=np.float32)
        n_rows = self.n_codes // self.coder.k
        codes = huffman_decode_columns(
            self.bitstream, n_rows, self.lengths, self.single)
        coeffs = dequantise(codes, self.coder)
        return self.coder.inverse_transform(coeffs).reshape(self.shape)

    # -- serialisation -------------------------------------------------
    def to_bytes(self) -> bytes:
        d, k = self.coder.d, self.coder.k
        header = (
            MAGIC_TENSOR
            + struct.pack("<BB", VERSION, len(self.shape))
            + struct.pack("<%dI" % len(self.shape), *self.shape)
            + struct.pack("<IIIII", d, k, self.n_codes, len(self.bitstream),
                          self.n_bits)
        )
        return (
            header
            + self.coder.mean.astype(np.float32).tobytes()
            + self.coder.basis.astype(np.float32).tobytes()
            + self.coder.scales.astype(np.float32).tobytes()
            + self.coder.bits.astype(np.uint8).tobytes()
            + self.single.astype(np.int16).tobytes()
            + self.lengths.astype(np.uint8).tobytes()
            + self.bitstream
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TransformCodedTensor":
        if data[:4] != MAGIC_TENSOR:
            raise ValueError("not a KVTC blob (bad magic)")
        version, ndim = struct.unpack_from("<BB", data, 4)
        if version != VERSION:
            raise ValueError("unsupported KVTC version %d" % version)
        off = 6
        shape = struct.unpack_from("<%dI" % ndim, data, off) if ndim else ()
        off += 4 * ndim
        d, k, n_codes, n_stream, n_bits = struct.unpack_from("<IIIII", data, off)
        off += 20
        mean = np.frombuffer(data, dtype=np.float32, count=d, offset=off).copy()
        off += 4 * d
        basis = np.frombuffer(
            data, dtype=np.float32, count=d * k, offset=off
        ).copy().reshape(d, k)
        off += 4 * d * k
        scales = np.frombuffer(data, dtype=np.float32, count=k, offset=off).copy()
        off += 4 * k
        bits = np.frombuffer(data, dtype=np.uint8, count=k, offset=off).copy()
        off += k
        single = np.frombuffer(data, dtype=np.int16, count=k, offset=off).copy()
        off += 2 * k
        lengths = np.frombuffer(
            data, dtype=np.uint8, count=k * 256, offset=off
        ).copy().reshape(k, 256)
        off += k * 256
        bitstream = data[off:off + n_stream]
        if len(bitstream) != n_stream:
            raise ValueError("KVTC blob truncated (bitstream length mismatch)")
        return cls(shape, Coder(mean, basis, bits, scales), lengths, bitstream,
                   n_codes, n_bits, single)


# ---------------------------------------------------------------------------
# Attention + the deterministic "reasoning" task (NumPy, P7 style).
# ---------------------------------------------------------------------------

def attention_float(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention over float32 KV (per head).

    Shapes (seq_q, heads, d), (seq_k, heads, d), (seq_k, heads,
    d); returns float32 (seq_q, heads, d). No masking, no
    dropout — the check is output error / top-1 agreement vs
    the uncompressed baseline, not a model result.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    d = q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    out = np.zeros_like(q)
    for h in range(q.shape[1]):
        scores = (q[:, h, :].astype(np.float64) @ k[:, h, :].astype(np.float64).T) * scale
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        out[:, h, :] = (probs @ v[:, h, :].astype(np.float64)).astype(np.float32)
    return out


def _vocab_projection(d: int) -> np.ndarray:
    """Fixed seeded random vocab projection (d, VOCAB_SIZE)."""
    rng = np.random.default_rng(VOCAB_SEED)
    return (rng.normal(size=(d, VOCAB_SIZE)) / math.sqrt(d)).astype(np.float32)


def reasoning_eval(kv_orig: dict, kv_recon: dict,
                   n_tokens: int = ATTN_CHECK_TOKENS) -> dict:
    """Ratio-vs-accuracy on the fixed attention task.

    Queries are the leading ``n_tokens`` K rows reused as Q
    (same convention as P7). Reports attention-output error
    (max abs / MSE) and two top-1 agreements vs the
    uncompressed baseline: attended-position agreement (which
    key position each query attends to most) and next-token
    argmax agreement through the fixed vocab projection.
    """
    n = min(n_tokens, kv_orig["K"].shape[0])
    q = kv_orig["K"][:n]
    out_orig = attention_float(q, kv_orig["K"][:n], kv_orig["V"][:n])
    out_recon = attention_float(q, kv_recon["K"][:n], kv_recon["V"][:n])
    diff = out_orig.astype(np.float64) - out_recon.astype(np.float64)

    # Attended-position top-1 per (query, head), original vs recon.
    pos_agree = 0
    pos_total = 0
    for h in range(q.shape[1]):
        s_orig = (q[:, h, :].astype(np.float64)
                  @ kv_orig["K"][:n, h, :].astype(np.float64).T) / math.sqrt(q.shape[-1])
        s_recon = (q[:, h, :].astype(np.float64)
                   @ kv_recon["K"][:n, h, :].astype(np.float64).T) / math.sqrt(q.shape[-1])
        pos_agree += int((s_orig.argmax(axis=1) == s_recon.argmax(axis=1)).sum())
        pos_total += s_orig.shape[0]
    wv = _vocab_projection(q.shape[-1])
    v_orig = (out_orig.reshape(-1, q.shape[-1]).astype(np.float64) @ wv.astype(np.float64)).argmax(axis=1)
    v_recon = (out_recon.reshape(-1, q.shape[-1]).astype(np.float64) @ wv.astype(np.float64)).argmax(axis=1)
    return {
        "n_tokens": n,
        "max_abs_error": float(np.abs(diff).max()),
        "mse": float(np.mean(diff ** 2)),
        "attended_position_top1": pos_agree / pos_total if pos_total else 1.0,
        "vocab_top1": float(np.mean(v_orig == v_recon)),
        "n_queries": int(n * q.shape[1]),
    }


# ---------------------------------------------------------------------------
# Snapshot store: per-session files, PCA basis cached per profile.
# ---------------------------------------------------------------------------

class Snapshot:
    """One transform-coded KV snapshot (a session at one turn)."""

    def __init__(self, profile_name, session_id, turn, n_tokens,
                 k_blob: TransformCodedTensor, v_blob: TransformCodedTensor):
        self.profile_name = str(profile_name)
        self.session_id = str(session_id)
        self.turn = int(turn)
        self.n_tokens = int(n_tokens)
        self.k_blob = k_blob
        self.v_blob = v_blob

    @property
    def shape(self):
        return self.k_blob.shape

    @property
    def combined_ratio(self) -> float:
        orig = self.k_blob.original_n_bytes + self.v_blob.original_n_bytes
        comp = self.k_blob.n_bytes + self.v_blob.n_bytes
        return orig / comp if comp else float("inf")

    def decode(self) -> dict[str, np.ndarray]:
        return {"K": self.k_blob.decode(), "V": self.v_blob.decode()}

    def to_bytes(self) -> bytes:
        header = {
            "profile": self.profile_name,
            "session_id": self.session_id,
            "turn": self.turn,
            "n_tokens": self.n_tokens,
        }
        hb = json.dumps(header, sort_keys=True).encode("utf-8")
        kb = self.k_blob.to_bytes()
        vb = self.v_blob.to_bytes()
        return (
            MAGIC_SNAPSHOT
            + struct.pack("<BI", VERSION, len(hb))
            + hb
            + struct.pack("<I", len(kb))
            + kb
            + struct.pack("<I", len(vb))
            + vb
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Snapshot":
        if data[:4] != MAGIC_SNAPSHOT:
            raise ValueError("not a KVSN snapshot (bad magic)")
        version, hlen = struct.unpack_from("<BI", data, 4)
        if version != VERSION:
            raise ValueError("unsupported KVSN version %d" % version)
        off = 9
        header = json.loads(data[off:off + hlen].decode("utf-8"))
        off += hlen
        (klen,) = struct.unpack_from("<I", data, off)
        off += 4
        k_blob = TransformCodedTensor.from_bytes(data[off:off + klen])
        off += klen
        (vlen,) = struct.unpack_from("<I", data, off)
        off += 4
        v_blob = TransformCodedTensor.from_bytes(data[off:off + vlen])
        return cls(header["profile"], header["session_id"], header["turn"],
                   header["n_tokens"], k_blob, v_blob)


class KVSnapshotStore:
    """Per-session snapshot files + per-profile PCA basis cache.

    Layout under ``root``: ``sessions/<session_id>/turn_<n:04d>.kvsn``
    and ``bases/<profile>.npz`` (mean, basis, bits, scales — the
    fitted coder, so a resumed session reloads it instead of
    refitting on the calibration slice). The basis cache is a
    cache, never the only copy: every snapshot file carries its
    own coder, so a snapshot loads even if the cache is gone.
    """

    def __init__(self, root: str):
        self.root = str(root)
        self._coder_cache: dict[str, dict[str, Coder]] = {}
        os.makedirs(os.path.join(self.root, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "bases"), exist_ok=True)

    # -- basis cache ----------------------------------------------------
    def _basis_path(self, profile_name: str) -> str:
        return os.path.join(self.root, "bases", "%s.npz" % profile_name)

    def _load_basis(self, profile_name: str):
        path = self._basis_path(profile_name)
        if not os.path.exists(path):
            return None
        z = np.load(path)
        return {
            "K": Coder(z["K_mean"], z["K_basis"], z["K_bits"], z["K_scales"]),
            "V": Coder(z["V_mean"], z["V_basis"], z["V_bits"], z["V_scales"]),
        }

    def _save_basis(self, profile_name: str, coders: dict) -> None:
        np.savez(
            self._basis_path(profile_name),
            K_mean=coders["K"].mean, K_basis=coders["K"].basis,
            K_bits=coders["K"].bits, K_scales=coders["K"].scales,
            V_mean=coders["V"].mean, V_basis=coders["V"].basis,
            V_bits=coders["V"].bits, V_scales=coders["V"].scales,
        )

    def get_coders(self, profile_name: str, kv: dict) -> dict[str, Coder]:
        """Coders for a profile: memory cache, disk cache, else fit.

        Fitting uses the first ``CALIBRATION_TOKENS`` tokens of
        the supplied KV (the brief calibration the plan asks
        for); the fitted coder is then cached in memory and on
        disk for the next session with the same profile.
        """
        if profile_name in self._coder_cache:
            return self._coder_cache[profile_name]
        cached = self._load_basis(profile_name)
        if cached is not None:
            self._coder_cache[profile_name] = cached
            return cached
        n_cal = min(CALIBRATION_TOKENS, kv["K"].shape[0])
        coders = {
            kind: fit_coder(kv[kind][:n_cal].reshape(-1, kv[kind].shape[-1]))
            for kind in ("K", "V")
        }
        self._save_basis(profile_name, coders)
        self._coder_cache[profile_name] = coders
        return coders

    # -- snapshots --------------------------------------------------------
    def _session_dir(self, session_id: str) -> str:
        path = os.path.join(self.root, "sessions", str(session_id))
        os.makedirs(path, exist_ok=True)
        return path

    def save(self, session_id: str, turn: int, kv: dict,
             profile_name: str) -> str:
        """Transform-code ``kv`` (a prefix) and write the snapshot file."""
        coders = self.get_coders(profile_name, kv)
        snap = Snapshot(
            profile_name, session_id, turn, int(kv["K"].shape[0]),
            TransformCodedTensor.encode(kv["K"], coders["K"]),
            TransformCodedTensor.encode(kv["V"], coders["V"]),
        )
        path = os.path.join(
            self._session_dir(session_id), "turn_%04d.kvsn" % int(turn)
        )
        with open(path, "wb") as fh:
            fh.write(snap.to_bytes())
        return path

    def list_turns(self, session_id: str) -> list[int]:
        d = os.path.join(self.root, "sessions", str(session_id))
        if not os.path.isdir(d):
            return []
        turns = []
        for name in sorted(os.listdir(d)):
            if name.startswith("turn_") and name.endswith(".kvsn"):
                try:
                    turns.append(int(name[5:9]))
                except ValueError:
                    pass
        return sorted(turns)

    def load(self, session_id: str, turn: int | None = None) -> Snapshot:
        """Load a snapshot (latest turn when ``turn`` is None)."""
        turns = self.list_turns(session_id)
        if not turns:
            raise FileNotFoundError("no snapshots for session %r" % session_id)
        turn = turns[-1] if turn is None else int(turn)
        path = os.path.join(
            self.root, "sessions", str(session_id), "turn_%04d.kvsn" % turn
        )
        with open(path, "rb") as fh:
            return Snapshot.from_bytes(fh.read())


# ---------------------------------------------------------------------------
# Phase 3 demo: a multi-turn coding session that survives a restart.
# ---------------------------------------------------------------------------

def _continuation(kv: dict, vocab_w: np.ndarray) -> tuple[int, np.ndarray]:
    """Next-token argmax from the last prefix token's attention output.

    Query = the last token's K row (all heads); the output is
    projected through the fixed vocab matrix and argmaxed.
    Returns (token_id, attention output for the last position).
    """
    q = kv["K"][-1:]  # (1, heads, d)
    out = attention_float(q, kv["K"], kv["V"])  # (1, heads, d)
    logits = out.reshape(-1, q.shape[-1]).astype(np.float64) @ vocab_w.astype(np.float64)
    # One argmax over the head-averaged logits: average the
    # per-head vocab logits so the continuation is a single id.
    token = int(logits.mean(axis=0).argmax())
    return token, out


def _continuation_diagnostics(kv_orig: dict, kv_recon: dict,
                              vocab_w: np.ndarray) -> dict:
    """Why an argmax did or did not survive reconstruction.

    The head-averaged vocab logits of the uninterrupted run
    have a top-2 gap; the reconstructed run perturbs every
    logit by up to ``max_logit_error``. Identical argmax is
    only guaranteed when the gap exceeds the perturbation —
    with a random (untrained) vocab projection the gaps are
    ~1e-3, the same order as the codec's logit error, so a
    flip is a property of the readout's margin, not of the
    codec's size. Returns the gap, the perturbation, and
    whether the argmax matched.
    """
    def _logits(kv):
        q = kv["K"][-1:]
        out = attention_float(q, kv["K"], kv["V"])
        lg = out.reshape(-1, q.shape[-1]).astype(np.float64) @ vocab_w.astype(np.float64)
        return lg.mean(axis=0)

    lo, lr = _logits(kv_orig), _logits(kv_recon)
    top2 = np.sort(lo)[::-1][:2]
    return {
        "top2_gap": float(top2[0] - top2[1]),
        "max_logit_error": float(np.abs(lo - lr).max()),
        "argmax_matches": bool(lo.argmax() == lr.argmax()),
    }


def run_session_demo(profile_name: str = "llama-7b-like",
                     workdir: str | None = None,
                     verbose: bool = True) -> dict:
    """Multi-turn session: snapshot each turn, restart, continue.

    The full KV for the session is generated once (seeded, so
    the uninterrupted run and the restarted run see the same
    tokens). Turns append 64 tokens to a 192-token prefix.
    After turn 2 the "process restarts": a fresh
    ``KVSnapshotStore`` (empty in-memory basis cache) reloads
    the latest snapshot from disk and the session continues
    from the reconstructed KV plus the same new tokens. The
    demo proves the resumed session continues from the
    snapshot with bounded attention-output error vs the
    uncompressed run, and reports the next-token argmax per
    restarted turn as measured — identical when the
    uninterrupted run's top-2 logit gap exceeds the codec's
    logit perturbation, flipped otherwise (see
    ``_continuation_diagnostics``). Nothing is re-prefilled:
    the resumed prefix KV comes from the snapshot file, never
    from recomputation.
    """
    profile = PROFILES[profile_name]
    d = profile["d_head"]
    vocab_w = _vocab_projection(d)
    full_kv = generate_synthetic_kv(profile_name)
    base_tokens, step, n_turns = 192, 64, 4
    restart_after_turn = 2

    tmp = workdir or tempfile.mkdtemp(prefix="kvcache-demo-")
    store = KVSnapshotStore(os.path.join(tmp, "store"))
    session_id = "coding-session-1"
    turns: list[dict] = []
    # Uninterrupted run: snapshot each turn, continue from the
    # full-precision prefix (the reference the restart must match).
    for turn in range(1, n_turns + 1):
        n_tok = base_tokens + (turn - 1) * step
        prefix = {"K": full_kv["K"][:n_tok], "V": full_kv["V"][:n_tok]}
        path = store.save(session_id, turn, prefix, profile_name)
        token, _ = _continuation(prefix, vocab_w)
        turns.append({
            "turn": turn, "n_tokens": n_tok,
            "snapshot_bytes": os.path.getsize(path),
            "continuation_uninterrupted": token,
            "continuation_restarted": None,
            "restarted": False,
        })
        if verbose:
            print("turn %d: %d tokens, snapshot %d bytes, continuation %d"
                  % (turn, n_tok, os.path.getsize(path), token))

    # Restart: fresh store object (fresh-process equivalent),
    # reload the turn-2 snapshot, continue turns 3-4 from the
    # reconstructed prefix + the same new tokens.
    restarted_store = KVSnapshotStore(os.path.join(tmp, "store"))
    snap = restarted_store.load(session_id, turn=restart_after_turn)
    recon = snap.decode()
    max_err = 0.0
    for turn in range(restart_after_turn + 1, n_turns + 1):
        n_tok = base_tokens + (turn - 1) * step
        # New tokens for this turn come from the same seeded
        # generation (their true KV); the prefix is the
        # reconstructed snapshot, never re-prefilled.
        continued = {
            "K": np.concatenate(
                [recon["K"], full_kv["K"][snap.n_tokens:n_tok]], axis=0),
            "V": np.concatenate(
                [recon["V"], full_kv["V"][snap.n_tokens:n_tok]], axis=0),
        }
        token, out_recon = _continuation(continued, vocab_w)
        orig_prefix = {"K": full_kv["K"][:n_tok], "V": full_kv["V"][:n_tok]}
        _, out_orig = _continuation(orig_prefix, vocab_w)
        max_err = max(max_err, float(
            np.abs(out_recon.astype(np.float64)
                   - out_orig.astype(np.float64)).max()))
        diag = _continuation_diagnostics(orig_prefix, continued, vocab_w)
        turns[turn - 1]["continuation_restarted"] = token
        turns[turn - 1]["restarted"] = True
        turns[turn - 1]["top2_gap"] = diag["top2_gap"]
        turns[turn - 1]["max_logit_error"] = diag["max_logit_error"]
        # Re-snapshot the continued session (turns after the
        # restart also persist, as in a real loop).
        restarted_store.save(session_id, turn, continued, profile_name)
        if verbose:
            print("turn %d (after restart): continuation %d "
                  "(uninterrupted %d)"
                  % (turn, token, turns[turn - 1]["continuation_uninterrupted"]))
    identical = all(
        t["continuation_restarted"] == t["continuation_uninterrupted"]
        for t in turns if t["restarted"]
    )
    return {
        "profile": profile_name,
        "session_id": session_id,
        "workdir": tmp,
        "turns": turns,
        "restart_after_turn": restart_after_turn,
        "restart_continuation_identical": bool(identical),
        "restart_max_abs_error": max_err,
        "snapshot_ratio_turn2": snap.combined_ratio,
        "basis_cached_on_disk": os.path.exists(
            os.path.join(tmp, "store", "bases", "%s.npz" % profile_name)),
    }


# ---------------------------------------------------------------------------
# Eval: ratio vs accuracy per profile + the restart demo.
# ---------------------------------------------------------------------------

def run_kvcache_eval(verbose: bool = True) -> dict:
    """Full P8 eval: transform-code each profile, measure ratio vs
    accuracy, then run the restart demo.

    For each profile: generate the synthetic KV, fit coders on
    the 128-token calibration slice, encode K and V, report the
    ratio (vs BF16), decode, and run the deterministic
    attention/reasoning check. The ~20x target is reported as
    measured — if it lands below, it lands below.
    """
    profiles: dict[str, dict] = {}
    for name in PROFILES:
        kv = generate_synthetic_kv(name)
        store = KVSnapshotStore(tempfile.mkdtemp(prefix="kvcache-eval-"))
        coders = store.get_coders(name, kv)
        entry: dict = {"shape": list(kv["K"].shape), "tensors": {}}
        blobs: dict[str, TransformCodedTensor] = {}
        t0 = time.perf_counter()
        for kind in ("K", "V"):
            blob = TransformCodedTensor.encode(kv[kind], coders[kind])
            blobs[kind] = blob
            # Serialisation round-trip through the blob bytes.
            rt = TransformCodedTensor.from_bytes(blob.to_bytes()).decode()
            entry["tensors"][kind] = {
                "original_bytes": blob.original_n_bytes,
                "compressed_bytes": blob.n_bytes,
                "ratio": blob.ratio,
                "k_components": int(coders[kind].k),
                "mean_bits": float(np.mean(coders[kind].bits)),
                "bits_per_code": blob.bits_per_code,
                "serialised_roundtrip_max_abs": float(
                    np.abs(rt.astype(np.float64)
                           - kv[kind].astype(np.float64)).max()),
            }
        entry["encode_wall_s"] = time.perf_counter() - t0
        total_orig = sum(b.original_n_bytes for b in blobs.values())
        total_comp = sum(b.n_bytes for b in blobs.values())
        entry["combined_ratio"] = total_orig / total_comp
        recon = {kind: blobs[kind].decode() for kind in ("K", "V")}
        entry["reasoning"] = reasoning_eval(kv, recon)
        profiles[name] = entry
        if verbose:
            r = entry["reasoning"]
            print(
                "%s %s: K %.2fx V %.2fx combined %.2fx; "
                "k=%d/%d, mean bits %.2f; attention max abs %.2e "
                "MSE %.2e; attended-position top-1 %.3f; vocab top-1 %.3f"
                % (
                    name, "x".join(str(s) for s in kv["K"].shape),
                    entry["tensors"]["K"]["ratio"],
                    entry["tensors"]["V"]["ratio"],
                    entry["combined_ratio"],
                    entry["tensors"]["K"]["k_components"],
                    entry["tensors"]["V"]["k_components"],
                    entry["tensors"]["K"]["mean_bits"],
                    r["max_abs_error"], r["mse"],
                    r["attended_position_top1"], r["vocab_top1"],
                )
            )
    demo = run_session_demo(verbose=verbose)
    return {
        "profiles": profiles,
        "demo": demo,
        "done_20x": all(p["combined_ratio"] >= 20.0 for p in profiles.values()),
        "note": (
            "Synthetic BF16-baseline KV only (fixed correlated "
            "profiles, seeded RNG) — no real model, no real "
            "reasoning benchmark. The restart is a fresh store "
            "object reloading from disk, not a separate process."
        ),
    }


def print_kvcache_report(result: dict) -> None:
    print("P8 — KV-cache snapshot store with transform coding "
          "(synthetic KV, NumPy)")
    print("NOTE: %s" % result["note"])
    for name, p in result["profiles"].items():
        print("\n%s  shape %s" % (name, "x".join(str(s) for s in p["shape"])))
        for kind in ("K", "V"):
            t = p["tensors"][kind]
            print(
                "  %s: %d -> %d bytes (%.2fx), k=%d PCA components, "
                "mean bits %.2f, Huffman %.2f bits/code, "
                "serialised round-trip max abs %.2e"
                % (
                    kind, t["original_bytes"], t["compressed_bytes"],
                    t["ratio"], t["k_components"], t["mean_bits"],
                    t["bits_per_code"], t["serialised_roundtrip_max_abs"],
                )
            )
        print("  combined ratio: %.2fx" % p["combined_ratio"])
        r = p["reasoning"]
        print(
            "  reasoning vs uncompressed: attention max abs %.2e, "
            "MSE %.2e, attended-position top-1 %.3f, vocab top-1 %.3f "
            "(%d queries)"
            % (
                r["max_abs_error"], r["mse"],
                r["attended_position_top1"], r["vocab_top1"],
                r["n_queries"],
            )
        )
    d = result["demo"]
    print("\nSession demo (%s, session %s):" % (d["profile"], d["session_id"]))
    for t in d["turns"]:
        print(
            "  turn %d: %d tokens, snapshot %d bytes, continuation "
            "uninterrupted=%d%s"
            % (
                t["turn"], t["n_tokens"], t["snapshot_bytes"],
                t["continuation_uninterrupted"],
                (", restarted=%d (top-2 gap %.1e, max logit error %.1e)"
                 % (t["continuation_restarted"], t["top2_gap"],
                    t["max_logit_error"]))
                if t["restarted"] else "",
            )
        )
    print(
        "  restart after turn %d: continuation identical=%s, "
        "max attention-output error after restart %.2e, "
        "turn-2 snapshot ratio %.2fx, basis cached on disk=%s"
        % (
            d["restart_after_turn"],
            d["restart_continuation_identical"],
            d["restart_max_abs_error"],
            d["snapshot_ratio_turn2"],
            d["basis_cached_on_disk"],
        )
    )
    print("\n20x on both synthetic profiles: %s" % result["done_20x"])
