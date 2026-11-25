"""P7 — GAAC lossless KV codec (offline prototype).

GAAC (arXiv:2610.38210) compresses BF16 KV-cache exponents
losslessly: per-channel base-plus-offset codes with
group-adaptive widths. Exponents are the compressible part —
in a real KV cache each channel's exponents cluster around a
few dominant values — while mantissas are stored verbatim,
so decompression is bit-for-bit and the codec composes with
every lossy technique. The plan's done condition is 1.4x
lossless compression reproduced on 2+ models, with attention
outputs bit-identical.

SUBSTITUTIONS, stated plainly (this VM has no torch, no GPU,
no real model checkpoints, and Triton/Metal are unavailable;
do NOT read this module as a run on real checkpoints):
- There are no real checkpoints here. The codec runs on
  SYNTHETIC BF16 KV-cache tensors with realistic exponent
  structure (fixed profiles below, seeded RNG): per-channel
  exponent clustering — a few dominant exponent values per
  channel plus a tail — mimicking two model shapes. The
  profile parameters were fixed BEFORE any ratio was
  measured, and the README reports the ratio as synthetic-KV,
  not real-checkpoint. If the measured ratio misses 1.4x it
  is reported as measured, not tuned into existence.
- No fused Triton/Metal kernel exists. Phase 2 is a CPU
  analogue only: ``attention_onthefly`` decodes exponents in
  tiles during the NumPy attention loop without ever
  materialising the full decompressed cache, with a measured
  memory/time tradeoff vs full decompression. The real fused
  kernel is NOT built.

Codec design (one design decision, documented because it is
where a naive reading of the paper goes wrong):
- The base is PER CHANNEL (one byte per channel), and the
  width is GROUP-ADAPTIVE (one width per group of
  ``group_size`` channels, the narrowest width covering every
  offset in the group). A shared base per group would be
  blown up by the inter-channel magnitude spread — real KV
  channels differ by several exponent steps from each other,
  so a group base would force a wide offset range and kill
  the ratio. The per-channel base costs one byte per channel
  (negligible at sequence length >> 1) and keeps the width
  driven only by the within-channel spread, which is what
  actually clusters. Offsets are ``exp - base[channel]``,
  packed MSB-first at the group's width; mantissa (7 bits)
  + sign (1 bit) are stored verbatim as one byte per element,
  which is exactly GAAC's exponent-only lossless claim.
- Format (``CompressedKV.to_bytes``): magic ``GAAC``,
  version 1, group size (u16), ndim (u8), shape (u32 each),
  per-channel bases (C bytes), per-group widths (n_groups
  bytes), verbatim sign+mantissa bytes (N*C), then each
  group's packed offsets byte-aligned in group order.
  Byte-aligning groups costs at most one byte per group and
  makes random-access row decoding possible, which the
  on-the-fly attention relies on.
"""

from __future__ import annotations

import struct
import time

import numpy as np

MAGIC = b"GAAC"
VERSION = 1
DEFAULT_GROUP_SIZE = 16

# ---------------------------------------------------------------------------
# Synthetic KV profiles.
#
# FIXED BEFORE MEASURING: these parameters were chosen from the
# qualitative structure of real KV caches (per-channel exponent
# clustering, a few dominant values plus a tail) and frozen;
# they were not tuned to hit a ratio. Profile A mimics a
# 7B-class shape (d_head 128, 32 heads), profile B a small-model
# shape (d_head 64, 16 heads) with tighter clustering. Each
# profile is a (K, V) pair of BF16 bit-pattern tensors of shape
# (seq, heads, d_head); the last axis is the channel axis.
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "llama-7b-like": {
        "seq": 1024,
        "heads": 32,
        "d_head": 128,
        "seed": 7,
        # Per-channel base exponent ~ N(127, 3), clipped.
        "base_mean": 127.0,
        "base_std": 3.0,
        "base_lo": 112,
        "base_hi": 140,
        # Exponent deltas above the channel base: (delta, prob).
        "deltas": ((0, 0.70), (1, 0.20), (2, 0.07), (3, 0.01),
                   (4, 0.01), (5, 0.005), (6, 0.005)),
        "sign_prob": 0.5,
    },
    "small-model-like": {
        "seq": 2048,
        "heads": 16,
        "d_head": 64,
        "seed": 11,
        "base_mean": 124.0,
        "base_std": 2.0,
        "base_lo": 114,
        "base_hi": 134,
        "deltas": ((0, 0.90), (1, 0.08), (2, 0.02)),
        "sign_prob": 0.5,
    },
}

# Attention checks run on this many leading tokens (all heads):
# full-sequence attention is the same code path, just slower.
ATTN_CHECK_TOKENS = 256
OTF_BLOCK_TOKENS = 128


def generate_synthetic_kv(profile_name: str) -> dict[str, np.ndarray]:
    """Generate a synthetic (K, V) BF16 bit-pattern pair.

    Per channel: a base exponent drawn once, then per element a
    delta above it from the profile's table (dominant value +
    tail). Mantissas are uniform 7-bit; signs are Bernoulli.
    K and V share the profile statistics but draw independently
    (different RNG streams, same seed family), as in a real
    cache where K and V have the same exponent structure but
    different values. Returns uint16 arrays of shape
    (seq, heads, d_head) holding raw BF16 bit patterns.
    """
    profile = PROFILES[profile_name]
    rng = np.random.default_rng(profile["seed"])
    seq, heads, d = profile["seq"], profile["heads"], profile["d_head"]
    n_rows = seq * heads
    deltas, probs = zip(*profile["deltas"])

    def _one_tensor(stream_seed: int) -> np.ndarray:
        r = np.random.default_rng(stream_seed)
        bases = np.clip(
            np.round(
                r.normal(profile["base_mean"], profile["base_std"], size=d)
            ).astype(np.int64),
            profile["base_lo"],
            profile["base_hi"],
        ).astype(np.uint8)  # (d,)
        delta_idx = r.choice(len(deltas), size=(n_rows, d), p=np.array(probs))
        delta = np.array(deltas, dtype=np.uint8)[delta_idx]
        exponents = (bases[None, :] + delta).astype(np.uint16)  # (rows, d)
        mantissa = r.integers(0, 128, size=(n_rows, d), dtype=np.uint16)
        sign = (r.random((n_rows, d)) < profile["sign_prob"]).astype(np.uint16)
        bits = (sign << 15) | (exponents << 7) | mantissa
        return bits.reshape(seq, heads, d).astype(np.uint16)

    return {
        "K": _one_tensor(profile["seed"] * 1000 + 1),
        "V": _one_tensor(profile["seed"] * 1000 + 2),
    }


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """Reinterpret BF16 bit patterns as float32 (exact)."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# Bit packing helpers (vectorised, MSB-first within each offset).
# ---------------------------------------------------------------------------

def _pack_offsets(offsets: np.ndarray, width: int) -> bytes:
    """Pack uint8 offsets (each < 2**width) into bytes, MSB-first."""
    if width == 0 or offsets.size == 0:
        return b""
    flat = offsets.reshape(-1)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint8)
    bits = ((flat[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits).tobytes()


def _unpack_offsets(data: bytes, count: int, width: int) -> np.ndarray:
    """Inverse of ``_pack_offsets``: ``count`` offsets of ``width`` bits."""
    if width == 0 or count == 0:
        return np.zeros(count, dtype=np.uint8)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bits = bits[: count * width].reshape(count, width)
    shifts = np.arange(width - 1, -1, -1, dtype=np.int32)
    return (bits.astype(np.int32) << shifts).sum(axis=1).astype(np.uint8)


def _width_for(max_offset: int) -> int:
    """Narrowest bit width covering offsets 0..max_offset (0..8)."""
    if max_offset <= 0:
        return 0
    return int(max_offset).bit_length()


class CompressedKV:
    """A GAAC-compressed BF16 tensor.

    ``bases`` is per-channel (uint8, length C); ``widths`` is
    per-group (uint8, length n_groups); ``verbatim`` holds the
    sign+mantissa byte per element, row-major over the flattened
    (N, C) view; ``packed`` holds each group's offsets, byte
    aligned, in group order. ``group_byte_offsets`` gives the
    byte start of each group inside ``packed`` (for row decode).
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        group_size: int,
        bases: np.ndarray,
        widths: np.ndarray,
        verbatim: np.ndarray,
        packed: bytes,
        group_byte_offsets: np.ndarray,
    ):
        self.shape = tuple(shape)
        self.group_size = int(group_size)
        self.bases = np.asarray(bases, dtype=np.uint8)
        self.widths = np.asarray(widths, dtype=np.uint8)
        self.verbatim = np.asarray(verbatim, dtype=np.uint8)
        self.packed = bytes(packed)
        self.group_byte_offsets = np.asarray(group_byte_offsets, dtype=np.int64)

    # -- size ---------------------------------------------------------
    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 0

    @property
    def n_bytes(self) -> int:
        header = 4 + 1 + 2 + 1 + 4 * len(self.shape)
        return header + len(self.bases) + len(self.widths) + len(self.verbatim) + len(self.packed)

    @property
    def original_n_bytes(self) -> int:
        return self.n_elements * 2

    @property
    def ratio(self) -> float:
        return self.original_n_bytes / self.n_bytes if self.n_bytes else float("inf")

    # -- serialisation -------------------------------------------------
    def to_bytes(self) -> bytes:
        header = (
            MAGIC
            + struct.pack("<BHB", VERSION, self.group_size, len(self.shape))
            + struct.pack("<%dI" % len(self.shape), *self.shape)
        )
        return (
            header
            + self.bases.tobytes()
            + self.widths.tobytes()
            + self.verbatim.tobytes()
            + self.packed
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompressedKV":
        if data[:4] != MAGIC:
            raise ValueError("not a GAAC blob (bad magic)")
        version, group_size, ndim = struct.unpack_from("<BHB", data, 4)
        if version != VERSION:
            raise ValueError("unsupported GAAC version %d" % version)
        off = 4 + 4
        shape = struct.unpack_from("<%dI" % ndim, data, off) if ndim else ()
        off += 4 * ndim
        n_channels = shape[-1] if shape else 0
        n_rows = int(np.prod(shape[:-1])) if len(shape) > 1 else (1 if shape else 0)
        n_groups = (n_channels + group_size - 1) // group_size if n_channels else 0
        bases = np.frombuffer(data, dtype=np.uint8, count=n_channels, offset=off).copy()
        off += n_channels
        widths = np.frombuffer(data, dtype=np.uint8, count=n_groups, offset=off).copy()
        off += n_groups
        n_elem = n_rows * n_channels
        verbatim = np.frombuffer(data, dtype=np.uint8, count=n_elem, offset=off).copy()
        off += n_elem
        packed = data[off:]
        group_byte_offsets = np.zeros(n_groups, dtype=np.int64)
        pos = 0
        for g in range(n_groups):
            group_byte_offsets[g] = pos
            n_in_group = n_rows * min(group_size, n_channels - g * group_size)
            pos += (n_in_group * int(widths[g]) + 7) // 8
        if pos != len(packed):
            raise ValueError("GAAC blob truncated / corrupt (packed length mismatch)")
        return cls(shape, group_size, bases, widths, verbatim, packed, group_byte_offsets)


def _group_slices(n_channels: int, group_size: int):
    """(start, stop) channel slices per group."""
    return [
        (g * group_size, min((g + 1) * group_size, n_channels))
        for g in range((n_channels + group_size - 1) // group_size)
    ]


def compress(bits: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE) -> CompressedKV:
    """Lossless GAAC compression of a BF16 bit-pattern tensor.

    ``bits`` is a uint16 array whose last axis is the channel
    axis; any leading axes are flattened into rows. Returns a
    ``CompressedKV``. Deterministic; no RNG, no model.
    """
    bits = np.asarray(bits, dtype=np.uint16)
    if bits.size == 0:
        n_channels = bits.shape[-1] if bits.ndim else 0
        n_groups = (n_channels + group_size - 1) // group_size if n_channels else 0
        return CompressedKV(
            bits.shape, group_size,
            np.zeros(n_channels, dtype=np.uint8),
            np.zeros(n_groups, dtype=np.uint8),
            np.zeros(0, dtype=np.uint8), b"", np.zeros(n_groups, dtype=np.int64),
        )
    shape = bits.shape
    n_channels = shape[-1]
    rows = bits.reshape(-1, n_channels)
    exponents = ((rows >> 7) & 0xFF).astype(np.uint8)
    # Verbatim byte per element: sign bit + 7 mantissa bits.
    verbatim = ((rows >> 8) & 0x80).astype(np.uint8) | (rows & 0x7F).astype(np.uint8)
    verbatim = np.ascontiguousarray(verbatim.reshape(-1))
    # Per-channel base = the channel's minimum exponent, so every
    # offset is >= 0 and the width is the within-channel spread.
    bases = exponents.min(axis=0)
    offsets = (exponents.astype(np.int16) - bases.astype(np.int16)).astype(np.uint8)
    widths = np.zeros(len(_group_slices(n_channels, group_size)), dtype=np.uint8)
    packed_parts: list[bytes] = []
    group_byte_offsets = np.zeros(len(widths), dtype=np.int64)
    pos = 0
    for g, (lo, hi) in enumerate(_group_slices(n_channels, group_size)):
        group_off = offsets[:, lo:hi]
        width = _width_for(int(group_off.max()) if group_off.size else 0)
        widths[g] = width
        group_byte_offsets[g] = pos
        part = _pack_offsets(group_off, width)
        packed_parts.append(part)
        pos += len(part)
    return CompressedKV(
        shape, group_size, bases, widths, verbatim,
        b"".join(packed_parts), group_byte_offsets,
    )


def decompress(blob: CompressedKV) -> np.ndarray:
    """Bit-exact inverse of ``compress`` (uint16, original shape)."""
    n_channels = blob.shape[-1] if blob.shape else 0
    n_rows = int(np.prod(blob.shape[:-1])) if len(blob.shape) > 1 else (1 if blob.shape else 0)
    if blob.n_elements == 0:
        return np.zeros(blob.shape, dtype=np.uint16)
    exponents = np.zeros((n_rows, n_channels), dtype=np.uint8)
    for g, (lo, hi) in enumerate(_group_slices(n_channels, blob.group_size)):
        width = int(blob.widths[g])
        n_in_group = n_rows * (hi - lo)
        start = int(blob.group_byte_offsets[g])
        n_bytes = (n_in_group * width + 7) // 8
        group_off = _unpack_offsets(blob.packed[start:start + n_bytes], n_in_group, width)
        exponents[:, lo:hi] = (group_off.reshape(n_rows, hi - lo)
                               + blob.bases[lo:hi][None, :]).astype(np.uint8)
    verbatim = blob.verbatim.reshape(n_rows, n_channels)
    sign = (verbatim & 0x80).astype(np.uint16) << 8
    mantissa = (verbatim & 0x7F).astype(np.uint16)
    bits = sign | (exponents.astype(np.uint16) << 7) | mantissa
    return bits.reshape(blob.shape)


def decode_rows(blob: CompressedKV, rows: np.ndarray) -> np.ndarray:
    """Decode only the given flattened rows (uint16, shape (len(rows), C)).

    Random access into the packed offset stream: each group's
    bytes are located via ``group_byte_offsets`` and only that
    group's slice is unpacked. This is what the on-the-fly
    attention uses — no full decompression happens.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n_channels = blob.shape[-1] if blob.shape else 0
    n_rows = int(np.prod(blob.shape[:-1])) if len(blob.shape) > 1 else 0
    if rows.size == 0 or n_channels == 0:
        return np.zeros((len(rows), n_channels), dtype=np.uint16)
    exponents = np.zeros((len(rows), n_channels), dtype=np.uint8)
    for g, (lo, hi) in enumerate(_group_slices(n_channels, blob.group_size)):
        width = int(blob.widths[g])
        gc = hi - lo
        n_in_group = n_rows * gc
        start = int(blob.group_byte_offsets[g])
        n_bytes = (n_in_group * width + 7) // 8
        group_off = _unpack_offsets(blob.packed[start:start + n_bytes], n_in_group, width)
        group_off = group_off.reshape(n_rows, gc)
        exponents[:, lo:hi] = group_off[rows] + blob.bases[lo:hi][None, :]
    verbatim = blob.verbatim.reshape(n_rows, n_channels)[rows]
    sign = (verbatim & 0x80).astype(np.uint16) << 8
    mantissa = (verbatim & 0x7F).astype(np.uint16)
    return sign | (exponents.astype(np.uint16) << 7) | mantissa


# ---------------------------------------------------------------------------
# Attention (NumPy scaled dot-product, float32) and the on-the-fly variant.
# ---------------------------------------------------------------------------

def attention(q_bits: np.ndarray, k_bits: np.ndarray, v_bits: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention over BF16 bit-pattern tensors.

    Shapes (seq_q, heads, d), (seq_k, heads, d), (seq_k, heads, d);
    returns float32 (seq_q, heads, d). Computed per head in
    float32; no masking, no dropout — the check is bit-identity
    of the output, not a model result.
    """
    q = bf16_bits_to_float32(q_bits)
    k = bf16_bits_to_float32(k_bits)
    v = bf16_bits_to_float32(v_bits)
    d = q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    out = np.zeros_like(q)
    for h in range(q.shape[1]):
        scores = (q[:, h, :] @ k[:, h, :].T) * scale
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        out[:, h, :] = probs @ v[:, h, :]
    return out


def attention_onthefly(
    k_blob: CompressedKV,
    v_blob: CompressedKV,
    q_bits: np.ndarray,
    block: int = OTF_BLOCK_TOKENS,
) -> tuple[np.ndarray, dict]:
    """Attention that decodes K/V exponents in tiles, on the fly.

    Phase 2 CPU analogue (NOT a Triton/Metal fused kernel — none
    is built here): the loop walks key blocks of ``block``
    tokens; for each block it decodes only that block's K/V rows
    from the compressed blobs (``decode_rows``), converts the
    tile to float32, and accumulates scores / the weighted sum.
    The full decompressed cache is never materialised: peak
    extra memory is the compressed blobs (already held) plus
    one tile of float32 K/V and the (seq_q, seq_k) score matrix
    the attention itself needs. One honest caveat: the weighted
    sum is accumulated block by block, so float32 summation
    order differs from a single BLAS ``probs @ V`` — the
    on-the-fly output is equivalent to the full path to float32
    rounding (max abs diff ~1e-6), not bit-identical. The
    bit-identical guarantee belongs to the codec itself
    (decompressed bytes == original bytes), which is what
    makes the full-path attention bit-identical.
    Returns (output, stats) where stats reports the tile peak
    bytes and wall time.
    """
    seq_q, heads, d = q_bits.shape
    seq_k = int(np.prod(k_blob.shape[:-1])) // heads
    q = bf16_bits_to_float32(q_bits)
    scale = 1.0 / (d ** 0.5)
    scores = np.zeros((seq_q, heads, seq_k), dtype=np.float32)
    t0 = time.perf_counter()
    for start in range(0, seq_k, block):
        rows = np.arange(start, min(start + block, seq_k))
        # Rows are flattened (seq, head) pairs: row r of the blob
        # is token (r // heads), head (r % heads).
        flat_rows = (rows[:, None] * heads + np.arange(heads)[None, :]).reshape(-1)
        k_tile = bf16_bits_to_float32(
            decode_rows(k_blob, flat_rows).reshape(len(rows), heads, d)
        )
        for h in range(heads):
            scores[:, h, start:start + len(rows)] = (
                q[:, h, :] @ k_tile[:, h, :].T
            ) * scale
    scores = scores - scores.max(axis=2, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=2, keepdims=True)
    out = np.zeros_like(q)
    for start in range(0, seq_k, block):
        rows = np.arange(start, min(start + block, seq_k))
        flat_rows = (rows[:, None] * heads + np.arange(heads)[None, :]).reshape(-1)
        v_tile = bf16_bits_to_float32(
            decode_rows(v_blob, flat_rows).reshape(len(rows), heads, d)
        )
        for h in range(heads):
            out[:, h, :] += probs[:, h, start:start + len(rows)] @ v_tile[:, h, :]
    wall_s = time.perf_counter() - t0
    # Peak extra memory beyond the blobs + q + output: one tile of
    # float32 K (decoded twice, once per pass) and the score/prob
    # matrices the attention itself needs.
    tile_bytes = block * heads * d * 4  # one float32 K or V tile
    stats = {
        "wall_s": wall_s,
        "tile_peak_extra_bytes": tile_bytes,
        "scores_bytes": int(scores.nbytes + probs.nbytes),
        "block_tokens": block,
    }
    return out, stats


# ---------------------------------------------------------------------------
# Eval: ratio per profile, bit-exactness, attention identity, on-the-fly.
# ---------------------------------------------------------------------------

def run_gaac_eval(verbose: bool = True) -> dict:
    """Full P7 eval on the two fixed synthetic KV profiles.

    For each profile: compress K and V, report the ratio, verify
    decompressed bytes == original bytes (exact), verify attention
    outputs from original vs decompressed KV are bit-identical,
    and run the on-the-fly attention for equivalence plus its
    memory/time tradeoff vs full decompression. Nothing here
    touches a real checkpoint or a GPU kernel.
    """
    profiles: dict[str, dict] = {}
    for name in PROFILES:
        kv = generate_synthetic_kv(name)
        entry: dict = {"shape": list(kv["K"].shape), "tensors": {}}
        blobs: dict[str, CompressedKV] = {}
        for kind in ("K", "V"):
            blob = compress(kv[kind])
            blobs[kind] = blob
            rt = decompress(blob)
            entry["tensors"][kind] = {
                "original_bytes": blob.original_n_bytes,
                "compressed_bytes": blob.n_bytes,
                "ratio": blob.ratio,
                "widths": {int(w): int((blob.widths == w).sum())
                           for w in np.unique(blob.widths)},
                "roundtrip_bit_exact": bool(np.array_equal(rt, kv[kind])),
                "serialised_roundtrip_bit_exact": bool(
                    np.array_equal(
                        decompress(CompressedKV.from_bytes(blob.to_bytes())),
                        kv[kind],
                    )
                ),
            }
        total_orig = sum(b.original_n_bytes for b in blobs.values())
        total_comp = sum(b.n_bytes for b in blobs.values())
        entry["combined_ratio"] = total_orig / total_comp

        # Attention checks on the leading tokens (all heads).
        n_tok = min(ATTN_CHECK_TOKENS, kv["K"].shape[0])
        q_bits = kv["K"][:n_tok]  # K slice reused as Q (same dtype/shape)
        k_orig, v_orig = kv["K"][:n_tok], kv["V"][:n_tok]
        k_dec = decompress(blobs["K"])[:n_tok]
        v_dec = decompress(blobs["V"])[:n_tok]
        out_orig = attention(q_bits, k_orig, v_orig)
        out_dec = attention(q_bits, k_dec, v_dec)
        entry["attention_bit_identical"] = bool(
            np.array_equal(out_orig, out_dec)
        ) and bool(np.array_equal(out_orig.view(np.uint8), out_dec.view(np.uint8)))

        # On-the-fly attention on the same slice, vs full decompress.
        k_slice = compress(k_orig)
        v_slice = compress(v_orig)
        t0 = time.perf_counter()
        k_dec_timed = decompress(blobs["K"])[:n_tok]
        v_dec_timed = decompress(blobs["V"])[:n_tok]
        out_full = attention(q_bits, k_dec_timed, v_dec_timed)
        full_s = time.perf_counter() - t0
        out_otf, otf_stats = attention_onthefly(k_slice, v_slice, q_bits)
        entry["onthefly_equivalent"] = bool(
            np.allclose(out_otf, out_full, rtol=1e-5, atol=1e-6)
        )
        entry["onthefly_max_abs_diff"] = float(
            np.abs(out_otf - out_full).max()
        )
        # Full-decompress path materialises both BF16 caches plus
        # their float32 copies; on-the-fly holds one float32 tile.
        n_elem_slice = k_orig.size
        full_extra = 2 * (n_elem_slice * 2 + n_elem_slice * 4)
        entry["onthefly"] = {
            "wall_s": otf_stats["wall_s"],
            "full_decompress_attention_wall_s": full_s,
            "tile_peak_extra_bytes": otf_stats["tile_peak_extra_bytes"],
            "full_path_extra_bytes": full_extra,
            "extra_bytes_ratio": full_extra / otf_stats["tile_peak_extra_bytes"],
            "block_tokens": otf_stats["block_tokens"],
        }
        profiles[name] = entry
        if verbose:
            print(
                "%s %s: K %.3fx V %.3fx combined %.3fx; "
                "round-trip bit-exact=%s; attention bit-identical=%s; "
                "on-the-fly equivalent=%s (max abs diff %.2e)"
                % (
                    name,
                    "x".join(str(s) for s in kv["K"].shape),
                    entry["tensors"]["K"]["ratio"],
                    entry["tensors"]["V"]["ratio"],
                    entry["combined_ratio"],
                    all(t["roundtrip_bit_exact"]
                        for t in entry["tensors"].values()),
                    entry["attention_bit_identical"],
                    entry["onthefly_equivalent"],
                    entry["onthefly_max_abs_diff"],
                )
            )
    return {
        "profiles": profiles,
        "done_1p4x_on_both": all(
            p["combined_ratio"] >= 1.4 for p in profiles.values()
        ),
        "note": (
            "Synthetic BF16 KV only (fixed profiles, seeded RNG) — "
            "no real checkpoints, no Triton/Metal kernel. "
            "On-the-fly attention is a NumPy CPU analogue."
        ),
    }


def print_gaac_report(result: dict) -> None:
    print("P7 — GAAC lossless KV codec (synthetic BF16 KV, NumPy)")
    print("NOTE: %s" % result["note"])
    for name, p in result["profiles"].items():
        print("\n%s  shape %s" % (name, "x".join(str(s) for s in p["shape"])))
        for kind in ("K", "V"):
            t = p["tensors"][kind]
            print(
                "  %s: %d -> %d bytes (%.3fx), group widths %s, "
                "round-trip bit-exact %s, serialised round-trip %s"
                % (
                    kind, t["original_bytes"], t["compressed_bytes"],
                    t["ratio"], t["widths"], t["roundtrip_bit_exact"],
                    t["serialised_roundtrip_bit_exact"],
                )
            )
        print("  combined ratio: %.3fx" % p["combined_ratio"])
        print("  attention bit-identical (original vs decompressed KV): %s"
              % p["attention_bit_identical"])
        o = p["onthefly"]
        print(
            "  on-the-fly attention equivalent: %s (max abs diff %.2e); "
            "wall %.2fs vs full decompress+attention %.2fs; peak extra "
            "%.1f KB vs %.1f KB full path (%.1fx less)"
            % (
                p["onthefly_equivalent"], p["onthefly_max_abs_diff"],
                o["wall_s"], o["full_decompress_attention_wall_s"],
                o["tile_peak_extra_bytes"] / 1024,
                o["full_path_extra_bytes"] / 1024,
                o["extra_bytes_ratio"],
            )
        )
    print("\n1.4x on both synthetic profiles: %s" % result["done_1p4x_on_both"])
