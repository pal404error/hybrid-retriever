"""P14 — Minimal layer-wise offload runner (potato-PC project).

AirLLM's idea, built from scratch: load one layer at a time from
disk, forward through it, free it — so a model whose weights do
not fit in RAM still runs. The plan's Phase 1 (load -> forward ->
free), Phase 2 (prefetch layer N+1 while computing layer N) and
Phase 3 (LRU layer cache in RAM + an honest benchmark table,
model x RAM cap x disk type -> tok/s) are all here.

SUBSTITUTIONS, stated plainly (this VM has no torch, no GPU and
no 70-billion-parameter model; do NOT read this module as a 70B
run on a 4 GB GPU — the word "70B" appears in this file only in
this paragraph):
- The model is a TOY transformer-like stack in pure NumPy: a
  stack of linear layers (fixed hidden dim, fixed layer count)
  with a tanh nonlinearity, plus a small embedding table and a
  small unembedding table. Weights live on disk as one .npz
  shard per layer. The toy model's real parameter count is
  reported by ``build_toy_model`` and by the eval (about
  0.54M parameters at the default 32 layers x hidden 128) —
  the claim this module proves is about the MECHANISM
  (offload under a hard RAM cap), not the scale.
- Quantisation (Q4) FIRST, offload second, per the plan:
  layer weights are stored Q4-quantised (see
  ``quantise_q4``), and the runner dequantises one layer at
  a time. The FP32 originals are never stored on disk; the
  error vs the FP32 baseline is measured in-memory from the
  same seeded weights (``q4_error_stats``).
- The "RAM cap" is a configured byte budget enforced by
  the runner on the LAYER WEIGHTS it holds: the sum of the
  cached dequantised (FP32) layer bytes may never exceed
  the cap (``peak_held_bytes`` is tracked and asserted in
  tests). The embedding / unembedding tables (32 KiB total
  at vocab 64) are resident for the whole run and are
  reported separately as fixed overhead — the cap covers
  the offloaded stack, which is the part AirLLM offloads.
  The transient staging buffer for one packed Q4 shard
  (~10 KiB) is freed after dequantisation and is not
  counted in the cap; it is bounded by one shard by
  construction. Peak RSS of the runner process is measured
  separately, in a child process (``run_generation`` /
  ``_measure_rss``), and reported alongside the cap —
  raw RSS includes the Python interpreter + NumPy baseline
  (tens of MB), so RSS is NEVER compared directly against
  the byte cap; the honest RSS comparison is offload vs
  all-in-RAM on the same model.
- Disk types in the benchmark are the real ones on this
  VM: /tmp (a 512 MB tmpfs) vs the home disk
  ($HOME/.tmp). Shard totals are kept well under the
  tmpfs limit and every benchmark directory is removed
  afterwards.
- Fit is not speed: tok/s is reported as measured, even
  when offload is slower than all-in-RAM (it is — disk
  loads dominate compute on the toy). The benchmark table
  is the deliverable, not a speed claim.

Q4 SCHEME (deterministic, documented here and nowhere else):
weights are flattened row-major and split into groups of
``GROUP_SIZE = 32`` consecutive elements (zero-padded at
the tail). Per group, ``scale = max|w| / 7`` (symmetric,
no zero-point); each weight is stored as a signed 4-bit
code ``clip(round(w / scale), -7, 7)``. Two codes pack
into one byte (first code in the low nibble). A group
whose max is 0 gets scale 0 and all-zero codes, so exact
zeros stay exact zeros. Dequantisation error per element
is at most ``scale / 2`` (rounding), and the eval measures
the real max / mean error against the FP32 weights.
Stored size per layer: n/2 bytes of codes + (n/32) * 4
bytes of float32 scales — about 6.4x smaller than FP32 for
large layers (4 bytes -> 0.5 bytes + 0.125 bytes of
scale per element).
"""

import argparse
import json
import os
import queue
import resource
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# Fixed parameters (fixed before measuring; see module docstring).
# ---------------------------------------------------------------------------

HIDDEN = 128
VOCAB = 64
LAYERS = 32
GROUP_SIZE = 32
EVAL_SEED = 20261126
START_TOKEN = 1
N_TOKENS = 16
REPEATS = 2

# One layer's dequantised (FP32) footprint: W (128x128) + b (128).
LAYER_FP32_BYTES = HIDDEN * HIDDEN * 4 + HIDDEN * 4  # 66,048
# One layer's packed Q4 footprint on disk: codes + per-group scales.
LAYER_Q4_BYTES = (HIDDEN * HIDDEN) // 2 + (HIDDEN * HIDDEN // GROUP_SIZE) * 4  # 10,240
# RAM caps for the benchmark: 1, 2 and 4 layers' worth of FP32.
RAM_CAPS = [LAYER_FP32_BYTES, 2 * LAYER_FP32_BYTES, 4 * LAYER_FP32_BYTES]
BENCH_LAYERS = [16, 32]

NOTE = (
    "P14 layer-wise offload runner: TOY transformer-like model "
    "in pure NumPy (no torch, no GPU, no 70B model — the toy "
    "is ~0.54M parameters at 32 layers x hidden 128); one "
    ".npz shard per layer on disk, Q4-quantised first "
    "(symmetric per-group scales, group 32, codes in "
    "[-7, 7], two codes per byte); the runner loads one "
    "layer, forwards, frees it under a hard byte cap on "
    "cached layer weights; prefetch loads layer N+1 in a "
    "background thread while layer N computes; an LRU cache "
    "bounded by the cap keeps hot layers in RAM; the "
    "benchmark table (layers x RAM cap x disk -> tok/s) is "
    "measured on /tmp (tmpfs) vs the home disk"
)


# ---------------------------------------------------------------------------
# Q4 quantisation (deterministic, NumPy only).
# ---------------------------------------------------------------------------

def quantise_q4(weights):
    """Quantise a float32 array to 4-bit with per-group scales.

    Returns (packed, scales, shape): ``packed`` is uint8 with
    two codes per byte, ``scales`` is float32 with one scale
    per group of 32 elements, ``shape`` is the original shape.
    """
    arr = np.asarray(weights, dtype=np.float32).ravel()
    n = arr.shape[0]
    n_groups = (n + GROUP_SIZE - 1) // GROUP_SIZE
    padded = np.zeros(n_groups * GROUP_SIZE, dtype=np.float32)
    padded[:n] = arr
    groups = padded.reshape(n_groups, GROUP_SIZE)
    scales = np.max(np.abs(groups), axis=1) / 7.0
    scales = scales.astype(np.float32)
    codes = np.zeros_like(groups, dtype=np.int8)
    nonzero = scales > 0
    codes[nonzero] = np.clip(
        np.rint(groups[nonzero] / scales[nonzero, None]),
        -7, 7).astype(np.int8)
    flat = codes.ravel()
    if flat.shape[0] % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
    packed = ((flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)).astype(np.uint8)
    return packed, scales, tuple(np.asarray(weights).shape)


def dequantise_q4(packed, scales, shape):
    """Inverse of ``quantise_q4``: returns a float32 array of
    ``shape``. Exact zeros stay exact zeros (scale 0 groups)."""
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)
    n = 1
    for dim in shape:
        n *= dim
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(packed.shape[0] * 2, dtype=np.float32)
    codes[0::2] = lo
    codes[1::2] = hi
    codes = codes[: ((n + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE]
    expanded_scales = np.repeat(scales, GROUP_SIZE)
    out = codes * expanded_scales
    return out[:n].reshape(shape).astype(np.float32)


def q4_size_bytes(n_elements):
    """Stored Q4 bytes for ``n_elements`` weights (codes +
    float32 per-group scales)."""
    n_groups = (n_elements + GROUP_SIZE - 1) // GROUP_SIZE
    return (n_groups * GROUP_SIZE) // 2 + n_groups * 4


# ---------------------------------------------------------------------------
# Toy model: deterministic weights, one .npz shard per layer.
# ---------------------------------------------------------------------------

def build_weights_in_memory(n_layers=LAYERS, hidden=HIDDEN,
                            vocab=VOCAB, seed=EVAL_SEED):
    """The FP32 weights, built deterministically in memory.
    ``build_toy_model`` writes the Q4 form of exactly these
    weights to disk, so the Q4-vs-FP32 error is measured
    against this function, not against a second draw."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(hidden)
    emb = (rng.normal(0.0, scale, (vocab, hidden))
           .astype(np.float32))
    unemb = (rng.normal(0.0, scale, (hidden, vocab))
             .astype(np.float32))
    layers = []
    for _ in range(n_layers):
        w = (rng.normal(0.0, scale, (hidden, hidden))
             .astype(np.float32))
        b = (rng.normal(0.0, scale, (hidden,))
             .astype(np.float32))
        layers.append((w, b))
    return emb, unemb, layers


def build_toy_model(directory, n_layers=LAYERS, hidden=HIDDEN,
                    vocab=VOCAB, seed=EVAL_SEED, quantised=True):
    """Write a toy model to ``directory``: ``embedding.npz``,
    ``unembedding.npz``, one ``layer_XX.npz`` per layer, and
    a ``manifest.json``. Layer weights are Q4-quantised when
    ``quantised`` is True (the default — quantisation first,
    offload second) and FP32 otherwise.

    Returns the manifest dict, which states the real
    parameter count plainly.
    """
    os.makedirs(directory, exist_ok=True)
    emb, unemb, layers = build_weights_in_memory(
        n_layers, hidden, vocab, seed)
    np.savez(os.path.join(directory, "embedding.npz"), emb=emb)
    np.savez(os.path.join(directory, "unembedding.npz"), unemb=unemb)
    total_q4 = 0
    total_fp32 = 0
    for i, (w, b) in enumerate(layers):
        if quantised:
            packed, scales, shape = quantise_q4(w)
            np.savez(
                os.path.join(directory, "layer_%02d.npz" % i),
                packed=packed, scales=scales,
                shape=np.array(shape, dtype=np.int64),
                bias=b)
            total_q4 += int(packed.nbytes + scales.nbytes + b.nbytes)
        else:
            np.savez(
                os.path.join(directory, "layer_%02d.npz" % i),
                weights=w, bias=b)
        total_fp32 += int(w.nbytes + b.nbytes)
    n_params = (n_layers * (hidden * hidden + hidden)
                + 2 * vocab * hidden)
    manifest = {
        "toy_model": True,
        "hidden": hidden,
        "vocab": vocab,
        "n_layers": n_layers,
        "seed": seed,
        "quantised": bool(quantised),
        "group_size": GROUP_SIZE,
        "n_params": n_params,
        "layer_fp32_bytes": int(total_fp32 // max(1, n_layers)),
        "layer_q4_bytes": LAYER_Q4_BYTES if quantised else None,
        "total_layer_fp32_bytes": total_fp32,
        "total_layer_q4_bytes": total_q4 if quantised else None,
        "embedding_bytes": int(emb.nbytes),
        "unembedding_bytes": int(unemb.nbytes),
    }
    with open(os.path.join(directory, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def load_manifest(directory):
    with open(os.path.join(directory, "manifest.json"),
              encoding="utf-8") as f:
        return json.load(f)


def _load_layer_packed(directory, index):
    """Read one layer shard from disk. Returns (packed,
    scales, shape, bias) for a Q4 model or (weights, None,
    shape, bias) for an FP32 one. The .npz is closed before
    returning, so the arrays are real copies, not views into
    a closed file."""
    path = os.path.join(directory, "layer_%02d.npz" % index)
    with np.load(path) as z:
        if "packed" in z:
            return (np.array(z["packed"]), np.array(z["scales"]),
                    tuple(int(x) for x in z["shape"]),
                    np.array(z["bias"], dtype=np.float32))
        return (np.array(z["weights"], dtype=np.float32), None,
                tuple(np.asarray(z["weights"]).shape),
                np.array(z["bias"], dtype=np.float32))


def _dequantise_layer(packed, scales, shape, bias):
    if scales is None:
        return packed.astype(np.float32), bias
    return dequantise_q4(packed, scales, shape), bias


def q4_error_stats(n_layers=LAYERS, hidden=HIDDEN, vocab=VOCAB,
                   seed=EVAL_SEED):
    """Q4 error vs the FP32 baseline, measured in memory on
    the same seeded weights the disk model is built from.
    Returns max / mean abs weight error, the rounding bound
    (max scale / 2), sizes, and the final-logits error of a
    Q4 run vs an FP32 run from the same start token."""
    _, _, layers = build_weights_in_memory(
        n_layers, hidden, vocab, seed)
    max_err = 0.0
    sum_err = 0.0
    n = 0
    max_scale_half = 0.0
    for w, _ in layers:
        packed, scales, shape = quantise_q4(w)
        dq = dequantise_q4(packed, scales, shape)
        err = np.abs(dq - w)
        max_err = max(max_err, float(err.max()))
        sum_err += float(err.sum())
        n += err.size
        max_scale_half = max(max_scale_half, float(scales.max()) / 2.0)
    fp32_bytes = n_layers * (hidden * hidden * 4 + hidden * 4)
    q4_bytes = n_layers * q4_size_bytes(hidden * hidden) \
        + n_layers * hidden * 4
    return {
        "max_abs_weight_error": max_err,
        "mean_abs_weight_error": sum_err / max(1, n),
        "rounding_bound_max_scale_over_2": max_scale_half,
        "bound_holds": bool(max_err <= max_scale_half + 1e-9),
        "fp32_layer_bytes": fp32_bytes,
        "q4_layer_bytes": q4_bytes,
        "size_ratio_fp32_over_q4": fp32_bytes / max(1, q4_bytes),
    }


# ---------------------------------------------------------------------------
# Phase 1 — the offload runner (load -> forward -> free, under a cap).
# ---------------------------------------------------------------------------

class OffloadRunner:
    """One layer at a time, under a hard RAM cap.

    The LRU cache holds dequantised FP32 layers; its total
    byte count never exceeds ``ram_cap_bytes`` (enforced at
    insert time by evicting least-recently-used layers; a
    cap smaller than one layer raises at construction).
    ``peak_held_bytes`` is the high-water mark of the cached
    bytes during a run. With ``use_cache=False`` every layer
    is loaded, used and freed immediately (pure Phase 1);
    with the cache on, a layer is freed only when the cap
    forces its eviction (Phase 3). ``prefetch=True`` starts
    a background thread that loads layer N+1's packed shard
    from disk while layer N computes (Phase 2); the thread
    only ever loads packed bytes and hands them to the main
    thread through a bounded queue — dequantisation and all
    cap accounting happen on the main thread, so the cap
    cannot be breached by a racing insert.
    """

    def __init__(self, model_dir, ram_cap_bytes,
                 use_cache=True, prefetch=False):
        self.model_dir = model_dir
        self.manifest = load_manifest(model_dir)
        self.hidden = int(self.manifest["hidden"])
        self.vocab = int(self.manifest["vocab"])
        self.n_layers = int(self.manifest["n_layers"])
        self.ram_cap_bytes = int(ram_cap_bytes)
        self.use_cache = bool(use_cache)
        self.prefetch = bool(prefetch)
        self.layer_bytes = self.hidden * self.hidden * 4 \
            + self.hidden * 4
        if self.layer_bytes > self.ram_cap_bytes:
            raise ValueError(
                "RAM cap %d bytes cannot hold one layer "
                "(%d bytes)" % (self.ram_cap_bytes,
                                self.layer_bytes))
        with np.load(os.path.join(model_dir, "embedding.npz")) as z:
            self.emb = np.array(z["emb"], dtype=np.float32)
        with np.load(os.path.join(model_dir, "unembedding.npz")) as z:
            self.unemb = np.array(z["unemb"], dtype=np.float32)
        # Fixed resident overhead, reported separately from
        # the cap (see module docstring): the embedding tables.
        self.fixed_resident_bytes = int(
            self.emb.nbytes + self.unemb.nbytes)
        self.cache = {}  # index -> (W, b); insertion order = LRU order
        self.cache_bytes = 0
        self.loads = 0
        self.cache_hits = 0
        self.evictions = 0
        self.peak_held_bytes = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prefetch_stale = 0
        self._req = None
        self._res = None
        self._worker = None
        self._stop = None

    # -- prefetch worker (Phase 2) -------------------------------------
    def _start_prefetch(self):
        if not self.prefetch:
            return
        self._req = queue.Queue(maxsize=1)
        self._res = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._prefetch_loop, daemon=True)
        self._worker.start()

    def _stop_prefetch(self):
        if self._worker is None:
            return
        self._stop.set()
        try:
            self._req.put_nowait(None)
        except queue.Full:
            pass  # worker sees _stop on its next get timeout
        self._worker.join(timeout=5.0)
        self._worker = None

    def _prefetch_loop(self):
        while True:
            try:
                index = self._req.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if index is None:
                return
            try:
                packed = _load_layer_packed(self.model_dir, index)
            except Exception:
                continue
            self.loads += 1
            try:
                # Non-blocking: if the main thread already
                # loaded this layer synchronously, the slot
                # is still full — drop the duplicate rather
                # than stalling the worker (and the main
                # thread, via the GIL) for seconds.
                self._res.put_nowait((index, packed))
            except queue.Full:
                self.prefetch_stale += 1

    def _request_prefetch(self, index):
        if not self.prefetch or self._req is None:
            return
        if not (0 <= index < self.n_layers):
            return
        try:
            self._req.put_nowait(index)
        except queue.Full:
            pass  # worker is busy; the layer loads synchronously

    def _take_prefetched(self, index):
        """Return the packed shard for ``index`` if the worker
        already delivered it, else None. A delivered shard
        for a different index is stale (the main thread
        already loaded that layer synchronously) and is
        discarded."""
        if not self.prefetch or self._res is None:
            return None
        try:
            got_index, packed = self._res.get_nowait()
        except queue.Empty:
            return None
        if got_index == index:
            self.prefetch_hits += 1
            return packed
        self.prefetch_stale += 1
        return None

    # -- cache (Phase 3) -------------------------------------------------
    def _cache_insert(self, index, layer):
        if not self.use_cache:
            return
        while (self.cache_bytes + self.layer_bytes
               > self.ram_cap_bytes):
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            self.cache_bytes -= self.layer_bytes
            self.evictions += 1
        self.cache[index] = layer
        self.cache_bytes += self.layer_bytes
        if self.cache_bytes > self.peak_held_bytes:
            self.peak_held_bytes = self.cache_bytes

    def _get_layer(self, index):
        """The layer, from cache or from disk. The returned
        arrays are the cached copies (never mutated in place
        by the forward pass)."""
        if index in self.cache:
            self.cache_hits += 1
            layer = self.cache.pop(index)
            self.cache[index] = layer  # mark most-recently-used
            return layer
        packed = self._take_prefetched(index)
        if packed is None:
            packed = _load_layer_packed(self.model_dir, index)
            self.loads += 1
            if self.prefetch:
                self.prefetch_misses += 1
        layer = _dequantise_layer(*packed)
        self._cache_insert(index, layer)
        return layer

    # -- forward / generate ----------------------------------------------
    def forward(self, hidden_state):
        """One full forward pass over the stack, from a hidden
        state (vocab embedding of the current token)."""
        self._start_prefetch()
        try:
            h = hidden_state
            for i in range(self.n_layers):
                self._request_prefetch(i + 1)
                w, b = self._get_layer(i)
                h = np.tanh(h @ w + b)
            return h
        finally:
            self._stop_prefetch()

    def logits(self, hidden_state):
        return self.forward(hidden_state) @ self.unemb

    def generate(self, start_token=START_TOKEN, n_tokens=N_TOKENS):
        """Greedy (argmax) generation. Returns (tokens,
        final_hidden, stats). Deterministic: same model,
        same start token -> same tokens, with or without
        prefetch and with or without the cache."""
        token = int(start_token)
        tokens = []
        h = None
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            h = self.forward(self.emb[token])
            token = int(np.argmax(h @ self.unemb))
            tokens.append(token)
        wall = time.perf_counter() - t0
        stats = self.stats()
        stats["tokens"] = len(tokens)
        stats["wall_sec"] = wall
        stats["tokens_per_sec"] = len(tokens) / max(wall, 1e-9)
        return tokens, h, stats

    def stats(self):
        return {
            "loads": self.loads,
            "cache_hits": self.cache_hits,
            "evictions": self.evictions,
            "peak_held_bytes": self.peak_held_bytes,
            "ram_cap_bytes": self.ram_cap_bytes,
            "cap_respected": bool(
                self.peak_held_bytes <= self.ram_cap_bytes),
            "fixed_resident_bytes": self.fixed_resident_bytes,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "prefetch_stale": self.prefetch_stale,
        }


class AllInRamModel:
    """The reference: every layer dequantised and held in RAM
    at once. Same weights, same forward math — the outputs
    the offload runner must match. Its held bytes are the
    full FP32 stack, which is the number the RAM cap is
    measured against."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.manifest = load_manifest(model_dir)
        self.hidden = int(self.manifest["hidden"])
        self.vocab = int(self.manifest["vocab"])
        self.n_layers = int(self.manifest["n_layers"])
        with np.load(os.path.join(model_dir, "embedding.npz")) as z:
            self.emb = np.array(z["emb"], dtype=np.float32)
        with np.load(os.path.join(model_dir, "unembedding.npz")) as z:
            self.unemb = np.array(z["unemb"], dtype=np.float32)
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(
                _dequantise_layer(
                    *_load_layer_packed(model_dir, i)))
        self.held_bytes = sum(
            int(w.nbytes + b.nbytes) for w, b in self.layers)

    def forward(self, hidden_state):
        h = hidden_state
        for w, b in self.layers:
            h = np.tanh(h @ w + b)
        return h

    def generate(self, start_token=START_TOKEN, n_tokens=N_TOKENS):
        token = int(start_token)
        tokens = []
        h = None
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            h = self.forward(self.emb[token])
            token = int(np.argmax(h @ self.unemb))
            tokens.append(token)
        wall = time.perf_counter() - t0
        return tokens, h, {
            "tokens": len(tokens),
            "wall_sec": wall,
            "tokens_per_sec": len(tokens) / max(wall, 1e-9),
            "held_bytes": self.held_bytes,
        }


# ---------------------------------------------------------------------------
# Child-process generation (peak RSS measured cleanly, without the
# parent's build-time allocations in the number).
# ---------------------------------------------------------------------------

def _cli_run(args):
    """`python -m hybrid_retriever.offload run ...`: generate
    in this process and print one JSON line with the tokens,
    the stats and this process's peak RSS (ru_maxrss)."""
    if args.all_in_ram:
        model = AllInRamModel(args.model_dir)
        tokens, _, stats = model.generate(
            start_token=args.start_token, n_tokens=args.tokens)
        stats["peak_rss_kb"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
        stats["mode"] = "all-in-ram"
    else:
        runner = OffloadRunner(
            args.model_dir, args.cap_bytes,
            use_cache=not args.no_cache, prefetch=args.prefetch)
        tokens, _, stats = runner.generate(
            start_token=args.start_token, n_tokens=args.tokens)
        stats["peak_rss_kb"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
        stats["mode"] = "offload-prefetch" if args.prefetch \
            else "offload"
    stats["tokens_generated"] = tokens
    print(json.dumps(stats, sort_keys=True))


def _measure_rss(model_dir, tokens, cap_bytes=None, prefetch=False,
                 all_in_ram=False):
    """Run one generation in a FRESH child process and return
    (stats dict, peak RSS in KiB) from that child's own
    rusage (via os.wait4, so the number belongs to exactly
    that child, not to the parent or to earlier children).
    The model is built by the caller, never in the child,
    so build-time allocations never enter the RSS number."""
    import subprocess
    cmd = [sys.executable, "-m", "hybrid_retriever.offload",
           "run", "--model-dir", model_dir,
           "--tokens", str(tokens),
           "--start-token", str(START_TOKEN)]
    if all_in_ram:
        cmd.append("--all-in-ram")
    else:
        cmd += ["--cap-bytes", str(cap_bytes)]
        if prefetch:
            cmd.append("--prefetch")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    # Output is one short JSON line: read both pipes to EOF
    # (the child has exited by then) BEFORE wait4, so the
    # child is reaped exactly once, by wait4.
    out = proc.stdout.read()
    proc.stderr.read()
    _, status, rusage = os.wait4(proc.pid, 0)
    if status != 0:
        raise RuntimeError(
            "offload child failed (status %d)" % status)
    stats = json.loads(out.strip().splitlines()[-1])
    return stats, int(rusage.ru_maxrss)


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="hybrid_retriever.offload",
        description="P14 toy layer-wise offload runner "
        "(NumPy, no torch, no GPU, no 70B model)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generate in this process "
                         "and print JSON incl. peak RSS")
    run.add_argument("--model-dir", required=True)
    run.add_argument("--tokens", type=int, default=N_TOKENS)
    run.add_argument("--start-token", type=int, default=START_TOKEN)
    run.add_argument("--cap-bytes", type=int, default=LAYER_FP32_BYTES)
    run.add_argument("--prefetch", action="store_true")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--all-in-ram", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        _cli_run(args)


if __name__ == "__main__":
    _main()


# ---------------------------------------------------------------------------
# Full eval: Q4 error, done-condition run, prefetch, benchmark table.
# ---------------------------------------------------------------------------

def _disk_dirs():
    """(tmpfs dir, home-disk dir): both fresh, both removed
    by the caller. /tmp is a 512 MB tmpfs on this VM; the
    home-disk side lives under $HOME/.tmp."""
    tmpfs = tempfile.mkdtemp(prefix="hr-offload-tmpfs-",
                             dir="/tmp")
    home_base = os.path.join(
        os.path.expanduser("~"), ".tmp")
    os.makedirs(home_base, exist_ok=True)
    disk = tempfile.mkdtemp(prefix="hr-offload-disk-",
                            dir=home_base)
    return tmpfs, disk


def _benchmark_cell(model_dir, cap_bytes, mode, n_tokens=N_TOKENS,
                    repeats=1):
    """One benchmark cell, in-process timing. ``mode`` is
    "offload", "offload+prefetch" or "all-in-ram". Returns a
    row dict; tok/s is the mean over repeats, everything
    else comes from repeat 0 (deterministic columns)."""
    runs = []
    for _ in range(repeats):
        if mode == "all-in-ram":
            model = AllInRamModel(model_dir)
            tokens, _, stats = model.generate(n_tokens=n_tokens)
            row = {
                "tokens_per_sec": stats["tokens_per_sec"],
                "tokens_generated": tokens,
                "loads": model.n_layers,
                "cache_hits": 0,
                "evictions": 0,
                "peak_held_bytes": stats["held_bytes"],
            }
        else:
            runner = OffloadRunner(
                model_dir, cap_bytes,
                use_cache=True,
                prefetch=(mode == "offload+prefetch"))
            tokens, _, stats = runner.generate(n_tokens=n_tokens)
            row = {
                "tokens_per_sec": stats["tokens_per_sec"],
                "tokens_generated": tokens,
                "loads": stats["loads"],
                "cache_hits": stats["cache_hits"],
                "evictions": stats["evictions"],
                "peak_held_bytes": stats["peak_held_bytes"],
            }
        runs.append(row)
    agg = dict(runs[0])
    agg["tokens_per_sec"] = float(np.mean(
        [r["tokens_per_sec"] for r in runs]))
    return agg


def run_offload_eval(n_tokens=N_TOKENS, repeats=REPEATS):
    """Full P14 eval. Builds the toy models on both disks,
    measures Q4 error, runs the done-condition case (a
    model whose total Q4 weights exceed the RAM cap) with
    output-match + cap + RSS evidence, compares prefetch
    on/off, and builds the benchmark table (layers x RAM
    cap x disk -> tok/s). All temp dirs are removed before
    returning."""
    tmpfs_base, disk_base = _disk_dirs()
    try:
        models = {}
        for n_layers in BENCH_LAYERS:
            for label, base in (("tmpfs", tmpfs_base),
                                ("disk", disk_base)):
                d = os.path.join(
                    base, "model_L%d" % n_layers)
                manifest = build_toy_model(d, n_layers=n_layers)
                models[(n_layers, label)] = (d, manifest)

        q4 = q4_error_stats(n_layers=LAYERS)
        # Logits error of Q4 vs FP32 from the same start token.
        fp32_dir = os.path.join(tmpfs_base, "model_L32_fp32")
        build_toy_model(fp32_dir, n_layers=LAYERS,
                        quantised=False)
        q4_dir, manifest32 = models[(LAYERS, "tmpfs")]
        ref = AllInRamModel(fp32_dir)
        q4ref = AllInRamModel(q4_dir)
        _, h_fp32, _ = ref.generate(n_tokens=1)
        _, h_q4, _ = q4ref.generate(n_tokens=1)
        logits_fp32 = h_fp32 @ ref.unemb
        logits_q4 = h_q4 @ q4ref.unemb
        q4["max_abs_logit_error_1_token"] = float(
            np.max(np.abs(logits_q4 - logits_fp32)))
        q4["mean_abs_logit_error_1_token"] = float(
            np.mean(np.abs(logits_q4 - logits_fp32)))

        # -- Done-condition run: 32 layers, 1-layer cap, tmpfs.
        cap1 = LAYER_FP32_BYTES
        total_q4 = manifest32["total_layer_q4_bytes"]
        runner = OffloadRunner(q4_dir, cap1, use_cache=True,
                               prefetch=True)
        tokens_off, h_off, stats_off = runner.generate(
            n_tokens=n_tokens)
        ref32 = AllInRamModel(q4_dir)
        tokens_ram, h_ram, stats_ram = ref32.generate(
            n_tokens=n_tokens)
        outputs_match = bool(
            tokens_off == tokens_ram
            and float(np.max(np.abs(h_off - h_ram))) <= 1e-5)
        max_hidden_diff = float(np.max(np.abs(h_off - h_ram)))
        rss_off, rss_off_kb = _measure_rss(
            q4_dir, n_tokens, cap_bytes=cap1, prefetch=True)
        rss_ram, rss_ram_kb = _measure_rss(
            q4_dir, n_tokens, all_in_ram=True)
        done = bool(
            outputs_match
            and stats_off["cap_respected"]
            and total_q4 > cap1)

        # -- Prefetch on/off (in-process, same model + cap).
        prefetch_rows = {}
        for mode in ("offload", "offload+prefetch"):
            rows = []
            for _ in range(repeats):
                rows.append(_benchmark_cell(
                    q4_dir, cap1, mode, n_tokens=n_tokens,
                    repeats=1))
            prefetch_rows[mode] = {
                "tokens_per_sec": float(np.mean(
                    [r["tokens_per_sec"] for r in rows])),
                "tokens_generated": rows[0]["tokens_generated"],
                "loads": rows[0]["loads"],
                "cache_hits": rows[0]["cache_hits"],
            }

        # -- Benchmark table: layers x RAM cap x disk -> tok/s.
        table = []
        for n_layers in BENCH_LAYERS:
            for label in ("tmpfs", "disk"):
                model_dir, manifest = models[(n_layers, label)]
                for cap in RAM_CAPS:
                    for mode in ("offload", "offload+prefetch",
                                 "all-in-ram"):
                        cell = _benchmark_cell(
                            model_dir, cap, mode,
                            n_tokens=n_tokens, repeats=1)
                        table.append({
                            "layers": n_layers,
                            "n_params": manifest["n_params"],
                            "ram_cap_bytes": cap,
                            "ram_cap_layers": cap // LAYER_FP32_BYTES,
                            "disk": label,
                            "mode": mode,
                            "tokens": n_tokens,
                            "tokens_per_sec": cell["tokens_per_sec"],
                            "tokens_generated":
                                cell["tokens_generated"],
                            "loads": cell["loads"],
                            "cache_hits": cell["cache_hits"],
                            "evictions": cell["evictions"],
                            "peak_held_bytes":
                                cell["peak_held_bytes"],
                        })
        return {
            "note": NOTE,
            "hidden": HIDDEN,
            "vocab": VOCAB,
            "eval_seed": EVAL_SEED,
            "n_tokens": n_tokens,
            "repeats": repeats,
            "layer_fp32_bytes": LAYER_FP32_BYTES,
            "layer_q4_bytes": LAYER_Q4_BYTES,
            "ram_caps": RAM_CAPS,
            "q4": q4,
            "done_condition": {
                "n_layers": LAYERS,
                "n_params": manifest32["n_params"],
                "ram_cap_bytes": cap1,
                "total_layer_q4_bytes": total_q4,
                "model_bigger_than_cap": bool(total_q4 > cap1),
                "outputs_match_all_in_ram": outputs_match,
                "max_abs_final_hidden_diff": max_hidden_diff,
                "tokens_offload": tokens_off,
                "tokens_all_in_ram": tokens_ram,
                "peak_held_bytes": stats_off["peak_held_bytes"],
                "cap_respected": stats_off["cap_respected"],
                "peak_rss_offload_kb": rss_off_kb,
                "peak_rss_all_in_ram_kb": rss_ram_kb,
                "rss_offload_minus_all_in_ram_kb":
                    rss_off_kb - rss_ram_kb,
                "offload_tokens_per_sec": stats_off["tokens_per_sec"],
                "all_in_ram_tokens_per_sec":
                    stats_ram["tokens_per_sec"],
            },
            "prefetch": prefetch_rows,
            "benchmark_table": table,
            "done_model_bigger_than_cap_runs_correctly": done,
        }
    finally:
        shutil.rmtree(tmpfs_base, ignore_errors=True)
        shutil.rmtree(disk_base, ignore_errors=True)


def print_offload_report(result):
    print("P14 — Minimal layer-wise offload runner (toy model, "
          "pure NumPy — no torch, no GPU, no 70B model)")
    print("NOTE: %s" % result["note"])
    q4 = result["q4"]
    print("\nQ4 quantisation (symmetric per-group, group 32, "
          "codes in [-7, 7], 2 codes/byte) vs FP32 baseline, "
          "%d layers:" % LAYERS)
    print("  max abs weight error %.6f (rounding bound "
          "max scale/2 = %.6f; bound holds: %s)"
          % (q4["max_abs_weight_error"],
             q4["rounding_bound_max_scale_over_2"],
             q4["bound_holds"]))
    print("  mean abs weight error %.6f; FP32 layers %d B -> "
          "Q4 %d B (%.2fx smaller)"
          % (q4["mean_abs_weight_error"],
             q4["fp32_layer_bytes"], q4["q4_layer_bytes"],
             q4["size_ratio_fp32_over_q4"]))
    print("  1-token logits error from quantisation: max %.6f, "
          "mean %.6f"
          % (q4["max_abs_logit_error_1_token"],
             q4["mean_abs_logit_error_1_token"]))
    dc = result["done_condition"]
    print("\nDone-condition run: %d-layer toy model "
          "(%d params), Q4 layer weights %d B, RAM cap %d B "
          "(1 layer); model bigger than cap: %s"
          % (dc["n_layers"], dc["n_params"],
             dc["total_layer_q4_bytes"], dc["ram_cap_bytes"],
             dc["model_bigger_than_cap"]))
    print("  outputs match all-in-RAM: %s (max abs final-"
          "hidden diff %.2e); tokens %s"
          % (dc["outputs_match_all_in_ram"],
             dc["max_abs_final_hidden_diff"],
             dc["tokens_offload"]))
    print("  peak layer bytes held %d B (cap %d B, respected: "
          "%s); peak RSS offload %d KiB vs all-in-RAM %d KiB "
          "(delta %+d KiB — RSS includes the interpreter + "
          "NumPy baseline, so it is compared offload vs "
          "all-in-RAM, never against the byte cap)"
          % (dc["peak_held_bytes"], dc["ram_cap_bytes"],
             dc["cap_respected"], dc["peak_rss_offload_kb"],
             dc["peak_rss_all_in_ram_kb"],
             dc["rss_offload_minus_all_in_ram_kb"]))
    print("  tok/s: offload (prefetch) %.1f vs all-in-RAM %.1f"
          % (dc["offload_tokens_per_sec"],
             dc["all_in_ram_tokens_per_sec"]))
    pf = result["prefetch"]
    print("\nPrefetch (32 layers, 1-layer cap, tmpfs, %d "
          "tokens, mean of %d runs): offload %.1f tok/s, "
          "offload+prefetch %.1f tok/s (%.2fx); tokens "
          "identical: %s"
          % (result["n_tokens"], result["repeats"],
             pf["offload"]["tokens_per_sec"],
             pf["offload+prefetch"]["tokens_per_sec"],
             pf["offload+prefetch"]["tokens_per_sec"]
             / max(pf["offload"]["tokens_per_sec"], 1e-9),
             pf["offload"]["tokens_generated"]
             == pf["offload+prefetch"]["tokens_generated"]))
    print("\nBenchmark table — layers x RAM cap x disk -> tok/s "
          "(%d tokens, tok/s is wall-clock; loads / hits / "
          "evictions are exact counts)"
          % result["n_tokens"])
    print("%-7s %-10s %-6s %-18s %10s %8s %10s %10s %12s" % (
        "layers", "cap", "disk", "mode", "tok/s", "loads",
        "hits", "evictions", "peak held B"))
    for row in result["benchmark_table"]:
        print("%-7d %-10s %-6s %-18s %10.1f %8d %10d %10d %12d" % (
            row["layers"],
            "%dL" % row["ram_cap_layers"],
            row["disk"], row["mode"], row["tokens_per_sec"],
            row["loads"], row["cache_hits"], row["evictions"],
            row["peak_held_bytes"]))
    print("\nDone condition (model bigger than cap runs "
          "correctly: outputs match all-in-RAM AND peak held "
          "<= cap): %s"
          % result["done_model_bigger_than_cap_runs_correctly"])
