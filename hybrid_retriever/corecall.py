"""P4 — REALM-style co-recall graph: memory as a lifecycle, not a store.

Source: REALM (arXiv:2609.16053). Retrieval reorganises memory, so
related memories cluster together over time.

Substitutions, stated plainly (there is no learned encoder here):
- REALM learns a retriever and re-encodes memories jointly. This
  module has no learned encoder and no LLM: co-recall is
  deterministic counting — pairs of records credited for the same
  answer accumulate credit weight (``corecall`` table) — and the
  "re-encoding" is the nightly job promoting those pairs to graph
  edges and re-linking clusters. The ONNX MiniLM vectors from P1
  are the only embeddings involved.
- There is no live traffic on this VM, so the eval's co-recall
  history is seeded synthetic traffic: the 48 keyword queries run
  3 rounds (simulating repeated use over days), each round logging
  co-credit through the same ``record_use`` + ``log_co_use`` path
  a live deployment would call after each answer. The held-out
  paraphrase + trap queries are never used for warm-up.

Why the gate is credit-weighted (measured, not assumed):
P1's graph leg chained *all* same-person records and hurt
retrieval (hybrid 0.631 vs vector-only 0.840 R@10 on the synthetic
corpus). Raw co-recall counts do not fix that: after 3 warm-up
rounds every co-credited pair — gold or distractor — reaches
count >= 3 (1302 pairs; gold-gold and other pairs have the same
count distribution). The credit weight does separate them: the
per-answer weight is min(credit_a, credit_b), so a pair only
accumulates when *both* records earned credit for the same
answer. wsum >= 1.5 keeps 118/146 gold pairs and 4/1156 other
pairs. That threshold is the promotion gate; expansion itself is
then bounded (top-k neighbours per seed, 1 hop by default) and
reads only promoted edges, so an unpromoted pair can never be
surfaced by the graph leg.

Phases
------
1. Heterogeneous graph: memory nodes + edges in the same SQLite
   file (the P1 ``edges`` table, extended by the ``corecall`` log
   and the ``reconsolidations`` audit table). Every answer logs
   which memories were used together (``log_co_use``).
2. Adaptive graph-search retrieval: graph-search atoms
   (neighbour expansion, temporal, type) composed per query, as an
   EXPANDER on top of P1's hybrid retriever via
   ``retrieve(..., graph_expander=...)`` — never a replacement.
   The default atom is selective neighbour expansion over promoted
   co-recall edges; temporal and typed atoms are available through
   ``compose_expansion``.
3. Nightly reconsolidation (``reconsolidate``): promote
   credit-weighted co-recall pairs to edges, re-link clusters by
   2-hop closure, prune stale / non-live co-recall edges, and
   merge near-duplicates inside clusters through P2's judge-gated
   ``propose_merge``. Every run is audited in ``reconsolidations``.
"""

from __future__ import annotations

import time

from .retrieval import DEFAULT_WEIGHTS, retrieve
from .store import Store

# Promotion gate: measured on the synthetic warm-up (see module
# docstring). Below this weight a pair is noise; above it, it is
# almost always a gold pair (118/146 vs 4/1156 at 1.5).
DEFAULT_MIN_WSUM = 1.5
DEFAULT_PROMOTE_MIN_COUNT = 2
# Expansion bounds: the P1 lesson. Indiscriminate chaining hurt,
# so expansion is top-k neighbours per seed and 1 hop by default.
DEFAULT_MAX_PER_SEED = 3
DEFAULT_MAX_HOPS = 1
# Retrieval-side floor on promoted edge weight. The promotion gate
# already did the strict filtering, so this is loose: it admits
# direct edges (weight >= 1.5) and closure edges (decayed to >= 0.75).
DEFAULT_RETRIEVAL_MIN_WEIGHT = 0.5
# 2-hop closure edges carry a decayed weight: they re-link cluster
# members that were never directly co-credited, but must not outrank
# a directly co-recalled pair.
DEFAULT_CLOSURE_DECAY = 0.5
# Near-duplicate merge inside a cluster (P2's Jaccard rule).
DEFAULT_MERGE_JACCARD = 0.9


# ----------------------------------------------------------------------
# Phase 1 — logging co-recall
# ----------------------------------------------------------------------
def log_co_use(store: Store, retrieved, credits: dict[int, float]) -> int:
    """Log credit-weighted co-recall for one answer.

    ``retrieved`` is the list of ``Retrieved`` the answer was built
    from; ``credits`` maps record id -> TRACE credit (as returned by
    ``lifecycle.record_use``). Every pair of *credited* records is
    logged with weight min(credit_a, credit_b). Records retrieved
    but not used (credit 0) contribute nothing — mere co-occurrence
    in a top-k list is not co-recall. Returns the number of pairs
    logged.
    """
    used = [r.id for r in retrieved if credits.get(r.id, 0.0) > 0]
    logged = 0
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            store.log_corecall(
                used[i], used[j], min(credits[used[i]], credits[used[j]])
            )
            logged += 1
    return logged


# ----------------------------------------------------------------------
# Phase 2 — expansion atoms
# ----------------------------------------------------------------------
def _adjacency_from_pairs(pairs) -> dict[int, list[tuple[int, float]]]:
    """Undirected adjacency {id: [(neighbour, weight)]} from (a, b, w)."""
    adj: dict[int, list[tuple[int, float]]] = {}
    for a, b, w in pairs:
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    return adj


def _bounded_expand(
    adj: dict[int, list[tuple[int, float]]],
    seed_ids: list[int],
    leg_k: int,
    max_per_seed: int,
    max_hops: int,
    hop_decay: float = 0.5,
    candidate_ids: list[int] | None = None,
    non_live: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Bounded multi-hop expansion over an adjacency map.

    Per hop, each frontier node contributes at most ``max_per_seed``
    neighbours (best weight first); deeper hops are decayed by
    ``hop_decay`` per hop. Seeds are never returned; live-only when
    ``non_live`` is given. Returns [(id, weight)] best weight first.
    """
    allowed = set(candidate_ids) if candidate_ids else None
    seeds = [s for s in seed_ids if s not in (non_live or set())]
    if not seeds:
        return []
    best: dict[int, float] = {}
    frontier = list(dict.fromkeys(seeds))  # dedupe, keep order
    seen = set(frontier)
    for hop in range(max_hops):
        decay = hop_decay**hop
        nxt: list[int] = []
        for node in frontier:
            neighbours = sorted(
                adj.get(node, []), key=lambda t: -t[1]
            )[:max_per_seed]
            for nb, w in neighbours:
                if nb in seen:
                    continue
                if allowed is not None and nb not in allowed:
                    continue
                if non_live is not None and nb in non_live:
                    continue
                w = w * decay
                if w > best.get(nb, 0.0):
                    best[nb] = w
                nxt.append(nb)
        if not nxt:
            break
        seen |= set(nxt)
        frontier = nxt
    return sorted(best.items(), key=lambda kv: -kv[1])[:leg_k]


def expand_from_edges(
    store: Store,
    seed_ids: list[int],
    leg_k: int = 50,
    min_weight: float = DEFAULT_RETRIEVAL_MIN_WEIGHT,
    max_per_seed: int = DEFAULT_MAX_PER_SEED,
    max_hops: int = DEFAULT_MAX_HOPS,
    candidate_ids: list[int] | None = None,
    kind: str = "co-recall",
) -> list[tuple[int, float]]:
    """Selective neighbour expansion over promoted graph edges.

    Only ``kind`` edges with weight >= ``min_weight`` are followed,
    at most ``max_per_seed`` per seed, ``max_hops`` hops. This is
    the P4 graph leg: unlike P1's ``store.neighbours`` it never
    follows an indiscriminate 'related' edge and never returns a
    neighbour the reconsolidation job has not promoted.
    """
    pairs = [
        (s, d, w)
        for s, d, w, _k in store.list_edges(kind=kind)
        if w >= min_weight
    ]
    return _bounded_expand(
        _adjacency_from_pairs(pairs),
        seed_ids,
        leg_k,
        max_per_seed,
        max_hops,
        candidate_ids=candidate_ids,
        non_live=store.non_live_ids(),
    )


def expand_from_log(
    store: Store,
    seed_ids: list[int],
    leg_k: int = 50,
    min_wsum: float = 1.0,
    max_per_seed: int = DEFAULT_MAX_PER_SEED,
    max_hops: int = DEFAULT_MAX_HOPS,
    candidate_ids: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Neighbour expansion straight from the raw co-recall log.

    The pre-reconsolidation ablation arm: the same bounded expansion,
    but over unpromoted ``corecall`` pairs. Comparing this against
    ``expand_from_edges`` isolates what the nightly job adds
    (promotion threshold + pruning + cluster closure + merges).
    """
    pairs = [
        (a, b, w)
        for a, b, _n, w in store.corecall_pairs(
            min_count=1, min_wsum=min_wsum
        )
    ]
    return _bounded_expand(
        _adjacency_from_pairs(pairs),
        seed_ids,
        leg_k,
        max_per_seed,
        max_hops,
        candidate_ids=candidate_ids,
        non_live=store.non_live_ids(),
    )


def expand_temporal(
    store: Store,
    seed_ids: list[int],
    leg_k: int = 50,
    window_days: float = 7.0,
    candidate_ids: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Temporal atom: live records created near a seed in time.

    Returns [(id, 1 - |dt| / window)] — closer in time scores
    higher — capped at ``leg_k``. Seeds themselves are excluded.
    """
    allowed = set(candidate_ids) if candidate_ids else None
    non_live = store.non_live_ids()
    window = window_days * 86400.0
    seeds = [
        store.get_record(s)
        for s in seed_ids
        if s not in non_live and store.get_record(s) is not None
    ]
    if not seeds:
        return []
    scored: dict[int, float] = {}
    for rec in store.list_live_records():
        if rec["id"] in {s["id"] for s in seeds}:
            continue
        if allowed is not None and rec["id"] not in allowed:
            continue
        dt = min(abs(rec["created_at"] - s["created_at"]) for s in seeds)
        if dt <= window:
            scored[rec["id"]] = 1.0 - dt / window
    return sorted(scored.items(), key=lambda kv: -kv[1])[:leg_k]


def expand_typed(
    store: Store,
    seed_ids: list[int],
    leg_k: int = 50,
    candidate_ids: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Typed atom: live records sharing a seed's type.

    Ranked by P2 importance (most useful first), capped at ``leg_k``.
    Seeds themselves are excluded.
    """
    allowed = set(candidate_ids) if candidate_ids else None
    non_live = store.non_live_ids()
    types = set()
    for s in seed_ids:
        rec = store.get_record(s)
        if rec is not None and s not in non_live:
            types.add(rec["type"])
    if not types:
        return []
    ids: list[int] = []
    for t in types:
        ids.extend(store.live_ids_by_type(t))
    ids = [
        i
        for i in dict.fromkeys(ids)
        if i not in set(seed_ids)
        and i not in non_live
        and (allowed is None or i in allowed)
    ]
    if not ids:
        return []
    imps = store.get_importances(ids)
    return sorted(
        [(i, imps.get(i, 0.5)) for i in ids], key=lambda kv: -kv[1]
    )[:leg_k]


_ATOMS = {
    "neighbour": expand_from_edges,
    "temporal": expand_temporal,
    "typed": expand_typed,
}


def compose_expansion(
    store: Store,
    seed_ids: list[int],
    atoms: tuple[str, ...] = ("neighbour",),
    leg_k: int = 50,
    candidate_ids: list[int] | None = None,
    **kwargs,
) -> list[tuple[int, float]]:
    """Compose graph-search atoms per query into one expansion list.

    Atoms run in the given order; the first atom to surface a record
    keeps its weight, later duplicates are dropped. Unknown atom
    names raise ValueError. ``kwargs`` are forwarded to every atom
    that accepts them (each atom takes only the parameters it
    understands — pass atom-specific options via ``make_expander``).
    """
    out: list[tuple[int, float]] = []
    seen: set[int] = set()
    for name in atoms:
        if name not in _ATOMS:
            raise ValueError("unknown expansion atom %r" % name)
        fn = _ATOMS[name]
        params = {
            k: v
            for k, v in kwargs.items()
            if k in fn.__code__.co_varnames
        }
        for rid, w in fn(
            store, seed_ids, leg_k, candidate_ids=candidate_ids, **params
        ):
            if rid not in seen:
                seen.add(rid)
                out.append((rid, w))
    return out[:leg_k]


def make_expander(
    store: Store,
    source: str = "edges",
    candidate_ids: list[int] | None = None,
    **kwargs,
):
    """Build a ``retrieve(..., graph_expander=...)`` callable.

    ``source="edges"`` expands over promoted co-recall edges
    (``expand_from_edges``); ``source="log"`` expands over the raw
    co-recall log (``expand_from_log``, the pre-reconsolidation
    ablation). ``kwargs`` are the expansion bounds
    (``min_weight`` / ``min_wsum``, ``max_per_seed``, ``max_hops``).
    """
    if source == "edges":
        def expander(seed_ids, leg_k):
            return expand_from_edges(
                store, seed_ids, leg_k,
                candidate_ids=candidate_ids, **kwargs
            )
    elif source == "log":
        def expander(seed_ids, leg_k):
            return expand_from_log(
                store, seed_ids, leg_k,
                candidate_ids=candidate_ids, **kwargs
            )
    else:
        raise ValueError("unknown expansion source %r" % source)
    return expander


def retrieve_expanded(
    store: Store,
    embedder,
    query: str,
    k: int = 10,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    source: str = "edges",
    use_mmr: bool = False,
    candidate_ids: list[int] | None = None,
    importance_weight: float = 0.0,
    **exp_kwargs,
):
    """Hybrid retrieval with the P4 graph expander as the graph leg.

    Identical to ``retrieve`` except the graph leg is selective
    co-recall expansion instead of P1's indiscriminate neighbour
    chaining. ``source`` / ``exp_kwargs`` are passed to
    ``make_expander``; everything else is passed straight through
    to ``retrieve``.
    """
    return retrieve(
        store,
        embedder,
        query,
        k=k,
        weights=weights,
        use_mmr=use_mmr,
        candidate_ids=candidate_ids,
        importance_weight=importance_weight,
        graph_expander=make_expander(
            store, source=source, candidate_ids=candidate_ids, **exp_kwargs
        ),
    )


# ----------------------------------------------------------------------
# Phase 3 — nightly reconsolidation
# ----------------------------------------------------------------------
def co_recall_clusters(
    store: Store,
    min_wsum: float = DEFAULT_MIN_WSUM,
    min_count: int = DEFAULT_PROMOTE_MIN_COUNT,
) -> list[list[int]]:
    """Connected components of the promoted-eligible co-recall graph.

    Union-find over pairs meeting the promotion gate; components of
    size >= 2 are the clusters related memories have formed through
    repeated joint use. Returns sorted member lists, largest first.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _n, _w in store.corecall_pairs(
        min_count=min_count, min_wsum=min_wsum
    ):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    clusters = [sorted(v) for v in groups.values() if len(v) >= 2]
    clusters.sort(key=len, reverse=True)
    return clusters


def graph_stats(store: Store) -> dict:
    """Snapshot of the graph for reports and the ``hr graph`` command."""
    edges = store.list_edges()
    by_kind: dict[str, int] = {}
    for _s, _d, _w, kind in edges:
        by_kind[kind] = by_kind.get(kind, 0) + 1
    pairs = store.corecall_pairs(min_count=1, min_wsum=0.0)
    return {
        "live_records": store.count_live(),
        "corecall_pairs": len(pairs),
        "corecall_pairs_ge_1.5": len(
            store.corecall_pairs(min_count=1, min_wsum=DEFAULT_MIN_WSUM)
        ),
        "edges_total": len(edges),
        "edges_by_kind": by_kind,
        "clusters": len(
            co_recall_clusters(store, DEFAULT_MIN_WSUM, DEFAULT_PROMOTE_MIN_COUNT)
        ),
        "reconsolidations": len(store.list_reconsolidations()),
    }


def reconsolidate(
    store: Store,
    embedder=None,
    min_wsum: float = DEFAULT_MIN_WSUM,
    min_count: int = DEFAULT_PROMOTE_MIN_COUNT,
    closure_decay: float = DEFAULT_CLOSURE_DECAY,
    merge_jaccard: float = DEFAULT_MERGE_JACCARD,
    dry_run: bool = False,
) -> dict:
    """The nightly job: promote, re-link, prune, merge.

    1. Merge near-duplicates inside co-recall clusters first
       (P2 judge-gated ``propose_merge`` — sources are superseded,
       never deleted), so promotion never links a record that is
       about to be superseded.
    2. Promote credit-weighted pairs meeting the gate to
       'co-recall' edges (both directions, weight = wsum).
    3. Re-link: within each cluster, add decayed 2-hop closure
       edges between members never directly co-credited, so a
       partially linked trio becomes fully linked.
    4. Prune: delete 'co-recall' edges whose pair no longer meets
       the gate (including closure edges whose endpoints were
       superseded by a merge) and any pointing at a non-live record.
    Dry-run reports without writing. Every real run is audited in
    ``reconsolidations``. Returns the summary dict.
    """
    # Imported lazily: lifecycle imports retrieval, and this module
    # is imported by retrieval's callers — a top-level import would
    # be a cycle (lifecycle -> retrieval -> corecall -> lifecycle).
    from .lifecycle import content_tokens, jaccard, propose_merge

    summary = {
        "merges_proposed": 0,
        "merges_committed": 0,
        "merges_would_commit": 0,
        "merges_rejected": 0,
        "promoted": 0,
        "closure_added": 0,
        "pruned": 0,
        "clusters": 0,
        "edges_before": len(store.list_edges(kind="co-recall")),
        "edges_after": 0,
        "dry_run": dry_run,
    }
    live = {r["id"] for r in store.list_live_records()}
    pairs = [
        (a, b, w)
        for a, b, _n, w in store.corecall_pairs(
            min_count=min_count, min_wsum=min_wsum
        )
        if a in live and b in live
    ]

    # 1. Near-duplicate merges inside co-recall clusters.
    tokens = {}
    for a, b, _w in pairs:
        for rid in (a, b):
            if rid not in tokens:
                rec = store.get_record(rid)
                tokens[rid] = (
                    content_tokens(rec["text"]) if rec else set()
                )
    merged_away: set[int] = set()
    for a, b, _w in sorted(pairs, key=lambda t: -t[2]):
        if a in merged_away or b in merged_away:
            continue
        if jaccard(tokens[a], tokens[b]) < merge_jaccard:
            continue
        summary["merges_proposed"] += 1
        res = propose_merge(store, embedder, [a, b], dry_run=dry_run)
        if res.status == "committed":
            summary["merges_committed"] += 1
            merged_away.update((a, b))
        elif res.status == "dry-run-pass":
            summary["merges_would_commit"] += 1
        else:
            summary["merges_rejected"] += 1
    if merged_away:
        live -= merged_away
        pairs = [(a, b, w) for a, b, w in pairs if a in live and b in live]

    # 2. Promote.
    clusters = co_recall_clusters(store, min_wsum, min_count)
    clusters = [
        [m for m in c if m in live] for c in clusters
    ]
    clusters = [c for c in clusters if len(c) >= 2]
    summary["clusters"] = len(clusters)
    keep: set[tuple[int, int]] = set()
    for a, b, w in pairs:
        keep.add((a, b))
        keep.add((b, a))
        summary["promoted"] += 1
        if not dry_run:
            store.add_edge(a, b, weight=w, kind="co-recall")
            store.add_edge(b, a, weight=w, kind="co-recall")

    # 3. 2-hop closure inside each cluster.
    adj: dict[int, dict[int, float]] = {}
    for a, b, w in pairs:
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w
    for cluster in clusters:
        for i, a in enumerate(cluster):
            for b in cluster[i + 1:]:
                if b in adj.get(a, {}):
                    continue
                via = [
                    min(adj[a][c], adj[c][b])
                    for c in adj.get(a, {})
                    if c in adj.get(b, {})
                ]
                if not via:
                    continue
                w = closure_decay * max(via)
                keep.add((a, b))
                keep.add((b, a))
                summary["closure_added"] += 1
                if not dry_run:
                    store.add_edge(a, b, weight=w, kind="co-recall")
                    store.add_edge(b, a, weight=w, kind="co-recall")

    # 4. Prune stale / non-live co-recall edges.
    for src, dst, _w, _k in store.list_edges(kind="co-recall"):
        if (src, dst) not in keep or src not in live or dst not in live:
            summary["pruned"] += 1
            if not dry_run:
                store.delete_edge(src, dst, "co-recall")
    summary["edges_after"] = len(store.list_edges(kind="co-recall"))
    if not dry_run:
        store.add_reconsolidation("reconsolidate", summary)
    return summary
