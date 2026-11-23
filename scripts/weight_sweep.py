"""Weight-sensitivity sweep on the synthetic corpus.

The default RRF weights (0.50 BM25 / 0.35 vector / 0.15 graph) come
from DocuSearch, tuned on their corpus. This script checks whether
they are right for *this* corpus: it re-scores the same questions
under several weight settings (and with the graph leg on/off) so the
README can report the sensitivity instead of asserting the defaults.

Usage: python scripts/weight_sweep.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hybrid_retriever.embeddings import get_embedder
from hybrid_retriever.eval import (
    average_scores,
    load_synthetic,
    score_modes,
)
from hybrid_retriever.retrieval import retrieve
from hybrid_retriever.store import Store
import tempfile

SETTINGS = [
    ("paper (0.50/0.35/0.15)", (0.50, 0.35, 0.15)),
    ("vector-heavy (0.25/0.60/0.15)", (0.25, 0.60, 0.15)),
    ("balanced (0.40/0.40/0.20)", (0.40, 0.40, 0.20)),
    ("no graph (0.50/0.50/0.00)", (0.50, 0.50, 0.00)),
    ("bm25-heavy (0.70/0.20/0.10)", (0.70, 0.20, 0.10)),
    ("vector only (0.00/1.00/0.00)", (0.00, 1.00, 0.00)),
]


def main() -> None:
    embedder = get_embedder()
    print("embedder:", embedder.name)
    records, queries = load_synthetic(
        "eval_data/synthetic_corpus.jsonl", "eval_data/synthetic_queries.json"
    )
    tmp = tempfile.mkdtemp(prefix="hr-sweep-")
    store = Store(os.path.join(tmp, "s.db"), embedder=embedder)
    rids = store.add_texts([r["text"] for r in records], embedder=embedder)
    rid_of_key = {r["key"]: rid for r, rid in zip(records, rids)}
    by_group: dict[str, list[int]] = {}
    for r, rid in zip(records, rids):
        by_group.setdefault(r.get("group", ""), []).append(rid)
    edges = []
    for group_rids in by_group.values():
        for a, b in zip(group_rids, group_rids[1:]):
            edges += [(a, b, 1.0, "related"), (b, a, 1.0, "related")]
    store.add_edges(edges)

    for name, weights in SETTINGS:
        per_q = []
        for q in queries:
            gold = {rid_of_key[k] for k in q["gold_keys"]}
            res = retrieve(
                store, embedder, q["question"], k=10,
                weights=weights, use_mmr=False,
            )
            per_q.append(
                score_modes({"hybrid": [r.id for r in res]}, gold)["hybrid"]
            )
        avg = {
            m: sum(p[m] for p in per_q) / len(per_q)
            for m in ["P@10", "R@10", "nDCG@10", "any-hit@5", "grounding@10"]
        }
        print(
            "%-32s P@10 %.3f  R@10 %.3f  nDCG %.3f  hit@5 %.3f  ground %.3f"
            % (name, avg["P@10"], avg["R@10"], avg["nDCG@10"],
               avg["any-hit@5"], avg["grounding@10"])
        )
    store.close()


if __name__ == "__main__":
    main()
