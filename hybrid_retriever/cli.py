"""Command line: index a corpus, run a query, run the eval."""

from __future__ import annotations

import argparse
import json
import os

from .embeddings import get_embedder
from .eval import print_report, run_longmemeval, run_synthetic
from .rerank import get_reranker
from .retrieval import retrieve
from .store import Store


def cmd_index(args) -> None:
    embedder = get_embedder()
    store = Store(args.db, embedder=embedder)
    with open(args.corpus, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    ids = store.add_texts(texts, embedder=embedder)
    print(
        "indexed %d records into %s (embedder=%s, vec_mode=%s)"
        % (len(ids), args.db, embedder.name, store.vec_mode)
    )
    store.close()


def cmd_query(args) -> None:
    embedder = get_embedder()
    store = Store(args.db, embedder=embedder)
    reranker = None
    if args.rerank:
        reranker = get_reranker()
    results = retrieve(
        store,
        embedder,
        args.query,
        k=args.k,
        use_mmr=not args.no_mmr,
        reranker=reranker,
    )
    for r in results:
        print(
            "#%-4d score=%.4f legs=%s  %s"
            % (r.id, r.score, ",".join(sorted(r.legs)), r.text[:160])
        )
    store.close()


def cmd_eval(args) -> None:
    embedder = get_embedder()
    reranker = None
    modes = ("bm25", "vector", "hybrid", "hybrid+mmr")
    if args.rerank:
        try:
            reranker = get_reranker()
            modes = modes + ("hybrid+rerank",)
        except RuntimeError as exc:
            print("rerank unavailable: %s" % exc)
    if args.dataset == "synthetic":
        result = run_synthetic(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            os.path.join("eval_data", "synthetic_queries.json"),
            embedder=embedder,
            reranker=reranker,
            modes=modes,
        )
    else:
        result = run_longmemeval(
            args.dataset,
            limit=args.limit,
            embedder=embedder,
            reranker=reranker,
            modes=modes,
        )
    print_report(result)


def main() -> None:
    p = argparse.ArgumentParser(prog="hr", description="SQLite hybrid retriever")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index a JSONL corpus (one {\"text\": ...} per line)")
    pi.add_argument("--db", required=True)
    pi.add_argument("--corpus", required=True)
    pi.set_defaults(func=cmd_index)

    pq = sub.add_parser("query", help="run one hybrid query")
    pq.add_argument("--db", required=True)
    pq.add_argument("--query", required=True)
    pq.add_argument("--k", type=int, default=10)
    pq.add_argument("--no-mmr", action="store_true")
    pq.add_argument("--rerank", action="store_true")
    pq.set_defaults(func=cmd_query)

    pe = sub.add_parser("eval", help="run the eval harness")
    pe.add_argument(
        "--dataset",
        default="synthetic",
        help="'synthetic' or a path to longmemeval_s_cleaned.json",
    )
    pe.add_argument("--limit", type=int, default=50)
    pe.add_argument("--rerank", action="store_true")
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
