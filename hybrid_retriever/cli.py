"""Command line: index a corpus, run a query, run the eval, sleep, credit, merges, consolidate, plan."""

from __future__ import annotations

import argparse
import json
import os

from .embeddings import get_embedder
from .eval import (
    print_report,
    print_segment_report,
    run_longmemeval,
    run_segment_eval,
    run_synthetic,
    run_synthetic_trace,
)
from .lifecycle import record_use, sleep
from .rerank import get_reranker
from .retrieval import retrieve
from .segments import consolidate_transcript, retrieve_planned
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
    if args.segments:
        result = run_segment_eval(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            os.path.join("eval_data", "synthetic_queries.json"),
            embedder=embedder,
            threshold=args.threshold,
            window=args.window,
        )
        print_segment_report(result)
        return
    if args.trace:
        result = run_synthetic_trace(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            os.path.join("eval_data", "synthetic_queries.json"),
            embedder=embedder,
            ucb_c=args.ucb_c,
        )
        print_report(result)
        return
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


def cmd_sleep(args) -> None:
    embedder = get_embedder()
    store = Store(args.db, embedder=embedder)
    summary = sleep(
        store,
        embedder=embedder,
        max_merges=args.max_merges,
        prune_days=args.prune_days,
        min_co_recall=args.min_co_recall,
        dry_run=args.dry_run,
    )
    print(
        "sleep%s: proposed=%d committed=%d%s rejected=%d pruned=%d "
        "co-recall pairs promoted=%d (live records: %d)"
        % (
            " (dry run)" if args.dry_run else "",
            summary["merges_proposed"],
            summary["merges_committed"],
            (
                " would-commit=%d" % summary["merges_would_commit"]
                if args.dry_run
                else ""
            ),
            summary["merges_rejected"],
            summary["pruned"],
            summary["co_recall_pairs_promoted"],
            store.count_live(),
        )
    )
    store.close()


def cmd_credit(args) -> None:
    """Retrieve for a query, then record TRACE credit from the answer."""
    embedder = get_embedder()
    store = Store(args.db, embedder=embedder)
    results = retrieve(store, embedder, args.query, k=args.k, use_mmr=False)
    credits = record_use(store, results, args.answer)
    for r in results:
        print(
            "#%-4d credit=%.3f  %s"
            % (r.id, credits.get(r.id, 0.0), r.text[:120])
        )
    store.close()


def cmd_consolidate(args) -> None:
    """Ingest a transcript file, consolidate its closed segments."""
    embedder = get_embedder()
    store = Store(args.db, embedder=embedder)
    with open(args.transcript, encoding="utf-8") as f:
        turns = [json.loads(line) for line in f if line.strip()]
    summary = consolidate_transcript(
        store,
        embedder,
        turns,
        threshold=args.threshold,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        fixed_window=args.fixed_window,
        dry_run=args.dry_run,
    )
    print(
        "consolidate%s: turns=%d segments=%d closed=%d consolidated=%d "
        "rejected=%d skipped-single-turn=%d left-open-turns=%d "
        "(live records: %d)"
        % (
            " (dry run)" if args.dry_run else "",
            summary["turns_ingested"],
            summary["segments"],
            summary["closed_segments"],
            summary["consolidated"],
            summary["rejected"],
            summary["skipped_single_turn"],
            summary["left_open"],
            store.count_live(),
        )
    )
    store.close()


def cmd_plan(args) -> None:
    """Show the query planner's route, then run it."""
    embedder = get_embedder()
    store = Store(args.db, embedder=embedder)
    results, plan = retrieve_planned(store, embedder, args.query, k=args.k)
    print(
        "route=%s weights=%s importance_weight=%.1f  (%s)"
        % (
            plan.route,
            "/".join("%.2f" % w for w in plan.weights),
            plan.importance_weight,
            plan.reason,
        )
    )
    for r in results:
        print(
            "#%-4d score=%.4f legs=%s  %s"
            % (r.id, r.score, ",".join(sorted(r.legs)), r.text[:160])
        )
    store.close()


def cmd_merges(args) -> None:
    store = Store(args.db)
    for m in store.list_merges(status=args.status):
        print(
            "merge #%-4d %-11s merged_record=%s sources=%s reasons=%s"
            % (
                m["id"],
                m["status"],
                m["merged_record_id"],
                m["merged_ids"],
                ",".join(m["certificate"].get("reasons", [])) or "-",
            )
        )
    store.close()


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
    pe.add_argument(
        "--trace",
        action="store_true",
        help="TRACE split eval on the synthetic corpus (warm-up on "
        "keyword queries, eval on paraphrase+trap)",
    )
    pe.add_argument(
        "--ucb-c", type=float, default=1.0,
        help="TRACE-UCB exploration constant for --trace (default 1.0)",
    )
    pe.add_argument(
        "--segments",
        action="store_true",
        help="P3 segment eval: boundaries, consolidation cost, "
        "retrieval after consolidation, planner routes",
    )
    pe.add_argument(
        "--threshold", type=float, default=None,
        help="boundary cosine threshold for --segments (default: "
        "tuned on the dev transcripts)",
    )
    pe.add_argument(
        "--window", type=int, default=None,
        help="fixed-window size for the --segments control (default: "
        "mean true segment length)",
    )
    pe.set_defaults(func=cmd_eval)

    ps = sub.add_parser(
        "sleep", help="offline consolidation: judge-gated merges, "
        "prune dead memories, promote co-recall edges"
    )
    ps.add_argument("--db", required=True)
    ps.add_argument("--max-merges", type=int, default=20)
    ps.add_argument("--prune-days", type=float, default=90.0)
    ps.add_argument("--min-co-recall", type=int, default=2)
    ps.add_argument("--dry-run", action="store_true")
    ps.set_defaults(func=cmd_sleep)

    pc = sub.add_parser(
        "credit", help="retrieve for a query, then record TRACE credit "
        "from the supplied answer text"
    )
    pc.add_argument("--db", required=True)
    pc.add_argument("--query", required=True)
    pc.add_argument("--answer", required=True)
    pc.add_argument("--k", type=int, default=10)
    pc.set_defaults(func=cmd_credit)

    pm = sub.add_parser("merges", help="list the merge audit table")
    pm.add_argument("--db", required=True)
    pm.add_argument("--status", default=None,
                    help="filter: committed, rejected, rolled-back")
    pm.set_defaults(func=cmd_merges)

    pc2 = sub.add_parser(
        "consolidate",
        help="P3 offline job: ingest a transcript (JSONL turns) and "
        "consolidate its closed segments into typed records",
    )
    pc2.add_argument("--db", required=True)
    pc2.add_argument("--transcript", required=True)
    pc2.add_argument("--threshold", type=float, default=0.30)
    pc2.add_argument("--min-turns", type=int, default=2)
    pc2.add_argument("--max-turns", type=int, default=12)
    pc2.add_argument(
        "--fixed-window", type=int, default=None,
        help="use fixed-window batching instead of semantic "
        "boundaries (the control)",
    )
    pc2.add_argument("--dry-run", action="store_true")
    pc2.set_defaults(func=cmd_consolidate)

    pp = sub.add_parser(
        "plan", help="P3 query planner: show the chosen route, then run it"
    )
    pp.add_argument("--db", required=True)
    pp.add_argument("--query", required=True)
    pp.add_argument("--k", type=int, default=10)
    pp.set_defaults(func=cmd_plan)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
