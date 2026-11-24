"""Command line: index a corpus, run a query, run the eval, sleep, credit, merges, consolidate, plan."""

from __future__ import annotations

import argparse
import json
import os

from .embeddings import get_embedder
from .eval import (
    print_compression_report,
    print_eviction_report,
    print_report,
    print_segment_report,
    run_compression_eval,
    run_eviction_eval,
    run_longmemeval,
    run_segment_eval,
    run_synthetic,
    run_synthetic_trace,
)
from .eviction import (
    evict_to_budget,
    explain_episode,
    ingest_session,
)
from .lifecycle import record_use, sleep
from .rerank import get_reranker
from .retrieval import retrieve
from .segments import consolidate_transcript, retrieve_planned
from .store import Store
from .verify import print_vbd_report, run_vbd_eval


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
    if args.eviction:
        result = run_eviction_eval(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            os.path.join("eval_data", "synthetic_queries.json"),
            embedder=embedder,
            n_tasks=args.tasks,
            budget_frac=args.budget_frac,
        )
        print_eviction_report(result)
        return
    if args.compression:
        result = run_compression_eval(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            os.path.join("eval_data", "synthetic_queries.json"),
            embedder=embedder,
            n_tasks=args.tasks,
            budget_frac=args.budget_frac,
        )
        print_compression_report(result)
        return
    if args.vbd:
        result = run_vbd_eval(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            embedder=embedder,
            n_tasks=args.tasks,
            k=args.k,
        )
        print_vbd_report(result)
        return
    if args.tobench:
        from .tobench import print_tobench_report, run_tobench_eval

        result = run_tobench_eval(n_tasks=args.tasks)
        print_tobench_report(result)
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


def cmd_ingest(args) -> None:
    """Ingest an annotated transcript (JSONL turns) as episodes."""
    store = Store(args.db, embedder=get_embedder())
    with open(args.transcript, encoding="utf-8") as f:
        turns = [json.loads(line) for line in f if line.strip()]
    ids = ingest_session(store, args.session, turns)
    print(
        "ingested %d turns into session %r "
        "(context tokens: %d)"
        % (
            len(ids),
            args.session,
            store.session_token_total(args.session),
        )
    )
    store.close()


def cmd_evict(args) -> None:
    """Evict a session's episodes to a token budget."""
    store = Store(args.db)
    summary = evict_to_budget(
        store,
        args.session,
        args.budget,
        protect_last=args.protect_last,
        dry_run=args.dry_run,
        policy=args.policy,
        compress=args.compress,
    )
    print(
        "evict%s (%s): total %d -> %d tokens (budget %d), "
        "evicted %d, user turns %d/%d, under budget: %s"
        % (
            " (dry run)" if args.dry_run else "",
            summary["policy"],
            summary["total_before"],
            summary["total_after"],
            summary["budget_tokens"],
            summary["n_evicted"],
            summary["user_turns_live"],
            summary["user_turns_total"],
            summary["under_budget"],
        )
    )
    if summary["n_compressed"]:
        print(
            "  compressed %d tool_result episodes, %d tokens saved"
            % (
                summary["n_compressed"],
                summary["tokens_saved_by_compression"],
            )
        )
    for ev in summary["evicted"]:
        print(
            "  episode #%d %-11s tier %d %d tokens: %s"
            % (
                ev["episode_id"],
                ev["type"],
                ev["tier"],
                ev["tokens"],
                ev["reason"],
            )
        )
    store.close()


def cmd_episodes(args) -> None:
    """List a session's episodes, or explain one."""
    store = Store(args.db)
    if args.explain is not None:
        print(explain_episode(store, args.explain))
        store.close()
        return
    for e in store.list_episodes(args.session, status=args.status):
        rec = store.get_record(e["record_id"])
        print(
            "#%-4d seq=%-3d %-11s %-8s %3d tokens persisted_by=%s  %s"
            % (
                e["id"],
                e["seq"],
                e["type"],
                e["status"],
                e["tokens"],
                e["persisted_by"],
                (rec["text"][:100] if rec else "?"),
            )
        )
    store.close()


def cmd_durable(args) -> None:
    """P22: forward to hybrid_retriever.durable's own CLI."""
    from .durable import main as durable_main

    raise SystemExit(durable_main(args.durable_args))


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
    pe.add_argument(
        "--eviction",
        action="store_true",
        help="P18 eviction eval: one long session under a token "
        "budget (CWL vs naive vs isolated sessions)",
    )
    pe.add_argument(
        "--compression",
        action="store_true",
        help="P19 compression A/B eval: compress+evict vs "
        "evict-only on synthetic tool-output tasks",
    )
    pe.add_argument(
        "--vbd",
        action="store_true",
        help="P20 verify-by-doing eval: causal lift of rule memories "
        "on computed-answer tasks (no-memory vs bm25 / vector / "
        "hybrid / oracle, plus leave-one-out adoption gate)",
    )
    pe.add_argument(
        "--tobench",
        action="store_true",
        help="P21 mini-TOBench eval: two deterministic agent policies "
        "on 12 tool-use tasks over the memory/files/tracker servers "
        "(no embedder needed)",
    )
    pe.add_argument(
        "--k", type=int, default=5,
        help="memories retrieved per task for --vbd (default 5)",
    )
    pe.add_argument(
        "--tasks", type=int, default=40,
        help="tasks in the --eviction demo (default 40)",
    )
    pe.add_argument(
        "--budget-frac", type=float, default=1.0 / 3.0,
        help="token budget as a fraction of the full session "
        "for --eviction (default 1/3)",
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

    pi2 = sub.add_parser(
        "ingest",
        help="P18: ingest an annotated transcript (JSONL turns) as "
        "typed episodes in a session",
    )
    pi2.add_argument("--db", required=True)
    pi2.add_argument("--session", required=True)
    pi2.add_argument("--transcript", required=True)
    pi2.set_defaults(func=cmd_ingest)

    pv = sub.add_parser(
        "evict",
        help="P18: evict a session's episodes to a token budget "
        "(CWL tiers, or --policy naive oldest-first)",
    )
    pv.add_argument("--db", required=True)
    pv.add_argument("--session", required=True)
    pv.add_argument("--budget", type=int, required=True)
    pv.add_argument("--protect-last", type=int, default=3)
    pv.add_argument(
        "--policy", default="cwl", choices=["cwl", "naive"]
    )
    pv.add_argument(
        "--compress",
        action="store_true",
        help="P19: compress a tool_result episode's output before "
        "evicting it (only if the session still does not fit)",
    )
    pv.add_argument("--dry-run", action="store_true")
    pv.set_defaults(func=cmd_evict)

    pe2 = sub.add_parser(
        "episodes",
        help="P18: list a session's episodes, or explain one "
        "(--explain <episode id>)",
    )
    pe2.add_argument("--db", required=True)
    pe2.add_argument("--session", required=True)
    pe2.add_argument("--status", default=None,
                     help="filter: live, evicted")
    pe2.add_argument("--explain", type=int, default=None)
    pe2.set_defaults(func=cmd_episodes)

    pd = sub.add_parser(
        "durable",
        help="P22: durable agent loop on DBOS (run/recover/approve/"
        "status/demo/approval-demo)",
    )
    pd.add_argument(
        "durable_args", nargs=argparse.REMAINDER,
        help="arguments forwarded to hybrid_retriever.durable",
    )
    pd.set_defaults(func=cmd_durable)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
