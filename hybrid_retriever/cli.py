"""Command line: index a corpus, run a query, run the eval, sleep, credit, merges, consolidate, plan."""

from __future__ import annotations

import argparse
import json
import os

from .corecall import graph_stats, reconsolidate
from .embeddings import get_embedder
from .eval import (
    print_compression_report,
    print_corecall_report,
    print_eviction_report,
    print_report,
    print_segment_report,
    run_compression_eval,
    run_corecall_eval,
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
    if args.gepa:
        from .gepa import print_gepa_report, run_gepa_demo, run_nightly, TraceStore
        import os as _os
        import tempfile as _tempfile

        if args.skill is None and args.traces is None:
            result = run_gepa_demo(nights=args.nights)
            print_gepa_report(result)
            return
        skill_path = args.skill or _os.path.join("skills", "tobench-agent", "SKILL.md")
        traces_path = args.traces or _os.path.join(
            _os.path.dirname(_os.path.abspath(skill_path)), "traces.db"
        )
        if not _os.path.exists(skill_path):
            from .gepa import render_skill

            _os.makedirs(_os.path.dirname(_os.path.abspath(skill_path)), exist_ok=True)
            with open(skill_path, "w", encoding="utf-8") as fh:
                fh.write(render_skill([]))
        store = TraceStore(traces_path)
        for night in range(args.nights):
            report = run_nightly(store, skill_path)
            print(
                "night %d: eval %d/%d, proposal %s -> %s"
                % (
                    night + 1,
                    report["eval_passed"],
                    report["eval_total"],
                    (report["proposal"] or {}).get("rule_id"),
                    report["verdict"],
                )
            )
        store.close()
        return
    if args.lessons:
        from .lessons import (
            print_lessons_report,
            run_lessons_demo,
            write_guidelines,
        )

        result = run_lessons_demo(k=args.lessons_k)
        print_lessons_report(result)
        if args.lessons_out:
            write_guidelines(args.lessons_out, result["guidelines"])
            print("guidelines written to %s" % args.lessons_out)
        return
    if args.lora:
        from .lora import (
            print_lora_report,
            print_planted_conflict_report,
            run_lora_eval,
            run_planted_conflict,
        )

        result = run_lora_eval(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            os.path.join("eval_data", "synthetic_queries.json"),
            embedder=embedder,
            rank=args.lora_rank,
        )
        print_lora_report(result)
        conflict = run_planted_conflict(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            os.path.join("eval_data", "synthetic_queries.json"),
            embedder=embedder,
            rank=args.lora_rank,
        )
        print_planted_conflict_report(conflict)
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
    if args.corecall:
        result = run_corecall_eval(
            os.path.join("eval_data", "synthetic_corpus.jsonl"),
            os.path.join("eval_data", "synthetic_queries.json"),
            embedder=embedder,
        )
        print_corecall_report(result)
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


def cmd_reconsolidate(args) -> None:
    """P4: nightly reconsolidation — promote / re-link / prune / merge."""
    embedder = get_embedder()
    store = Store(args.db, embedder=embedder)
    summary = reconsolidate(
        store,
        embedder=embedder,
        min_wsum=args.min_wsum,
        min_count=args.min_count,
        closure_decay=args.closure_decay,
        dry_run=args.dry_run,
    )
    print(
        "reconsolidate%s: promoted=%d closure=%d pruned=%d clusters=%d "
        "merges proposed=%d committed=%d%s rejected=%d "
        "co-recall edges: %d -> %d (live records: %d)"
        % (
            " (dry run)" if args.dry_run else "",
            summary["promoted"],
            summary["closure_added"],
            summary["pruned"],
            summary["clusters"],
            summary["merges_proposed"],
            summary["merges_committed"],
            (
                " would-commit=%d" % summary["merges_would_commit"]
                if args.dry_run
                else ""
            ),
            summary["merges_rejected"],
            summary["edges_before"],
            summary["edges_after"],
            store.count_live(),
        )
    )
    store.close()


def cmd_graph(args) -> None:
    """P4: snapshot of the co-recall graph in a store."""
    store = Store(args.db)
    stats = graph_stats(store)
    print(
        "graph: live=%d corecall pairs=%d (>=1.5: %d) edges=%d %s "
        "clusters=%d reconsolidations=%d"
        % (
            stats["live_records"],
            stats["corecall_pairs"],
            stats["corecall_pairs_ge_1.5"],
            stats["edges_total"],
            stats["edges_by_kind"],
            stats["clusters"],
            stats["reconsolidations"],
        )
    )
    store.close()


def cmd_lora(args) -> None:
    """P6: distil a versioned LoRA artifact from the synthetic corpus
    (the per-project swap: a different --project/--version loads a
    different artifact with ``load_student``)."""
    import tempfile

    from .eval import load_synthetic
    from .lora import (
        artifact_path,
        build_distillation_examples,
        build_fact_index,
        save_student,
        train_student,
    )

    embedder = get_embedder()
    records, queries = load_synthetic(
        os.path.join("eval_data", "synthetic_corpus.jsonl"),
        os.path.join("eval_data", "synthetic_queries.json"),
    )
    tmp = tempfile.mkdtemp(prefix="hr-lora-train-")
    store = Store(os.path.join(tmp, "lora.db"), embedder=embedder)
    rids = store.add_texts(
        [r["text"] for r in records], type_="fact", embedder=embedder
    )
    index = build_fact_index(records, queries, rids)
    texts, labels = build_distillation_examples(index, queries)
    student = train_student(
        texts,
        labels,
        embedder,
        rank=args.rank,
        project=args.project,
        version=args.version,
        fact_set_hash=index.fact_set_hash(),
    )
    out = args.out or artifact_path(args.project, args.version, ".")
    save_student(student, out)
    print(
        "lora %s v%d: %d facts, %d train examples, rank=%d, "
        "%d params, train acc=%.3f, audit threshold=%.3f -> %s"
        % (
            student.project,
            student.version,
            len(student.fact_ids),
            student.n_train,
            student.rank,
            student.n_params,
            student.train_accuracy,
            student.audit_threshold,
            out,
        )
    )
    store.close()


def cmd_lessons(args) -> None:
    """P5: run the lessons-miner demo and optionally persist the
    mined guidelines file (a natural .larp import)."""
    import shutil

    from .lessons import print_lessons_report, run_lessons_demo

    result = run_lessons_demo(k=args.k)
    print_lessons_report(result)
    if args.out:
        shutil.copyfile(result["guidelines_path"], args.out)
        print("guidelines written to %s" % args.out)


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
        "--gepa",
        action="store_true",
        help="P23 GEPA-style loop: N nights of trace -> diagnose -> "
        "propose one skill edit -> A/B on the fixed P21 set, keep "
        "only if it wins (no --skill/--traces runs the temp-dir "
        "demo incl. the planted bad-edit trial)",
    )
    pe.add_argument(
        "--nights", type=int, default=4,
        help="nights for --gepa (default 4)",
    )
    pe.add_argument(
        "--skill", default=None,
        help="SKILL.md path for --gepa (default: temp-dir demo skill)",
    )
    pe.add_argument(
        "--traces", default=None,
        help="trace-store SQLite path for --gepa "
        "(default: temp-dir demo store)",
    )
    pe.add_argument(
        "--lessons",
        action="store_true",
        help="P5 IBM-style lessons miner: mine lessons from the "
        "no-rule run, merge into guidelines, A/B none / retrieved / "
        "full injection on the fixed P21 set, and trial the planted "
        "backfire guideline through the gate",
    )
    pe.add_argument(
        "--lessons-k", type=int, default=2,
        help="guidelines retrieved per task for --lessons (default 2)",
    )
    pe.add_argument(
        "--lessons-out", default=None,
        help="write the mined guidelines file here for --lessons",
    )
    pe.add_argument(
        "--lora",
        action="store_true",
        help="P6 LoRA weight memory: HaluMem-style three-arm eval "
        "(retrieval / student / student+verify) plus the planted-"
        "conflict demo on the synthetic corpus",
    )
    pe.add_argument(
        "--lora-rank", type=int, default=32,
        help="adapter rank for --lora (default 32)",
    )
    pe.add_argument(
        "--corecall",
        action="store_true",
        help="P4 co-recall graph eval: hybrid / vector / naive / raw / "
        "reconsolidated expansion on the synthetic corpus (warm-up on "
        "keyword queries, eval on paraphrase+trap)",
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

    pr = sub.add_parser(
        "reconsolidate",
        help="P4: nightly reconsolidation — promote co-recall pairs "
        "to edges, re-link clusters, prune stale edges, merge "
        "near-duplicates",
    )
    pr.add_argument("--db", required=True)
    pr.add_argument("--dry-run", action="store_true")
    pr.add_argument("--min-wsum", type=float, default=1.5)
    pr.add_argument("--min-count", type=int, default=2)
    pr.add_argument("--closure-decay", type=float, default=0.5)
    pr.set_defaults(func=cmd_reconsolidate)

    pg = sub.add_parser(
        "graph", help="P4: snapshot of the co-recall graph in a store"
    )
    pg.add_argument("--db", required=True)
    pg.set_defaults(func=cmd_graph)

    pl = sub.add_parser(
        "lora",
        help="P6: distil a versioned LoRA artifact from the synthetic "
        "corpus (per-project swap via load_student)",
    )
    pl.add_argument("--project", default="synthetic")
    pl.add_argument("--version", type=int, default=1)
    pl.add_argument("--rank", type=int, default=32)
    pl.add_argument(
        "--out", default=None,
        help="artifact path (default: lora_<project>_v<version>.npz)",
    )
    pl.set_defaults(func=cmd_lora)

    pl2 = sub.add_parser(
        "lessons",
        help="P5: mine lessons from the no-rule run, merge into "
        "guidelines, A/B none / retrieved / full injection, trial "
        "the planted backfire guideline through the gate",
    )
    pl2.add_argument(
        "--k", type=int, default=2,
        help="guidelines retrieved per task in the retrieved arm "
        "(default 2)",
    )
    pl2.add_argument(
        "--out", default=None,
        help="write the mined guidelines file here",
    )
    pl2.set_defaults(func=cmd_lessons)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
