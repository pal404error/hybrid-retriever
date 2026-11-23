"""P19 tests: type detection, per-type compressors, faithfulness,
eviction-loop integration, and the originals-stay-intact rule."""

import json
import sqlite3

import pytest

from hybrid_retriever.compress import (
    compress_text,
    detect_type,
    verify_no_invention,
)
from hybrid_retriever.embeddings import HashedEmbedder
from hybrid_retriever.eviction import (
    estimate_tokens,
    evict_to_budget,
    ingest_session,
)
from hybrid_retriever.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "t.db"), embedder=HashedEmbedder())
    yield s
    s.close()


def _json_tool(n_items=12, gold="the user prefers rust for backend work", gold_index=5):
    """A JSON API response: a homogeneous list, one gold item,
    filler items, a long string, nulls, and a duplicate."""
    items = []
    for i in range(n_items):
        items.append(
            {
                "id": i,
                "note": "unrelated entry %d about topic %d" % (i, i % 5),
                "score": 0.1 * i,
                "detail": "x" * 200,
            }
        )
    items[gold_index]["note"] = gold
    items.append(dict(items[3]))  # exact duplicate
    payload = {
        "status": "ok",
        "count": len(items),
        "results": items,
        "empty": None,
        "notes": "",
    }
    return "tool: " + json.dumps(payload)


def _log_tool():
    lines = []
    for i in range(30):
        lines.append(
            "2026-11-24T10:%02d:00Z INFO fetched record %d" % (i % 60, i)
        )
    lines[3] = lines[3]  # distinct
    for _ in range(5):
        lines.append("2026-11-24T10:00:00Z INFO heartbeat ok")
    lines.append("2026-11-24T10:00:00Z ERROR timeout on shard 7")
    return "tool: " + "\n".join(lines)


def _table_tool():
    rows = ["| id | note | score |", "|---|---|---|"]
    for i in range(15):
        rows.append("| %d | unrelated entry %d | %.1f |" % (i, i, 0.1 * i))
    rows.append("| 3 | unrelated entry 3 | 0.3 |")  # duplicate row
    return "tool: " + "\n".join(rows)


def _code_tool():
    return '''tool: """Module docstring to strip."""
import os  # inline comment


def fetch(n):
    """Fetch n records."""
    # a full-line comment
    long_text = "%s"
    return [i for i in range(n)]
''' % ("y" * 150)


# ----------------------------------------------------------------------
# Type detection
# ----------------------------------------------------------------------
def test_detect_json_table_log_code_text():
    assert detect_type(_json_tool()) == "json"
    assert detect_type(_table_tool()) == "table"
    assert detect_type(_log_tool()) == "log"
    assert detect_type(_code_tool()) == "code"
    assert detect_type("tool: just a short sentence here") == "text"
    assert detect_type('{"a": 1}') == "json"  # no prefix needed
    assert detect_type("plain text, no structure at all") == "text"


def test_detect_json_needs_valid_json():
    # Starts with { but is not JSON -> not json.
    assert detect_type("{not json at all") == "text"


# ----------------------------------------------------------------------
# JSON compressor
# ----------------------------------------------------------------------
def test_json_collapses_homogeneous_list():
    res = compress_text(_json_tool())
    assert res.type == "json"
    assert res.tokens_saved > 0
    out = json.loads(res.text[len("tool: "):])
    # Null/empty keys dropped; the list collapsed to head+tail+marker.
    assert "empty" not in out and "notes" not in out
    assert len(out["results"]) == 2 + 1 + 2  # head + tail + marker + dup marker
    assert any("more items" in str(v) for v in out["results"][-2].values())
    # The gold item sits at index 5 — inside the collapsed middle,
    # so it is lost. Collapse trades middle items for tokens;
    # faithfulness records the cost honestly (< 1.0).
    assert "rust for backend work" not in res.text
    assert res.faithfulness < 1.0
    assert verify_no_invention(_json_tool(), res.text) == []


def test_json_gold_in_kept_head_survives():
    res = compress_text(_json_tool(gold_index=1))
    assert "rust for backend work" in res.text
    assert res.faithfulness > 0.5


def test_json_long_string_truncates_with_marker():
    res = compress_text(_json_tool())
    assert "(200 chars)" in res.text
    assert "x" * 200 not in res.text


# ----------------------------------------------------------------------
# Table / log / code / text compressors
# ----------------------------------------------------------------------
def test_table_dedupes_and_caps_rows():
    res = compress_text(_table_tool())
    assert res.type == "table"
    assert res.tokens_saved > 0
    assert "more rows" in res.text
    assert "duplicate rows removed" in res.text
    assert verify_no_invention(_table_tool(), res.text) == []


def test_log_folds_repeats():
    res = compress_text(_log_tool())
    assert res.type == "log"
    assert "(x5)" in res.text  # heartbeat folded
    assert "timeout on shard 7" in res.text  # rare line kept
    assert res.tokens_saved > 0
    assert verify_no_invention(_log_tool(), res.text) == []


def test_code_strips_docstrings_and_comments():
    res = compress_text(_code_tool())
    assert res.type == "code"
    assert "Module docstring to strip" not in res.text
    assert "Fetch n records" not in res.text
    assert "a full-line comment" not in res.text
    assert "inline comment" not in res.text
    # Code itself survives, including the line with the comment.
    assert "import os" in res.text
    assert "def fetch(n):" in res.text
    assert "return [i for i in range(n)]" in res.text
    assert "(150 chars)" in res.text
    assert res.faithfulness == 1.0
    assert verify_no_invention(_code_tool(), res.text) == []


def test_text_extracts_and_dedupes():
    body = " ".join(
        ["The first sentence matters."] * 2
        + ["Sentence number %d here." % i for i in range(8)]
        + ["The last sentence matters."]
    )
    res = compress_text("tool: " + body)
    assert res.type == "text"
    assert res.text.count("The first sentence matters.") == 1
    assert "sentences omitted" in res.text
    assert "The last sentence matters." in res.text
    assert res.tokens_saved > 0


def test_small_text_is_returned_unchanged():
    short = "tool: hello world, a tiny result"
    res = compress_text(short)
    assert res.text == short
    assert res.tokens_saved == 0


def test_prefix_is_preserved():
    res = compress_text(_json_tool())
    assert res.text.startswith("tool: ")


# ----------------------------------------------------------------------
# Eviction-loop integration
# ----------------------------------------------------------------------
def _session_with_big_tool_result(store):
    turns = [
        {"role": "user", "text": "what do the records say?"},
        {
            "role": "assistant",
            "text": "looking it up",
            "episode_type": "tool_call",
        },
        {"role": "tool", "text": _json_tool()[len("tool: "):]},
        {"role": "assistant", "text": "noted", "episode_type": "chatter"},
    ]
    # ingest_session adds the "tool: " prefix via turn_text.
    return ingest_session(store, "s", turns)


def test_compress_before_evict_keeps_episode_live(store):
    ep_ids = _session_with_big_tool_result(store)
    total_before = store.session_token_total("s")
    # The tool_call is protected while its tool_result lives
    # (the result depends on the call), so only the chatter can
    # be evicted before the tool_result is compressed. The budget
    # below accounts for exactly that: chatter evicted, then the
    # tool_result's compression alone brings the session under
    # budget, so the episode stays live. protect_last=0: the
    # 4-turn session would otherwise be entirely inside the
    # active window.
    tool_ep = store.get_episode(ep_ids[2])
    raw = store.get_record(tool_ep["record_id"])["text"]
    res = compress_text(raw)
    assert res.tokens_saved > 0
    chatter_tokens = store.get_episode(ep_ids[3])["tokens"]
    budget = total_before - chatter_tokens - res.tokens_saved
    summary = evict_to_budget(
        store, "s", budget, protect_last=0, compress=True
    )
    assert summary["n_compressed"] == 1
    assert summary["n_evicted"] == 1  # chatter only
    assert summary["under_budget"] is True
    assert summary["tokens_saved_by_compression"] == res.tokens_saved
    # The episode is still live, at its compressed size.
    live = store.list_episodes("s", status="live")
    assert len(live) == 3  # user + tool_call + compressed tool_result
    ep = store.get_episode(ep_ids[2])
    assert ep["status"] == "live"
    assert ep["tokens"] == res.tokens_after
    assert ep["compressed_text"] == res.text
    # The audit row exists; the record text is untouched.
    rows = store.list_compressions(session_id="s")
    assert len(rows) == 1
    assert rows[0]["type"] == "json"
    assert store.get_record(tool_ep["record_id"])["text"] == raw
    # The agent sees the compressed text; the original is on demand.
    assert store.episode_context_text(ep_ids[2]) == res.text


def test_still_over_budget_after_compression_evicts(store):
    ep_ids = _session_with_big_tool_result(store)
    # Budget below even the compressed total: compress first,
    # then evict the (compressed) episode.
    summary = evict_to_budget(store, "s", 10, protect_last=0, compress=True)
    assert summary["n_compressed"] == 1
    assert summary["n_evicted"] == 3  # chatter + tool_call + tool_result
    assert ep_ids[2] in [e["episode_id"] for e in summary["evicted"]]
    # The eviction record notes the compression happened first.
    from hybrid_retriever.eviction import explain_episode

    assert "compressed" in explain_episode(store, ep_ids[2])


def test_compress_dry_run_writes_nothing(store):
    ep_ids = _session_with_big_tool_result(store)
    total_before = store.session_token_total("s")
    summary = evict_to_budget(
        store, "s", 10, protect_last=0, compress=True, dry_run=True
    )
    assert summary["would_compress"] == 1
    assert summary["would_evict"] == 3
    assert store.list_compressions() == []
    assert store.list_evictions() == []
    assert store.session_token_total("s") == total_before
    assert store.get_episode(ep_ids[2])["compressed_text"] is None


def test_compress_false_is_p18_behavior(store):
    ep_ids = _session_with_big_tool_result(store)
    summary = evict_to_budget(store, "s", 10, protect_last=0, compress=False)
    assert summary["n_compressed"] == 0
    assert summary["tokens_saved_by_compression"] == 0
    assert summary["n_evicted"] == 3
    assert store.list_compressions() == []


def test_uncompressible_tool_result_is_evicted_not_compressed(store):
    turns = [
        {"role": "user", "text": "hi"},
        {"role": "tool", "text": "tiny result"},
    ]
    ep_ids = ingest_session(store, "s", turns)
    summary = evict_to_budget(store, "s", 5, protect_last=0, compress=True)
    assert summary["n_compressed"] == 0
    assert summary["n_evicted"] == 1
    assert summary["evicted"][0]["episode_id"] == ep_ids[1]


def test_compressed_episode_is_never_recompressed(store):
    """Regression: an episode compressed in one eviction call
    must not be 'compressed' again in a later call — the second
    pass would re-count the same savings against the original
    record text (phantom compression) and log a duplicate row.
    Instead the already-compressed episode is evicted."""
    ep_ids = _session_with_big_tool_result(store)
    tool_ep = store.get_episode(ep_ids[2])
    raw = store.get_record(tool_ep["record_id"])["text"]
    res = compress_text(raw)
    total_before = store.session_token_total("s")
    # The tool_call is protected while its tool_result lives
    # (the result depends on the call), so only the chatter can
    # be evicted before the tool_result is compressed.
    chatter_tokens = store.get_episode(ep_ids[3])["tokens"]
    budget = total_before - chatter_tokens - res.tokens_saved
    first = evict_to_budget(
        store, "s", budget, protect_last=0, compress=True
    )
    assert first["n_compressed"] == 1
    assert first["n_evicted"] == 1  # chatter only
    assert first["under_budget"] is True
    # Tighten the budget: the compressed episode is now the
    # victim again, and must be evicted, not re-compressed.
    # (Its tool_call follows once the result is gone.)
    second = evict_to_budget(
        store, "s", 10, protect_last=0, compress=True
    )
    assert second["n_compressed"] == 0
    assert ep_ids[2] in [e["episode_id"] for e in second["evicted"]]
    assert len(store.list_compressions(session_id="s")) == 1


def test_migration_adds_columns_to_old_db(tmp_path):
    """A DB created by P18-era code gains the P19 columns."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE episodes(id INTEGER PRIMARY KEY, "
        "session_id TEXT, seq INTEGER, record_id INTEGER, "
        "type TEXT, tokens INTEGER, status TEXT, persisted_by "
        "INTEGER, depends_on TEXT, created_at REAL)"
    )
    conn.commit()
    conn.close()
    s = Store(path, embedder=HashedEmbedder())
    cols = {
        r[1] for r in s.conn.execute("PRAGMA table_info(episodes)")
    }
    assert "compressed_text" in cols
    assert "compressed_tokens" in cols
    tables = {
        r[0]
        for r in s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "compressions" in tables
    s.close()


# ----------------------------------------------------------------------
# A/B harness smoke test
# ----------------------------------------------------------------------
def test_compression_eval_smoke():
    """4 tasks, hashed embedder (fast): the harness runs, the
    compress arm compresses and saves tokens, nothing is
    invented, user turns and dependencies survive."""
    from hybrid_retriever.eval import run_compression_eval

    result = run_compression_eval(
        "eval_data/synthetic_corpus.jsonl",
        "eval_data/synthetic_queries.json",
        embedder=HashedEmbedder(),
        n_tasks=4,
        verbose=False,
    )
    arms = result["arms"]
    assert arms["isolated"]["gold_survival_context"] == 1.0
    c = arms["compress"]
    assert c["compressed_total"] > 0
    assert c["tokens_saved_by_compression"] > 0
    assert c["no_invention_violations"] == 0
    assert 0.0 <= c["gold_survival_context"] <= 1.0
    assert 0.0 <= c["gold_survival_mean_over_time"] <= 1.0
    assert 0.0 < c["faithfulness"] <= 1.0
    assert c["latency_ms_mean"] >= 0.0
    assert c["user_turns_live"] == 4
    assert c["dependency_violations"] == 0
    assert c["under_budget"] is True
