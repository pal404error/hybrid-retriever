"""SQLite store: records + FTS5 (BM25) + vectors + a minimal graph.

Schema
------
records(id, text, type, tags, created_at, meta)
  tags/meta are JSON strings. ``type`` is free-form: fact / preference /
  episode / correction, matching the .larp record types.
records_fts
  FTS5 external-content table on records.text, kept in sync by triggers.
  This is the BM25 leg.
record_vecs (sqlite-vec) OR embeddings (NumPy fallback)
  Vectors live in a ``vec0`` virtual table when the ``sqlite_vec``
  extension loads; otherwise as BLOBs in ``embeddings`` and search is
  exact NumPy cosine. The two paths expose the same API, and the
  constructor records which one is active in ``self.vec_mode``.
edges(src, dst, weight, kind)
  Minimal graph for the neighbour-expansion leg: 1-hop neighbours of
  the top BM25 / vector hits, ranked by edge weight. v1 keeps the
  graph as an expander only — it never invents records, it only
  re-surfaces ones linked to a hit.

P2 (MEMOIR-style lifecycle) adds, all on records unless noted:
  status       'live' | 'superseded' | 'pending' | 'pruned' | 'archived'.
               Search legs only ever return 'live' records; the other
               statuses are audit history (superseded originals,
               pruned dead memories, rolled-back merges).
  superseded_by  id of the merged record that replaced this one.
  importance   learned prior, updated by TRACE-UCB (lifecycle.py).
               Neutral prior 0.5; mean credit + exploration bonus.
  credit       cumulative credit attributed by record_use.
  uses         times the record was retrieved for an answer.
  last_used    timestamp of the last retrieval.
merges(id, merged_ids, merged_record_id, merged_text, certificate, status, created_at)
  Audit table for judge-gated consolidation: one row per proposed
  merge, committed / rejected / rolled-back, with the full
  sufficiency certificate as JSON.
co_recall(a, b, count)
  How often two records were used together for one answer
  (a < b). The sleep job promotes count >= 2 pairs to 'co-recall'
  edges; the online path only ever writes this table, never edges.

P4 (REALM-style co-recall graph) adds, all in the same SQLite file:
corecall(a, b, count, wsum)
  Credit-weighted co-recall log (a < b): ``count`` is how many
  answers credited both records, ``wsum`` is the sum over those
  answers of min(credit_a, credit_b). Raw counts do not separate
  gold pairs from same-person distractors (probe: both reach
  count 15 over 3 warm-up rounds); the credit weight does
  (wsum >= 1.5 keeps 118/146 gold pairs, 4/1156 others).
  Promotion to 'co-recall' edges reads this table, never the raw
  co_recall counts.
reconsolidations(id, kind, detail, created_at)
  Audit table for the nightly reconsolidation job (P4 phase 3):
  one row per run, detail as JSON (promoted / pruned / closure /
  merges / clusters).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
  INSERT INTO records_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
  INSERT INTO records_fts(records_fts, rowid, text)
  VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN
  INSERT INTO records_fts(records_fts, rowid, text)
  VALUES ('delete', old.id, old.text);
  INSERT INTO records_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def _fts_query(query: str) -> str | None:
    """Turn free text into a safe FTS5 OR query of quoted terms.

    Returns None when the query has no indexable tokens.
    """
    tokens = _TOKEN_RE.findall(query.lower())
    if not tokens:
        return None
    return " OR ".join('"%s"' % t for t in tokens)


class Store:
    """A single SQLite file holding records, their vectors, and edges."""

    def __init__(self, path: str, embedder=None) -> None:
        self.path = path
        self.embedder = embedder
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._vec_dim: int | None = None
        self.vec_mode = self._detect_vec_mode()
        self._init_schema()
        self._migrate()

    # P2 columns added after v1; existing files are migrated in place.
    _P2_COLUMNS = {
        "status": "TEXT NOT NULL DEFAULT 'live'",
        "superseded_by": "INTEGER",
        "importance": "REAL NOT NULL DEFAULT 0.5",
        "credit": "REAL NOT NULL DEFAULT 0.0",
        "uses": "INTEGER NOT NULL DEFAULT 0",
        "last_used": "REAL",
    }

    # P19 columns on episodes; existing files are migrated in place.
    _P19_EPISODE_COLUMNS = {
        "compressed_text": "TEXT",
        "compressed_tokens": "INTEGER",
    }

    def _migrate(self) -> None:
        have = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(records)")
        }
        for name, decl in self._P2_COLUMNS.items():
            if name not in have:
                self.conn.execute(
                    "ALTER TABLE records ADD COLUMN %s %s" % (name, decl)
                )
        ep_have = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(episodes)")
        }
        for name, decl in self._P19_EPISODE_COLUMNS.items():
            if name not in ep_have:
                self.conn.execute(
                    "ALTER TABLE episodes ADD COLUMN %s %s"
                    % (name, decl)
                )
        self.conn.commit()

    # -- setup ---------------------------------------------------------
    def _detect_vec_mode(self) -> str:
        try:
            import sqlite_vec  # noqa: F401

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            return "sqlite-vec"
        except Exception:
            return "numpy"

    def _init_schema(self) -> None:
        c = self.conn
        c.execute(
            """CREATE TABLE IF NOT EXISTS records(
                 id INTEGER PRIMARY KEY,
                 text TEXT NOT NULL,
                 type TEXT NOT NULL DEFAULT 'fact',
                 tags TEXT NOT NULL DEFAULT '[]',
                 created_at REAL NOT NULL,
                 meta TEXT NOT NULL DEFAULT '{}',
                 status TEXT NOT NULL DEFAULT 'live',
                 superseded_by INTEGER,
                 importance REAL NOT NULL DEFAULT 0.5,
                 credit REAL NOT NULL DEFAULT 0.0,
                 uses INTEGER NOT NULL DEFAULT 0,
                 last_used REAL)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS merges(
                 id INTEGER PRIMARY KEY,
                 merged_ids TEXT NOT NULL,
                 merged_record_id INTEGER,
                 merged_text TEXT NOT NULL DEFAULT '',
                 certificate TEXT NOT NULL DEFAULT '{}',
                 status TEXT NOT NULL DEFAULT 'proposed',
                 created_at REAL NOT NULL)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS co_recall(
                 a INTEGER NOT NULL,
                 b INTEGER NOT NULL,
                 count INTEGER NOT NULL DEFAULT 0,
                 PRIMARY KEY (a, b))"""
        )
        # P4 (REALM-style co-recall graph): credit-weighted co-recall
        # log. Created here so existing files gain it on open, exactly
        # like every other table in this schema.
        c.execute(
            """CREATE TABLE IF NOT EXISTS corecall(
                 a INTEGER NOT NULL,
                 b INTEGER NOT NULL,
                 count INTEGER NOT NULL DEFAULT 0,
                 wsum REAL NOT NULL DEFAULT 0.0,
                 PRIMARY KEY (a, b))"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS reconsolidations(
                 id INTEGER PRIMARY KEY,
                 kind TEXT NOT NULL,
                 detail TEXT NOT NULL DEFAULT '{}',
                 created_at REAL NOT NULL)"""
        )
        c.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts "
            "USING fts5(text, content='records', content_rowid='id')"
        )
        c.executescript(_FTS_TRIGGERS)
        c.execute(
            """CREATE TABLE IF NOT EXISTS edges(
                 src INTEGER NOT NULL, dst INTEGER NOT NULL,
                 weight REAL NOT NULL DEFAULT 1.0,
                 kind TEXT NOT NULL DEFAULT 'related',
                 PRIMARY KEY (src, dst, kind))"""
        )
        # P18 (CWL structured context eviction): episodes are the
        # context — typed, dependency-linked units of one session's
        # trajectory, each linked to its turn record. Records are the
        # memory; eviction removes from context, never deletes memory.
        c.execute(
            """CREATE TABLE IF NOT EXISTS episodes(
                 id INTEGER PRIMARY KEY,
                 session_id TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 record_id INTEGER NOT NULL,
                 type TEXT NOT NULL,
                 tokens INTEGER NOT NULL,
                 status TEXT NOT NULL DEFAULT 'live',
                 persisted_by INTEGER,
                 depends_on TEXT NOT NULL DEFAULT '[]',
                 compressed_text TEXT,
                 compressed_tokens INTEGER,
                 created_at REAL NOT NULL)"""
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS episodes_session "
            "ON episodes(session_id, seq)"
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS evictions(
                 id INTEGER PRIMARY KEY,
                 episode_id INTEGER NOT NULL,
                 session_id TEXT NOT NULL,
                 reason TEXT NOT NULL,
                 tier INTEGER NOT NULL,
                 tokens_freed INTEGER NOT NULL,
                 budget_tokens INTEGER NOT NULL,
                 total_tokens_before INTEGER NOT NULL,
                 created_at REAL NOT NULL)"""
        )
        # P19 (type-aware compression): one row per compression
        # applied to an episode — the measurement log for the A/B
        # harness, next to the evictions table. The compressed
        # text itself lives on the episode row (compressed_text /
        # compressed_tokens columns, migrated in _migrate); the
        # record text is never rewritten.
        c.execute(
            """CREATE TABLE IF NOT EXISTS compressions(
                 id INTEGER PRIMARY KEY,
                 episode_id INTEGER NOT NULL,
                 session_id TEXT NOT NULL,
                 type TEXT NOT NULL,
                 tokens_before INTEGER NOT NULL,
                 tokens_after INTEGER NOT NULL,
                 units_before INTEGER NOT NULL,
                 units_after INTEGER NOT NULL,
                 latency_ms REAL NOT NULL,
                 created_at REAL NOT NULL)"""
        )
        if self.vec_mode == "numpy":
            c.execute(
                """CREATE TABLE IF NOT EXISTS embeddings(
                     record_id INTEGER PRIMARY KEY,
                     dim INTEGER NOT NULL, vec BLOB NOT NULL)"""
            )
        c.commit()

    def _ensure_vec_table(self, dim: int) -> None:
        if self.vec_mode != "sqlite-vec":
            return
        if self._vec_dim == dim:
            return
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS record_vecs "
            "USING vec0(embedding float[%d])" % dim
        )
        self._vec_dim = dim

    # -- writes --------------------------------------------------------
    def add_record(
        self,
        text: str,
        type_: str = "fact",
        tags: list[str] | None = None,
        meta: dict | None = None,
        embedding: np.ndarray | None = None,
        status: str = "live",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO records(text, type, tags, created_at, meta, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                text,
                type_,
                json.dumps(tags or []),
                time.time(),
                json.dumps(meta or {}),
                status,
            ),
        )
        rid = int(cur.lastrowid)
        if embedding is not None:
            self._store_vector(rid, embedding)
        elif self.embedder is not None:
            self._store_vector(rid, self.embedder.embed([text])[0])
        self.conn.commit()
        return rid

    def add_texts(
        self, texts: list[str], type_: str = "fact", embedder=None
    ) -> list[int]:
        """Batch insert; embeds in one call when an embedder is given."""
        emb = embedder or self.embedder
        vecs = emb.embed(texts) if emb is not None else [None] * len(texts)
        return [
            self.add_record(t, type_=type_, embedding=v)
            for t, v in zip(texts, vecs)
        ]

    def _store_vector(self, rid: int, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32)
        if self.vec_mode == "sqlite-vec":
            import sqlite_vec

            self._ensure_vec_table(int(vec.shape[0]))
            self.conn.execute(
                "INSERT OR REPLACE INTO record_vecs(rowid, embedding) "
                "VALUES (?, ?)",
                (rid, sqlite_vec.serialize_float32(vec)),
            )
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(record_id, dim, vec) "
                "VALUES (?, ?, ?)",
                (rid, int(vec.shape[0]), vec.tobytes()),
            )

    def add_edge(
        self, src: int, dst: int, weight: float = 1.0, kind: str = "related"
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO edges(src, dst, weight, kind) "
            "VALUES (?, ?, ?, ?)",
            (src, dst, weight, kind),
        )
        self.conn.commit()

    def add_edges(self, edges: list[tuple[int, int, float, str]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO edges(src, dst, weight, kind) "
            "VALUES (?, ?, ?, ?)",
            edges,
        )
        self.conn.commit()

    # -- P2: lifecycle -------------------------------------------------
    def non_live_ids(self) -> set[int]:
        """Ids of records the search legs must never return."""
        return {
            int(r[0])
            for r in self.conn.execute(
                "SELECT id FROM records WHERE status != 'live'"
            )
        }

    def count_live(self) -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM records WHERE status = 'live'"
            ).fetchone()[0]
        )

    def set_status(
        self, rid: int, status: str, superseded_by: int | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE records SET status = ?, superseded_by = ? WHERE id = ?",
            (status, superseded_by, rid),
        )
        self.conn.commit()

    def set_status_many(
        self, rids: list[int], status: str, superseded_by: int | None = None
    ) -> None:
        self.conn.executemany(
            "UPDATE records SET status = ?, superseded_by = ? WHERE id = ?",
            [(status, superseded_by, rid) for rid in rids],
        )
        self.conn.commit()

    def record_credit_update(
        self, rid: int, uses: int, credit: float, importance: float
    ) -> None:
        self.conn.execute(
            "UPDATE records SET uses = ?, credit = ?, importance = ?, "
            "last_used = ? WHERE id = ?",
            (uses, credit, importance, time.time(), rid),
        )
        self.conn.commit()

    def get_importances(self, rids: list[int]) -> dict[int, float]:
        if not rids:
            return {}
        rows = self.conn.execute(
            "SELECT id, importance FROM records WHERE id IN (%s)"
            % ",".join("?" * len(rids)),
            rids,
        ).fetchall()
        return {int(r): float(i) for r, i in rows}

    def total_uses(self) -> int:
        row = self.conn.execute("SELECT COALESCE(SUM(uses), 0) FROM records").fetchone()
        return int(row[0])

    def log_co_recall(self, a: int, b: int) -> None:
        if a == b:
            return
        lo, hi = (a, b) if a < b else (b, a)
        self.conn.execute(
            "INSERT INTO co_recall(a, b, count) VALUES (?, ?, 1) "
            "ON CONFLICT(a, b) DO UPDATE SET count = count + 1",
            (lo, hi),
        )
        self.conn.commit()

    def co_recall_pairs(self, min_count: int = 2) -> list[tuple[int, int, int]]:
        return [
            (int(a), int(b), int(n))
            for a, b, n in self.conn.execute(
                "SELECT a, b, count FROM co_recall WHERE count >= ? "
                "ORDER BY count DESC",
                (min_count,),
            ).fetchall()
        ]

    # -- P4: credit-weighted co-recall log + reconsolidation audit ----
    def log_corecall(self, a: int, b: int, weight: float) -> None:
        """Log one co-credited use of the pair (a, b).

        ``weight`` is min(credit_a, credit_b) for that answer: a pair
        only accumulates weight when *both* records actually earned
        credit for the same answer, which is what separates jointly
        useful memories from records that merely co-occurred in a
        top-k list. Self-pairs and non-positive weights are ignored.
        """
        if a == b or weight <= 0:
            return
        lo, hi = (a, b) if a < b else (b, a)
        self.conn.execute(
            "INSERT INTO corecall(a, b, count, wsum) VALUES (?, ?, 1, ?) "
            "ON CONFLICT(a, b) DO UPDATE SET count = count + 1, "
            "wsum = wsum + excluded.wsum",
            (lo, hi, float(weight)),
        )
        self.conn.commit()

    def corecall_pairs(
        self, min_count: int = 1, min_wsum: float = 0.0
    ) -> list[tuple[int, int, int, float]]:
        """[(a, b, count, wsum)] with count >= min_count and
        wsum >= min_wsum, best weight first."""
        return [
            (int(a), int(b), int(n), float(w))
            for a, b, n, w in self.conn.execute(
                "SELECT a, b, count, wsum FROM corecall "
                "WHERE count >= ? AND wsum >= ? ORDER BY wsum DESC",
                (min_count, min_wsum),
            ).fetchall()
        ]

    def corecall_weight(self, a: int, b: int) -> float:
        """Accumulated credit weight for one pair (0.0 if never logged)."""
        if a == b:
            return 0.0
        lo, hi = (a, b) if a < b else (b, a)
        row = self.conn.execute(
            "SELECT wsum FROM corecall WHERE a = ? AND b = ?", (lo, hi)
        ).fetchone()
        return float(row[0]) if row is not None else 0.0

    def list_edges(
        self, kind: str | None = None
    ) -> list[tuple[int, int, float, str]]:
        sql = "SELECT src, dst, weight, kind FROM edges"
        params: list = []
        if kind is not None:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY weight DESC"
        return [
            (int(s), int(d), float(w), k)
            for s, d, w, k in self.conn.execute(sql, params).fetchall()
        ]

    def delete_edge(self, src: int, dst: int, kind: str) -> None:
        self.conn.execute(
            "DELETE FROM edges WHERE src = ? AND dst = ? AND kind = ?",
            (src, dst, kind),
        )
        self.conn.commit()

    def add_reconsolidation(self, kind: str, detail: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO reconsolidations(kind, detail, created_at) "
            "VALUES (?, ?, ?)",
            (kind, json.dumps(detail), time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_reconsolidations(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, kind, detail, created_at FROM reconsolidations "
            "ORDER BY id"
        ).fetchall()
        return [
            {
                "id": int(r[0]),
                "kind": r[1],
                "detail": json.loads(r[2]),
                "created_at": r[3],
            }
            for r in rows
        ]

    def list_live_records(
        self, older_than_days: float | None = None
    ) -> list[dict]:
        """Live records, optionally only those older than N days.

        Used by the sleep job's pruning pass.
        """
        sql = (
            "SELECT id, text, type, tags, created_at, meta, status, "
            "superseded_by, importance, credit, uses, last_used "
            "FROM records WHERE status = 'live'"
        )
        params: list = []
        if older_than_days is not None:
            sql += " AND created_at < ?"
            params.append(time.time() - older_than_days * 86400.0)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row) -> dict:
        return {
            "id": row[0],
            "text": row[1],
            "type": row[2],
            "tags": json.loads(row[3]),
            "created_at": row[4],
            "meta": json.loads(row[5]),
            "status": row[6],
            "superseded_by": row[7],
            "importance": row[8],
            "credit": row[9],
            "uses": row[10],
            "last_used": row[11],
        }

    def add_merge(
        self,
        merged_ids: list[int],
        merged_record_id: int | None,
        merged_text: str,
        certificate: dict,
        status: str,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO merges(merged_ids, merged_record_id, merged_text, "
            "certificate, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                json.dumps(merged_ids),
                merged_record_id,
                merged_text,
                json.dumps(certificate),
                status,
                time.time(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_merge(self, merge_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id, merged_ids, merged_record_id, merged_text, "
            "certificate, status, created_at FROM merges WHERE id = ?",
            (merge_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "merged_ids": json.loads(row[1]),
            "merged_record_id": row[2],
            "merged_text": row[3],
            "certificate": json.loads(row[4]),
            "status": row[5],
            "created_at": row[6],
        }

    def list_merges(self, status: str | None = None) -> list[dict]:
        sql = (
            "SELECT id, merged_ids, merged_record_id, merged_text, "
            "certificate, status, created_at FROM merges"
        )
        params: list = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "id": r[0],
                "merged_ids": json.loads(r[1]),
                "merged_record_id": r[2],
                "merged_text": r[3],
                "certificate": json.loads(r[4]),
                "status": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]

    def set_merge_status(self, merge_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE merges SET status = ? WHERE id = ?", (status, merge_id)
        )
        self.conn.commit()

    # -- P18: episodes (context) and evictions (audit) ----------------
    def add_episode(
        self,
        session_id: str,
        record_id: int,
        type_: str,
        tokens: int,
        depends_on: list[int] | None = None,
        persisted_by: int | None = None,
    ) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) FROM episodes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seq = int(row[0]) + 1
        cur = self.conn.execute(
            "INSERT INTO episodes(session_id, seq, record_id, type, "
            "tokens, status, persisted_by, depends_on, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'live', ?, ?, ?)",
            (
                session_id,
                seq,
                record_id,
                type_,
                tokens,
                persisted_by,
                json.dumps(depends_on or []),
                time.time(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def _row_to_episode(self, row) -> dict:
        return {
            "id": row[0],
            "session_id": row[1],
            "seq": row[2],
            "record_id": row[3],
            "type": row[4],
            "tokens": row[5],
            "status": row[6],
            "persisted_by": row[7],
            "depends_on": json.loads(row[8]),
            "compressed_text": row[9],
            "compressed_tokens": row[10],
            "created_at": row[11],
        }

    def get_episode(self, episode_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id, session_id, seq, record_id, type, tokens, "
            "status, persisted_by, depends_on, compressed_text, compressed_tokens, created_at FROM episodes "
            "WHERE id = ?",
            (episode_id,),
        ).fetchone()
        return self._row_to_episode(row) if row is not None else None

    def list_episodes(
        self, session_id: str, status: str | None = None
    ) -> list[dict]:
        sql = (
            "SELECT id, session_id, seq, record_id, type, tokens, "
            "status, persisted_by, depends_on, compressed_text, compressed_tokens, created_at FROM episodes "
            "WHERE session_id = ?"
        )
        params: list = [session_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY seq"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def session_token_total(self, session_id: str) -> int:
        """Tokens of the session's live episodes — its context size."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM episodes "
            "WHERE session_id = ? AND status = 'live'",
            (session_id,),
        ).fetchone()
        return int(row[0])

    def set_episode_status(self, episode_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE episodes SET status = ? WHERE id = ?",
            (status, episode_id),
        )
        self.conn.commit()

    def set_episode_persisted(
        self, episode_id: int, record_id: int
    ) -> None:
        self.conn.execute(
            "UPDATE episodes SET persisted_by = ? WHERE id = ?",
            (record_id, episode_id),
        )
        self.conn.commit()

    # -- P19: compression (context only — records are never rewritten) --
    def set_episode_compressed(
        self,
        episode_id: int,
        compressed_text: str,
        compressed_tokens: int,
    ) -> None:
        """Store a compressed rendering on the episode and shrink
        its token count, so the session total reflects what the
        agent actually sees. The linked record keeps the original
        text, retrievable on demand."""
        self.conn.execute(
            "UPDATE episodes SET compressed_text = ?, "
            "compressed_tokens = ?, tokens = ? WHERE id = ?",
            (
                compressed_text,
                compressed_tokens,
                compressed_tokens,
                episode_id,
            ),
        )
        self.conn.commit()

    def add_compression(
        self,
        episode_id: int,
        session_id: str,
        type_: str,
        tokens_before: int,
        tokens_after: int,
        units_before: int,
        units_after: int,
        latency_ms: float,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO compressions(episode_id, session_id, type, "
            "tokens_before, tokens_after, units_before, units_after, "
            "latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode_id,
                session_id,
                type_,
                tokens_before,
                tokens_after,
                units_before,
                units_after,
                latency_ms,
                time.time(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_compressions(
        self, session_id: str | None = None
    ) -> list[dict]:
        sql = (
            "SELECT id, episode_id, session_id, type, tokens_before, "
            "tokens_after, units_before, units_after, latency_ms, "
            "created_at FROM compressions"
        )
        params: list = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY id"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "id": r[0],
                "episode_id": r[1],
                "session_id": r[2],
                "type": r[3],
                "tokens_before": r[4],
                "tokens_after": r[5],
                "units_before": r[6],
                "units_after": r[7],
                "latency_ms": r[8],
                "created_at": r[9],
            }
            for r in rows
        ]

    def episode_context_text(self, episode_id: int) -> str | None:
        """The text the agent sees for an episode: its compressed
        rendering when one exists, else the linked record's text."""
        ep = self.get_episode(episode_id)
        if ep is None:
            return None
        if ep.get("compressed_text"):
            return ep["compressed_text"]
        rec = self.get_record(ep["record_id"])
        return rec["text"] if rec else None

    def episodes_depending_on(
        self, episode_id: int, session_id: str | None = None
    ) -> list[dict]:
        """Live episodes whose depends_on lists this episode."""
        eps = self.list_episodes(session_id, status="live") if (
            session_id is not None
        ) else [
            self._row_to_episode(r)
            for r in self.conn.execute(
                "SELECT id, session_id, seq, record_id, type, tokens, "
                "status, persisted_by, depends_on, compressed_text, "
                "compressed_tokens, created_at "
                "FROM episodes WHERE status = 'live' ORDER BY seq"
            ).fetchall()
        ]
        return [e for e in eps if episode_id in e["depends_on"]]

    def add_eviction(
        self,
        episode_id: int,
        session_id: str,
        reason: str,
        tier: int,
        tokens_freed: int,
        budget_tokens: int,
        total_tokens_before: int,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO evictions(episode_id, session_id, reason, "
            "tier, tokens_freed, budget_tokens, total_tokens_before, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode_id,
                session_id,
                reason,
                tier,
                tokens_freed,
                budget_tokens,
                total_tokens_before,
                time.time(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_evictions(
        self, session_id: str | None = None
    ) -> list[dict]:
        sql = (
            "SELECT id, episode_id, session_id, reason, tier, "
            "tokens_freed, budget_tokens, total_tokens_before, "
            "created_at FROM evictions"
        )
        params: list = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY id"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "id": r[0],
                "episode_id": r[1],
                "session_id": r[2],
                "reason": r[3],
                "tier": r[4],
                "tokens_freed": r[5],
                "budget_tokens": r[6],
                "total_tokens_before": r[7],
                "created_at": r[8],
            }
            for r in rows
        ]

    def context_record_ids(self, session_id: str) -> list[int]:
        """Record ids the agent can use right now: live episodes'
        records (the context) plus live non-episode records (the
        consolidated memory). Evicted episodes' records are out.
        """
        live_ep = self.conn.execute(
            "SELECT record_id FROM episodes WHERE session_id = ? "
            "AND status = 'live'",
            (session_id,),
        ).fetchall()
        ids = {int(r[0]) for r in live_ep}
        mem = self.conn.execute(
            "SELECT id FROM records WHERE status = 'live' "
            "AND type != 'episode'"
        ).fetchall()
        ids |= {int(r[0]) for r in mem}
        return sorted(ids)

    # -- reads ---------------------------------------------------------
    def doc_freq(self, term: str) -> int:
        """How many records contain ``term`` (FTS5, exact term).

        P3 planner signal: a query whose content terms mostly appear
        in the corpus is lexical (BM25 can answer); one whose terms
        mostly don't is a paraphrase (only the vector leg can).
        """
        row = self.conn.execute(
            "SELECT COUNT(*) FROM records_fts WHERE records_fts MATCH ?",
            ('"%s"' % term.lower(),),
        ).fetchone()
        return int(row[0])

    def recent_live_ids(
        self, days: float = 7.0, limit: int = 200
    ) -> list[int]:
        """Live record ids created in the last ``days`` days, newest first."""
        rows = self.conn.execute(
            "SELECT id FROM records WHERE status = 'live' "
            "AND created_at >= ? ORDER BY created_at DESC LIMIT ?",
            (time.time() - days * 86400.0, limit),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def live_ids_by_type(self, type_: str) -> list[int]:
        """Live record ids of one type (P3 planner's typed route)."""
        rows = self.conn.execute(
            "SELECT id FROM records WHERE status = 'live' AND type = ?",
            (type_,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def count(self) -> int:
        return int(
            self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        )

    def get_record(self, rid: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id, text, type, tags, created_at, meta, status, "
            "superseded_by, importance, credit, uses, last_used "
            "FROM records WHERE id = ?",
            (rid,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def fetch(self, rids: list[int]) -> list[dict]:
        return [r for r in (self.get_record(i) for i in rids) if r is not None]

    def bm25_search(
        self, query: str, k: int, candidate_ids: list[int] | None = None
    ) -> list[tuple[int, float]]:
        """FTS5 BM25. Returns [(record_id, bm25_score)] ascending (lower is better).

        Scores are returned as-is (negative = more relevant in FTS5's
        bm25(); callers that need a similarity should negate them).
        """
        fts_q = _fts_query(query)
        if fts_q is None:
            return []
        non_live = self.non_live_ids()
        # Fetch extra rows when some must be filtered out in Python,
        # so the caller still gets up to k live hits.
        limit = k * 5 if non_live else k
        sql = (
            "SELECT rowid, bm25(records_fts) FROM records_fts "
            "WHERE records_fts MATCH ?"
        )
        params: list = [fts_q]
        if candidate_ids:
            sql += " AND rowid IN (%s)" % ",".join(
                "?" * len(candidate_ids)
            )
            params.extend(candidate_ids)
        sql += " ORDER BY bm25(records_fts) LIMIT ?"
        params.append(limit)
        rows = [
            (int(r), float(s))
            for r, s in self.conn.execute(sql, params).fetchall()
        ]
        if non_live:
            rows = [x for x in rows if x[0] not in non_live]
        return rows[:k]

    def vector_search(
        self,
        query_vec: np.ndarray,
        k: int,
        candidate_ids: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Cosine similarity, descending. Returns [(record_id, cosine)]."""
        query_vec = np.asarray(query_vec, dtype=np.float32)
        non_live = self.non_live_ids()
        if self.vec_mode == "sqlite-vec":
            import sqlite_vec

            fetch_k = k * 5 if (candidate_ids or non_live) else k
            sql = (
                "SELECT rowid, vec_distance_cosine(embedding, ?) AS d "
                "FROM record_vecs ORDER BY d LIMIT ?"
            )
            rows = self.conn.execute(
                sql, [sqlite_vec.serialize_float32(query_vec), fetch_k]
            ).fetchall()
            # d is NULL when the query vector has norm 0 (0/0 in cosine);
            # sim <= 0 means no similarity at all, so drop those rows.
            out = [
                (int(r), 1.0 - float(d))
                for r, d in rows
                if d is not None and float(d) < 1.0
            ]
            if non_live:
                out = [x for x in out if x[0] not in non_live]
            if candidate_ids:
                allowed = set(candidate_ids)
                out = [x for x in out if x[0] in allowed][:k]
            return out[:k]
        # NumPy exact path.
        rows = self.conn.execute(
            "SELECT record_id, vec FROM embeddings"
        ).fetchall()
        if not rows:
            return []
        ids = [int(r) for r, _ in rows]
        mat = np.stack(
            [np.frombuffer(v, dtype=np.float32) for _, v in rows]
        )
        if non_live:
            keep = [i for i, rid in enumerate(ids) if rid not in non_live]
            if not keep:
                return []
            ids = [ids[i] for i in keep]
            mat = mat[keep]
        if candidate_ids:
            allowed = set(candidate_ids)
            keep = [i for i, rid in enumerate(ids) if rid in allowed]
            if not keep:
                return []
            ids = [ids[i] for i in keep]
            mat = mat[keep]
        sims = mat @ query_vec
        # Cosine <= 0 means no similarity at all; drop those rows so the
        # vector leg never returns the whole store on a no-overlap query.
        order = np.argsort(-sims)
        out = [
            (ids[i], float(sims[i])) for i in order if sims[i] > 0
        ][:k]
        return out

    def get_vector(self, rid: int) -> np.ndarray | None:
        if self.vec_mode == "sqlite-vec":
            # SELECT on a vec0 column returns the raw float32 blob.
            row = self.conn.execute(
                "SELECT embedding FROM record_vecs WHERE rowid = ?", (rid,)
            ).fetchone()
            if row is None:
                return None
            return np.frombuffer(row[0], dtype=np.float32).copy()
        row = self.conn.execute(
            "SELECT vec FROM embeddings WHERE record_id = ?", (rid,)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def neighbours(
        self, seed_ids: list[int], k: int
    ) -> list[tuple[int, float]]:
        """1-hop graph neighbours of the seeds, by edge weight desc.

        Seeds themselves are excluded; duplicates collapse to the max
        weight across incoming edges.
        """
        if not seed_ids:
            return []
        non_live = self.non_live_ids()
        placeholders = ",".join("?" * len(seed_ids))
        rows = self.conn.execute(
            "SELECT dst, MAX(weight) FROM edges WHERE src IN (%s) "
            "AND dst NOT IN (%s) GROUP BY dst ORDER BY 2 DESC LIMIT ?"
            % (placeholders, placeholders),
            (*seed_ids, *seed_ids, k * 5 if non_live else k),
        ).fetchall()
        out = [(int(r), float(w)) for r, w in rows]
        if non_live:
            out = [x for x in out if x[0] not in non_live]
        return out[:k]

    def close(self) -> None:
        self.conn.close()
