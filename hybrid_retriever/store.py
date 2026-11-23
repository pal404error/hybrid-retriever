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
