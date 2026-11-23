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
                 meta TEXT NOT NULL DEFAULT '{}')"""
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
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO records(text, type, tags, created_at, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                text,
                type_,
                json.dumps(tags or []),
                time.time(),
                json.dumps(meta or {}),
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

    # -- reads ---------------------------------------------------------
    def count(self) -> int:
        return int(
            self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        )

    def get_record(self, rid: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id, text, type, tags, created_at, meta FROM records "
            "WHERE id = ?",
            (rid,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "text": row[1],
            "type": row[2],
            "tags": json.loads(row[3]),
            "created_at": row[4],
            "meta": json.loads(row[5]),
        }

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
        params.append(k)
        return [
            (int(r), float(s))
            for r, s in self.conn.execute(sql, params).fetchall()
        ]

    def vector_search(
        self,
        query_vec: np.ndarray,
        k: int,
        candidate_ids: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Cosine similarity, descending. Returns [(record_id, cosine)]."""
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if self.vec_mode == "sqlite-vec":
            import sqlite_vec

            fetch_k = k * 5 if candidate_ids else k
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
        placeholders = ",".join("?" * len(seed_ids))
        rows = self.conn.execute(
            "SELECT dst, MAX(weight) FROM edges WHERE src IN (%s) "
            "AND dst NOT IN (%s) GROUP BY dst ORDER BY 2 DESC LIMIT ?"
            % (placeholders, placeholders),
            (*seed_ids, *seed_ids, k),
        ).fetchall()
        return [(int(r), float(w)) for r, w in rows]

    def close(self) -> None:
        self.conn.close()
