"""P19 — type-aware tool-output compressor.

Headroom-style compression, honest edition: compress by content
type (JSON schema/key pruning + value summarisation, tables
row/column pruning, logs dedup + tail, text extractive keep,
Python AST pruning), deterministically, with no LLM and no model.

Design rules
------------
* Extractive only. The compressed text is built from pieces of
  the original plus count markers (``... N more rows ...`` and the
  like). Nothing is invented, so faithfulness is checkable by
  construction — ``verify_no_invention`` proves it for any output.
* The original is never touched. Compression lives on the
  *episode* (the context), never on the *record* (the memory):
  ``Store.set_episode_compressed`` stores the compressed text
  beside the episode, and the record text stays retrievable on
  demand via ``episode_context_text``'s ``original`` mode.
* Turn text carries a ``"role: "`` prefix (see
  ``segments.turn_text``); detection and compression run on the
  body, the prefix is re-attached to the output.
* A compression only counts when it saves real tokens: at least
  10 tokens and at least 10% of the original. Anything smaller
  is returned unchanged, so the eviction loop never pays for
  noise.

Faithfulness
------------
Every compressor reports ``units_before`` / ``units_after`` —
the countable facts of its type (JSON leaf paths, table cells,
distinct log lines, code lines, sentences) — and
``faithfulness = units_after / units_before``. A truncated
string value still counts as retained (its head survives and
the marker says so); a collapsed list item does not.
"""

from __future__ import annotations

import ast
import io
import json
import math
import re
import tokenize

from .eviction import estimate_tokens

# A compression must save at least this many tokens and this
# fraction of the original to count; anything smaller is noise.
MIN_SAVED_TOKENS = 10
MIN_SAVED_RATIO = 0.10

# JSON lists of homogeneous dicts collapse to this many kept
# items (head) + this many (tail); the middle becomes one
# marker line naming the shared keys.
JSON_KEEP_HEAD = 2
JSON_KEEP_TAIL = 1
# Long string values truncate to this many characters (+ marker).
JSON_TRUNCATE_CHARS = 120
CODE_TRUNCATE_CHARS = 80
# Tables keep this many rows (head) + this many (tail).
TABLE_KEEP_HEAD = 5
TABLE_KEEP_TAIL = 2
# Logs keep this many lines (head) + this many (tail) when every
# line is distinct and nothing folds.
LOG_KEEP_HEAD = 10
LOG_KEEP_TAIL = 5
# Text keeps this many sentences (head) + this many (tail).
TEXT_KEEP_HEAD = 3
TEXT_KEEP_TAIL = 2

_PREFIX_RE = re.compile(r"^[a-z_]+: ")
_LOG_RE = re.compile(
    r"^\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}:\d{2}(:\d{2})?"
    r"|\[?(INFO|DEBUG|WARN|WARNING|ERROR|CRITICAL|TRACE)\b)"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_prefix(text: str) -> tuple[str, str]:
    """Split a leading ``role: `` prefix off a turn's text."""
    m = _PREFIX_RE.match(text)
    if m and len(text) > m.end():
        return text[: m.end()], text[m.end():]
    return "", text


# ----------------------------------------------------------------------
# Type detection
# ----------------------------------------------------------------------
def _is_table(lines: list[str]) -> bool:
    """>=3 rows with a consistent column count via | or ,."""
    if len(lines) < 3:
        return False
    for sep in ("|", ","):
        counts = []
        for line in lines:
            parts = [p for p in line.split(sep)]
            # A pipe table's outer pipes make empty edge cells.
            if sep == "|":
                parts = [p for p in parts if p.strip() != ""]
            counts.append(len(parts))
        if len(set(counts)) == 1 and counts[0] >= 2:
            return True
    return False


def _log_ratio(lines: list[str]) -> float:
    if not lines:
        return 0.0
    hits = sum(1 for line in lines if _LOG_RE.match(line))
    return hits / len(lines)


def _looks_like_code(body: str) -> bool:
    """Parses as Python and contains a def/class/import."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
    return False


def detect_type(text: str) -> str:
    """Classify tool output: json | table | log | code | text."""
    _, body = _split_prefix(text)
    s = body.strip()
    if s[:1] in ("{", "["):
        try:
            json.loads(s)
            return "json"
        except ValueError:
            pass
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if _is_table(lines):
        return "table"
    if len(lines) >= 3 and _log_ratio(lines) >= 0.6:
        return "log"
    if _looks_like_code(body):
        return "code"
    return "text"


# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------
def _json_leaves(value, path: str = "") -> list[str]:
    """Leaf paths of a JSON value, as ``a.b[0].c`` strings."""
    leaves: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            leaves.extend(_json_leaves(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            leaves.extend(_json_leaves(v, f"{path}[{i}]"))
    else:
        leaves.append(path)
    return leaves


def _truncate_str(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return "%s... (%d chars)" % (value[:limit], len(value))


def _collapse_json(value):
    """Recursively collapse a JSON value.

    * dicts: drop null/empty values; truncate long strings.
    * lists of homogeneous dicts (same key set, >=4 items):
      keep head + tail, middle becomes one marker naming the
      shared keys and the count.
    * lists: dedupe identical items (keep first + marker).
    * long strings truncate with a marker.
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if v is None or v == "" or v == [] or v == {}:
                continue
            out[k] = _collapse_json(v)
        return out
    if isinstance(value, list):
        items = [_collapse_json(v) for v in value]
        # Dedupe identical items, keep first occurrence.
        seen: list = []
        dups = 0
        for item in items:
            if item in seen:
                dups += 1
            else:
                seen.append(item)
        items = seen
        if (
            len(items) >= 4
            and all(isinstance(i, dict) for i in items)
            and len({tuple(sorted(i.keys())) for i in items}) == 1
        ):
            keys = sorted(items[0].keys())
            kept = items[:JSON_KEEP_HEAD] + items[-JSON_KEEP_TAIL:]
            kept.append(
                {
                    "...": "%d more items with the same keys: %s"
                    % (len(items) - len(kept), ", ".join(keys))
                }
            )
            items = kept
        if dups:
            items.append({"...": "%d duplicate items removed" % dups})
        return items
    if isinstance(value, str):
        return _truncate_str(value, JSON_TRUNCATE_CHARS)
    return value


def _json_retained(original, collapsed) -> int:
    """Leaf paths whose value survives verbatim or as a marked
    prefix in the collapsed value."""
    def _leaves_with_values(value, path: str = ""):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from _leaves_with_values(
                    v, f"{path}.{k}" if path else str(k)
                )
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield from _leaves_with_values(v, f"{path}[{i}]")
        else:
            yield path, value

    collapsed_text = json.dumps(collapsed, sort_keys=True)
    retained = 0
    for path, value in _leaves_with_values(original):
        if isinstance(value, str):
            head = value[:JSON_TRUNCATE_CHARS]
            if head in collapsed_text:
                retained += 1
        else:
            if json.dumps(value, sort_keys=True) in collapsed_text:
                retained += 1
    return retained


def _compress_json(body: str) -> tuple[str, int, int]:
    original = json.loads(body)
    collapsed = _collapse_json(original)
    text = json.dumps(collapsed, indent=1, sort_keys=True)
    before = len(_json_leaves(original))
    after = _json_retained(original, collapsed)
    return text, before, after


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def _parse_table(lines: list[str]) -> tuple[str, list[list[str]]]:
    """Detect the separator and split rows into cells."""
    for sep in ("|", ","):
        rows = []
        for line in lines:
            parts = line.split(sep)
            if sep == "|":
                parts = [p for p in parts if p.strip() != ""]
            rows.append([p.strip() for p in parts])
        if len({len(r) for r in rows}) == 1 and len(rows[0]) >= 2:
            return sep, rows
    return "|", [line.split("|") for line in lines]


def _compress_table(body: str) -> tuple[str, int, int]:
    lines = [ln for ln in body.splitlines() if ln.strip()]
    sep, rows = _parse_table(lines)
    # Drop a Markdown separator row (|---|---|) if present.
    rows = [
        r
        for r in rows
        if not all(re.match(r"^:?-+:?$", c) for c in r)
    ]
    header = rows[0]
    data = rows[1:]
    # Drop columns that are entirely empty (header included).
    keep_cols = [
        j
        for j in range(len(header))
        if any(r[j] for r in rows)
    ]
    rows = [[r[j] for j in keep_cols] for r in rows]
    header = rows[0]
    data = rows[1:]
    # Dedupe identical rows, keep first occurrence.
    seen: set[tuple[str, ...]] = set()
    uniq: list[list[str]] = []
    dups = 0
    for row in data:
        key = tuple(row)
        if key in seen:
            dups += 1
        else:
            seen.add(key)
            uniq.append(row)
    data = uniq
    total_rows = len(data)
    if total_rows > TABLE_KEEP_HEAD + TABLE_KEEP_TAIL:
        kept = data[:TABLE_KEEP_HEAD] + data[-TABLE_KEEP_TAIL:]
        omitted = total_rows - len(kept)
    else:
        kept = data
        omitted = 0
    out_rows = [header] + kept
    if omitted:
        out_rows.append(
            ["... %d more rows ..." % omitted]
            + [""] * (len(header) - 1)
        )
    if dups:
        out_rows.append(
            ["... %d duplicate rows removed ..." % dups]
            + [""] * (len(header) - 1)
        )
    if sep == "|":
        text = "\n".join(
            "| " + " | ".join(r) + " |" for r in out_rows
        )
    else:
        text = "\n".join(",".join(r) for r in out_rows)
    before = sum(len(r) for r in rows)
    after = sum(len(r) for r in out_rows if not r[0].startswith("..."))
    # Header cells survive by construction; marker rows carry no
    # original cells.
    after = min(after, before)
    return text, before, after


# ----------------------------------------------------------------------
# Logs
# ----------------------------------------------------------------------
def _compress_log(body: str) -> tuple[str, int, int]:
    lines = [ln for ln in body.splitlines() if ln.strip()]
    # Fold repeats: keep the first occurrence, count the rest.
    counts: dict[str, int] = {}
    order: list[str] = []
    for line in lines:
        if line not in counts:
            counts[line] = 0
            order.append(line)
        counts[line] += 1
    folded = [
        "%s (x%d)" % (line, counts[line]) if counts[line] > 1 else line
        for line in order
    ]
    if len(folded) > LOG_KEEP_HEAD + LOG_KEEP_TAIL:
        kept = folded[:LOG_KEEP_HEAD] + folded[-LOG_KEEP_TAIL:]
        omitted = len(folded) - len(kept)
        kept.append("... %d more lines ..." % omitted)
        folded = kept
    before = len(order)
    after = len([ln for ln in folded if not ln.startswith("...")])
    return "\n".join(folded), before, after


# ----------------------------------------------------------------------
# Code (Python AST: strip docstrings/comments, truncate literals)
# ----------------------------------------------------------------------
def _docstring_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """(start, end) line ranges of module/class/function docstrings."""
    spans = []

    def _check(node):
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            spans.append((first.lineno, first.end_lineno or first.lineno))

    _check(tree)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            _check(node)
    return spans


def _compress_code(body: str) -> tuple[str, int, int]:
    tree = ast.parse(body)
    doc_lines: set[int] = set()
    for start, end in _docstring_spans(tree):
        doc_lines.update(range(start, end + 1))
    lines = body.splitlines()
    try:
        tokens = list(
            tokenize.generate_tokens(io.StringIO(body).readline)
        )
    except (tokenize.TokenError, SyntaxError, IndentationError):
        tokens = []
    for tok in tokens:
        row = tok.start[0] - 1
        if row < 0 or row >= len(lines):
            continue
        if tok.type == tokenize.COMMENT:
            # Strip from the comment column only; code before
            # the comment on the same line survives.
            lines[row] = lines[row][: tok.start[1]].rstrip()
        elif tok.type == tokenize.STRING:
            raw = tok.string
            # Only truncate single-line literals; multi-line
            # literals (including docstrings) are blanked below.
            if (
                tok.start[0] == tok.end[0]
                and len(raw) > CODE_TRUNCATE_CHARS + 2
            ):
                quote = raw[0]
                triple = raw[:3] in ('"""', "'''")
                q = raw[:3] if triple else quote
                inner = (
                    raw[len(q): -len(q)]
                    if len(raw) > 2 * len(q)
                    else ""
                )
                truncated = "%s%s... (%d chars)%s" % (
                    q,
                    inner[:CODE_TRUNCATE_CHARS],
                    len(inner),
                    q,
                )
                lines[row] = lines[row].replace(raw, truncated, 1)
    for ln in doc_lines:
        if 1 <= ln <= len(lines):
            lines[ln - 1] = ""
    # Units are the code lines: non-blank, non-comment-only,
    # excluding docstring lines. Truncation keeps the line, so
    # code faithfulness is 1.0 by construction — the compressor
    # removes only docstrings, comments and blank lines.
    cleaned: list[str] = []
    units = 0
    for i, line in enumerate(lines, start=1):
        if i in doc_lines:
            continue
        if not line.strip() or line.strip().startswith("#"):
            continue
        units += 1
        cleaned.append(line)
    return "\n".join(cleaned), units, units


# ----------------------------------------------------------------------
# Text (extractive keep)
# ----------------------------------------------------------------------
def _sentences(body: str) -> list[str]:
    parts = [
        p.strip()
        for p in _SENTENCE_RE.split(body)
        if p.strip()
    ]
    return parts


def _compress_text(body: str) -> tuple[str, int, int]:
    sents = _sentences(body)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in sents:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    if len(uniq) > TEXT_KEEP_HEAD + TEXT_KEEP_TAIL:
        kept = uniq[:TEXT_KEEP_HEAD] + uniq[-TEXT_KEEP_TAIL:]
        omitted = len(uniq) - len(kept)
        text = " ".join(kept) + " [... %d sentences omitted ...]" % omitted
    else:
        kept = uniq
        text = " ".join(kept)
    return text, len(uniq), len(kept)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
class CompressionResult:
    """One compression: what was done and what it saved."""

    def __init__(
        self,
        type_: str,
        text: str,
        tokens_before: int,
        tokens_after: int,
        units_before: int,
        units_after: int,
    ):
        self.type = type_
        self.text = text
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.units_before = units_before
        self.units_after = units_after

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def faithfulness(self) -> float:
        if self.units_before == 0:
            return 1.0
        return self.units_after / self.units_before

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "text": self.text,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "units_before": self.units_before,
            "units_after": self.units_after,
            "faithfulness": self.faithfulness,
        }


def compress_text(text: str) -> CompressionResult:
    """Compress one tool output by its detected type.

    Returns the original unchanged (type ``"text"``) when
    compression would save less than ``MIN_SAVED_TOKENS`` and
    ``MIN_SAVED_RATIO`` — the eviction loop must never pay for
    noise.
    """
    prefix, body = _split_prefix(text)
    tokens_before = estimate_tokens(text)
    type_ = detect_type(text)
    if type_ == "json":
        out, before, after = _compress_json(body)
    elif type_ == "table":
        out, before, after = _compress_table(body)
    elif type_ == "log":
        out, before, after = _compress_log(body)
    elif type_ == "code":
        out, before, after = _compress_code(body)
    else:
        out, before, after = _compress_text(body)
    candidate = prefix + out
    tokens_after = estimate_tokens(candidate)
    saved = tokens_before - tokens_after
    if (
        saved < MIN_SAVED_TOKENS
        or saved < MIN_SAVED_RATIO * tokens_before
    ):
        return CompressionResult(
            "text", text, tokens_before, tokens_before, 0, 0
        )
    return CompressionResult(
        type_, candidate, tokens_before, tokens_after, before, after
    )


def verify_no_invention(original: str, compressed: str) -> list[str]:
    """Lines in ``compressed`` that are neither verbatim from the
    original nor a ``...`` count marker.

    Extractive-only compression must return an empty list.
    Comparison is whitespace-normalised, so a re-serialised JSON
    value (``"count": 42`` vs ``"count":42``) still matches its
    original. Marker lines contain ``...`` plus a count word
    (more / removed / omitted / chars / lines / duplicate).
    """
    _, orig_body = _split_prefix(original)
    _, comp_body = _split_prefix(compressed)

    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s).rstrip(",")

    orig_lines = {_norm(ln) for ln in orig_body.splitlines()}
    orig_norm = _norm(orig_body)
    bad: list[str] = []
    for line in comp_body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "..." in stripped and any(
            w in stripped
            for w in (
                "more",
                "removed",
                "omitted",
                "chars",
                "lines",
                "duplicate",
            )
        ):
            continue
        # A folded log line: "line (xN)" — the line itself is
        # original, the (xN) is the fold count.
        folded = re.sub(r"\s*\(x\d+\)$", "", stripped)
        if _norm(folded) in orig_lines or _norm(folded) in orig_norm:
            continue
        if _norm(stripped) in orig_lines or _norm(stripped) in orig_norm:
            continue
        # A truncated value: its head survives in the original.
        head = stripped.split("...")[0].strip().strip('",')
        if head and _norm(head) in orig_norm:
            continue
        bad.append(line)
    return bad


def context_text(
    original: str, compressed: str | None
) -> str:
    """The text the agent actually sees for an episode: the
    compressed version when one exists, else the original."""
    return compressed if compressed else original
