"""Audit the judge-gated merge: 100 proposals, zero-loss check.

Builds a store from the synthetic corpus (records are single
sentences, so multi-sentence sources are composed from each
(person, attribute) group's 3 records):

- 96 positive proposals: for each of the 48 groups, two variants
  (A="S1 S2" + B="S2 S3", and A'="S1 S2 S3" + B'="S1 S3"). The
  sentence-union draft must compress (length <= 0.90), keep every
  source sentence (faithfulness 1.0, coverage 1.0), and commit.
  A committed merge is audited sentence by sentence: any source
  sentence missing from the merged text is a LOSS.
- 4 negative proposals that must be rejected: unrelated sources
  (length), invented content (faithfulness), mixed types, single
  source.

Exit code 0 only if losses == 0 and every proposal behaved as
expected. Prints which embedder was used; default is the ONNX
MiniLM (the production path), --embedder hashed runs the lexical
fallback instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hybrid_retriever.embeddings import HashedEmbedder, get_embedder
from hybrid_retriever.lifecycle import (
    draft_merge_text,
    propose_merge,
    split_sentences,
)
from hybrid_retriever.store import Store


def group_key(key: str) -> str:
    return "-".join(key.split("-")[:-1])


def norm(sentence: str) -> str:
    return " ".join(sentence.strip().lower().rstrip(".!?").split())


def audit_committed(store, result, sources):
    """Return the list of source sentences lost by this merge."""
    merged_sentences = {norm(s) for s in split_sentences(result.merged_text)}
    lost = []
    for text in sources:
        for sentence in split_sentences(text):
            if norm(sentence) not in merged_sentences:
                lost.append(sentence)
    return lost


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", default="onnx", choices=["onnx", "hashed"])
    args = p.parse_args()
    embedder = HashedEmbedder() if args.embedder == "hashed" else get_embedder()
    print("embedder: %s" % embedder.name)

    corpus_path = os.path.join(
        os.path.dirname(__file__), "..", "eval_data", "synthetic_corpus.jsonl"
    )
    with open(corpus_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    groups: dict[str, list[str]] = {}
    for r in records:
        if r["key"].startswith("trap-"):
            continue
        groups.setdefault(group_key(r["key"]), []).append(r["text"])
    groups = {k: v for k, v in groups.items() if len(v) == 3}
    print("audit groups: %d" % len(groups))

    tmp = tempfile.mkdtemp(prefix="hr-audit-")
    store = Store(os.path.join(tmp, "audit.db"), embedder=embedder)

    proposed = committed = rejected = 0
    losses: list[tuple[int, str]] = []
    unexpected: list[tuple[int, str]] = []

    def run_case(source_texts, expect, merged_text=None, types=None):
        nonlocal proposed, committed, rejected
        ids = []
        for i, text in enumerate(source_texts):
            t = types[i] if types else "fact"
            ids.append(store.add_record(text, type_=t))
        proposed += 1
        res = propose_merge(store, embedder, ids, merged_text=merged_text)
        if expect == "commit":
            if res.status != "committed":
                unexpected.append(
                    (proposed, "expected commit, got %s (%s)"
                     % (res.status, res.certificate.get("reasons")))
                )
                rejected += 1
                return
            committed += 1
            for sentence in audit_committed(store, res, source_texts):
                losses.append((proposed, sentence))
        else:
            if res.status != "rejected":
                unexpected.append(
                    (proposed, "expected reject, got %s" % res.status)
                )
                committed += 1
                return
            rejected += 1
            reasons = res.certificate.get("reasons", [])
            if expect not in reasons and expect not in str(reasons):
                unexpected.append(
                    (proposed, "expected reason %r, got %r"
                     % (expect, reasons))
                )

    for s1, s2, s3 in groups.values():
        run_case([s1 + " " + s2, s2 + " " + s3], "commit")
        run_case([s1 + " " + s2 + " " + s3, s1 + " " + s3], "commit")

    # Negatives.
    g0 = list(groups.values())[0]
    g1 = list(groups.values())[1]
    run_case([g0[0], g1[0]], "length_ratio")
    run_case(
        [g0[0], g0[1]],
        "faithfulness",
        merged_text=g0[0] + " And the moon is made of green cheese.",
    )
    run_case([g0[0], g0[1]], "mixed record types", types=["fact", "episode"])
    run_case([g0[0]], "fewer than 2 live sources")

    store.close()
    print(
        "proposed=%d committed=%d rejected=%d losses=%d unexpected=%d"
        % (proposed, committed, rejected, len(losses), len(unexpected))
    )
    for n, sentence in losses[:10]:
        print("  LOSS in case %d: %s" % (n, sentence))
    for n, why in unexpected[:10]:
        print("  UNEXPECTED in case %d: %s" % (n, why))
    return 0 if not losses and not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())
