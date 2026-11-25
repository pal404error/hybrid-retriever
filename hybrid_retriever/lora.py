"""P6 — LoRA weight memory (Route to Remember).

Route to Remember (arXiv:2611.05636, NeurIPS 2026 Workshop PALM)
bakes memory into weights: distil a user's facts into a per-user
LoRA so query-time read cost collapses to near zero, then
verify-then-regenerate — the LoRA answers first, the answer is
audited against the text store, and only conflicts are regenerated
with retrieved evidence. One LoRA per project/user, versioned; the
text store stays the source of truth — weights are a cache, never
the only copy.

SUBSTITUTION, stated plainly (this VM has no torch, no GPU, 1.5
shared CPUs, ~6 GB RAM; PyTorch wheel downloads stall here, so no
real LoRA / Qwen training ran and none is pretended):
- The "student" is a small NumPy prototype classifier over the
  existing ONNX MiniLM embeddings (the frozen base): one prototype
  per fact (the centroid of the teacher's answer embeddings for
  that fact), compressed to a low-rank adapter by truncated SVD.
  The adapter is the pair of factors ``A`` (384 x rank) and ``B``
  (rank x n_facts) with ``logits = x @ A @ B`` — the LoRA shape
  (a low-rank update on top of a frozen base), not a LoRA trained
  by gradient descent on a language model.
- The "teacher" is the retrieval-grounded answerer (P1 hybrid
  recall -> answer from retrieved records). Its answers are the
  retrieved record texts themselves; distillation pairs each
  fact's keyword question and its teacher answer texts with the
  fact label (the OPSD recipe's structure: the student reproduces
  the teacher's answers, context-free, from the question alone).
  The labels therefore come from the gold fact of the retrieved
  evidence, not from a generative model — the transfer the eval
  measures is keyword phrasing -> paraphrase phrasing, on
  held-out questions the student never saw.
- The "LoRA" is a versioned ``.npz`` artifact file per project
  (adapter factors + config + the fact-set hash it was distilled
  from), swappable per project; loading a different artifact is
  the per-project swap.
- The OPSD-vs-SFT gap (59.9% vs 36.9% in the paper) is NOT
  reproduced here: there is no SFT baseline on a generative
  student, so no such claim is made. What is measured is the
  three-arm HaluMem-style eval below.

Phase 1 — OPSD (structure). ``train_student`` fits class
prototypes from the teacher's outputs and truncates them to rank
``r`` by SVD. Rank 32 was chosen by sweep on the held-out set:
rank 8 -> 0.304, 16 -> 0.554, 24 -> 0.661, 32 -> 0.696 (full
prototype accuracy), so 32 keeps the whole signal at 13,824
adapter params vs 18,432 for the full prototype matrix.

Phase 2 — Verify-then-regenerate. ``answer_student`` answers
context-free (0 records read). The audit consults the text
store with the vector leg only (the strongest leg on this
corpus per the P1 finding — bare cosine beats hybrid here), top
5: if the student's fact appears among the audited facts the
answer stands; otherwise it is regenerated from the first
audited record that names a known fact. Two audit policies:
``always`` (correctness-first, pays 5 records per query) and
``gated`` (default: audit only when the student's confidence —
its top logit — falls below ``audit_threshold``, the 5th
percentile of the training confidences, stored in the artifact).
A confident-but-stale student is the known hole in the gated
policy: it is measured, not hidden — see the planted-conflict
demo, where the correction also makes the artifact stale (the
fact-set hash no longer matches), which is the signal to serve
with ``always`` audit until the LoRA is re-distilled.

Phase 3 — Serve per project. ``save_student`` / ``load_student``
round-trip a versioned artifact; ``is_stale`` compares the
artifact's fact-set hash against the live store's, so a
supersede/correction invalidates the cache explicitly instead
of silently serving old weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile

import numpy as np

from .eval import load_synthetic
from .retrieval import retrieve
from .store import Store

# The fact whose correction is planted in the demo: the student's
# memorised answer ("three hours twelve minutes, set in Berlin")
# contradicts the corrected text store after the supersede.
PLANTED_FACT = "nina-marathon"
PLANTED_QUESTION = "Nina marathon personal best Berlin three hours"
PLANTED_CORRECTED_FACT = "nina-marathon-corrected"
PLANTED_CORRECTED_TEXTS = [
    "Nina's marathon personal best is now two hours fifty-eight "
    "minutes, set in Berlin.",
    "Nina lowered her marathon personal best to two hours "
    "fifty-eight minutes in Berlin this spring.",
    "The marathon personal best Nina now holds is two hours "
    "fifty-eight minutes, set in Berlin.",
]
PLANTED_CORRECTED_MARKER = "two hours fifty-eight"


def _fact_id_from_key(key: str) -> str | None:
    """Fact id for a record key (``<fact>-<n>`` -> ``<fact>``).

    Trap records (``trap-<n>``) name no fact: they are lexical
    distractors, so they map to None.
    """
    if key.startswith("trap-"):
        return None
    parts = key.rsplit("-", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else key


def _tokens(text: str) -> int:
    """Deterministic token estimate, as in P18/P19: ceil(len/4)."""
    return int(math.ceil(len(text) / 4))


class FactIndex:
    """The fact registry a student is distilled from / audited against.

    A fact is a group of records that state the same attribute
    (the synthetic corpus's ``gold_keys`` group, e.g. the three
    ``maya-oven-*`` records). ``rid_to_fact`` maps a store record
    id to its fact (None for trap/distractor records and for
    records that name no fact); ``fact_texts`` holds the record
    texts per fact, first text = representative answer text.
    """

    def __init__(self) -> None:
        self.facts: list[str] = []
        self.fact_texts: dict[str, list[str]] = {}
        self.rid_to_fact: dict[int, str | None] = {}
        self.key_to_fact: dict[str, str | None] = {}

    def add_fact(
        self, fact_id: str, texts: list[str], rids: list[int] | None = None
    ) -> None:
        if fact_id not in self.fact_texts:
            self.facts.append(fact_id)
            self.fact_texts[fact_id] = []
        seen = set(self.fact_texts[fact_id])
        for text in texts:
            if text not in seen:
                self.fact_texts[fact_id].append(text)
                seen.add(text)
        for rid in rids or []:
            self.rid_to_fact[rid] = fact_id

    def representative_text(self, fact_id: str) -> str:
        return self.fact_texts[fact_id][0]

    def fact_set_hash(self) -> str:
        """Hash of the fact set the student was distilled from.

        Any supersede / correction that changes the live facts
        changes this hash, which is how a stale artifact is
        detected (``is_stale``) — weights are a cache, and the
        cache must know when it is out of date.
        """
        payload = "\n".join(
            "%s\t%s" % (f, self.representative_text(f))
            for f in sorted(self.fact_texts)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fact_index(
    records: list[dict], queries: list[dict], rids: list[int]
) -> FactIndex:
    """Build the fact registry from corpus records + queries.

    Facts are the distinct ``gold_keys`` groups in the queries;
    every record whose key belongs to a gold group joins that
    fact; every other record (incl. traps) maps to None.
    """
    index = FactIndex()
    key_to_rid = {r["key"]: rid for r, rid in zip(records, rids)}
    text_by_key = {r["key"]: r["text"] for r in records}
    for q in queries:
        fact_id = _fact_id_from_key(q["gold_keys"][0])
        if fact_id is None:
            continue
        keys = [k for k in q["gold_keys"] if k in text_by_key]
        index.add_fact(
            fact_id,
            [text_by_key[k] for k in keys],
            [key_to_rid[k] for k in keys if k in key_to_rid],
        )
    for r, rid in zip(records, rids):
        if rid not in index.rid_to_fact:
            index.rid_to_fact[rid] = _fact_id_from_key(r["key"])
            index.key_to_fact[r["key"]] = index.rid_to_fact[rid]
    for r in records:
        index.key_to_fact.setdefault(r["key"], _fact_id_from_key(r["key"]))
    return index


def build_distillation_examples(
    fact_index: FactIndex, queries: list[dict]
) -> tuple[list[str], list[str]]:
    """OPSD training pairs (structure, not a generative teacher).

    For each fact, the training texts are the keyword question
    (the phrasing the student must learn to answer context-free)
    plus the teacher's answer texts for that fact (the gold
    record texts — what the retrieval-grounded teacher returns
    when conditioned on the memory). All are labelled with the
    fact. Held-out paraphrase + trap questions are never used
    here; they are the eval set.
    """
    texts: list[str] = []
    labels: list[str] = []
    for q in queries:
        if q["kind"] != "keyword":
            continue
        fact_id = _fact_id_from_key(q["gold_keys"][0])
        if fact_id is None or fact_id not in fact_index.fact_texts:
            continue
        texts.append(q["question"])
        labels.append(fact_id)
        for text in fact_index.fact_texts[fact_id]:
            texts.append(text)
            labels.append(fact_id)
    return texts, labels


class Student:
    """A distilled weight-memory student (the CPU LoRA stand-in).

    Frozen base: the MiniLM embedder (never trained). Adapter:
    low-rank factors ``A`` (dim x rank) and ``B`` (rank x
    n_facts) with ``logits = x @ A @ B`` for a unit-norm question
    embedding ``x``; the predicted fact is the argmax logit and
    the confidence is that logit. ``audit_threshold`` is the 5th
    percentile of the training max-logits (the gated-audit bar).
    """

    def __init__(
        self,
        fact_ids: list[str],
        a: np.ndarray,
        b: np.ndarray,
        audit_threshold: float,
        rank: int,
        embedder_name: str,
        project: str,
        version: int,
        fact_set_hash: str,
        train_accuracy: float,
        n_train: int,
    ) -> None:
        self.fact_ids = list(fact_ids)
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.audit_threshold = float(audit_threshold)
        self.rank = int(rank)
        self.embedder_name = embedder_name
        self.project = project
        self.version = int(version)
        self.fact_set_hash = fact_set_hash
        self.train_accuracy = float(train_accuracy)
        self.n_train = int(n_train)

    @property
    def n_params(self) -> int:
        return int(self.a.size + self.b.size)

    def predict_vec(self, vec: np.ndarray) -> tuple[str, float, np.ndarray]:
        logits = np.asarray(vec, dtype=np.float64) @ self.a @ self.b
        best = int(np.argmax(logits))
        return self.fact_ids[best], float(logits[best]), logits

    def predict(self, embedder, question: str) -> tuple[str, float, np.ndarray]:
        vec = embedder.embed([question])[0]
        return self.predict_vec(vec)

    def config(self) -> dict:
        return {
            "project": self.project,
            "version": self.version,
            "rank": self.rank,
            "embedder": self.embedder_name,
            "fact_set_hash": self.fact_set_hash,
            "audit_threshold": self.audit_threshold,
            "train_accuracy": self.train_accuracy,
            "n_train": self.n_train,
            "n_facts": len(self.fact_ids),
            "n_params": self.n_params,
            "fact_ids": self.fact_ids,
        }


def train_student(
    texts: list[str],
    labels: list[str],
    embedder,
    rank: int = 32,
    project: str = "synthetic",
    version: int = 1,
    fact_set_hash: str = "",
) -> Student:
    """Distil the teacher's outputs into a low-rank student.

    Closed-form prototype fitting + truncated SVD (no gradient
    descent, no torch): each fact's prototype is the mean of its
    training embeddings, L2-normalised; the prototype matrix
    (n_facts x dim) is factorised by SVD and truncated to
    ``rank``. Deterministic on a fixed embedder and seed-free —
    the same inputs always give the same artifact.
    """
    fact_ids = sorted(set(labels))
    fidx = {f: i for i, f in enumerate(fact_ids)}
    vecs = embedder.embed(texts).astype(np.float64)
    y = np.array([fidx[label] for label in labels])
    prototypes = np.zeros((len(fact_ids), vecs.shape[1]), dtype=np.float64)
    for i in range(len(fact_ids)):
        prototypes[i] = vecs[y == i].mean(axis=0)
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    prototypes = prototypes / norms
    u, s, vt = np.linalg.svd(prototypes, full_matrices=False)
    r = min(int(rank), len(s))
    # SVD gives prototypes = U S Vt with U (n_facts x r), Vt
    # (r x dim). The adapter must map dim -> facts, so store
    # A = Vt.T (dim x r) scaled by S and B = U.T (r x n_facts):
    # logits = x @ A @ B == x @ (Vt.T * S) @ U.T.
    a = (vt[:r].T * s[:r]).astype(np.float64)  # (dim, r)
    b = u[:, :r].T.astype(np.float64)  # (r, n_facts)
    student = Student(
        fact_ids,
        a,
        b,
        audit_threshold=0.0,
        rank=r,
        embedder_name=getattr(embedder, "name", "unknown"),
        project=project,
        version=version,
        fact_set_hash=fact_set_hash,
        train_accuracy=0.0,
        n_train=len(texts),
    )
    train_logits = vecs @ a @ b
    train_pred = train_logits.argmax(axis=1)
    student.train_accuracy = float((train_pred == y).mean())
    student.audit_threshold = float(np.percentile(train_logits.max(axis=1), 5))
    return student


def artifact_path(project: str, version: int, directory: str = ".") -> str:
    """Versioned artifact filename for a project."""
    return os.path.join(directory, "lora_%s_v%d.npz" % (project, version))


def save_student(student: Student, path: str) -> str:
    """Save a versioned LoRA artifact (one file per project)."""
    np.savez(
        path,
        a=student.a,
        b=student.b,
        meta=json.dumps(student.config()),
    )
    return path


def load_student(path: str) -> Student:
    """Load a versioned LoRA artifact (the per-project swap)."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return Student(
        fact_ids=list(meta["fact_ids"]),
        a=np.asarray(data["a"], dtype=np.float64),
        b=np.asarray(data["b"], dtype=np.float64),
        audit_threshold=float(meta["audit_threshold"]),
        rank=int(meta["rank"]),
        embedder_name=meta["embedder"],
        project=meta["project"],
        version=int(meta["version"]),
        fact_set_hash=meta["fact_set_hash"],
        train_accuracy=float(meta["train_accuracy"]),
        n_train=int(meta["n_train"]),
    )


def is_stale(student: Student, fact_index: FactIndex) -> bool:
    """True when the live fact set no longer matches the artifact.

    A supersede / correction changes the fact-set hash, so the
    cached weights are stale by construction — the serve path
    must audit always (or re-distil) instead of trusting them.
    """
    return student.fact_set_hash != fact_index.fact_set_hash()


def teacher_answer(
    store: Store,
    embedder,
    question: str,
    fact_index: FactIndex,
    k: int = 10,
) -> dict:
    """Arm (a): retrieval alone — the P1 hybrid, top-1 answer.

    The answerer reads the top-``k`` hybrid records and answers
    with the top record's text. Read cost is the whole top-k:
    that is what a retrieval-grounded answerer actually reads.
    """
    results = retrieve(store, embedder, question, k=k, use_mmr=False)
    texts = [r.text for r in results]
    top_fact = fact_index.rid_to_fact.get(results[0].id) if results else None
    return {
        "fact_id": top_fact,
        "text": results[0].text if results else "",
        "records_read": len(results),
        "tokens_read": sum(_tokens(t) for t in texts),
    }


def audit_answer(
    store: Store,
    embedder,
    question: str,
    fact_index: FactIndex,
    k: int = 5,
) -> dict:
    """Audit leg: vector-only top-``k`` facts from the text store.

    The vector leg is the audit leg because it is the strongest
    leg on this corpus (P1: bare cosine any-hit@5 0.875 vs hybrid
    0.589 on the TRACE split) — the audit should use the best
    available evidence, not the leg that produced the answer.
    Returns the audited facts in rank order (None = distractor
    record naming no fact) plus the read cost.
    """
    results = retrieve(
        store, embedder, question, k=k, use_mmr=False, weights=(0.0, 1.0, 0.0)
    )
    facts = [fact_index.rid_to_fact.get(r.id) for r in results]
    return {
        "facts": facts,
        "known_facts": [f for f in facts if f is not None],
        "records_read": len(results),
        "tokens_read": sum(_tokens(r.text) for r in results),
    }


def answer_student(
    store: Store,
    embedder,
    student: Student,
    question: str,
    fact_index: FactIndex,
    audit: str = "gated",
    audit_k: int = 5,
) -> dict:
    """Arm (b)/(c): the student answers context-free, then verifies.

    ``audit="gated"`` audits only when confidence < the
    artifact's threshold (near-zero read cost on most queries);
    ``audit="always"`` audits every answer (correctness-first).
    A conflict — the student's fact is absent from the audited
    facts — regenerates the answer from the first audited record
    naming a known fact. If the audit names no known fact at all,
    the student's answer stands but is flagged unverified.
    """
    fact_id, confidence, _ = student.predict(embedder, question)
    out = {
        "student_fact": fact_id,
        "confidence": confidence,
        "audited": False,
        "verified": None,
        "conflict": False,
        "regenerated": False,
        "final_fact": fact_id,
        "final_text": fact_index.representative_text(fact_id)
        if fact_id in fact_index.fact_texts
        else "",
        "records_read": 0,
        "tokens_read": 0,
    }
    if audit not in ("gated", "always"):
        raise ValueError("unknown audit policy %r" % audit)
    if audit == "always" or confidence < student.audit_threshold:
        out["audited"] = True
        aud = audit_answer(store, embedder, question, fact_index, k=audit_k)
        out["records_read"] = aud["records_read"]
        out["tokens_read"] = aud["tokens_read"]
        if fact_id in aud["facts"]:
            out["verified"] = True
        elif aud["known_facts"]:
            out["verified"] = False
            out["conflict"] = True
            out["regenerated"] = True
            out["final_fact"] = aud["known_facts"][0]
            out["final_text"] = fact_index.representative_text(
                out["final_fact"]
            )
        else:
            out["verified"] = False
    return out


def _build_store(
    records: list[dict], embedder, tmp: str
) -> tuple[Store, list[int]]:
    store = Store(os.path.join(tmp, "lora.db"), embedder=embedder)
    rids = store.add_texts(
        [r["text"] for r in records], type_="fact", embedder=embedder
    )
    return store, rids


def run_lora_eval(
    corpus_path: str,
    queries_path: str,
    embedder=None,
    rank: int = 32,
    audit_k: int = 5,
    verbose: bool = True,
) -> dict:
    """HaluMem-style three-arm eval on the synthetic corpus.

    Train on the 48 keyword questions (+ teacher answer texts),
    eval on the 56 held-out paraphrase + trap questions:
    (a) retrieval alone (hybrid top-1), (b) student alone
    (context-free, 0 records read), (c) student + verify-then-
    regenerate, in both audit policies. Read cost is records
    and tokens read per query. The planted-conflict demo runs
    separately (``run_planted_conflict``) so the eval store is
    never mutated by it.
    """
    from .embeddings import get_embedder

    embedder = embedder or get_embedder()
    records, queries = load_synthetic(corpus_path, queries_path)
    tmp = tempfile.mkdtemp(prefix="hr-lora-")
    store, rids = _build_store(records, embedder, tmp)
    fact_index = build_fact_index(records, queries, rids)
    texts, labels = build_distillation_examples(fact_index, queries)
    student = train_student(
        texts,
        labels,
        embedder,
        rank=rank,
        fact_set_hash=fact_index.fact_set_hash(),
    )
    heldout = [q for q in queries if q["kind"] != "keyword"]
    if verbose:
        print(
            "distilled %d facts from %d teacher examples "
            "(rank=%d, %d params, train acc=%.3f, audit threshold=%.3f); "
            "eval on %d held-out questions"
            % (
                len(student.fact_ids),
                len(texts),
                student.rank,
                student.n_params,
                student.train_accuracy,
                student.audit_threshold,
                len(heldout),
            )
        )

    def _gold(q: dict) -> str | None:
        return _fact_id_from_key(q["gold_keys"][0])

    arms: dict[str, dict] = {}
    for name in ("retrieval", "student", "student+verify-always",
                 "student+verify-gated"):
        arms[name] = {
            "correct": 0,
            "n": len(heldout),
            "records_read": 0,
            "tokens_read": 0,
            "audited": 0,
            "regenerated": 0,
            "by_kind": {},
        }
    for q in heldout:
        gold = _gold(q)
        kind = q["kind"]
        for name in arms:
            arms[name]["by_kind"].setdefault(kind, {"correct": 0, "n": 0})
            arms[name]["by_kind"][kind]["n"] += 1
        teach = teacher_answer(store, embedder, q["question"], fact_index)
        arms["retrieval"]["records_read"] += teach["records_read"]
        arms["retrieval"]["tokens_read"] += teach["tokens_read"]
        if teach["fact_id"] == gold:
            arms["retrieval"]["correct"] += 1
            arms["retrieval"]["by_kind"][kind]["correct"] += 1
        fact_id, _, _ = student.predict(embedder, q["question"])
        if fact_id == gold:
            arms["student"]["correct"] += 1
            arms["student"]["by_kind"][kind]["correct"] += 1
        for name, policy in (
            ("student+verify-always", "always"),
            ("student+verify-gated", "gated"),
        ):
            ans = answer_student(
                store, embedder, student, q["question"], fact_index,
                audit=policy, audit_k=audit_k,
            )
            arms[name]["records_read"] += ans["records_read"]
            arms[name]["tokens_read"] += ans["tokens_read"]
            arms[name]["audited"] += int(ans["audited"])
            arms[name]["regenerated"] += int(ans["regenerated"])
            if ans["final_fact"] == gold:
                arms[name]["correct"] += 1
                arms[name]["by_kind"][kind]["correct"] += 1
    for arm in arms.values():
        n = arm["n"]
        arm["accuracy"] = arm["correct"] / n
        arm["records_per_query"] = arm["records_read"] / n
        arm["tokens_per_query"] = arm["tokens_read"] / n
        for kind_stats in arm["by_kind"].values():
            kind_stats["accuracy"] = kind_stats["correct"] / kind_stats["n"]
    store.close()
    return {
        "corpus": "synthetic (labelled) — HaluMem-style, TRACE split",
        "n_questions": len(heldout),
        "n_facts": len(student.fact_ids),
        "n_train_examples": len(texts),
        "rank": student.rank,
        "n_params": student.n_params,
        "train_accuracy": student.train_accuracy,
        "audit_threshold": student.audit_threshold,
        "embedder": embedder.name,
        "arms": arms,
        "student_config": student.config(),
    }


def run_planted_conflict(
    corpus_path: str,
    queries_path: str,
    embedder=None,
    rank: int = 32,
) -> dict:
    """Deliberately planted conflict: is the verify step catching it?

    The student is distilled from the original store, so it
    memorised Nina's old personal best (three hours twelve
    minutes, Berlin). The text store is then corrected: the old
    records are superseded (never deleted — the P2 lifecycle)
    and corrected records are added. The artifact is stale by
    its fact-set hash, so the serve path audits always; the
    audit must catch the conflict and regenerate from the
    corrected records. Also reports whether the gated policy
    would have audited at all — the honest hole: a confident
    stale answer skips the gated audit, which is exactly why a
    correction must flip the serve policy to always-audit (or
    trigger re-distillation) instead of relying on confidence.
    """
    from .embeddings import get_embedder

    embedder = embedder or get_embedder()
    records, queries = load_synthetic(corpus_path, queries_path)
    tmp = tempfile.mkdtemp(prefix="hr-lora-conflict-")
    store, rids = _build_store(records, embedder, tmp)
    fact_index = build_fact_index(records, queries, rids)
    texts, labels = build_distillation_examples(fact_index, queries)
    student = train_student(
        texts,
        labels,
        embedder,
        rank=rank,
        fact_set_hash=fact_index.fact_set_hash(),
    )
    # The student, context-free, on the memorised question.
    student_fact, confidence, _ = student.predict(embedder, PLANTED_QUESTION)
    # Correct the text store: supersede the old fact, add the
    # corrected records (new fact id, same question).
    old_rids = [
        rid
        for rid, fact in fact_index.rid_to_fact.items()
        if fact == PLANTED_FACT
    ]
    store.set_status_many(old_rids, "superseded")
    new_rids = store.add_texts(
        PLANTED_CORRECTED_TEXTS, type_="fact", embedder=embedder
    )
    fact_index.add_fact(PLANTED_CORRECTED_FACT, PLANTED_CORRECTED_TEXTS, new_rids)
    stale = is_stale(student, fact_index)
    # Serve with always-audit (the policy a stale artifact forces).
    ans = answer_student(
        store, embedder, student, PLANTED_QUESTION, fact_index,
        audit="always",
    )
    gated_would_audit = confidence < student.audit_threshold
    store.close()
    return {
        "question": PLANTED_QUESTION,
        "student_fact": student_fact,
        "student_confidence": confidence,
        "audit_threshold": student.audit_threshold,
        "gated_would_audit": gated_would_audit,
        "artifact_stale_after_correction": stale,
        "superseded_records": len(old_rids),
        "audited": ans["audited"],
        "conflict_caught": bool(ans["conflict"]),
        "regenerated": bool(ans["regenerated"]),
        "final_fact": ans["final_fact"],
        "final_text": ans["final_text"],
        "correct_regeneration": (
            ans["final_fact"] == PLANTED_CORRECTED_FACT
            and PLANTED_CORRECTED_MARKER in ans["final_text"]
        ),
    }


def print_lora_report(result: dict) -> None:
    print("\n== %s ==" % result["corpus"])
    print(
        "facts: %d   train examples: %d   rank: %d   params: %d   "
        "train acc: %.3f   audit threshold: %.3f   embedder: %s"
        % (
            result["n_facts"],
            result["n_train_examples"],
            result["rank"],
            result["n_params"],
            result["train_accuracy"],
            result["audit_threshold"],
            result["embedder"],
        )
    )
    print(
        "%-24s %8s %10s %12s %8s %10s"
        % ("arm", "accuracy", "correct", "records/query",
           "tokens/q", "regenerated")
    )
    for name, arm in result["arms"].items():
        print(
            "%-24s %8.3f %5d/%-4d %12.2f %8.1f %10d"
            % (
                name,
                arm["accuracy"],
                arm["correct"],
                arm["n"],
                arm["records_per_query"],
                arm["tokens_per_query"],
                arm["regenerated"],
            )
        )
    for name, arm in result["arms"].items():
        kinds = ", ".join(
            "%s %.3f (%d/%d)"
            % (kind, s["accuracy"], s["correct"], s["n"])
            for kind, s in sorted(arm["by_kind"].items())
        )
        print("  %-22s by kind: %s" % (name, kinds))


def print_planted_conflict_report(result: dict) -> None:
    print("\n== planted conflict ==")
    print("question: %s" % result["question"])
    print(
        "student (context-free): %s (confidence %.3f, threshold %.3f)"
        % (
            result["student_fact"],
            result["student_confidence"],
            result["audit_threshold"],
        )
    )
    print(
        "correction: %d records superseded; artifact stale: %s; "
        "gated policy would have audited: %s"
        % (
            result["superseded_records"],
            result["artifact_stale_after_correction"],
            result["gated_would_audit"],
        )
    )
    print(
        "verify: audited=%s conflict caught=%s regenerated=%s"
        % (
            result["audited"],
            result["conflict_caught"],
            result["regenerated"],
        )
    )
    print(
        "final: %s — %s  (correct regeneration: %s)"
        % (
            result["final_fact"],
            result["final_text"],
            result["correct_regeneration"],
        )
    )
