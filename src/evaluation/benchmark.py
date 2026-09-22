"""Reproducible question-answering samples and SQuAD-style scoring."""

from __future__ import annotations

import hashlib
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence


@dataclass(frozen=True)
class BenchmarkCase:
    example_id: str
    question: str
    context: str
    references: tuple[str, ...]
    relevant_doc_id: int
    answerable: bool


@dataclass(frozen=True)
class BenchmarkSet:
    corpus: tuple[str, ...]
    document_hashes: tuple[str, ...]
    cases: tuple[BenchmarkCase, ...]
    seed: int


def build_benchmark(
    examples: Sequence[dict[str, Any]],
    question_count: int,
    corpus_size: int,
    seed: int,
) -> BenchmarkSet:
    """Choose the document collection before sampling questions from it.

    A case uses one unique source passage. Half the questions are unanswerable
    (rounded down for odd counts). The other documents serve as distractors.
    """
    if question_count <= 0 or corpus_size <= 0:
        raise ValueError("question_count and corpus_size must be positive")
    if corpus_size < question_count:
        raise ValueError("corpus_size must be at least as large as question_count")

    contexts = list(dict.fromkeys(
        example["context"] for example in examples if example.get("context")
    ))
    if corpus_size > len(contexts):
        raise ValueError(
            f"requested {corpus_size} documents, but only {len(contexts)} distinct contexts exist"
        )

    rng = random.Random(seed)
    corpus = tuple(rng.sample(contexts, corpus_size))
    context_to_id = {context: index for index, context in enumerate(corpus)}
    selected: list[dict[str, Any]] = []
    used_contexts: set[str] = set()

    for answerable, target in ((True, (question_count + 1) // 2), (False, question_count // 2)):
        candidates = [
            example for example in examples
            if example.get("context") in context_to_id
            and bool(example.get("answers")) == answerable
            and example.get("question")
        ]
        rng.shuffle(candidates)
        chosen = 0
        for example in candidates:
            if example["context"] in used_contexts:
                continue
            selected.append(example)
            used_contexts.add(example["context"])
            chosen += 1
            if chosen == target:
                break
        if chosen != target:
            category = "answerable" if answerable else "unanswerable"
            raise ValueError(
                f"only {chosen} distinct {category} contexts available; "
                "increase corpus_size or reduce question_count"
            )

    rng.shuffle(selected)
    cases = tuple(
        BenchmarkCase(
            example_id=str(example["id"]),
            question=example["question"],
            context=example["context"],
            references=tuple(example.get("answers") or ()),
            relevant_doc_id=context_to_id[example["context"]],
            answerable=bool(example.get("answers")),
        )
        for example in selected
    )
    hashes = tuple(hashlib.sha256(context.encode("utf-8")).hexdigest() for context in corpus)
    return BenchmarkSet(corpus=corpus, document_hashes=hashes, cases=cases, seed=seed)


def _answer_tokens(text: str) -> list[str]:
    lowered = text.lower().translate(str.maketrans("", "", string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", lowered)
    return without_articles.split()


def score_answer(
    prediction: str,
    references: Sequence[str],
    abstained: bool = False,
) -> dict[str, float]:
    """Return the best exact match and token F1 over all reference answers."""
    predicted_tokens = _answer_tokens("" if abstained else prediction)
    gold_answers = references or ("",)
    exact_scores = []
    f1_scores = []
    for reference in gold_answers:
        reference_tokens = _answer_tokens(reference)
        exact_scores.append(float(predicted_tokens == reference_tokens))
        if not predicted_tokens or not reference_tokens:
            f1_scores.append(float(not predicted_tokens and not reference_tokens))
            continue
        overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
        precision = overlap / len(predicted_tokens)
        recall = overlap / len(reference_tokens)
        f1_scores.append(2 * precision * recall / (precision + recall) if overlap else 0.0)
    return {"exact_match": max(exact_scores), "f1": max(f1_scores)}


def summarize_results(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """Summarize answer quality and retrieval separately by answerability."""
    fields = (
        "exact_match", "f1", "retrieval_hit", "evidence_hit", "final_evidence_hit",
        "abstained", "verified"
    )

    def summarize(group: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
        summary: dict[str, float | int | None] = {"count": len(group)}
        for field in fields:
            summary[field] = fmean(float(row[field]) for row in group) if group else None
        summary["revision_rate"] = (
            fmean(float(row["revision_iterations"] > 0) for row in group) if group else None
        )
        # Exact match is deliberately strict: a nonexact answer is not
        # necessarily factually incorrect (e.g. a useful partial answer).
        summary["verified_nonexact_rate"] = (
            fmean(float(row["verified"] and row["exact_match"] == 0) for row in group)
            if group else None
        )
        summary["false_accept_rate"] = (
            fmean(float(row["verified"] and not row["abstained"]) for row in group)
            if group and all(not row["answerable"] for row in group) else None
        )
        summary["unanswerable_answer_rate"] = (
            fmean(float(not row["abstained"] and bool(row["generated"].strip())) for row in group)
            if group and all(not row["answerable"] for row in group) else None
        )
        return summary

    return {
        "overall": summarize(rows),
        "answerable": summarize([row for row in rows if row["answerable"]]),
        "unanswerable": summarize([row for row in rows if not row["answerable"]]),
    }
