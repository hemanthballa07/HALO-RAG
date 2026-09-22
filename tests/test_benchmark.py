"""Behavioral checks for reproducible question-answering benchmarks."""

import pytest

from src.evaluation.benchmark import build_benchmark, score_answer, summarize_results
from src.evaluation.metrics import EvaluationMetrics


def sample_examples():
    examples = []
    for index in range(12):
        context = f"Passage {index} describes location {index}."
        examples.append({
            "id": f"answer-{index}",
            "question": f"Which location is described in passage {index}?",
            "context": context,
            "answers": [f"location {index}", f"Location {index}"],
        })
        examples.append({
            "id": f"unanswerable-{index}",
            "question": f"Who wrote passage {index}?",
            "context": context,
            "answers": [],
        })
    return examples


def test_build_benchmark_is_seeded_balanced_and_uses_distinct_contexts():
    first = build_benchmark(sample_examples(), question_count=6, corpus_size=10, seed=42)
    second = build_benchmark(sample_examples(), question_count=6, corpus_size=10, seed=42)

    assert first == second
    assert len(first.corpus) == 10
    assert len(first.cases) == 6
    assert sum(case.answerable for case in first.cases) == 3
    assert len({case.relevant_doc_id for case in first.cases}) == 6
    assert all(first.corpus[case.relevant_doc_id] == case.context for case in first.cases)
    assert len(first.document_hashes) == 10


def test_build_benchmark_rejects_a_trivial_or_impossible_corpus():
    with pytest.raises(ValueError, match="at least as large"):
        build_benchmark(sample_examples(), question_count=6, corpus_size=5, seed=42)
    with pytest.raises(ValueError, match="only 12 distinct"):
        build_benchmark(sample_examples(), question_count=6, corpus_size=13, seed=42)


def test_answer_scoring_uses_all_references_and_handles_no_answer():
    assert score_answer("New York", ["NYC", "New York City"]) == {
        "exact_match": 0.0,
        "f1": pytest.approx(0.8),
    }
    assert score_answer("", []) == {"exact_match": 1.0, "f1": 1.0}
    assert score_answer("France", [], abstained=True) == {"exact_match": 1.0, "f1": 1.0}
    assert score_answer("France", []) == {"exact_match": 0.0, "f1": 0.0}
    assert EvaluationMetrics().f1_score("", "") == 1.0


def test_summary_reports_answerable_and_unanswerable_separately():
    rows = [
        {"answerable": True, "exact_match": 1.0, "f1": 1.0, "retrieval_hit": 1.0,
         "evidence_hit": 1.0,
         "final_evidence_hit": 1.0,
         "abstained": False, "verified": True, "revision_iterations": 0,
         "generated": "right answer"},
        {"answerable": True, "exact_match": 0.0, "f1": 0.5, "retrieval_hit": 0.0,
         "evidence_hit": 0.0,
         "final_evidence_hit": 0.0,
         "abstained": True, "verified": False, "revision_iterations": 1,
         "generated": "I cannot answer"},
        {"answerable": False, "exact_match": 1.0, "f1": 1.0, "retrieval_hit": 1.0,
         "evidence_hit": 0.0,
         "final_evidence_hit": 0.0,
         "abstained": True, "verified": False, "revision_iterations": 1,
         "generated": "I cannot answer"},
        {"answerable": False, "exact_match": 0.0, "f1": 0.0, "retrieval_hit": 1.0,
         "evidence_hit": 1.0,
         "final_evidence_hit": 0.0,
         "abstained": False, "verified": True, "revision_iterations": 0,
         "generated": "a wrong answer"},
    ]

    summary = summarize_results(rows)

    assert summary["overall"]["count"] == 4
    assert summary["answerable"]["exact_match"] == 0.5
    assert summary["unanswerable"]["exact_match"] == 0.5
    assert summary["overall"]["retrieval_hit"] == 0.75
    assert summary["overall"]["evidence_hit"] == 0.5
    assert summary["overall"]["final_evidence_hit"] == 0.25
    assert summary["unanswerable"]["false_accept_rate"] == 0.5
    assert summary["unanswerable"]["unanswerable_answer_rate"] == 0.5
    assert summary["overall"]["verified_nonexact_rate"] == 0.25
    assert summary["answerable"]["false_accept_rate"] is None
