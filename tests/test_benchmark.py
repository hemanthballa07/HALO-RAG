"""Behavioral checks for reproducible question-answering benchmarks."""

import json

import pytest

from experiments.summarize_paired_benchmarks import combine_runs
from src.evaluation.benchmark import (
    BenchmarkCase, build_benchmark, case_result_record, score_answer, source_fingerprint,
    summarize_results,
)
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


def test_case_record_keeps_evidence_and_claim_scores_for_audit():
    case = BenchmarkCase(
        example_id="sample", question="Which fort?", context="The source passage.",
        references=(), relevant_doc_id=7, answerable=False,
    )
    result = {
        "initial_retrieved_docs": [7], "initial_reranked_docs": [7],
        "retrieved_docs": [7], "reranked_docs": [7],
        "reranked_texts": ["The British captured Fort Beauséjour."],
        "generated_text": "Fort Beauséjour", "claims": ["Fort Beauséjour"],
        "verification_results": {"verification_results": [{"claim": "Fort Beauséjour",
                                                       "entailment_score": 1.0}]},
        "verified": True, "abstained": False, "revision_iterations": 0,
    }

    record = case_result_record(case, result, {"exact_match": 0.0, "f1": 0.0}, 1.2345)

    assert record["source_context"] == "The source passage."
    assert record["final_evidence_texts"] == ["The British captured Fort Beauséjour."]
    assert record["claims"] == ["Fort Beauséjour"]
    assert record["claim_verification"][0]["entailment_score"] == 1.0
    assert record["final_evidence_hit"] == 1.0
    assert record["latency_seconds"] == 1.234


def test_source_fingerprint_changes_with_benchmark_code(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    experiment_dir = tmp_path / "experiments"
    experiment_dir.mkdir()
    source_file = source_dir / "pipeline.py"
    source_file.write_text("answer = 1\n", encoding="utf-8")
    runner = experiment_dir / "run_representative_benchmark.py"
    runner.write_text("run()\n", encoding="utf-8")

    first = source_fingerprint(tmp_path)
    source_file.write_text("answer = 2\n", encoding="utf-8")

    assert source_fingerprint(tmp_path) != first
    second = source_fingerprint(tmp_path)
    runner.write_text("run(strict=True)\n", encoding="utf-8")
    assert source_fingerprint(tmp_path) != second


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


def test_combined_benchmarks_keep_splits_pairs_and_latency(tmp_path):
    def make_row(example_id, answerable, correct, verified, abstained, latency):
        return {
            "example_id": example_id,
            "answerable": answerable,
            "exact_match": float(correct),
            "f1": float(correct),
            "retrieval_hit": 1.0,
            "evidence_hit": 1.0,
            "verified": verified,
            "abstained": abstained,
            "generated": "" if abstained else "answer",
            "latency_seconds": latency,
        }

    paths = []
    for seed in (1, 2):
        answer_id, no_answer_id = f"{seed}-answer", f"{seed}-no-answer"
        rows = {
            "baseline": [
                make_row(answer_id, True, True, True, False, 2.0),
                make_row(no_answer_id, False, False, True, False, 4.0),
            ],
            "revision": [
                make_row(answer_id, True, True, True, False, 3.0),
                make_row(no_answer_id, False, True, False, True, 5.0),
            ],
            "focused": [
                make_row(answer_id, True, True, True, False, 1.0),
                make_row(no_answer_id, False, seed == 1, False, seed == 1, 1.5),
            ],
        }
        payload = {
            "metadata": {
                "dataset": "squad_v2", "split": "validation", "corpus_size": 100,
                "top_k_retrieve": 20, "top_k_rerank": 5, "max_revisions": 1,
                "config_sha256": "config", "source_sha256": "source",
                "models": {"generator": "test"},
                "package_versions": {"torch": "test"},
                "seed": seed, "question_count": 2,
                "question_ids": [answer_id, no_answer_id],
            },
            "cases": rows,
            "summary": {},
        }
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = combine_runs(paths)

    assert result["summary"]["focused"]["overall"]["exact_match"] == 0.75
    assert result["summary"]["focused"]["unanswerable"]["false_accept_rate"] == 0.0
    assert result["summary"]["focused"]["overall"]["latency_p95_seconds"] == 1.5
    assert result["paired_vs_baseline"]["focused"]["improved"] == 1
    assert result["duplicate_question_ids_across_runs"] == []


def test_combined_benchmarks_reject_protocol_changes(tmp_path):
    first = {
        "metadata": {
            "dataset": "squad_v2", "split": "validation", "corpus_size": 100,
            "top_k_retrieve": 20, "top_k_rerank": 5, "max_revisions": 1,
            "config_sha256": "config", "source_sha256": "source", "models": {},
            "package_versions": {"torch": "test"}, "seed": 1,
            "question_count": 1, "question_ids": ["one"],
        },
        "cases": {"baseline": [{"example_id": "one"}]},
        "summary": {},
    }
    second = json.loads(json.dumps(first))
    second["metadata"]["seed"] = 2
    second["metadata"]["corpus_size"] = 200
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible benchmark protocol"):
        combine_runs([first_path, second_path])


def test_combined_benchmarks_reject_code_changes(tmp_path):
    payload = {
        "metadata": {
            "dataset": "squad_v2", "split": "validation", "corpus_size": 100,
            "top_k_retrieve": 20, "top_k_rerank": 5, "max_revisions": 1,
            "config_sha256": "config", "source_sha256": "first", "models": {},
            "package_versions": {}, "seed": 1, "question_count": 0,
            "question_ids": [],
        },
        "cases": {"baseline": []}, "summary": {},
    }
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(payload), encoding="utf-8")
    payload["metadata"]["seed"] = 2
    payload["metadata"]["source_sha256"] = "second"
    second_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible benchmark protocol"):
        combine_runs([first_path, second_path])

    payload["metadata"].pop("source_sha256")
    second_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata missing source_sha256"):
        combine_runs([second_path])
