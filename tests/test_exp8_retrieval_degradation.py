"""Retrieval stress tests must actually remove gold passages."""

from types import SimpleNamespace

import pytest

from experiments import exp8_stress_test as exp8


def test_single_gold_passage_can_be_removed():
    pipeline = object.__new__(exp8.DegradedRetrievalPipeline)
    pipeline.corpus = ["gold", "distractor", "another distractor"]
    retrieved = [(0, "gold", 1.0), (1, "distractor", 0.8)]

    degraded = pipeline.degrade_retrieval(retrieved, [0], 0.0)

    assert [doc[0] for doc in degraded] == [1, 2]
    assert retrieved[0][0] == 0
    assert pipeline.degrade_retrieval(retrieved, [0], 1.0) == retrieved


def test_degradation_fails_without_a_distractor():
    pipeline = object.__new__(exp8.DegradedRetrievalPipeline)
    pipeline.corpus = ["gold"]

    with pytest.raises(ValueError, match="without an irrelevant document"):
        pipeline.degrade_retrieval([(0, "gold", 1.0)], [0], 0.0)


def test_generate_applies_query_target_before_reranking():
    pipeline = object.__new__(exp8.DegradedRetrievalPipeline)
    pipeline.corpus = ["gold", "distractor"]
    pipeline.target_recall_at_20 = 1.0
    pipeline.enable_revision = False
    pipeline.retriever = SimpleNamespace(
        retrieve=lambda _query, top_k: [(0, "gold", 1.0), (1, "distractor", 0.8)]
    )
    pipeline.reranker = SimpleNamespace(
        rerank=lambda _query, texts, top_k: [(0, texts[0], 1.0)]
    )
    pipeline.generator = SimpleNamespace(generate=lambda _query, _context, **_kwargs: "answer")
    pipeline.claim_extractor = SimpleNamespace(extract_claims=lambda _answer: [])
    pipeline.verifier = SimpleNamespace(
        verify_generation=lambda _answer, _texts, _claims: {
            "verified": True, "verification_results": []
        }
    )

    result = pipeline.generate("question", relevant_docs=[0], target_recall_at_20=0.0)

    assert result["retrieved_docs"] == [1]
    assert result["reranked_docs"] == [1]
    assert result["context"] == "distractor"


def test_run_level_targets_are_reproducible_and_nested():
    relevant_docs = [[idx] for idx in range(10)]
    high = exp8.allocate_query_recall_targets(relevant_docs, 0.7, seed=42)
    low = exp8.allocate_query_recall_targets(relevant_docs, 0.3, seed=42)

    assert sum(high) == 7
    assert sum(low) == 3
    assert all(low_target <= high_target for low_target, high_target in zip(low, high))
    assert high == exp8.allocate_query_recall_targets(relevant_docs, 0.7, seed=42)


@pytest.mark.parametrize("target", [0.95, 0.85, 0.75, 0.65])
def test_configured_targets_degrade_single_gold_queries(target):
    relevant_docs = [[idx] for idx in range(100)]

    targets = exp8.allocate_query_recall_targets(relevant_docs, target, seed=42)

    assert sum(targets) == round(target * 100)
    assert 0.0 in targets


def test_retrieval_degradation_passes_per_query_targets(monkeypatch):
    applied_targets = []

    class FakePipeline:
        def __init__(self, **_kwargs):
            pass

        def generate(self, _query, *, target_recall_at_20, **_kwargs):
            applied_targets.append(target_recall_at_20)
            return {
                "retrieved_docs": [],
                "verification_results": {"verification_results": []},
                "generated_text": "",
            }

    class FakeEvaluator:
        def compute_all_metrics(self, **_kwargs):
            return {
                "recall@20": 0.0,
                "factual_precision": 0.0,
                "verified_f1": 0.0,
                "hallucination_rate": 0.0,
                "exact_match": 0.0,
                "f1_score": 0.0,
            }

    monkeypatch.setattr(exp8, "DegradedRetrievalPipeline", FakePipeline)
    monkeypatch.setattr(exp8, "EvaluationMetrics", FakeEvaluator)
    monkeypatch.setattr(exp8, "resolve_device", lambda _device: "cpu")

    exp8.run_retrieval_degradation_test(
        queries=[f"question {idx}" for idx in range(10)],
        ground_truths=["answer"] * 10,
        relevant_docs=[[idx] for idx in range(10)],
        corpus=["passage"] * 10,
        config={"experiments": {"device": "cpu"}},
        target_recalls=[0.7],
    )

    assert len(applied_targets) == 10
    assert sum(applied_targets) == 7


def test_retrieval_degradation_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must have equal lengths"):
        exp8.run_retrieval_degradation_test(
            queries=["question"],
            ground_truths=[],
            relevant_docs=[[0]],
            corpus=["passage"],
            config={},
            target_recalls=[0.7],
        )
