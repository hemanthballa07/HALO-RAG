"""Revision abstentions should not count discarded claims as final hallucinations."""

import pytest

from experiments import exp6_iterative_training as exp6
from experiments import exp8_stress_test as exp8
from src.evaluation import EvaluationMetrics


class FakePipeline:
    abstained = False

    def __init__(self, **_kwargs):
        pass

    def generate(self, _query, **_kwargs):
        return {
            "retrieved_docs": [0],
            "reranked_texts": ["Paris is in France."],
            "verification_results": {
                "verification_results": [{"claim": "rejected", "is_entailed": False}]
            },
            "generated_text": "I cannot provide a confident answer.",
            "abstained": self.abstained,
        }


class ScoringEvaluator:
    def compute_all_metrics(self, **kwargs):
        return {
            "hallucination_rate": EvaluationMetrics().hallucination_rate(
                kwargs["verification_results"], abstained=kwargs["abstained"]
            ),
            "factual_precision": 0.0,
            "verified_f1": 0.0,
            "exact_match": 0.0,
            "f1_score": 0.0,
        }


@pytest.mark.parametrize("abstained,expected", [(True, 0.0), (False, 1.0)])
def test_iterative_validation_uses_abstention_flag(abstained, expected):
    pipeline = FakePipeline()
    pipeline.abstained = abstained

    metrics = exp6.evaluate_iteration(
        pipeline=pipeline,
        queries=["Where is Paris?"],
        ground_truths=["France"],
        relevant_docs=[[0]],
        corpus=["Paris is in France."],
        evaluator=ScoringEvaluator(),
    )

    assert metrics["hallucination_rate"] == expected


@pytest.mark.parametrize("abstained,expected", [(True, 0.0), (False, 1.0)])
def test_stress_baseline_uses_abstention_flag(monkeypatch, abstained, expected):
    class BaselinePipeline(FakePipeline):
        pass

    BaselinePipeline.abstained = abstained
    monkeypatch.setattr(exp8, "SelfVerificationRAGPipeline", BaselinePipeline)
    monkeypatch.setattr(exp8, "EvaluationMetrics", ScoringEvaluator)
    monkeypatch.setattr(exp8, "resolve_device", lambda _device: "cpu")

    metrics = exp8.run_baseline_test(
        queries=["Where is Paris?"],
        ground_truths=["France"],
        relevant_docs=[[0]],
        corpus=["Paris is in France."],
        config={"experiments": {"device": "cpu"}},
    )

    assert metrics["hallucination_rate"] == expected
