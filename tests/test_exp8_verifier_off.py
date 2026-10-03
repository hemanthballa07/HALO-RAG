"""The pure RAG comparison keeps independent factuality scoring."""

from types import SimpleNamespace

from experiments import exp8_stress_test as exp8


def test_verifier_off_keeps_normal_threshold_for_scoring(monkeypatch):
    class FakePipeline:
        def __init__(self, **kwargs):
            assert kwargs["enable_revision"] is False
            self.verifier = SimpleNamespace(threshold=0.75)

        def generate(self, _query, **_kwargs):
            return {
                "retrieved_docs": [0],
                "reranked_texts": ["passage"],
                "generated_text": "unsupported claim",
                "verification_results": {
                    "verification_results": [
                        {"is_entailed": 0.2 >= self.verifier.threshold}
                    ]
                },
            }

    class FakeEvaluator:
        def compute_all_metrics(self, **kwargs):
            assert kwargs["verification_results"][0]["is_entailed"] is False
            return {
                "factual_precision": 0.0,
                "hallucination_rate": 1.0,
                "verified_f1": 0.0,
                "exact_match": 0.0,
                "f1_score": 0.0,
            }

    monkeypatch.setattr(exp8, "SelfVerificationRAGPipeline", FakePipeline)
    monkeypatch.setattr(exp8, "EvaluationMetrics", FakeEvaluator)
    monkeypatch.setattr(exp8, "resolve_device", lambda _device: "cpu")

    result = exp8.run_verifier_off_test(
        queries=["question"],
        ground_truths=["answer"],
        relevant_docs=[[0]],
        corpus=["passage"],
        config={"experiments": {"device": "cpu"}},
    )

    assert result["factual_precision"] == 0.0
    assert result["hallucination_rate"] == 1.0
