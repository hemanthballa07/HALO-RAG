"""Self-consistency retrieval scores must use the selected sample's documents."""

from types import SimpleNamespace

import pytest

from experiments import exp5_self_consistency as exp5
from src.evaluation import EvaluationMetrics


def test_selected_answer_uses_its_own_retrieval(monkeypatch):
    class FakePipeline:
        def __init__(self, **_kwargs):
            self.calls = 0
            self.claim_extractor = SimpleNamespace(extract_claims=lambda text: [text])
            self.verifier = SimpleNamespace(
                verify_generation=lambda *_args: {
                    "verification_results": [{"is_entailed": True}]
                }
            )

        def generate(self, _query, **_kwargs):
            doc_id = self.calls
            self.calls += 1
            answer = "Paris" if doc_id in {2, 3} else "London"
            return {
                "retrieved_docs": [doc_id],
                "reranked_texts": [f"passage {doc_id}"],
                "generated_text": answer,
                "verification_results": {
                    "verification_results": [{"label": "ENTAILED", "is_entailed": True}]
                },
            }

    monkeypatch.setattr(exp5, "SelfVerificationRAGPipeline", FakePipeline)
    monkeypatch.setattr(exp5, "resolve_device", lambda _device: "cpu")

    run = exp5.run_self_consistency_experiment(
        queries=["Which city?"],
        ground_truths=["Paris"],
        relevant_docs=[[2]],
        corpus=[f"passage {idx}" for idx in range(5)],
        config={"experiments": {"device": "cpu"}},
        k=3,
    )
    results = run["individual_results"]

    assert run["config"]["aggregation_method"] == "majority_vote"
    assert results["greedy"][0]["metrics"]["recall@5"] == 0.0
    assert results["self_consistency"][0]["generated"] == "Paris"
    assert results["self_consistency"][0]["metrics"]["recall@5"] == 1.0


@pytest.mark.parametrize("ground_truth", ["Paris", "London"])
def test_majority_selection_does_not_use_ground_truth(ground_truth):
    class FakePipeline:
        def __init__(self):
            self.calls = 0

        def generate(self, _query, **_kwargs):
            doc_id = self.calls
            self.calls += 1
            answer = "Paris" if doc_id < 2 else "London"
            return {
                "retrieved_docs": [doc_id],
                "generated_text": answer,
                "verification_results": {
                    "verification_results": [{"label": "ENTAILED"}]
                },
            }

    result = exp5.generate_with_self_consistency(
        FakePipeline(), "Which city?", ground_truth, EvaluationMetrics(), k=3
    )

    assert result["final_answer"] == "Paris"
    assert result["selected_sample"]["retrieved_docs"] == [0]


def test_ground_truth_scoring_method_is_rejected():
    with pytest.raises(ValueError, match="Unsupported aggregation method"):
        exp5.generate_with_self_consistency(
            object(), "query", "answer", EvaluationMetrics(),
            aggregation_method="highest_verified_f1",
        )


def test_configured_factual_precision_selection(monkeypatch):
    class FakePipeline:
        def __init__(self, **_kwargs):
            self.calls = 0
            self.claim_extractor = SimpleNamespace(extract_claims=lambda text: [text])
            self.verifier = SimpleNamespace(
                verify_generation=lambda *_args: {"verification_results": []}
            )

        def generate(self, _query, **_kwargs):
            doc_id = self.calls
            self.calls += 1
            answer = "Paris" if doc_id == 4 else "London"
            labels = ["ENTAILED"] if doc_id == 4 else ["ENTAILED", "REFUTED"]
            return {
                "retrieved_docs": [doc_id],
                "generated_text": answer,
                "verification_results": {
                    "verification_results": [
                        {"label": label, "is_entailed": label == "ENTAILED"}
                        for label in labels
                    ]
                },
            }

    monkeypatch.setattr(exp5, "SelfVerificationRAGPipeline", FakePipeline)
    monkeypatch.setattr(exp5, "resolve_device", lambda _device: "cpu")

    run = exp5.run_self_consistency_experiment(
        queries=["Which city?"],
        ground_truths=["London"],
        relevant_docs=[[4]],
        corpus=[f"passage {idx}" for idx in range(5)],
        config={"experiments": {"device": "cpu", "exp5": {
            "aggregation_method": "highest_factual_precision"
        }}},
        k=3,
        factual_precision_threshold=0.0,
    )

    assert run["config"]["aggregation_method"] == "highest_factual_precision"
    assert run["individual_results"]["self_consistency"][0]["generated"] == "Paris"
