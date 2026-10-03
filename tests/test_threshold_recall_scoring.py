"""Threshold sweeps must score ground-truth claims against retrieved context."""

from types import SimpleNamespace

import pytest

from experiments import exp3_threshold_tuning as exp3
from experiments import exp8_stress_test as exp8


@pytest.mark.parametrize("experiment", [exp3, exp8])
def test_threshold_sweep_scores_factual_recall(monkeypatch, experiment):
    class FakePipeline:
        def __init__(self, **_kwargs):
            self.claim_extractor = SimpleNamespace(extract_claims=lambda text: [text])
            self.verifier = self
            self.threshold = 0.0

        def set_entailment_threshold(self, threshold):
            self.threshold = threshold

        def is_entailed(self, claim, context):
            assert claim == "Paris is in France."
            assert context == "Paris is in France."
            return self.threshold < 0.5, 1.0

        def generate(self, _query, **_kwargs):
            return {
                "retrieved_docs": [0],
                "reranked_texts": ["Paris is in France."],
                "verification_results": {"verification_results": []},
                "generated_text": "Paris",
            }

    monkeypatch.setattr(experiment, "SelfVerificationRAGPipeline", FakePipeline)
    monkeypatch.setattr(experiment, "resolve_device", lambda _device: "cpu")
    kwargs = {
        "queries": ["Where is Paris?"],
        "ground_truths": ["Paris is in France."],
        "relevant_docs": [[0]],
        "corpus": ["Paris is in France."],
        "config": {"experiments": {"device": "cpu"}},
        "thresholds": [0.4, 0.8],
    }

    if experiment is exp3:
        results = exp3.run_threshold_tuning(**kwargs)["threshold_results"]
        recall = lambda threshold: results[threshold]["aggregated_metrics"]["factual_recall"]["mean"]
    else:
        results = exp8.run_tau_sweep_stress_test(**kwargs)
        recall = lambda threshold: results[threshold]["factual_recall"]

    assert recall(0.4) == 1.0
    assert recall(0.8) == 0.0
