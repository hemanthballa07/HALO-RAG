"""Experiment artifacts retain the run identity checked by the final reader."""

import json
from types import SimpleNamespace

import pytest

from experiments import exp4_revision_strategies as exp4
from experiments import exp7_ablation_study as exp7
from experiments import exp8_stress_test as exp8


METADATA = {
    "dataset": "squad_v2",
    "split": "validation",
    "sample_limit": 2,
    "total_queries": 2,
    "seed": 42,
}


def test_revision_artifact_keeps_run_metadata(tmp_path, monkeypatch):
    class FakePipeline:
        def __init__(self, **kwargs):
            self.revision = kwargs["enable_revision"]

        def evaluate(self, query, _ground_truth, _relevant_docs):
            score = (0.7 if self.revision else 0.4)
            if query == "second":
                score += 0.15 if self.revision else 0.1
            return {
                "metrics": {
                    "factual_precision": score,
                    "hallucination_rate": 1 - score,
                    "verified_f1": score,
                    "f1_score": score,
                },
                "revision_iterations": int(self.revision),
            }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(exp4, "SelfVerificationRAGPipeline", FakePipeline)
    monkeypatch.setattr(exp4, "resolve_device", lambda _device: "cpu")
    exp4.run_revision_strategies_experiment(
        queries=["first", "second"],
        ground_truths=["a", "b"],
        relevant_docs=[[0], [1]],
        corpus=["a", "b"],
        config={"experiments": {"device": "cpu"}, "generation": {
            "qlora": {"training_enabled": False}}, "revision": {"max_iterations": 1}},
        metadata=METADATA,
    )

    result = json.loads((tmp_path / "results/metrics/exp4_revision_strategies.json").read_text())
    assert result["metadata"] == METADATA


def test_ablation_artifact_keeps_run_metadata(tmp_path):
    exp7.save_results(
        {"aggregated": {"full": {"verified_f1": {"mean": 0.5}}},
         "drops": {}, "variants": ["full"]},
        metadata=METADATA,
        output_dir=str(tmp_path),
    )

    result = json.loads((tmp_path / "exp7_ablation.json").read_text())
    assert result["metadata"] == METADATA


def test_stress_artifact_keeps_run_metadata(tmp_path):
    exp8.save_stress_test_results(
        tau_results={}, retrieval_results={}, baseline_results={}, verifier_off_results={},
        metadata=METADATA, output_dir=str(tmp_path),
    )

    result = json.loads((tmp_path / "exp8_stress.json").read_text())
    assert result["metadata"] == METADATA


def test_ablation_fails_when_a_query_fails(monkeypatch):
    class FailingPipeline:
        def __init__(self, **_kwargs):
            pass

        def generate(self, *_args, **_kwargs):
            raise RuntimeError("generation failed")

    monkeypatch.setattr(exp7, "AblationPipeline", FailingPipeline)
    monkeypatch.setattr(exp7, "resolve_device", lambda _device: "cpu")
    with pytest.raises(RuntimeError, match="Failed to process query 0 in full"):
        exp7.run_ablation_study(
            queries=["question"], ground_truths=["answer"], relevant_docs=[[0]],
            corpus=["answer"], config={"experiments": {"device": "cpu"}},
            limit=1,
        )


@pytest.mark.parametrize("runner_name,extra,expected", [
    ("run_tau_sweep_stress_test", {"thresholds": [0.75]}, "τ-sweep failed"),
    ("run_retrieval_degradation_test", {"target_recalls": [0.95]},
     "Retrieval degradation failed"),
    ("run_verifier_off_test", {}, "Verifier-off evaluation failed"),
    ("run_baseline_test", {}, "Stress-test baseline evaluation failed"),
])
def test_stress_tests_fail_when_a_query_fails(monkeypatch, runner_name, extra, expected):
    class FailingPipeline:
        def __init__(self, **_kwargs):
            self.verifier = SimpleNamespace(threshold=0.75)

        def set_entailment_threshold(self, _threshold):
            pass

        def generate(self, *_args, **_kwargs):
            raise RuntimeError("generation failed")

    monkeypatch.setattr(exp8, "SelfVerificationRAGPipeline", FailingPipeline)
    monkeypatch.setattr(exp8, "DegradedRetrievalPipeline", FailingPipeline)
    monkeypatch.setattr(exp8, "resolve_device", lambda _device: "cpu")
    runner = getattr(exp8, runner_name)
    with pytest.raises(RuntimeError, match=expected):
        runner(
            queries=["question"], ground_truths=["answer"], relevant_docs=[[0]],
            corpus=["answer"], config={"experiments": {"device": "cpu"}},
            limit=1, **extra,
        )
