"""Experiment artifacts retain the run identity checked by the final reader."""

import json
import sys
from types import SimpleNamespace

import pytest

from experiments import exp1_baseline as exp1
from experiments import exp3_threshold_tuning as exp3
from experiments import exp4_revision_strategies as exp4
from experiments import exp5_self_consistency as exp5
from experiments import exp6_iterative_training as exp6
from experiments import exp7_ablation_study as exp7
from experiments import exp8_stress_test as exp8
from experiments import exp9_complete_pipeline as exp9


METADATA = {
    "dataset": "squad_v2",
    "split": "validation",
    "sample_limit": 2,
    "total_queries": 2,
    "seed": 42,
}


def test_baseline_fails_when_a_query_fails(monkeypatch):
    class FailingPipeline:
        def __init__(self, **_kwargs):
            self.claim_extractor = SimpleNamespace(extract_claims=lambda _text: [])
            self.verifier = SimpleNamespace()

        def generate(self, query, **_kwargs):
            if query == "second":
                raise RuntimeError("generation failed")
            return {
                "generated_text": "answer",
                "retrieved_docs": [0],
                "verification_results": {"verification_results": []},
            }

    class FakeEvaluator:
        def compute_all_metrics(self, **_kwargs):
            return {"f1_score": 1.0}

    monkeypatch.setattr(exp1, "SelfVerificationRAGPipeline", FailingPipeline)
    monkeypatch.setattr(exp1, "EvaluationMetrics", FakeEvaluator)
    monkeypatch.setattr(exp1, "resolve_device", lambda _device: "cpu")

    with pytest.raises(RuntimeError, match="Baseline failed to process query 1"):
        exp1.run_baseline_experiment(
            queries=["first", "second"],
            ground_truths=["answer", "answer"],
            relevant_docs=[[0], [0]],
            corpus=["answer"],
            config={"experiments": {"device": "cpu"}},
        )


def test_complete_pipeline_fails_when_a_query_fails(monkeypatch):
    class FailingPipeline:
        def __init__(self, **_kwargs):
            self.claim_extractor = SimpleNamespace(extract_claims=lambda _text: [])
            self.verifier = SimpleNamespace()

        def generate(self, query, **_kwargs):
            if query == "second":
                raise RuntimeError("generation failed")
            return {
                "generated_text": "answer",
                "retrieved_docs": [0],
                "verification_results": {"verification_results": []},
            }

    class FakeEvaluator:
        def compute_all_metrics(self, **_kwargs):
            return {"f1_score": 1.0}

    monkeypatch.setattr(exp9, "SelfVerificationRAGPipeline", FailingPipeline)
    monkeypatch.setattr(exp9, "EvaluationMetrics", FakeEvaluator)
    monkeypatch.setattr(exp9, "resolve_device", lambda _device: "cpu")

    with pytest.raises(RuntimeError, match="Complete pipeline failed to process query 1"):
        exp9.run_complete_pipeline_experiment(
            queries=["first", "second"],
            ground_truths=["answer", "answer"],
            relevant_docs=[[0], [0]],
            corpus=["answer"],
            config={"experiments": {"device": "cpu"}},
        )


def test_threshold_tuning_fails_when_a_query_fails(monkeypatch):
    class FailingPipeline:
        def __init__(self, **_kwargs):
            self.claim_extractor = SimpleNamespace(extract_claims=lambda _text: [])
            self.verifier = SimpleNamespace()

        def set_entailment_threshold(self, _threshold):
            pass

        def generate(self, query, **_kwargs):
            if query == "second":
                raise RuntimeError("generation failed")
            return {
                "generated_text": "answer",
                "retrieved_docs": [0],
                "verification_results": {"verification_results": []},
            }

    class FakeEvaluator:
        def compute_all_metrics(self, **_kwargs):
            return {"f1_score": 1.0}

    monkeypatch.setattr(exp3, "SelfVerificationRAGPipeline", FailingPipeline)
    monkeypatch.setattr(exp3, "EvaluationMetrics", FakeEvaluator)
    monkeypatch.setattr(exp3, "resolve_device", lambda _device: "cpu")

    with pytest.raises(RuntimeError, match="Threshold 0.75 failed to process query 1"):
        exp3.run_threshold_tuning(
            queries=["first", "second"],
            ground_truths=["answer", "answer"],
            relevant_docs=[[0], [0]],
            corpus=["answer"],
            config={"experiments": {"device": "cpu"}},
            thresholds=[0.75],
        )


def test_self_consistency_fails_when_a_query_fails(monkeypatch):
    class FailingPipeline:
        def __init__(self, **_kwargs):
            self.claim_extractor = SimpleNamespace(extract_claims=lambda _text: [])
            self.verifier = SimpleNamespace(
                verify_generation=lambda *_args: {"verification_results": []}
            )

        def generate(self, query, **_kwargs):
            if query == "second":
                raise RuntimeError("generation failed")
            return {
                "generated_text": "answer",
                "retrieved_docs": [0],
                "verification_results": {"verification_results": []},
            }

    class FakeEvaluator:
        def compute_all_metrics(self, **_kwargs):
            return {"f1_score": 1.0}

    monkeypatch.setattr(exp5, "SelfVerificationRAGPipeline", FailingPipeline)
    monkeypatch.setattr(exp5, "EvaluationMetrics", FakeEvaluator)
    monkeypatch.setattr(exp5, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(exp5, "generate_with_self_consistency", lambda *_args, **_kwargs: {
        "final_answer": "answer",
        "samples": [{"verified_f1": 1.0, "retrieved_docs": [0], "retrieved_texts": []}],
        "filtered_samples": [{"verified_f1": 1.0, "retrieved_docs": [0], "retrieved_texts": []}],
    })

    with pytest.raises(RuntimeError, match="Self-consistency failed to process query 1"):
        exp5.run_self_consistency_experiment(
            queries=["first", "second"],
            ground_truths=["answer", "answer"],
            relevant_docs=[[0], [0]],
            corpus=["answer"],
            config={"experiments": {"device": "cpu"}},
        )


def test_iterative_validation_fails_when_a_query_fails():
    class FailingPipeline:
        def generate(self, query, **_kwargs):
            if query == "second":
                raise RuntimeError("generation failed")
            return {
                "generated_text": "answer",
                "retrieved_docs": [0],
                "verification_results": {"verification_results": []},
            }

    class FakeEvaluator:
        def compute_all_metrics(self, **_kwargs):
            return {"f1_score": 1.0}

    with pytest.raises(RuntimeError, match="Validation failed to process query 1"):
        exp6.evaluate_iteration(
            pipeline=FailingPipeline(),
            queries=["first", "second"],
            ground_truths=["answer", "answer"],
            relevant_docs=[[0], [0]],
            corpus=["answer"],
            evaluator=FakeEvaluator(),
        )


def test_iterative_validation_runs_when_module_is_imported():
    class FakePipeline:
        def generate(self, _query, **_kwargs):
            return {
                "generated_text": "answer",
                "retrieved_docs": [0],
                "verification_results": {"verification_results": []},
            }

    class FakeEvaluator:
        def compute_all_metrics(self, **_kwargs):
            return {"f1_score": 1.0}

    metrics = exp6.evaluate_iteration(
        pipeline=FakePipeline(),
        queries=["question"],
        ground_truths=["answer"],
        relevant_docs=[[0]],
        corpus=["answer"],
        evaluator=FakeEvaluator(),
    )
    assert metrics["f1_score"] == 1.0


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


def test_iterative_training_dry_run_caps_both_splits(monkeypatch, capsys):
    calls = []
    saved = []
    config = {
        "datasets": {"active": "squad_v2"},
        "experiments": {"device": "cpu", "exp6": {"train_limit": 10000}},
    }

    def load_examples(_config, split, limit):
        calls.append((split, limit))
        return [{"id": split, "question": split, "context": split}]

    def prepare_examples(examples):
        return (
            [example["question"] for example in examples],
            ["answer"] * len(examples),
            [[index] for index in range(len(examples))],
            [example["context"] for example in examples],
        )

    monkeypatch.setattr(exp6, "load_config", lambda _path: config)
    monkeypatch.setattr(exp6, "resolve_device", lambda _preferred: "cpu")
    monkeypatch.setattr(exp6, "load_dataset_from_config", load_examples)
    monkeypatch.setattr(exp6, "prepare_for_experiments", prepare_examples)
    monkeypatch.setattr(exp6, "run_iterative_training", lambda **_kwargs: {
        "iteration_results": {0: {"metrics": {"f1_score": 0.5}}},
        "total_iterations": 0,
    })
    monkeypatch.setattr(exp6, "save_results", lambda results: saved.append(results))
    monkeypatch.setattr(exp6, "plot_iteration_curves", lambda _results: None)
    monkeypatch.setattr(sys, "argv", [
        "exp6_iterative_training.py", "--iterations", "0", "--limit", "2",
        "--dry-run", "--no-wandb",
    ])

    exp6.main()

    assert calls == [("train", 100), ("validation", 2)]
    assert saved[0]["metadata"]["train_limit"] == 100
    assert saved[0]["metadata"]["val_limit"] == 2
    output = capsys.readouterr().out
    assert "from dry-run cap" in output
    assert "Acceptance criteria are not evaluated for a baseline-only run" in output


def test_empty_verified_set_stops_training_before_writing(monkeypatch):
    writes = []
    monkeypatch.setattr(exp6, "collect_verified_data", lambda **_kwargs: [])
    monkeypatch.setattr(exp6, "save_verified_data", lambda *_args: writes.append(True))

    with pytest.raises(RuntimeError, match="Iteration 1 collected no verified training examples"):
        exp6.collect_verified_training_data(
            pipeline=None,
            queries=["question"],
            ground_truths=["answer"],
            relevant_docs=[[0]],
            corpus=["answer"],
            iteration=1,
            config={"verification": {"accept_min": 0.85}},
        )

    assert writes == []
