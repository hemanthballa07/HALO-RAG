"""Checks for fresh per-seed artifacts in the final experiment runner."""

import json
import sys

import pytest

from experiments.final_result_reader import RESULT_FILES, load_metrics, select_metrics
from experiments import run_final_experiments as runner


@pytest.mark.parametrize("experiment,payload,expected", [
    ("exp1_baseline", {"aggregated_metrics": {"f1_score": {"mean": 0.1}}}, 0.1),
    ("exp2_retrieval_comparison", {
        "aggregated_metrics": {"hybrid_rerank": {"f1_score": {"mean": 0.2}}}
    }, 0.2),
    ("exp3_threshold_tuning", {
        "threshold_results": {"0.75": {"aggregated_metrics": {"f1_score": {"mean": 0.3}}}}
    }, 0.3),
    ("exp4_revision_strategies", {"revision_metrics": {"f1_score": {"mean": 0.4}}}, 0.4),
    ("exp5_self_consistency", {
        "aggregated_metrics": {"self_consistency": {"f1_score": {"mean": 0.5}}}
    }, 0.5),
    ("exp6_iterative_training", {
        "iteration_results": {"0": {"metrics": {"f1_score": 0.1}},
                              "2": {"metrics": {"f1_score": 0.6}}},
        "total_iterations": 2,
    }, 0.6),
    ("exp7_ablation_study", {"aggregated": {"full": {"f1_score": {"mean": 0.7}}}}, 0.7),
    ("exp8_stress_test", {"baseline": {"f1_score": 0.8}}, 0.8),
])
def test_reader_selects_the_right_saved_variant(experiment, payload, expected):
    assert select_metrics(experiment, payload, 0.75)["f1_score"] == expected


def test_reader_rejects_missing_threshold_and_empty_metrics():
    with pytest.raises(ValueError, match="exactly one result for threshold"):
        select_metrics("exp3_threshold_tuning", {"threshold_results": {}}, 0.75)
    with pytest.raises(ValueError, match="no numeric metrics"):
        select_metrics("exp8_stress_test", {"baseline": {}}, 0.75)
    with pytest.raises(ValueError, match="did not process every query"):
        select_metrics("exp1_baseline", {
            "aggregated_metrics": {"f1_score": {"mean": 0.5}},
            "total_queries": 10, "processed_queries": 9,
        }, 0.75)


def test_reader_checks_embedded_seed_when_available(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps({
        "metadata": {"seed": 42, "split": "validation"},
        "aggregated_metrics": {"f1_score": {"mean": 0.5}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact seed 42 does not match 123"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=123)
    assert load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=42)["f1_score"] == 0.5


def test_runner_archives_each_seed_before_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    source = tmp_path / "results/metrics" / RESULT_FILES["exp1_baseline"]
    source.parent.mkdir(parents=True)

    def run_experiment(_name, seed, *_args):
        source.write_text(json.dumps({
            "aggregated_metrics": {"f1_score": {"mean": seed / 10}}
        }), encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr(runner, "run_experiment", run_experiment)
    results, failures, artifacts = runner.aggregate_results_across_seeds(
        ["exp1_baseline"], [4, 6], archive_dir=tmp_path / "archive"
    )

    assert failures == []
    assert results["exp1_baseline"]["f1_score"]["values"] == [0.4, 0.6]
    assert artifacts["exp1_baseline"]["4"]["metrics"]["f1_score"] == 0.4
    assert (tmp_path / artifacts["exp1_baseline"]["4"]["path"]).exists()
    assert (tmp_path / artifacts["exp1_baseline"]["6"]["path"]).exists()


def test_runner_rejects_a_success_without_new_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    source = tmp_path / "results/metrics" / RESULT_FILES["exp1_baseline"]
    source.parent.mkdir(parents=True)
    source.write_text('{"aggregated_metrics": {"f1_score": {"mean": 0.9}}}', encoding="utf-8")
    monkeypatch.setattr(runner, "run_experiment", lambda *_args: {"status": "success"})

    results, failures, artifacts = runner.aggregate_results_across_seeds(
        ["exp1_baseline"], [42], archive_dir=tmp_path / "archive"
    )

    assert results == {}
    assert "no fresh metrics artifact" in failures[0]
    assert artifacts["exp1_baseline"] == {}


@pytest.mark.parametrize("write_result,extra_args,expected_status,exit_code", [
    (False, [], "incomplete", 1),
    (True, [], "diagnostic", 0),
    (True, ["--dry-run"], "diagnostic", 0),
])
def test_main_does_not_publish_incomplete_or_diagnostic_results(
    tmp_path, monkeypatch, write_result, extra_args, expected_status, exit_code
):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    source = tmp_path / "results/metrics" / RESULT_FILES["exp1_baseline"]
    source.parent.mkdir(parents=True)

    def run_experiment(_name, _seed, *_args):
        if write_result:
            source.write_text('{"aggregated_metrics": {"f1_score": {"mean": 0.5}}}',
                              encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr(runner, "run_experiment", run_experiment)
    monkeypatch.setattr(runner, "load_config", lambda _path: {"verification": {"threshold": 0.75}})
    monkeypatch.setattr(sys, "argv", ["run_final_experiments.py", "--experiments", "exp1_baseline",
                                       "--seeds", "42", *extra_args])

    assert runner.main() == exit_code
    manifests = list((tmp_path / "results/metrics/final_runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding="utf-8"))["status"] == expected_status
    assert not (tmp_path / "results/metrics/final_summary.csv").exists()


def test_missing_plot_set_is_not_partly_published(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    figures = tmp_path / "results/figures"
    figures.mkdir(parents=True)
    (figures / "exp2_retrieval_bars.png").write_bytes(b"plot")

    missing = runner.copy_key_plots_to_final()

    assert missing
    assert not (figures / "final").exists()
