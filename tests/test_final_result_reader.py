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
        "metadata": {"seed": 42, "split": "validation", "sample_limit": 2,
                     "total_queries": 2},
        "aggregated_metrics": {"f1_score": {"mean": 0.5}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact seed 42 does not match 123"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=123)
    assert load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=42)["f1_score"] == 0.5


def test_reader_requires_provenance_for_archived_runs(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps({
        "aggregated_metrics": {"f1_score": {"mean": 0.5}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="missing run metadata"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=42,
                     expected_split="validation")

    artifact.write_text(json.dumps({
        "metadata": {"seed": 42, "split": "validation", "sample_limit": 2},
        "aggregated_metrics": {"f1_score": {"mean": 0.5}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="missing total_queries"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=42,
                     expected_split="validation")

    artifact.write_text(json.dumps({
        "metadata": {"seed": 42, "split": "validation", "sample_limit": 2,
                     "total_queries": 2},
        "total_queries": 3,
        "aggregated_metrics": {"f1_score": {"mean": 0.5}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="query count disagrees"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=42,
                     expected_split="validation")


def test_reader_checks_dataset_and_sample_limit(tmp_path):
    artifact = tmp_path / "result.json"
    payload = {
        "metadata": {"dataset": "squad_v2", "seed": 42, "split": "validation",
                     "sample_limit": 2, "total_queries": 2},
        "aggregated_metrics": {"f1_score": {"mean": 0.5}},
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact dataset squad_v2 does not match hotpotqa"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_dataset="hotpotqa")
    with pytest.raises(ValueError, match="artifact sample_limit 2 does not match 3"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_sample_limit=3,
                     check_sample_limit=True)
    assert load_metrics(artifact, "exp1_baseline", 0.75, expected_dataset="squad_v2",
                        expected_sample_limit=2, check_sample_limit=True)["f1_score"] == 0.5

    payload["metadata"]["total_queries"] = 3
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="query count exceeds sample_limit"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=42)

    payload["metadata"]["sample_limit"] = True
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid sample_limit"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_seed=42)


def test_reader_checks_unlimited_run_metadata(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps({
        "metadata": {"dataset": "squad_v2", "seed": 42, "split": "validation",
                     "sample_limit": 2, "total_queries": 2},
        "aggregated_metrics": {"f1_score": {"mean": 0.5}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact sample_limit 2 does not match None"):
        load_metrics(artifact, "exp1_baseline", 0.75, expected_sample_limit=None,
                     check_sample_limit=True)


def test_runner_archives_each_seed_before_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    source = tmp_path / "results/metrics" / RESULT_FILES["exp1_baseline"]
    source.parent.mkdir(parents=True)

    def run_experiment(_name, seed, *_args):
        source.write_text(json.dumps({
            "metadata": {"seed": seed, "split": "validation", "sample_limit": None,
                         "total_queries": 10},
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


def test_runner_rejects_an_artifact_with_the_wrong_configured_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    source = tmp_path / "results/metrics" / RESULT_FILES["exp1_baseline"]
    source.parent.mkdir(parents=True)

    def run_experiment(*_args):
        source.write_text(json.dumps({
            "metadata": {"dataset": "squad_v2", "seed": 42, "split": "validation",
                         "sample_limit": 1, "total_queries": 1},
            "aggregated_metrics": {"f1_score": {"mean": 0.5}},
        }), encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr(runner, "run_experiment", run_experiment)
    results, failures, artifacts = runner.aggregate_results_across_seeds(
        ["exp1_baseline"], [42], archive_dir=tmp_path / "archive",
        expected_dataset="squad_v2", configured_limit=2,
    )

    assert results == {}
    assert "sample_limit 1 does not match 2" in failures[0]
    assert artifacts["exp1_baseline"] == {}


@pytest.mark.parametrize("write_plot", [False, True])
def test_runner_requires_a_fresh_plot_for_each_seed(tmp_path, monkeypatch, write_plot):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    source = tmp_path / "results/metrics" / RESULT_FILES["exp2_retrieval_comparison"]
    source.parent.mkdir(parents=True)
    plot = tmp_path / "results/figures/exp2_retrieval_bars.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"old plot")

    def run_experiment(_name, seed, *_args):
        source.write_text(json.dumps({
            "metadata": {"seed": seed, "split": "validation", "sample_limit": None,
                         "total_queries": 10},
            "aggregated_metrics": {"hybrid_rerank": {"f1_score": {"mean": 0.5}}},
        }), encoding="utf-8")
        if write_plot:
            plot.write_bytes(b"new plot")
        return {"status": "success"}

    monkeypatch.setattr(runner, "run_experiment", run_experiment)
    results, failures, artifacts = runner.aggregate_results_across_seeds(
        ["exp2_retrieval_comparison"], [42], archive_dir=tmp_path / "archive",
        require_plots=True,
    )

    if write_plot:
        assert failures == []
        assert results["exp2_retrieval_comparison"]["f1_score"]["values"] == [0.5]
        assert "42" in artifacts["exp2_retrieval_comparison"]
    else:
        assert failures == [
            "exp2_retrieval_comparison (seed 42): no fresh plot exp2_retrieval_bars.png"
        ]
        assert results == {}
        assert artifacts["exp2_retrieval_comparison"] == {}


def test_runner_rejects_a_plot_missing_from_a_later_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    source = tmp_path / "results/metrics" / RESULT_FILES["exp2_retrieval_comparison"]
    source.parent.mkdir(parents=True)
    plot = tmp_path / "results/figures/exp2_retrieval_bars.png"
    plot.parent.mkdir(parents=True)

    def run_experiment(_name, seed, *_args):
        source.write_text(json.dumps({
            "metadata": {"seed": seed, "split": "validation", "sample_limit": None,
                         "total_queries": 10},
            "aggregated_metrics": {"hybrid_rerank": {"f1_score": {"mean": seed / 100}}},
        }), encoding="utf-8")
        if seed == 42:
            plot.write_bytes(b"first seed")
        return {"status": "success"}

    monkeypatch.setattr(runner, "run_experiment", run_experiment)
    results, failures, artifacts = runner.aggregate_results_across_seeds(
        ["exp2_retrieval_comparison"], [42, 123], archive_dir=tmp_path / "archive",
        require_plots=True,
    )

    assert failures == [
        "exp2_retrieval_comparison (seed 123): no fresh plot exp2_retrieval_bars.png"
    ]
    assert results["exp2_retrieval_comparison"]["f1_score"]["values"] == [0.42]
    assert list(artifacts["exp2_retrieval_comparison"]) == ["42"]


def test_runner_records_the_subprocess_failure_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    monkeypatch.setattr(
        runner, "run_experiment",
        lambda *_args: {"status": "error", "error": "Traceback\nPermissionError: cache is read-only\n"},
    )

    results, failures, artifacts = runner.aggregate_results_across_seeds(
        ["exp1_baseline"], [42], archive_dir=tmp_path / "archive"
    )

    assert results == {}
    assert failures == ["exp1_baseline (seed 42): experiment failed: PermissionError: cache is read-only"]
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
    config = tmp_path / "config/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("verification:\n  threshold: 0.75\ndatasets:\n  active: squad_v2\n",
                      encoding="utf-8")

    def run_experiment(_name, _seed, _config_path, _split, _limit, dry_run):
        if write_result:
            source.write_text(json.dumps({
                "metadata": {"dataset": "squad_v2", "seed": 42, "split": "validation",
                             "sample_limit": 30 if dry_run else None, "total_queries": 10},
                "aggregated_metrics": {"f1_score": {"mean": 0.5}},
            }), encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr(runner, "run_experiment", run_experiment)
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


def test_configuration_change_marks_run_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    config = tmp_path / "config/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("verification:\n  threshold: 0.75\ndatasets:\n  active: squad_v2\n",
                      encoding="utf-8")
    source = tmp_path / "results/metrics" / RESULT_FILES["exp1_baseline"]
    source.parent.mkdir(parents=True)

    def run_experiment(*_args):
        source.write_text(json.dumps({
            "metadata": {"dataset": "squad_v2", "seed": 42, "split": "validation",
                         "sample_limit": None, "total_queries": 10},
            "aggregated_metrics": {"f1_score": {"mean": 0.5}},
        }), encoding="utf-8")
        config.write_text(config.read_text(encoding="utf-8") + "# changed\n",
                          encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr(runner, "run_experiment", run_experiment)
    monkeypatch.setattr(sys, "argv", ["run_final_experiments.py", "--experiments", "exp1_baseline",
                                       "--seeds", "42"])

    assert runner.main() == 1
    manifest_path = next((tmp_path / "results/metrics/final_runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert "configuration changed during the run" in manifest["failures"]


def test_runner_checks_training_hardware_before_creating_a_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    config = tmp_path / "config/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "verification:\n  threshold: 0.75\n"
        "datasets:\n  active: squad_v2\n"
        "experiments:\n  device: auto\n  exp6:\n    iterations: 3\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "resolve_device", lambda _preferred: "cpu")
    monkeypatch.setattr(runner, "qlora_supported", lambda _device: False)
    monkeypatch.setattr(sys, "argv", [
        "run_final_experiments.py", "--experiments", "exp1_baseline",
        "exp6_iterative_training", "--dry-run",
    ])

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2
    assert "needs CUDA and bitsandbytes" in capsys.readouterr().err
    assert not (tmp_path / "results/metrics/final_runs").exists()


def test_full_run_requires_training_iterations():
    config = {"experiments": {"exp6": {"iterations": 0}}}
    with pytest.raises(ValueError, match="at least one"):
        runner.check_training_readiness(config, ["exp6_iterative_training"],
                                        "validation", diagnostic=False)
    runner.check_training_readiness(config, ["exp6_iterative_training"],
                                    "validation", diagnostic=True)


def test_reader_rejects_a_short_training_artifact(tmp_path):
    artifact = tmp_path / "exp6.json"
    artifact.write_text(json.dumps({
        "metadata": {"seed": 42, "split": "validation", "sample_limit": 2,
                     "total_queries": 2},
        "iteration_results": {"1": {"metrics": {"f1_score": 0.5}}},
        "total_iterations": 1,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match configured 3"):
        load_metrics(artifact, "exp6_iterative_training", 0.75, expected_seed=42,
                     expected_split="validation", expected_iterations=3)


def test_full_run_rejects_dirty_source_before_creating_a_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    config = tmp_path / "config/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "verification:\n  threshold: 0.75\n"
        "datasets:\n  active: squad_v2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "check_training_readiness", lambda *_args: None)
    monkeypatch.setattr(runner, "repository_commit", lambda _root: "12345678")
    monkeypatch.setattr(runner, "repository_is_clean", lambda _root: False)
    monkeypatch.setattr(sys, "argv", ["run_final_experiments.py"])

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2
    assert "clean Git worktree" in capsys.readouterr().err
    assert not (tmp_path / "results/metrics/final_runs").exists()


def test_full_run_marks_midrun_source_changes_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "project_root", tmp_path)
    config = tmp_path / "config/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "verification:\n  threshold: 0.75\n"
        "datasets:\n  active: squad_v2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "check_training_readiness", lambda *_args: None)
    monkeypatch.setattr(runner, "repository_commit", lambda _root: "12345678")
    clean_states = iter([True, False])
    monkeypatch.setattr(runner, "repository_is_clean", lambda _root: next(clean_states))

    def aggregate_results_across_seeds(**kwargs):
        kwargs["archive_dir"].mkdir(parents=True)
        return {}, [], {}

    monkeypatch.setattr(runner, "aggregate_results_across_seeds", aggregate_results_across_seeds)
    monkeypatch.setattr(sys, "argv", ["run_final_experiments.py"])

    assert runner.main() == 1
    manifest_path = next((tmp_path / "results/metrics/final_runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert "repository worktree changed during the run" in manifest["failures"]
    assert not (tmp_path / "results/metrics/final_summary.csv").exists()
