"""Checks that release records require consistent complete run artifacts."""

import csv
import hashlib
import json
import statistics

import pytest

from experiments.final_result_reader import RESULT_FILES
from scripts.create_results_lock import SUMMARY_METRICS, create_results_lock


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_payload(experiment, seed, value):
    metric = {"f1_score": {"mean": value}}
    payloads = {
        "exp1_baseline": {"aggregated_metrics": metric},
        "exp2_retrieval_comparison": {"aggregated_metrics": {"hybrid_rerank": metric}},
        "exp3_threshold_tuning": {
            "threshold_results": {"0.75": {"aggregated_metrics": metric}}
        },
        "exp4_revision_strategies": {"revision_metrics": metric},
        "exp5_self_consistency": {"aggregated_metrics": {"self_consistency": metric}},
        "exp6_iterative_training": {
            "iteration_results": {"3": {"metrics": {"f1_score": value}}},
            "total_iterations": 3,
        },
        "exp7_ablation_study": {"aggregated": {"full": metric},
                                "commit_hash": "12345678"},
        "exp8_stress_test": {"baseline": {"f1_score": value},
                             "commit_hash": "12345678"},
    }
    metadata = {
        "dataset": "squad_v2", "seed": seed, "split": "validation",
        "sample_limit": 5000, "total_queries": 10,
    }
    if experiment not in {"exp7_ablation_study", "exp8_stress_test"}:
        metadata["commit_hash"] = "12345678"
    return {**payloads[experiment], "metadata": metadata}


def complete_run(root):
    config_path = root / "config/config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "datasets:\n  active: squad_v2\n  sample_limit: 5000\n"
        "verification:\n  threshold: 0.75\n",
        encoding="utf-8",
    )
    run_dir = root / "results/metrics/final_runs/test-run"
    run_dir.mkdir(parents=True)
    seeds = [42, 123, 456]
    artifacts = {}
    aggregated = {}
    for experiment in RESULT_FILES:
        artifacts[experiment] = {}
        values = []
        for seed in seeds:
            value = seed / 1000
            values.append(value)
            path = run_dir / f"{experiment}_seed{seed}.json"
            path.write_text(json.dumps(artifact_payload(experiment, seed, value)),
                            encoding="utf-8")
            artifacts[experiment][str(seed)] = {
                "path": str(path.relative_to(root)),
                "sha256": hash_file(path),
                "metrics": {"f1_score": value},
            }
        aggregated[experiment] = {
            "f1_score": {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "values": values,
                "n": len(values),
            }
        }
    manifest = {
        "status": "complete", "failures": [], "dry_run": False, "limit": None,
        "experiments": list(RESULT_FILES), "seeds": seeds,
        "config_path": "config/config.yaml", "config_sha256": hash_file(config_path),
        "dataset": "squad_v2", "split": "validation", "selected_threshold": 0.75,
        "commit_hash": "12345678", "artifacts": artifacts,
        "aggregated_results": aggregated,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    canonical = root / "results/metrics/final_aggregated_results.json"
    canonical.write_text(json.dumps(manifest), encoding="utf-8")
    summary = root / "results/metrics/final_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", *SUMMARY_METRICS))
        writer.writeheader()
        for experiment in RESULT_FILES:
            stats = aggregated[experiment]["f1_score"]
            writer.writerow({
                "experiment": experiment,
                "f1_score": f"{stats['mean']:.4f} ± {stats['std']:.4f}",
            })
    return manifest_path


def test_complete_run_produces_a_traceable_lock(tmp_path):
    manifest = complete_run(tmp_path)
    output = tmp_path / "RESULTS_LOCK.md"

    create_results_lock(manifest, output, root=tmp_path)

    content = output.read_text(encoding="utf-8")
    assert "validated complete Exp1-8 run" in content
    assert "Human evaluation is separate" in content
    assert "Configured verification threshold: 0.75" in content
    assert "exp8_stress_test" in content
    with pytest.raises(FileExistsError, match="already exists"):
        create_results_lock(manifest, output, root=tmp_path)


def test_incomplete_manifest_never_creates_a_lock(tmp_path):
    manifest_path = complete_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "incomplete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "RESULTS_LOCK.md"

    with pytest.raises(ValueError, match="does not describe a complete run"):
        create_results_lock(manifest_path, output, root=tmp_path)
    assert not output.exists()


def test_changed_seed_artifact_or_config_is_rejected(tmp_path):
    manifest_path = complete_run(tmp_path)
    output = tmp_path / "RESULTS_LOCK.md"
    artifact = manifest_path.parent / "exp1_baseline_seed42.json"
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        create_results_lock(manifest_path, output, root=tmp_path)
    assert not output.exists()

    manifest_path = complete_run(tmp_path / "second")
    config = tmp_path / "second/config/config.yaml"
    config.write_text(config.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        create_results_lock(manifest_path, tmp_path / "second/RESULTS_LOCK.md",
                            root=tmp_path / "second")


@pytest.mark.parametrize("field,value,error", [
    ("dataset", "hotpotqa", "artifact dataset hotpotqa does not match squad_v2"),
    ("sample_limit", 100, "artifact sample_limit 100 does not match 5000"),
])
def test_release_rechecks_artifact_dataset_and_limit(tmp_path, field, value, error):
    manifest_path = complete_run(tmp_path)
    artifact = manifest_path.parent / "exp1_baseline_seed42.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["metadata"][field] = value
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["exp1_baseline"]["42"]["sha256"] = hash_file(artifact)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results/metrics/final_aggregated_results.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=error):
        create_results_lock(manifest_path, tmp_path / "RESULTS_LOCK.md", root=tmp_path)
    assert not (tmp_path / "RESULTS_LOCK.md").exists()


def test_stale_final_summary_is_rejected(tmp_path):
    manifest_path = complete_run(tmp_path)
    summary = tmp_path / "results/metrics/final_summary.csv"
    summary.write_text(summary.read_text(encoding="utf-8").replace("0.2070", "0.9999"),
                       encoding="utf-8")

    with pytest.raises(ValueError, match="final summary disagrees"):
        create_results_lock(manifest_path, tmp_path / "RESULTS_LOCK.md", root=tmp_path)


def test_manifest_metrics_must_match_archived_json(tmp_path):
    manifest_path = complete_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["exp1_baseline"]["42"]["metrics"]["f1_score"] = 0.99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    canonical = tmp_path / "results/metrics/final_aggregated_results.json"
    canonical.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics disagree with archived JSON"):
        create_results_lock(manifest_path, tmp_path / "RESULTS_LOCK.md", root=tmp_path)


def test_release_rejects_zero_training_iterations(tmp_path):
    manifest_path = complete_run(tmp_path)
    config = tmp_path / "config/config.yaml"
    config.write_text(config.read_text(encoding="utf-8") +
                      "experiments:\n  exp6:\n    iterations: 0\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_sha256"] = hash_file(config)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results/metrics/final_aggregated_results.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="positive Experiment 6 training iterations"):
        create_results_lock(manifest_path, tmp_path / "RESULTS_LOCK.md", root=tmp_path)


def test_release_rejects_wrong_training_iteration_artifact(tmp_path):
    manifest_path = complete_run(tmp_path)
    artifact = manifest_path.parent / "exp6_iterative_training_seed42.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["total_iterations"] = 1
    payload["iteration_results"] = {"1": {"metrics": {"f1_score": 0.042}}}
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["exp6_iterative_training"]["42"]["sha256"] = hash_file(artifact)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results/metrics/final_aggregated_results.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="do not match configured 3"):
        create_results_lock(manifest_path, tmp_path / "RESULTS_LOCK.md", root=tmp_path)


def test_release_rejects_an_artifact_from_another_commit(tmp_path):
    manifest_path = complete_run(tmp_path)
    artifact = manifest_path.parent / "exp7_ablation_study_seed42.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["commit_hash"] = "87654321"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["exp7_ablation_study"]["42"]["sha256"] = hash_file(artifact)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results/metrics/final_aggregated_results.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="artifact commit does not match"):
        create_results_lock(manifest_path, tmp_path / "RESULTS_LOCK.md", root=tmp_path)
