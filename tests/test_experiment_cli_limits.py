"""Experiment entry points forward the effective sample limit to data loading."""

import importlib
import sys

import pytest


@pytest.mark.parametrize("module_name", [
    "exp1_baseline", "exp2_retrieval_comparison", "exp3_threshold_tuning",
    "exp4_revision_strategies", "exp5_self_consistency", "exp7_ablation_study",
    "exp8_stress_test", "exp9_complete_pipeline",
])
@pytest.mark.parametrize("extra_args,expected_limit", [
    ([], 2),
    (["--limit", "6"], 6),
    (["--dry-run", "--limit", "6"], 6),
])
def test_experiment_passes_effective_limit_to_loader(
    monkeypatch, module_name, extra_args, expected_limit
):
    module = importlib.import_module(f"experiments.{module_name}")
    monkeypatch.setattr(module, "load_config", lambda _path: {"datasets": {"sample_limit": 2}})
    monkeypatch.setenv("WANDB_DISABLED", "true")

    class ReachedLoader(Exception):
        pass

    def load_dataset(_config, *, split, limit):
        assert split == "validation"
        assert limit == expected_limit
        raise ReachedLoader

    monkeypatch.setattr(module, "load_dataset_from_config", load_dataset)
    monkeypatch.setattr(sys, "argv", [module_name, "--no-wandb", *extra_args])
    with pytest.raises(ReachedLoader):
        module.main()
