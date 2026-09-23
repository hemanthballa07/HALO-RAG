"""Sample limits used by experiment commands."""

import importlib
import sys

import pytest

from src.utils.cli import resolve_sample_limit


@pytest.mark.parametrize("requested,dry_run,cap,configured,expected", [
    (2, True, 30, 1000, 2),
    (100, True, 30, 1000, 30),
    (None, True, 20, 1000, 20),
    (2, False, 30, 1000, 2),
    (None, False, 30, 1000, 1000),
    (None, False, 30, None, None),
])
def test_resolve_sample_limit(requested, dry_run, cap, configured, expected):
    assert resolve_sample_limit(requested, dry_run, cap, configured) == expected


@pytest.mark.parametrize("requested", [0, -1])
def test_resolve_sample_limit_rejects_nonpositive_values(requested):
    with pytest.raises(ValueError, match="--limit must be a positive integer"):
        resolve_sample_limit(requested, True, 30)


@pytest.mark.parametrize("module_name", [
    "exp1_baseline", "exp2_retrieval_comparison", "exp3_threshold_tuning",
    "exp4_revision_strategies", "exp5_self_consistency", "exp7_ablation_study",
    "exp8_stress_test",
])
@pytest.mark.parametrize("extra_args,expected_limit", [([], 2), (["--limit", "6"], 6)])
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
