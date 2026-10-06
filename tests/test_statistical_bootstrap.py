"""Bootstrap methods must report the interval they claim to compute."""

import json

import numpy as np
import pytest

from src.evaluation.statistical_testing import StatisticalTester


def test_bca_differs_from_percentile_on_skewed_data():
    values = [0.0] * 20 + [1.0, 2.0, 10.0]
    tester = StatisticalTester()
    state = np.random.get_state()
    try:
        np.random.seed(7)
        percentile = tester.bootstrap_ci(values, n_iterations=5000, method="percentile")
        np.random.seed(7)
        bca = tester.bootstrap_ci(values, n_iterations=5000, method="bca")
        np.random.seed(7)
        assert tester.bootstrap_ci(values, n_iterations=5000, method="bca") == bca
    finally:
        np.random.set_state(state)

    assert bca[0] > percentile[0]
    assert bca[1] > percentile[1]


def test_bca_constant_data_has_a_point_interval():
    assert StatisticalTester().bootstrap_ci([2.0, 2.0, 2.0], method="bca") == (2.0, 2.0)


@pytest.mark.parametrize("kwargs,expected", [
    ({"confidence": 0.0}, "confidence"),
    ({"confidence": 1.0}, "confidence"),
    ({"n_iterations": 0}, "n_iterations"),
    ({"method": "unknown"}, "Unknown method"),
])
def test_bootstrap_rejects_invalid_options(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        StatisticalTester().bootstrap_ci([1.0, 2.0, 3.0], **kwargs)


@pytest.mark.parametrize("alpha", [0.0, 1.0, float("nan")])
def test_significance_level_must_be_a_probability(alpha):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        StatisticalTester(alpha=alpha)


@pytest.mark.parametrize("confidence", [0.0, 1.0, float("nan")])
def test_mean_interval_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        StatisticalTester().mean_std_ci([1.0, 2.0], confidence=confidence)


def test_lower_is_better_comparison_tests_for_a_reduction():
    baseline = [0.8, 0.7, 0.9, 0.6, 0.85]
    proposed = [0.3, 0.25, 0.4, 0.2, 0.3]

    comparison = StatisticalTester().compare_metrics(
        baseline, proposed, "hallucination_rate", higher_is_better=False
    )

    assert comparison["higher_is_better"] is False
    assert comparison["improvement"] == pytest.approx(0.48)
    assert comparison["improvement_pct"] > 0
    assert comparison["is_significant"] is True
    assert comparison["p_value"] < 0.05

    worse = StatisticalTester().compare_metrics(
        proposed, baseline, "hallucination_rate", higher_is_better=False
    )
    assert worse["improvement"] == pytest.approx(-0.48)
    assert worse["is_significant"] is False


def test_default_comparison_still_tests_for_an_increase():
    comparison = StatisticalTester().compare_metrics(
        [0.3, 0.25, 0.4, 0.2, 0.3], [0.8, 0.7, 0.9, 0.6, 0.85]
    )

    assert comparison["higher_is_better"] is True
    assert comparison["improvement"] == pytest.approx(0.48)
    assert comparison["is_significant"] is True


@pytest.mark.parametrize("baseline,proposed", [
    ([1.0], [1.0]),
    ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
])
def test_undefined_comparison_statistics_are_valid_json(baseline, proposed):
    comparison = StatisticalTester().compare_metrics(baseline, proposed)

    assert comparison["t_statistic"] is None
    assert comparison["p_value"] is None
    assert comparison["is_significant"] is False
    json.dumps(comparison, allow_nan=False)
