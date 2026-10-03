"""Bootstrap methods must report the interval they claim to compute."""

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
