"""Fast checks for metric and statistical helper behavior."""

import json
import unittest
from math import log2

from src.evaluation import EvaluationMetrics, StatisticalTester


class EvaluationMetricTests(unittest.TestCase):
    def setUp(self):
        self.metrics = EvaluationMetrics()

    def test_recall_at_k(self):
        self.assertAlmostEqual(self.metrics.recall_at_k([0, 1, 2, 3, 4], [1, 3, 5], 5), 2 / 3)

    def test_ndcg_counts_each_relevant_document_once(self):
        self.assertAlmostEqual(self.metrics.ndcg_at_k([1, 1], [1], 2), 1.0)
        expected = (1 + 1 / log2(4)) / (1 + 1 / log2(3))
        self.assertAlmostEqual(self.metrics.ndcg_at_k([1, 1, 2], [1, 2], 3), expected)

    def test_verified_f1(self):
        self.assertAlmostEqual(self.metrics.verified_f1(0.60, 0.70), 0.42)

    def test_coverage(self):
        coverage = self.metrics.coverage(
            "Paris is the capital of France",
            ["Paris is a city in France. The capital of France is Paris."],
        )
        self.assertGreater(coverage, 0.8)


class StatisticalHelperTests(unittest.TestCase):
    def test_summary_statistics_are_finite(self):
        mean, standard_deviation, interval = StatisticalTester().mean_std_ci(
            [0.8, 0.85, 0.9, 0.88, 0.92]
        )
        self.assertGreater(mean, 0)
        self.assertGreater(standard_deviation, 0)
        self.assertLess(interval[0], interval[1])

    def test_comparison_is_json_serializable(self):
        comparison = StatisticalTester().compare_metrics(
            [0.2, 0.4, 0.3], [0.6, 0.75, 0.95]
        )
        self.assertIs(type(comparison["is_significant"]), bool)
        json.dumps(comparison)


if __name__ == "__main__":
    unittest.main()
