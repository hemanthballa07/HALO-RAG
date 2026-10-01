"""Evaluation metrics and statistical testing utilities."""

from .metrics import EvaluationMetrics

__all__ = ["EvaluationMetrics", "StatisticalTester"]


def __getattr__(name: str):
    """Load the statistical helper only when its SciPy dependency is needed."""
    if name == "StatisticalTester":
        from .statistical_testing import StatisticalTester

        return StatisticalTester
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
