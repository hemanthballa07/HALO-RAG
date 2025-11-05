"""
Statistical Testing Module
Implements t-tests and bootstrap confidence intervals.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional, Dict


class StatisticalTester:
    """
    Statistical testing for experiment results.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def t_test(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = "two-sided"
    ) -> Tuple[float, float, bool]:
        """
        Perform independent samples t-test.
        
        Args:
            group1: First group of scores
            group2: Second group of scores
            alternative: "two-sided", "less", or "greater"
        
        Returns:
            Tuple of (t_statistic, p_value, is_significant)
        """
        t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
        is_significant = p_value < self.alpha
        
        return float(t_stat), float(p_value), is_significant
    
    def paired_t_test(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = "two-sided"
    ) -> Tuple[float, float, bool]:
        """
        Perform paired samples t-test.
        
        Args:
            group1: First group of scores
            group2: Second group of scores
            alternative: "two-sided", "less", or "greater"
        
        Returns:
            Tuple of (t_statistic, p_value, is_significant)
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have same length for paired t-test")
        
        t_stat, p_value = stats.ttest_rel(group1, group2, alternative=alternative)
        is_significant = p_value < self.alpha
        
        return float(t_stat), float(p_value), is_significant
    
    def bootstrap_ci(
        self,
        data: List[float],
        confidence: float = 0.95,
        n_iterations: int = 1000,
        method: str = "percentile"
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            data: Sample data
            confidence: Confidence level (default: 0.95)
            n_iterations: Number of bootstrap iterations
            method: "percentile" or "bca"
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) == 0:
            return (0.0, 0.0)
        
        n = len(data)
        alpha = 1 - confidence
        
        # Bootstrap sampling
        bootstrap_samples = []
        for _ in range(n_iterations):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        bootstrap_samples = np.array(bootstrap_samples)
        
        if method == "percentile":
            lower = np.percentile(bootstrap_samples, alpha / 2 * 100)
            upper = np.percentile(bootstrap_samples, (1 - alpha / 2) * 100)
        elif method == "bca":
            # Bias-corrected and accelerated
            lower, upper = self._bca_ci(data, bootstrap_samples, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(lower), float(upper)
    
    def _bca_ci(
        self,
        data: List[float],
        bootstrap_samples: np.ndarray,
        alpha: float
    ) -> Tuple[float, float]:
        """Compute bias-corrected and accelerated (BCa) confidence interval."""
        # Simplified BCa - for full implementation, need jackknife
        # Using percentile method as approximation
        lower = np.percentile(bootstrap_samples, alpha / 2 * 100)
        upper = np.percentile(bootstrap_samples, (1 - alpha / 2) * 100)
        return lower, upper
    
    def mean_std_ci(
        self,
        data: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Compute mean, std, and confidence interval.
        
        Args:
            data: Sample data
            confidence: Confidence level
        
        Returns:
            Tuple of (mean, std, (lower_ci, upper_ci))
        """
        if len(data) == 0:
            return (0.0, 0.0, (0.0, 0.0))
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Standard error
        se = std / np.sqrt(len(data))
        
        # t-statistic for confidence interval
        t_critical = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        margin = t_critical * se
        lower_ci = mean - margin
        upper_ci = mean + margin
        
        return float(mean), float(std), (float(lower_ci), float(upper_ci))
    
    def compare_metrics(
        self,
        baseline_scores: List[float],
        proposed_scores: List[float],
        metric_name: str = "metric"
    ) -> Dict[str, any]:
        """
        Compare baseline vs proposed scores with statistical testing.
        
        Args:
            baseline_scores: Baseline scores
            proposed_scores: Proposed method scores
            metric_name: Name of metric
        
        Returns:
            Dictionary with comparison results
        """
        # Means and stds
        baseline_mean, baseline_std, baseline_ci = self.mean_std_ci(baseline_scores)
        proposed_mean, proposed_std, proposed_ci = self.mean_std_ci(proposed_scores)
        
        # Improvement
        improvement = proposed_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean * 100) if baseline_mean > 0 else 0.0
        
        # T-test
        t_stat, p_value, is_significant = self.t_test(
            baseline_scores,
            proposed_scores,
            alternative="less"  # Test if proposed > baseline
        )
        
        return {
            "metric": metric_name,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "baseline_ci": baseline_ci,
            "proposed_mean": proposed_mean,
            "proposed_std": proposed_std,
            "proposed_ci": proposed_ci,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "alpha": self.alpha
        }

