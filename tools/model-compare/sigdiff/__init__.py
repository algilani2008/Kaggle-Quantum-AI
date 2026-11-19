"""sigdiff: is model B really better than model A, or is it just noise?

Paired significance tests and confidence intervals for comparing machine
learning models on the same test set or the same cross-validation folds.
"""
from .compare import Comparison, compare, compare_many, holm
from .cv import corrected_resampled_ttest, five_by_two_cv_ttest, naive_ttest
from .metrics import METRICS, Metric, get_metric
from .paired import TestResult, mcnemar, paired_bootstrap, permutation_test

__version__ = "0.1.0"

__all__ = [
    "Comparison", "compare", "compare_many", "holm",
    "corrected_resampled_ttest", "five_by_two_cv_ttest", "naive_ttest",
    "METRICS", "Metric", "get_metric",
    "TestResult", "mcnemar", "paired_bootstrap", "permutation_test",
]
