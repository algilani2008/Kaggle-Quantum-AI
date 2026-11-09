"""leakcheck: find data leakage between train and test sets before it fools your validation score."""
from .core import (Finding, Report, Thresholds, audit, check_adversarial, check_cross_duplicates,
                   check_group_overlap, check_id_like, check_target_leakage, check_time_order,
                   check_train_duplicates, infer_task)

__version__ = "0.1.0"
__all__ = ["audit", "Report", "Finding", "Thresholds", "infer_task", "check_adversarial",
           "check_cross_duplicates", "check_group_overlap", "check_id_like", "check_target_leakage",
           "check_time_order", "check_train_duplicates"]
