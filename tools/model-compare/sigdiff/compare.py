"""High-level helpers: compare two models, or many models against a baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import get_metric
from .paired import mcnemar, paired_bootstrap, permutation_test


@dataclass
class Comparison:
    metric: str
    greater_is_better: bool
    n: int
    score_a: float
    score_b: float
    diff: float            # score_b - score_a
    ci_low: float          # bootstrap CI of diff
    ci_high: float
    p_value: float         # from `method`
    method: str
    alpha: float
    verdict: str
    name_a: str = "A"
    name_b: str = "B"

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        arrow = "higher is better" if self.greater_is_better else "lower is better"
        return (
            f"{self.metric} ({arrow}), n={self.n}\n"
            f"  {self.name_a}: {self.score_a:.5f}\n"
            f"  {self.name_b}: {self.score_b:.5f}\n"
            f"  {self.name_b} - {self.name_a}: {self.diff:+.5f}  "
            f"[{100 * (1 - self.alpha):.0f}% CI {self.ci_low:+.5f}, {self.ci_high:+.5f}]\n"
            f"  p-value ({self.method}): {self.p_value:.4g}\n"
            f"  verdict: {self.verdict}"
        )


def _verdict(diff, p, alpha, gib, name_a, name_b):
    if not np.isfinite(p) or p >= alpha:
        return f"no significant difference at alpha={alpha}"
    better = name_b if (diff > 0) == gib else name_a
    return f"{better} is better (p={p:.3g} < {alpha})"


def compare(y, pred_a, pred_b, metric="accuracy", *, method="bootstrap", alpha=0.05,
            n_resamples=2000, groups=None, seed: Optional[int] = 0,
            name_a="A", name_b="B") -> Comparison:
    """Compare two models' predictions on the same test set.

    ``method`` picks where the p-value comes from: ``"bootstrap"`` (paired
    bootstrap), ``"permutation"`` (paired swap test) or ``"mcnemar"`` (exact
    McNemar, accuracy only). The confidence interval always comes from the
    paired bootstrap. ``groups`` makes resampling respect clusters of rows
    (e.g. several rows from the same user or the same day).
    """
    m = get_metric(metric)
    y = np.asarray(y)
    boot = paired_bootstrap(y, pred_a, pred_b, m, n_resamples=n_resamples, alpha=alpha, groups=groups, seed=seed)
    if method == "bootstrap":
        p = boot.p_value
    elif method == "permutation":
        p = permutation_test(y, pred_a, pred_b, m, n_resamples=n_resamples, groups=groups, seed=seed).p_value
    elif method == "mcnemar":
        if not m.needs_labels:
            raise ValueError("McNemar's test only applies to accuracy")
        if groups is not None:
            raise ValueError("McNemar's test assumes independent rows; use bootstrap/permutation with groups")
        p = mcnemar(y, pred_a, pred_b).p_value
    else:
        raise ValueError("method must be 'bootstrap', 'permutation' or 'mcnemar'")
    sa, sb = m.score(y, np.asarray(pred_a)), m.score(y, np.asarray(pred_b))
    return Comparison(m.name, m.greater_is_better, int(y.shape[0]), float(sa), float(sb), float(sb - sa),
                      boot.ci_low, boot.ci_high, float(p), method, alpha,
                      _verdict(sb - sa, p, alpha, m.greater_is_better, name_a, name_b), name_a, name_b)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    k = p.size
    order = np.argsort(p)
    adj = np.empty(k)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        adj[i] = running
    return adj


def compare_many(y, preds: Dict[str, np.ndarray], baseline: str, metric="accuracy", *,
                 method="bootstrap", alpha=0.05, n_resamples=2000, groups=None,
                 seed: Optional[int] = 0) -> List[Comparison]:
    """Compare every model in ``preds`` against ``baseline`` with Holm correction.

    When you try many models, some will beat the baseline by luck alone. The
    Holm correction keeps the chance of *any* false "better" claim below alpha.
    The returned comparisons carry Holm-adjusted p-values and verdicts.
    """
    if baseline not in preds:
        raise ValueError(f"baseline {baseline!r} not in preds")
    others = [k for k in preds if k != baseline]
    if not others:
        raise ValueError("need at least one model besides the baseline")
    res = [compare(y, preds[baseline], preds[k], metric, method=method, alpha=alpha, n_resamples=n_resamples,
                   groups=groups, seed=seed, name_a=baseline, name_b=k) for k in others]
    adj = holm([r.p_value for r in res])
    m = get_metric(metric)
    for r, pa in zip(res, adj):
        r.p_value = float(pa)
        r.method = f"{r.method}+holm"
        r.verdict = _verdict(r.diff, pa, alpha, m.greater_is_better, r.name_a, r.name_b)
    return res
