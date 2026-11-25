"""Validation statistics. Pure functions: arrays in, numbers out."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, chi2
from sklearn.metrics import roc_auc_score


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    """Max distance between cumulative bad and good distributions over the PD ranking."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def hosmer_lemeshow(y, p, groups: int = 10) -> tuple[float, float, pd.DataFrame]:
    """Returns (statistic, p-value, decile table)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["g"] = pd.qcut(d["p"].rank(method="first"), groups, labels=False)
    t = d.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), pred=("p", "sum"))
    exp_ = t["pred"]
    stat = float((((t["obs"] - exp_) ** 2) / (exp_ * (1 - exp_ / t["n"]))).sum())
    pval = float(1 - chi2.cdf(stat, groups - 2))
    t["obs_rate"] = t["obs"] / t["n"]
    t["pred_rate"] = t["pred"] / t["n"]
    return stat, pval, t


def grade_binomial(y, p, grades) -> pd.DataFrame:
    """Per-grade one-sided binomial test: is the observed default count too high for the PD?"""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "grade": np.asarray(grades)})
    rows = []
    for g, s in d.groupby("grade"):
        n, k, pd_ = len(s), int(s["y"].sum()), float(s["p"].mean())
        pval = binomtest(k, n, pd_, alternative="greater").pvalue
        rows.append({"grade": g, "n": n, "defaults": k, "avg_pd": pd_, "odr": k / n, "p_value": pval})
    return pd.DataFrame(rows).set_index("grade")


def oe_ratio(y, p) -> float:
    """Observed defaults / expected defaults (sum of PD). 1.0 = perfectly calibrated in total."""
    return float(np.sum(y) / np.sum(p))


def jeffreys_pvalue(defaults: int, n: int, pd_: float) -> float:
    """One-sided Jeffreys test of PD underestimation (as in ECB IRB validation reporting).

    Posterior of the true default rate is Beta(D + 0.5, N - D + 0.5). The p-value is the
    posterior probability that the true rate is at or below the predicted PD; a small value
    means the PD is too low. Like the binomial test, it assumes independent defaults.
    """
    return float(beta.cdf(pd_, defaults + 0.5, n - defaults + 0.5))


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor of each column (regress it on the others)."""
    out = {}
    A = X.to_numpy(dtype=float)
    for j, col in enumerate(X.columns):
        y = A[:, j]
        others = np.column_stack([np.ones(len(A)), np.delete(A, j, axis=1)])
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ coef
        r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
        out[col] = float(1 / max(1 - r2, 1e-12))
    return pd.Series(out).sort_values(ascending=False)


def band_status(value: float, green: tuple[float, float], amber: tuple[float, float]) -> str:
    if green[0] <= value <= green[1]:
        return "GREEN"
    return "AMBER" if amber[0] <= value <= amber[1] else "RED"


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of a continuous variable (bins from the expected sample)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1)
    return _psi_from_counts(e, a)


def _as_categories(x) -> pd.Series:
    """Categorical values as plain strings, with missing values as their own category."""
    s = pd.Series(x).astype(object)
    return s.where(s.notna(), "MISSING").astype(str)


def psi_categorical(expected, actual) -> float:
    e_s, a_s = _as_categories(expected), _as_categories(actual)
    cats = sorted(set(e_s) | set(a_s))
    e = e_s.value_counts().reindex(cats, fill_value=0).to_numpy()
    a = a_s.value_counts().reindex(cats, fill_value=0).to_numpy()
    return _psi_from_counts(e, a)


def _psi_from_counts(e: np.ndarray, a: np.ndarray) -> float:
    pe = np.clip(e / e.sum(), 1e-6, None)
    pa = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def traffic_light(value: float, green: float, amber: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "GREEN" if value >= green else "AMBER" if value >= amber else "RED"
    return "GREEN" if value <= green else "AMBER" if value <= amber else "RED"
