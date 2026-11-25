"""Weight-of-Evidence binning with monotonic bad-rate enforcement.

Convention: WoE = ln(%good / %bad). Positive WoE = lower risk.
Missing values always get their own bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MISSING = "MISSING"
_SMOOTH = 0.5  # Laplace smoothing so empty cells never give infinite WoE


@dataclass
class FeatureBins:
    name: str
    categorical: bool
    edges: list[float] = field(default_factory=list)  # numeric interior cut points
    woe: dict[str, float] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    iv: float = 0.0

    def labels(self, x: pd.Series) -> pd.Series:
        if self.categorical:
            return x.astype(str).where(x.notna(), MISSING)
        idx = np.searchsorted(self.edges, x.to_numpy(), side="right")
        out = pd.Series([f"bin_{i}" for i in idx], index=x.index)
        return out.where(x.notna(), MISSING)

    def transform(self, x: pd.Series) -> pd.Series:
        lab = self.labels(x)
        # Unseen categories get WoE 0 (neutral) - flagged by data-quality tests upstream.
        return lab.map(self.woe).fillna(0.0).astype(float)


def _woe_table(labels: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.crosstab(labels, y).reindex(columns=[0, 1], fill_value=0)
    t.columns = ["good", "bad"]
    t["n"] = t["good"] + t["bad"]
    t["bad_rate"] = t["bad"] / t["n"]
    g = (t["good"] + _SMOOTH) / (t["good"].sum() + _SMOOTH * len(t))
    b = (t["bad"] + _SMOOTH) / (t["bad"].sum() + _SMOOTH * len(t))
    t["woe"] = np.log(g / b)
    t["iv"] = (g - b) * t["woe"]
    return t


def _monotone_edges(x: pd.Series, y: pd.Series, edges: list[float]) -> list[float]:
    """Merge adjacent bins until bad rate is monotonic in the direction of the trend."""
    direction = np.sign(spearmanr(x, y).statistic) or 1.0
    edges = list(edges)
    while edges:
        idx = np.searchsorted(edges, x.to_numpy(), side="right")
        rates = pd.Series(y.to_numpy()).groupby(idx).mean().sort_index().to_numpy()
        diffs = np.diff(rates) * direction
        if (diffs >= 0).all():
            break
        # Remove the edge between the worst-violating adjacent pair
        edges.pop(int(np.argmin(diffs)))
    return edges


def fit_feature(
    x: pd.Series, y: pd.Series, categorical: bool, n_bins: int, min_share: float
) -> FeatureBins:
    fb = FeatureBins(name=str(x.name), categorical=categorical)
    if not categorical:
        xn = x.dropna()
        qs = np.unique(np.quantile(xn, np.linspace(0, 1, n_bins + 1)[1:-1]))
        edges = [float(e) for e in qs]
        # Drop edges that would create bins smaller than min_share
        while edges:
            idx = np.searchsorted(edges, xn.to_numpy(), side="right")
            shares = np.bincount(idx, minlength=len(edges) + 1) / len(xn)
            if shares.min() >= min_share:
                break
            k = int(np.argmin(shares))
            edges.pop(min(k, len(edges) - 1))
        fb.edges = _monotone_edges(xn, y.loc[xn.index], edges)
    t = _woe_table(fb.labels(x), y)
    fb.table = t
    fb.woe = t["woe"].to_dict()
    fb.iv = float(t["iv"].sum())
    return fb


class WoETransformer:
    def __init__(self, features, categorical, n_bins: int, min_share: float):
        self.features = list(features)
        self.categorical = set(categorical)
        self.n_bins = n_bins
        self.min_share = min_share
        self.bins: dict[str, FeatureBins] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WoETransformer:
        for f in self.features:
            self.bins[f] = fit_feature(
                X[f], y, f in self.categorical, self.n_bins, self.min_share
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: self.bins[f].transform(X[f]) for f in self.features}, index=X.index)

    def iv(self) -> pd.Series:
        return pd.Series({f: b.iv for f, b in self.bins.items()}).sort_values(ascending=False)
