"""Champion (WoE logistic scorecard) and challenger (gradient boosting) PD models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from pdmodel.config import SETTINGS, ModelConfig
from pdmodel.woe import WoETransformer


class PDScorecard:
    """Champion model: WoE-binned logistic regression, scaled to scorecard points."""

    def __init__(self, cfg: ModelConfig = SETTINGS.model, iv_min: float = SETTINGS.thresholds.iv_min):
        self.cfg = cfg
        self.iv_min = iv_min
        self.woe = WoETransformer(cfg.features, cfg.categorical, cfg.n_bins, cfg.min_bin_share)
        self.selected: list[str] = []
        self.dropped_for_sign: list[tuple[str, float]] = []
        self.lr = LogisticRegression(C=1e6, max_iter=1000)  # effectively unpenalised
        self.grade_edges: np.ndarray | None = None
        # Optional PD calibration layer: PD = sigmoid(a + b * logit(raw PD)), fitted on a
        # hold-out calibration sample. Scores and grades stay on the raw (ranking) scale.
        self.calibration: tuple[float, float] | None = None

    def fit(self, df: pd.DataFrame) -> PDScorecard:
        X, y = df[list(self.cfg.features)], df[self.cfg.target]
        self.woe.fit(X, y)
        iv = self.woe.iv()
        self.selected = [f for f in self.cfg.features if iv[f] >= self.iv_min]
        Xw = self.woe.transform(X)
        self.lr.fit(Xw[self.selected], y)
        # Standard scorecard practice: with WoE = ln(good/bad) every coefficient must be
        # negative. Drop the most positive (counter-intuitive) variable and refit until none
        # remain; dropped variables are reported, not hidden.
        self.dropped_for_sign = []
        while len(self.selected) > 1 and (self.lr.coef_[0] >= 0).any():
            j = int(self.lr.coef_[0].argmax())
            self.dropped_for_sign.append((self.selected[j], float(self.lr.coef_[0][j])))
            self.selected = [f for i, f in enumerate(self.selected) if i != j]
            self.lr.fit(Xw[self.selected], y)
        # Rating grades: score deciles on the development sample (grade 1 = best)
        s = self.score(df)
        self.grade_edges = np.quantile(s, np.linspace(0, 1, self.cfg.n_grades + 1)[1:-1])
        return self

    def predict_pd_raw(self, df: pd.DataFrame) -> np.ndarray:
        Xw = self.woe.transform(df[list(self.cfg.features)])[self.selected]
        return self.lr.predict_proba(Xw)[:, 1]

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.predict_pd_raw(df)
        if self.calibration is None:
            return raw
        a, b = self.calibration
        z = np.log(np.clip(raw, 1e-12, 1 - 1e-12) / np.clip(1 - raw, 1e-12, 1))
        return 1.0 / (1.0 + np.exp(-(a + b * z)))

    def recalibrate(self, df: pd.DataFrame) -> tuple[float, float]:
        """Logistic recalibration (intercept and slope) on a hold-out sample."""
        raw = np.clip(self.predict_pd_raw(df), 1e-12, 1 - 1e-12)
        z = np.log(raw / (1 - raw)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, df[self.cfg.target])
        self.calibration = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
        return self.calibration

    def score(self, df: pd.DataFrame) -> np.ndarray:
        pd_ = np.clip(self.predict_pd_raw(df), 1e-9, 1 - 1e-9)
        factor = self.cfg.pdo / np.log(2)
        offset = self.cfg.base_score - factor * np.log(self.cfg.base_odds)
        return offset + factor * np.log((1 - pd_) / pd_)

    def grade(self, df: pd.DataFrame) -> np.ndarray:
        # Higher score = lower risk; grade 1 = best
        return self.cfg.n_grades - np.searchsorted(self.grade_edges, self.score(df), side="right")

    def coefficients(self) -> pd.Series:
        return pd.Series(self.lr.coef_[0], index=self.selected)


class Challenger:
    """Independent benchmark model on raw features (no WoE), per SR 26-2 / SR 11-7 benchmarking."""

    def __init__(self, cfg: ModelConfig = SETTINGS.model, seed: int = 0):
        self.cfg = cfg
        self.clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            categorical_features="from_dtype", random_state=seed,
        )

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[list(self.cfg.features)].copy()
        for c in self.cfg.categorical:
            X[c] = X[c].astype("category")
        return X

    def fit(self, df: pd.DataFrame) -> Challenger:
        self.clf.fit(self._X(df), df[self.cfg.target])
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._X(df))[:, 1]
