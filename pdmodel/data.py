"""Data layer: a synthetic loan book plus a schema check every source must pass.

The synthetic generator exists so the full pipeline runs end-to-end with no
downloads. To use real data (e.g. Freddie Mac Single-Family Loan-Level or
Lending Club), write a loader that returns the same columns and pass it through
`validate_schema` - nothing downstream needs to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdmodel.config import SETTINGS, ModelConfig

SCHEMA: dict[str, tuple[float, float] | None] = {
    # column: (min, max) for numeric range checks, None for categorical
    "fico": (300, 850),
    "dti": (0, 100),
    "ltv": (1, 150),
    "income": (0, 5_000_000),
    "loan_amount": (500, 2_000_000),
    "emp_length": (0, 50),
    "delinq_2y": (0, 30),
    "purpose": None,
    "unemployment": (0, 30),
}

PURPOSES = ("debt_consolidation", "home_improvement", "credit_card", "small_business", "other")
_PURPOSE_EFFECT = {
    "debt_consolidation": 0.10,
    "home_improvement": -0.15,
    "credit_card": 0.0,
    "small_business": 0.55,
    "other": 0.20,
}


def quarters(start: str = "2018Q1", end: str = "2023Q4") -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="Q")]


def unemployment_path(qs: list[str]) -> np.ndarray:
    """Stylised US unemployment: low pre-2020, COVID spike, then normalisation."""
    path = []
    for q in qs:
        year, qn = int(q[:4]), int(q[-1])
        if year < 2020:
            u = 3.9 - 0.1 * (year - 2018) - 0.05 * qn
        elif year == 2020:
            u = {1: 3.8, 2: 13.0, 3: 8.8, 4: 6.8}[qn]
        elif year == 2021:
            u = {1: 6.2, 2: 5.9, 3: 5.1, 4: 4.2}[qn]
        else:
            u = 3.7 + 0.08 * (year - 2022) + 0.03 * qn
        path.append(u)
    return np.array(path)


def true_logit(df: pd.DataFrame) -> np.ndarray:
    """Data-generating process. Includes non-linearities the scorecard must approximate."""
    emp = df["emp_length"].fillna(2.0)
    z = (
        -3.35
        - 0.0135 * (df["fico"] - 700)
        + 0.030 * (df["dti"] - 30)
        + 0.012 * (df["ltv"] - 80)
        + 0.45 * (df["ltv"] > 95)
        - 0.35 * np.log(df["income"] / 70_000)
        + 0.15 * np.log(df["loan_amount"] / 20_000)
        - 0.03 * emp
        + 0.30 * df["emp_length"].isna()
        + 0.35 * df["delinq_2y"]
        + df["purpose"].map(_PURPOSE_EFFECT)
        + 0.18 * (df["unemployment"] - 4.0)
    )
    return z.to_numpy()


def generate_synthetic(rng: np.random.Generator, n: int = 60_000) -> pd.DataFrame:
    qs = quarters()
    unemp = dict(zip(qs, unemployment_path(qs), strict=True))
    q = rng.choice(qs, size=n)
    late = np.array([int(x[:4]) >= 2022 for x in q])

    # Mild population drift in the OOT window: slightly weaker FICO, higher DTI.
    fico = np.clip(rng.normal(715 - 8 * late, 50), 450, 850).round()
    dti = np.clip(rng.gamma(6.0, 5.0 + 0.4 * late), 1, 65).round(1)
    ltv = np.clip(rng.normal(78, 14, n), 20, 110).round(1)
    income = np.exp(rng.normal(np.log(72_000), 0.5, n)).round(-2)
    loan_amount = np.clip(np.exp(rng.normal(np.log(18_000), 0.6, n)), 1_000, 300_000).round(-2)
    emp_length = rng.integers(0, 21, n).astype(float)
    emp_length[rng.random(n) < 0.03] = np.nan  # realistic missingness
    delinq_2y = rng.poisson(0.25, n)
    purpose = rng.choice(PURPOSES, size=n, p=[0.45, 0.15, 0.22, 0.06, 0.12])

    df = pd.DataFrame(
        {
            "loan_id": np.arange(n),
            "orig_quarter": q,
            "fico": fico,
            "dti": dti,
            "ltv": ltv,
            "income": income,
            "loan_amount": loan_amount,
            "emp_length": emp_length,
            "delinq_2y": delinq_2y,
            "purpose": purpose,
            "unemployment": [unemp[x] for x in q],
        }
    )
    pd_true = 1.0 / (1.0 + np.exp(-true_logit(df)))
    df["default_12m"] = (rng.random(n) < pd_true).astype(int)
    return df.sort_values(["orig_quarter", "loan_id"]).reset_index(drop=True)


def validate_schema(df: pd.DataFrame, cfg: ModelConfig = SETTINGS.model) -> list[str]:
    """Return a list of problems. Empty list means the data passes."""
    problems: list[str] = []
    required = set(SCHEMA) | {cfg.target, cfg.date_col}
    missing = required - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    for col, rng_ in SCHEMA.items():
        if rng_ is None:
            continue
        lo, hi = rng_
        s = df[col].dropna()
        bad = ((s < lo) | (s > hi)).sum()
        if bad:
            problems.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    if not set(df[cfg.target].unique()) <= {0, 1}:
        problems.append(f"{cfg.target} must be binary 0/1")
    if df["loan_id"].duplicated().any():
        problems.append("duplicate loan_id")
    unknown = set(df["purpose"].unique()) - set(PURPOSES)
    if unknown:
        problems.append(f"unknown purpose values: {sorted(unknown)}")
    return problems


def split(df: pd.DataFrame, rng: np.random.Generator,
          cfg: ModelConfig = SETTINGS.model) -> dict[str, pd.DataFrame]:
    """Samples keyed by name: train/test (random split of dev), optional calib, OOT segments."""

    def window(lo: str, hi: str) -> pd.DataFrame:
        d = df[cfg.date_col]
        return df[(d >= lo) & (d <= hi)].copy()

    dev = window(*cfg.dev)
    is_test = rng.random(len(dev)) < cfg.test_share
    out = {"train": dev[~is_test].copy(), "test": dev[is_test].copy()}
    if cfg.calib:
        out["calib"] = window(*cfg.calib)
    for name, lo, hi in cfg.oot:
        out[name] = window(lo, hi)
    return out


FREDDIE_RANGES = {"fico": (300, 850), "dti": (0, 65), "ltv": (1, 200), "cltv": (1, 200),
                  "orig_rate": (0, 20), "orig_upb": (1_000, 5_000_000)}


def validate_freddie(df: pd.DataFrame, cfg: ModelConfig) -> list[str]:
    problems: list[str] = []
    missing = (set(cfg.features) | {cfg.target, cfg.date_col, "loan_id"}) - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    for col, (lo, hi) in FREDDIE_RANGES.items():
        s = df[col].dropna()
        bad = int(((s < lo) | (s > hi)).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    if not set(df[cfg.target].unique()) <= {0, 1}:
        problems.append(f"{cfg.target} must be binary 0/1")
    if df["loan_id"].duplicated().any():
        problems.append("duplicate loan_id")
    return problems
