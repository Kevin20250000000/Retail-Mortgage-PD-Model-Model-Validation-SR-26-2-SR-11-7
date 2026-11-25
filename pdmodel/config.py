"""Single source of truth for model settings and validation thresholds.

Thresholds are kept here (not scattered in tests) so a reviewer can audit every pass/fail
rule in one place. Changing a threshold or a sample window is a *material change* and
requires human review (see AGENTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEED = 20261122


def make_rng(seed: int = SEED) -> np.random.Generator:
    """The only sanctioned way to create randomness in this repo."""
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class ModelConfig:
    source: str = "synthetic"
    features: tuple[str, ...] = ("fico", "dti", "ltv", "income", "loan_amount",
                                 "emp_length", "delinq_2y", "purpose")
    categorical: tuple[str, ...] = ("purpose",)
    target: str = "default_12m"
    date_col: str = "orig_quarter"
    # Sample windows on date_col (inclusive). calib is optional.
    dev: tuple[str, str] = ("2018Q1", "2021Q4")
    calib: tuple[str, str] | None = None
    oot: tuple[tuple[str, str, str], ...] = (("oot", "2022Q1", "2023Q4"),)
    test_share: float = 0.3
    # Sensitivity shocks: (label, column, additive shock); PD is expected to rise.
    shocks: tuple[tuple[str, str, float], ...] = (
        ("fico -40", "fico", -40), ("dti +10", "dti", 10), ("ltv +15", "ltv", 15),
        ("delinq_2y +1", "delinq_2y", 1),
    )
    n_bins: int = 6
    min_bin_share: float = 0.05
    # Scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0
    n_grades: int = 10


FREDDIE_MODEL = ModelConfig(
    source="freddie",
    # cltv and mi_pct dropped (collinear with ltv: wrong-sign coefficient in run 1);
    # orig_rate replaced by sato (raw rate tracks the market cycle, CSI 3.1 in run 1).
    features=("fico", "dti", "ltv", "sato", "orig_upb", "num_borrowers",
              "occupancy", "loan_purpose", "first_time_buyer", "property_type", "channel"),
    categorical=("occupancy", "loan_purpose", "first_time_buyer", "property_type", "channel"),
    dev=("2014Q1", "2017Q4"),
    calib=("2018Q1", "2018Q4"),
    oot=(("oot1_2019", "2019Q1", "2019Q4"),
         ("oot2_2020_21", "2020Q1", "2021Q4"),
         ("oot3_2022_23", "2022Q1", "2023Q4")),
    shocks=(("fico -40", "fico", -40), ("dti +10", "dti", 10), ("ltv +15", "ltv", 15),
            ("sato +0.5pp", "sato", 0.5)),
)


@dataclass(frozen=True)
class Thresholds:
    """Traffic-light thresholds. green >= green, amber >= amber, otherwise red."""

    # Discrimination (higher is better)
    auc_green: float = 0.70
    auc_amber: float = 0.65
    ks_green: float = 0.30
    ks_amber: float = 0.20
    # Max AUC drop from training to the worst OOT segment
    auc_decay_max: float = 0.05
    # Stability (lower is better)
    psi_green: float = 0.10
    psi_amber: float = 0.25
    # Calibration. Hosmer-Lemeshow p-value (informational: over-rejects in large samples).
    hl_pvalue_min: float = 0.05
    # O/E = observed defaults / expected defaults (sum of PD). Relative, so it works for
    # low-default portfolios where an absolute bps gap is meaningless.
    oe_green: tuple[float, float] = (0.80, 1.25)
    oe_amber: tuple[float, float] = (0.67, 1.50)
    # Jeffreys test (one-sided, H0: PD not underestimated): p-value thresholds
    jeffreys_red: float = 0.01
    jeffreys_amber: float = 0.05
    # Recalibration slope: ~1 means the development model's risk spread holds on the
    # calibration sample; far from 1 means it is over- or under-confident.
    slope_green: tuple[float, float] = (0.80, 1.25)
    slope_amber: tuple[float, float] = (0.60, 1.50)
    # Multicollinearity among model inputs (WoE scale)
    vif_green: float = 5.0
    vif_amber: float = 10.0
    # Minimum information value for a feature to be kept
    iv_min: float = 0.02
    # Challenger: champion may trail challenger AUC by at most this much
    challenger_gap_max: float = 0.03
    # Regression vs approved baseline
    baseline_tol: float = 0.01


@dataclass(frozen=True)
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)


SETTINGS = Settings()
FREDDIE_SETTINGS = Settings(model=FREDDIE_MODEL)
