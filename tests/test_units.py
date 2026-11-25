"""Unit tests for metrics, binning and data checks, against cases with known answers."""

import numpy as np
import pandas as pd
import pytest

from pdmodel import metrics as m
from pdmodel.config import make_rng
from pdmodel.data import generate_synthetic, validate_schema
from pdmodel.woe import MISSING, fit_feature


# ---------- metrics: known answers ----------
def test_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert m.auc(y, p) == 1.0 and m.gini(y, p) == 1.0 and m.ks(y, p) == pytest.approx(1.0)


def test_random_ranking_auc_near_half():
    rng = make_rng(1)
    y = rng.integers(0, 2, 20_000)
    assert abs(m.auc(y, rng.random(20_000)) - 0.5) < 0.02


def test_psi_identical_is_zero_and_shift_is_large():
    rng = make_rng(2)
    x = rng.normal(size=50_000)
    assert m.psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert m.psi(x, x + 1.0) > 0.25


def test_hosmer_lemeshow_accepts_calibrated_and_rejects_biased():
    rng = make_rng(3)
    p = rng.uniform(0.01, 0.2, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    assert m.hosmer_lemeshow(y, p)[1] > 0.01
    assert m.hosmer_lemeshow(y, p * 0.6)[1] < 1e-6


def test_oe_and_jeffreys():
    y = np.array([1] * 30 + [0] * 9970)          # 30 defaults in 10,000
    assert m.oe_ratio(y, np.full(10_000, 0.003)) == pytest.approx(1.0)
    assert m.jeffreys_pvalue(30, 10_000, 0.003) > 0.3      # PD 0.30% consistent with 30 defaults
    assert m.jeffreys_pvalue(30, 10_000, 0.0015) < 0.001   # PD 0.15% clearly too low


def test_vif_detects_collinearity():
    rng = make_rng(9)
    a = rng.normal(size=5_000)
    X = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.1, size=5_000), "c": rng.normal(size=5_000)})
    v = m.vif(X)
    assert v["a"] > 50 and v["c"] < 1.1


def test_psi_categorical_handles_missing_values():
    a = pd.Series(["Y", "N", None, "N"] * 100, dtype="string")
    b = pd.Series(["Y", None, None, "N"] * 100, dtype="string")
    assert m.psi_categorical(a, a) == pytest.approx(0.0, abs=1e-9)
    assert m.psi_categorical(a, b) > 0


def test_recalibration_fixes_level_but_keeps_ranking():
    from pdmodel.config import FREDDIE_SETTINGS
    from pdmodel.model import PDScorecard
    rng = make_rng(11)
    n = 40_000
    df = pd.DataFrame({"fico": rng.normal(740, 50, n), "dti": rng.normal(35, 9, n), "ltv": rng.normal(78, 12, n),
                       "sato": rng.normal(0, 0.4, n), "orig_upb": rng.normal(250_000, 80_000, n),
                       "num_borrowers": rng.choice([1.0, 2.0], n), "occupancy": "P", "loan_purpose": "P",
                       "first_time_buyer": "N", "property_type": "SF", "channel": "R"})
    z = -5 - 0.03 * (df["fico"] - 740) + 0.04 * (df["dti"] - 35)
    df["default_12m"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    model = PDScorecard(FREDDIE_SETTINGS.model).fit(df.iloc[: n // 2])
    later = df.iloc[n // 2 :].copy()
    later["default_12m"] = (rng.random(len(later)) < 3 / (1 + np.exp(-z.iloc[n // 2 :]))).astype(int)  # 3x level
    before = m.oe_ratio(later["default_12m"], model.predict_pd(later))
    auc_before = m.auc(later["default_12m"], model.predict_pd(later))
    model.recalibrate(later)
    assert before > 2.0
    assert m.oe_ratio(later["default_12m"], model.predict_pd(later)) == pytest.approx(1.0, abs=0.05)
    assert m.auc(later["default_12m"], model.predict_pd(later)) == pytest.approx(auc_before)


def test_wrong_sign_variable_is_dropped_and_recorded():
    """A suppressor: first-time buyers have lower FICO (riskier on their own) but are
    safer once FICO is controlled for, so its WoE coefficient comes out positive."""
    from pdmodel.config import FREDDIE_SETTINGS
    from pdmodel.model import PDScorecard
    rng = make_rng(21)
    n = 60_000
    fico = rng.normal(740, 50, n)
    ftb = np.where(rng.random(n) < 1 / (1 + np.exp((fico - 700) / 15)), "Y", "N")
    z = -4.0 - 0.035 * (fico - 740) - 1.2 * (ftb == "Y")
    df = pd.DataFrame({"fico": fico, "dti": rng.normal(35, 9, n), "ltv": rng.normal(78, 12, n),
                       "sato": rng.normal(0, 0.4, n), "orig_upb": rng.normal(250_000, 80_000, n),
                       "num_borrowers": rng.choice([1.0, 2.0], n), "occupancy": "P", "loan_purpose": "P",
                       "first_time_buyer": ftb, "property_type": "SF", "channel": "R"})
    df["default_12m"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    model = PDScorecard(FREDDIE_SETTINGS.model).fit(df)
    assert [f for f, _ in model.dropped_for_sign] == ["first_time_buyer"]
    assert (model.coefficients() < 0).all()


def test_traffic_light():
    assert m.traffic_light(0.75, 0.70, 0.65) == "GREEN"
    assert m.traffic_light(0.67, 0.70, 0.65) == "AMBER"
    assert m.traffic_light(0.30, 0.10, 0.25, higher_is_better=False) == "RED"


# ---------- WoE binning ----------
def test_woe_monotonic_and_missing_bin():
    rng = make_rng(4)
    x = pd.Series(rng.normal(size=20_000), name="x")
    y = pd.Series((rng.random(20_000) < 1 / (1 + np.exp(2 - x))).astype(int))
    x.iloc[:500] = np.nan
    fb = fit_feature(x, y, categorical=False, n_bins=8, min_share=0.05)
    t = fb.table.drop(index=MISSING)
    t = t.loc[sorted(t.index, key=lambda s: int(s.split("_")[1]))]
    assert np.all(np.diff(t["bad_rate"]) >= 0)
    assert MISSING in fb.table.index
    assert fb.iv > 0.1


def test_unseen_category_gets_neutral_woe():
    x = pd.Series(["a", "b", "a", "b"] * 100, name="c")
    y = pd.Series([0, 1, 0, 0] * 100)
    fb = fit_feature(x, y, categorical=True, n_bins=5, min_share=0.05)
    assert fb.transform(pd.Series(["zzz"])).iloc[0] == 0.0


# ---------- data quality catches bad inputs ----------
def test_schema_flags_injected_errors():
    df = generate_synthetic(make_rng(5), n=2_000)
    assert validate_schema(df) == []
    bad = df.copy()
    bad.loc[0, "fico"] = 900
    bad.loc[1, "loan_id"] = bad.loc[2, "loan_id"]
    bad.loc[3, "purpose"] = "crypto"
    problems = " ".join(validate_schema(bad))
    assert "fico" in problems and "duplicate" in problems and "purpose" in problems
