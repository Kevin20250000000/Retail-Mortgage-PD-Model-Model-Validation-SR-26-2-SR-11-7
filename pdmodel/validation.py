"""Runs the full validation and returns every check as structured evidence.

Each check has an id, the model risk area it belongs to, the measured value, the rule
applied and a GREEN / AMBER / RED status. Tests and the HTML report both read from this
one result, so the report can never disagree with CI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdmodel import metrics as m
from pdmodel.config import SETTINGS, Settings, make_rng
from pdmodel.data import generate_synthetic, split, validate_freddie, validate_schema
from pdmodel.model import Challenger, PDScorecard

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "processed"


@dataclass
class Check:
    id: str
    area: str  # Data, Conceptual Soundness, Outcomes Analysis, Ongoing Monitoring, Benchmarking
    description: str
    value: float
    rule: str
    status: str


@dataclass
class ValidationResult:
    checks: list[Check] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    artefacts: dict[str, Any] = field(default_factory=dict)

    def add(self, *args) -> None:
        self.checks.append(Check(*args))

    def by_id(self) -> dict[str, Check]:
        return {c.id: c for c in self.checks}

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.checks])

    def key_metrics(self) -> dict[str, float]:
        return {c.id: round(float(c.value), 6) for c in self.checks}


def load_data(settings: Settings, rng: np.random.Generator, data_dir: Path | None,
              raw_dir: Path | None = None) -> pd.DataFrame:
    if settings.model.source == "synthetic":
        return generate_synthetic(rng)
    from pdmodel.freddie.ingest import load_processed  # polars only needed for real data
    from pdmodel.freddie.macro import add_sato
    data_dir = Path(data_dir or DEFAULT_DATA)
    return add_sato(load_processed(data_dir), Path(raw_dir or data_dir.parent / "raw"))


def run_validation(settings: Settings = SETTINGS, seed: int | None = None,
                   data_dir: Path | None = None, df: pd.DataFrame | None = None,
                   raw_dir: Path | None = None) -> ValidationResult:
    cfg, th = settings.model, settings.thresholds
    rng = make_rng() if seed is None else make_rng(seed)
    if df is None:
        df = load_data(settings, rng, data_dir, raw_dir)
    res = ValidationResult()
    res.artefacts["source"] = cfg.source

    # ---------------- Data ----------------
    problems = validate_schema(df, cfg) if cfg.source == "synthetic" else validate_freddie(df, cfg)
    res.add("DQ-01", "Data", "Schema and range checks", float(len(problems)),
            "== 0 problems", "GREEN" if not problems else "RED")
    miss = df[list(cfg.features)].isna().mean().max()
    res.add("DQ-02", "Data", "Max missing share across features", miss,
            "<= 10%", "GREEN" if miss <= 0.10 else "RED")
    samples = split(df, rng, cfg)
    empty = [k for k, d in samples.items() if d.empty]
    if empty:
        raise ValueError(f"No loans in sample(s) {empty}. Check that every vintage in the sample "
                         f"windows was downloaded and ingested (python -m pdmodel.freddie).")
    oot_names = [n for n, _, _ in cfg.oot]
    eval_names = ["test"] + (["calib"] if cfg.calib else []) + oot_names
    res.artefacts.update(samples=samples, problems=problems, oot_names=oot_names, eval_names=eval_names)
    train = samples["train"]
    oot_all = pd.concat([samples[n] for n in oot_names])

    # ---------------- Development ----------------
    champ = PDScorecard(cfg, th.iv_min).fit(train)
    chall = Challenger(cfg).fit(train)
    res.artefacts.update(champion=champ, challenger=chall)
    y = {k: d[cfg.target].to_numpy() for k, d in samples.items()}
    p_raw = {k: champ.predict_pd_raw(d) for k, d in samples.items()}
    if cfg.calib:
        a, b = champ.recalibrate(samples["calib"])
        res.add("CAL-SLOPE", "Outcomes Analysis", f"Recalibration slope on calibration sample (intercept {a:.2f})",
                b, f"GREEN {th.slope_green[0]}-{th.slope_green[1]}, AMBER {th.slope_amber[0]}-{th.slope_amber[1]}",
                m.band_status(b, th.slope_green, th.slope_amber))
    p = {k: champ.predict_pd(d) for k, d in samples.items()}
    res.artefacts.update(y=y, p=p, p_raw=p_raw)
    if cfg.calib:
        res.tables["recalibration"] = pd.DataFrame([
            {"sample": k, "default_rate": y[k].mean(), "avg_pd_before": p_raw[k].mean(),
             "avg_pd_after": p[k].mean(), "O/E_before": m.oe_ratio(y[k], p_raw[k]),
             "O/E_after": m.oe_ratio(y[k], p[k])}
            for k in samples
        ]).set_index("sample")

    # Default-definition sensitivity: same model, alternative label that also counts
    # 90+ DPD during forbearance / disaster as default.
    if "forbearance_dq_excluded" in df.columns:
        rows = []
        for k, d in samples.items():
            alt = (d[cfg.target].astype(bool) | d["forbearance_dq_excluded"].astype(bool)).to_numpy().astype(int)
            rows.append({"sample": k, "default_rate": y[k].mean(), "default_rate_incl_forbearance": alt.mean(),
                         "AUC": m.auc(y[k], p[k]), "AUC_incl_forbearance": m.auc(alt, p[k]),
                         "O/E": m.oe_ratio(y[k], p[k]), "O/E_incl_forbearance": m.oe_ratio(alt, p[k])})
        res.tables["default_definition"] = pd.DataFrame(rows).set_index("sample")

    # ---------------- Conceptual soundness ----------------
    iv = champ.woe.iv()
    res.tables["iv"] = iv.rename("IV").to_frame()
    res.add("CS-01", "Conceptual Soundness", "Features retained (IV >= threshold)",
            float(len(champ.selected)), f">= 4 features with IV >= {th.iv_min}",
            "GREEN" if len(champ.selected) >= 4 else "RED")
    coefs = champ.coefficients()
    res.tables["coefficients"] = coefs.rename("coef").to_frame()
    res.tables["dropped_for_sign"] = pd.DataFrame(
        champ.dropped_for_sign, columns=["feature", "coef_before_drop"]).set_index("feature")
    # With WoE = ln(good/bad), every coefficient must be negative when predicting default.
    wrong_sign = int((coefs >= 0).sum())
    res.add("CS-02", "Conceptual Soundness", "Coefficients with wrong sign", float(wrong_sign),
            "== 0", "GREEN" if wrong_sign == 0 else "RED")
    non_mono = 0
    for f in champ.selected:
        b = champ.woe.bins[f]
        if b.categorical:
            continue
        t = b.table.drop(index="MISSING", errors="ignore")
        t = t.loc[sorted(t.index, key=lambda s: int(s.split("_")[1]))]
        r = t["bad_rate"].to_numpy()
        if not (np.all(np.diff(r) >= 0) or np.all(np.diff(r) <= 0)):
            non_mono += 1
    res.add("CS-03", "Conceptual Soundness", "Numeric features with non-monotonic bad rate",
            float(non_mono), "== 0", "GREEN" if non_mono == 0 else "RED")

    # Multicollinearity on the model's own (WoE) inputs
    v = m.vif(champ.woe.transform(train[list(cfg.features)])[champ.selected])
    res.tables["vif"] = v.rename("VIF").to_frame()
    res.add("CS-05", "Conceptual Soundness", f"Max variance inflation factor ({v.idxmax()})",
            float(v.max()), f"GREEN <= {th.vif_green}, AMBER <= {th.vif_amber}",
            m.traffic_light(float(v.max()), th.vif_green, th.vif_amber, higher_is_better=False))

    base = samples["test"]
    rows = []
    for name, col, delta in cfg.shocks:
        shocked = base.copy()
        shocked[col] = shocked[col] + delta
        chg = float(champ.predict_pd(shocked).mean() - champ.predict_pd(base).mean())
        ok = chg > 0 or col not in champ.selected
        rows.append({"shock": name, "avg_pd_change_bps": chg * 1e4, "intuitive": bool(ok)})
    sens = pd.DataFrame(rows).set_index("shock")
    res.tables["sensitivity"] = sens
    bad = int((~sens["intuitive"]).sum())
    res.add("CS-04", "Conceptual Soundness", "Counter-intuitive sensitivity responses",
            float(bad), "== 0", "GREEN" if bad == 0 else "RED")

    # ---------------- Outcomes analysis: discrimination ----------------
    disc = pd.DataFrame([
        {"sample": k, "n": len(y[k]), "defaults": int(y[k].sum()), "default_rate": y[k].mean(),
         "avg_pd": p[k].mean(), "O/E": m.oe_ratio(y[k], p[k]), "Brier": m.brier(y[k], p[k]),
         "AUC": m.auc(y[k], p[k]), "Gini": m.gini(y[k], p[k]), "KS": m.ks(y[k], p[k])}
        for k in ["train"] + eval_names
    ]).set_index("sample")
    res.tables["discrimination"] = disc
    for k in eval_names:
        a, s = disc.loc[k, "AUC"], disc.loc[k, "KS"]
        res.add(f"OA-AUC-{k}", "Outcomes Analysis", f"AUC ({k})", a,
                f"GREEN >= {th.auc_green}, AMBER >= {th.auc_amber}",
                m.traffic_light(a, th.auc_green, th.auc_amber))
        res.add(f"OA-KS-{k}", "Outcomes Analysis", f"KS ({k})", s,
                f"GREEN >= {th.ks_green}, AMBER >= {th.ks_amber}",
                m.traffic_light(s, th.ks_green, th.ks_amber))
    worst = disc.loc[oot_names, "AUC"].idxmin()
    decay = disc.loc["train", "AUC"] - disc.loc[worst, "AUC"]
    res.add("OA-DECAY", "Outcomes Analysis", f"AUC decay train -> worst OOT ({worst})", decay,
            f"<= {th.auc_decay_max}", "GREEN" if decay <= th.auc_decay_max else "RED")

    # ---------------- Outcomes analysis: calibration ----------------
    for k in eval_names:
        _, pval, hl = m.hosmer_lemeshow(y[k], p[k])
        res.tables[f"hl_{k}"] = hl
        res.add(f"CAL-HL-{k}", "Outcomes Analysis", f"Hosmer-Lemeshow p-value ({k})", pval,
                f">= {th.hl_pvalue_min}", "GREEN" if pval >= th.hl_pvalue_min else "AMBER")
        oe = m.oe_ratio(y[k], p[k])
        res.add(f"CAL-OE-{k}", "Outcomes Analysis", f"O/E ratio, observed / expected defaults ({k})",
                oe, f"GREEN {th.oe_green[0]}-{th.oe_green[1]}, AMBER {th.oe_amber[0]}-{th.oe_amber[1]}",
                m.band_status(oe, th.oe_green, th.oe_amber))
        jp = m.jeffreys_pvalue(int(y[k].sum()), len(y[k]), float(p[k].mean()))
        res.add(f"CAL-JEF-{k}", "Outcomes Analysis", f"Jeffreys test p-value, PD underestimation ({k})",
                jp, f"GREEN >= {th.jeffreys_amber}, AMBER >= {th.jeffreys_red}",
                m.traffic_light(jp, th.jeffreys_amber, th.jeffreys_red))
    y_oot, p_oot = oot_all[cfg.target].to_numpy(), champ.predict_pd(oot_all)
    gb = m.grade_binomial(y_oot, p_oot, champ.grade(oot_all))
    res.tables["grade_binomial_oot"] = gb
    under = int((gb["p_value"] < 0.01).sum())
    res.add("CAL-GRADE", "Outcomes Analysis", "Grades where PD is significantly too low (all OOT, 1%)",
            float(under), "== 0", "GREEN" if under == 0 else "RED")
    rank_breaks = int((np.diff(gb["odr"].to_numpy()) < 0).sum())
    res.add("CAL-RANK", "Outcomes Analysis", "Rank-order breaks in observed default rate by grade",
            float(rank_breaks), "<= 1", "GREEN" if rank_breaks <= 1 else "AMBER")

    # ---------------- Ongoing monitoring: stability ----------------
    s_dev = champ.score(train)
    res.artefacts.update(score_dev=s_dev, score_oot=champ.score(oot_all))
    for k in oot_names:
        v = m.psi(s_dev, champ.score(samples[k]))
        res.add(f"MON-PSI-{k}", "Ongoing Monitoring", f"Score PSI (dev -> {k})", v,
                f"GREEN <= {th.psi_green}, AMBER <= {th.psi_amber}",
                m.traffic_light(v, th.psi_green, th.psi_amber, higher_is_better=False))
    csi = {}
    for k in oot_names:
        for f in champ.selected:
            a, b = train[f], samples[k][f]
            csi[(f, k)] = (m.psi_categorical(a, b) if f in cfg.categorical
                           else m.psi(a.dropna(), b.dropna()))
    csi_t = pd.Series(csi).unstack()[oot_names]
    res.tables["csi"] = csi_t
    (wf, wk), wv = csi_t.stack().idxmax(), float(csi_t.stack().max())
    res.add("MON-CSI", "Ongoing Monitoring", f"Max characteristic stability index ({wf}, {wk})",
            wv, f"GREEN <= {th.psi_green}, AMBER <= {th.psi_amber}",
            m.traffic_light(wv, th.psi_green, th.psi_amber, higher_is_better=False))

    # ---------------- Benchmarking ----------------
    a_champ, a_ch = m.auc(y_oot, p_oot), m.auc(y_oot, chall.predict_pd(oot_all))
    res.tables["benchmark"] = pd.DataFrame(
        {"AUC (all OOT)": [a_champ, a_ch]},
        index=["Champion: WoE scorecard", "Challenger: gradient boosting"])
    res.add("BM-01", "Benchmarking", "Challenger AUC minus champion AUC (all OOT)", a_ch - a_champ,
            f"<= {th.challenger_gap_max}",
            "GREEN" if a_ch - a_champ <= th.challenger_gap_max else "AMBER")
    return res
