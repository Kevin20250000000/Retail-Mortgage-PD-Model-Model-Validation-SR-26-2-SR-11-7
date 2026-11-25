"""Fake files in the exact Freddie Mac sample layout, for tests and CI (no real data needed).

`write_fake_vintages` writes sample_orig_YYYY.txt / sample_perf_YYYY.txt with realistic
relationships (FICO, DTI, LTV drive default; 2020 forbearance; prepayments; defects).
`edge_cases` returns hand-built loans whose correct labels are known exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from pdmodel.freddie.layout import ORIG_COLUMNS, PERF_COLUMNS

STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "WA", "AZ", "CO"]


def _add_months(yyyymm: int, k: int) -> int:
    idx = (yyyymm // 100) * 12 + (yyyymm % 100 - 1) + k
    return (idx // 12) * 100 + idx % 12 + 1


def _orig_row(loan_id, fpd, fico, dti, ltv, upb, rate, occ, purpose, ftb, ptype, channel, state, nb, harp=""):
    f = dict.fromkeys(ORIG_COLUMNS, "")
    f.update(fico=str(fico), first_payment_date=str(fpd), first_time_buyer=ftb,
             maturity_date=str(_add_months(fpd, 359)), msa="", mi_pct="0" if ltv <= 80 else "25",
             num_units="1", occupancy=occ, cltv=str(ltv), dti=str(dti), orig_upb=str(upb),
             ltv=str(ltv), orig_rate=f"{rate:.3f}", channel=channel, ppm_flag="N",
             amortization_type="FRM", state=state, property_type=ptype, postal_code="00000",
             loan_id=loan_id, loan_purpose=purpose, orig_term="360", num_borrowers=str(nb),
             seller_name="Other sellers", super_conforming="", pre_harp_loan_id="",
             program_indicator="9", harp_indicator=harp, valuation_method="9", io_indicator="N",
             vantage_score="9999")
    return "|".join(f[c] for c in ORIG_COLUMNS)


def _perf_row(loan_id, period, age, dq="00", zb="", disaster="", plan=""):
    f = dict.fromkeys(PERF_COLUMNS, "")
    f.update(loan_id=loan_id, period=str(period), current_upb="0" if zb else "200000.00",
             dq_status=dq, loan_age=str(age), remaining_months=str(360 - age),
             zero_balance_code=zb, current_rate="4.000", dq_disaster=disaster,
             assistance_plan=plan, servicer_name="Other servicers")
    return "|".join(f[c] for c in PERF_COLUMNS)


def write_fake_vintages(out: Path, years=range(2014, 2024), n_per_year: int = 4_000,
                        seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for year in years:
        orig_lines, perf_lines = [], []
        for i in range(n_per_year):
            q = int(rng.integers(1, 5))
            loan_id = f"F{year % 100:02d}Q{q}{i:07d}"
            fpd = _add_months(year * 100 + 1, 3 * (q - 1) + 2)  # ~2 months after quarter start
            fico = int(np.clip(rng.normal(752 - 4 * (year >= 2022), 45), 600, 840))
            dti = int(np.clip(rng.normal(35, 9), 5, 50))
            ltv = int(np.clip(rng.normal(77, 14), 20, 97))
            upb = int(np.clip(rng.normal(250, 90), 40, 800)) * 1000
            rate = float({2020: 3.2, 2021: 3.0, 2022: 5.2, 2023: 6.7}.get(year, 4.2) + rng.normal(0, .3))
            occ = rng.choice(["P", "I", "S"], p=[0.88, 0.08, 0.04])
            purpose = rng.choice(["P", "C", "N"], p=[0.5, 0.25, 0.25])
            ftb = rng.choice(["Y", "N", "9"], p=[0.25, 0.73, 0.02])  # 9 = not available
            ptype = rng.choice(["SF", "PU", "CO"], p=[0.65, 0.25, 0.10])
            channel = rng.choice(["R", "C", "B"], p=[0.5, 0.35, 0.15])
            nb = int(rng.choice([1, 2], p=[0.5, 0.5]))
            state = rng.choice(STATES)
            fico_s = 9999 if rng.random() < 0.004 else fico
            harp = "Y" if (year <= 2018 and rng.random() < 0.003) else ""
            ltv_s = int(rng.integers(150, 250)) if harp else ltv
            orig_lines.append(_orig_row(loan_id, fpd, fico_s, dti, ltv_s, upb, rate, occ,
                                        purpose, ftb, ptype, channel, state, nb, harp))

            z = (-5.0 - 0.022 * (fico - 750) + 0.035 * (dti - 35) + 0.03 * (ltv - 80)
                 + 0.5 * (occ == "I") + 0.3 * (nb == 1) + 0.4 * (purpose == "C"))
            pd_ = 1 / (1 + np.exp(-z))
            default = rng.random() < pd_
            d_month = int(rng.integers(3, 12)) if default else None
            prepay_m = int(rng.integers(1, 12)) if rng.random() < (0.10 if year in (2019, 2020) else 0.04) else None
            defect = rng.random() < 0.002
            # 2020 forbearance: 90+ DPD while in forbearance, then cure
            forb = (not default) and rng.random() < 0.03 and year in (2019, 2020)

            for m in range(14):
                period = _add_months(fpd, m)
                dq, zb, plan, dis = "00", "", "", ""
                if default and m >= d_month - 2:
                    dq = f"{min(m - (d_month - 2) + 1, 99):02d}"
                    if m == d_month + 1 and rng.random() < 0.3:
                        zb = rng.choice(["03", "09"])
                if forb and 2020 * 100 + 4 <= period <= 2020 * 100 + 12:
                    dq, plan = f"{min((period % 100) - 3, 5):02d}", "F"
                if prepay_m is not None and m == prepay_m and not (default and m >= d_month - 2):
                    zb = "01"
                if defect and m == 6:
                    zb = "96"
                perf_lines.append(_perf_row(loan_id, period, m + 1, dq, zb, dis, plan))
                if zb:
                    break
        (out / f"sample_orig_{year}.txt").write_text("\n".join(orig_lines) + "\n")
        (out / f"sample_perf_{year}.txt").write_text("\n".join(perf_lines) + "\n")
    write_fake_pmms(out)


def write_fake_pmms(out: Path) -> None:
    """Monthly MORTGAGE30US-style CSV (FRED download format) matching the fake rate levels."""
    levels = {2020: 3.2, 2021: 3.0, 2022: 5.2, 2023: 6.7}
    rows = ["observation_date,MORTGAGE30US"]
    for y in range(2013, 2026):
        for mth in range(1, 13):
            rows.append(f"{y}-{mth:02d}-01,{levels.get(y, 4.2):.2f}")
    (out / "MORTGAGE30US.csv").write_text("\n".join(rows) + "\n")


def edge_cases() -> tuple[pl.LazyFrame, pl.LazyFrame, int, dict[str, dict]]:
    """Hand-built loans with known labels. Returns orig, perf, data_cutoff, expected."""
    fpd, cutoff = 202001, 202312
    orig, perf, exp = [], [], {}

    def loan(name, months, expected):
        orig.append({"loan_id": name, "first_payment_date": fpd})
        for m, (dq, zb, plan) in enumerate(months):
            perf.append({"loan_id": name, "period": str(_add_months(fpd, m)), "dq_status": dq,
                         "zero_balance_code": zb, "dq_disaster": "", "assistance_plan": plan})
        exp[name] = expected

    cur = ("00", "", "")
    loan("F20Q1CURRENT", [cur] * 12, {"default_12m": 0, "dropped": False})
    loan("F20Q1DQ90", [cur] * 5 + [("01", "", ""), ("02", "", ""), ("03", "", "")] + [("04", "", "")] * 4,
         {"default_12m": 1, "dropped": False})
    loan("F20Q1DQ60ONLY", [cur] * 9 + [("01", "", ""), ("02", "", ""), cur],
         {"default_12m": 0, "dropped": False})
    loan("F20Q1REO", [cur] * 3 + [("RA", "", "")], {"default_12m": 1, "dropped": False})
    loan("F20Q1SHORTSALE", [cur] * 4 + [("00", "03", "")], {"default_12m": 1, "dropped": False})
    loan("F20Q1PREPAID", [cur] * 4 + [("00", "01", "")], {"default_12m": 0, "prepaid": True, "dropped": False})
    loan("F20Q1FORBEAR", [cur] * 3 + [("01", "", "F"), ("02", "", "F"), ("03", "", "F"), ("04", "", "F")] + [cur] * 5,
         {"default_12m": 0, "forbearance_dq_excluded": True, "dropped": False})
    loan("F20Q1DEFECT", [cur] * 6 + [("00", "96", "")], {"dropped": True})
    loan("F20Q1GAP", [cur] * 6, {"censored": True, "dropped": True})
    loan("F20Q1LATEDQ", [cur] * 12 + [("03", "", "")], {"default_12m": 0, "dropped": False})  # month 13: outside window

    o = pl.LazyFrame(orig)
    p = pl.LazyFrame(perf)
    return o, p, cutoff, exp
