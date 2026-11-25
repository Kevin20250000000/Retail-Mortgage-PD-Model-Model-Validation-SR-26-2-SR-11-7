# Retail Mortgage PD Model & Model Validation (SR 26-2 / SR 11-7)

A 12-month Probability of Default (PD) model for **newly originated mortgages**, built on the
Freddie Mac Single-Family Loan-Level Dataset, with an **independent-style validation suite that
runs on every change**. Validation is structured around SR 26-2 (the April 2026 interagency
guidance that replaced SR 11-7): data, conceptual soundness, outcomes analysis, ongoing
monitoring, benchmarking. Every check is a test, and CI won't merge a change while any check
is RED and undocumented.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                                   # merge gate: runs without any real data
python -m pdmodel.run                       # demo on synthetic data -> reports/synthetic/
```

## Run on Freddie Mac data

1. Register (free) at Freddie Mac's Single-Family Loan-Level Dataset page and download the
   **sample** files for 2014–2023 (`sample_2014.zip` … `sample_2023.zip`).
2. Put the zips, unopened, in `data/raw/`. Also download the FRED series **MORTGAGE30US**
   (Download → CSV) into `data/raw/MORTGAGE30US.csv`; it is used to compute the spread at
   origination. The data is licensed and is **not** committed (`data/` is in `.gitignore`).
3. Build the loan-level table, then validate:

```bash
python -m pdmodel.freddie --raw data/raw --out data/processed   # ~1 min for 10 vintages
python -m pdmodel.run --source freddie                          # -> reports/freddie/
```

The ingest step prints, per vintage: loans, loans kept, defaults, default rate, censored,
prepaid and forbearance cases. It checks the file layout first and stops with a clear message
if Freddie Mac has changed it (edit `pdmodel/freddie/layout.py`).

### Default definition

Within the first 12 monthly periods after the first payment date:

- 90+ days past due, or REO acquisition (`RA`), or a credit-loss zero balance
  (02 third-party sale, 03 short sale / charge-off, 09 REO disposition).
- 90+ DPD reported while in **forbearance** (assistance plan `F`) or flagged as disaster-related
  is not counted as default (switch: `--count-forbearance-dq`).
- Loans **prepaid** inside the window are non-defaults (switch: `--drop-prepaid`).
- Loans without a fully observed 12-month window (**right-censoring**), repurchased
  defects (zero balance 96) and **HARP** relief refinances are excluded.

### Features

FICO, DTI, LTV, **SATO** (note rate minus the PMMS 30-year rate in the origination month),
loan amount, number of borrowers, occupancy, purpose, first-time buyer, property type, channel.
Run 1 used the raw note rate, CLTV and MI %. Validation flagged two problems: the raw rate
drifted with the rate cycle (CSI 3.1, score PSI red in 2020–23), and CLTV/MI % duplicated LTV,
which gave MI % a wrong-sign coefficient. Remediation: SATO replaces the raw rate, CLTV and MI %
are dropped, and a VIF check was added.

### Sample windows

| Sample | Vintages | Purpose |
|---|---|---|
| Development | 2014–2017 | Train / test (70/30 random) |
| Validation & calibration | 2018 | Hold-out for calibration |
| OOT I | 2019 | Pre-pandemic |
| OOT II | 2020–2021 | Pandemic shock and forbearance |
| OOT III | 2022–2023 | Rate-hiking cycle |

## What the validation checks

| Area | Checks |
|---|---|
| Data | Schema and ranges, missingness |
| Conceptual soundness | Information value, coefficient signs, monotonic binning, VIF, shock sensitivity |
| Outcomes analysis | AUC / KS / Gini per sample; calibration by O/E ratio and Jeffreys test (primary) and Hosmer-Lemeshow; Brier score; per-grade binomial test; rank ordering; AUC decay |
| Ongoing monitoring | Score PSI and characteristic stability (CSI) for each OOT segment |
| Benchmarking | Gradient-boosting challenger vs WoE scorecard champion |

## Design

| Module | Role |
|---|---|
| `pdmodel/freddie/` | File layout, layout self-check, per-vintage Polars ingestion to Parquet, default labels |
| `pdmodel/woe.py` | Weight-of-Evidence binning with monotonic bad-rate enforcement and a missing bin |
| `pdmodel/model.py` | Champion: WoE logistic scorecard (points, PDO 20, 10 grades). Challenger: gradient boosting |
| `pdmodel/metrics.py` | AUC, Gini, KS, Hosmer-Lemeshow, per-grade binomial test, PSI/CSI |
| `pdmodel/validation.py` | Runs every check and returns structured evidence with GREEN/AMBER/RED status |
| `pdmodel/config.py` | Features, sample windows and all thresholds in one auditable place |
| `baseline/<source>/` | Approved metric baseline and findings register |
| `tests/` | Merge gate, known-answer unit tests, label edge cases, reproducibility, regression vs baseline |

## How the workflow keeps the model honest

- **Errors are made impossible, not just discouraged.** Global RNG calls are banned by a lint rule and a test, and every number is reproducible.
- **Every validation step is a test.** A RED check fails CI unless it is a documented finding with rationale, remediation, owner and review date. A finding that is no longer RED also fails CI.
- **Label logic is tested on known cases.** These include 90+ DPD, REO, short sale, 60 DPD only, prepayment, forbearance, defect, data gap, and delinquency just outside the window.
- **Silent changes are caught.** Metrics are compared with an approved baseline. Updating it is a material change.
- **Reviewers read evidence.** CI regenerates the report and charts for every PR.

## Limitations
- Sample files (50,000 loans per vintage), agency-conforming fixed-rate loans only.
- Origination scorecard without macro inputs; point-in-time PD drifts with the economy.
- Binomial tests assume independent defaults; correlated defaults make them over-reject.
- 12-month PD only: no lifetime PD, LGD or EAD.

## Roadmap
1. Monotonic XGBoost challenger with SHAP
2. Macro stress testing: state unemployment (FRED) and house prices (FHFA), Fed scenarios
3. Independent replication of the champion in a separate validation module
4. SR 26-2-aligned validation report with findings register and approval conclusion
