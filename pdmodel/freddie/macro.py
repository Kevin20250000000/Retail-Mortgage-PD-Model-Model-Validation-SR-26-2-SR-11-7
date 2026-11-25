"""Spread at origination (SATO) from the Freddie Mac PMMS 30-year rate (FRED series MORTGAGE30US).

The note rate mixes two things: the market rate level when the loan was made, and the
borrower-specific risk premium. Only the second is borrower risk. Across 2014-2023 the
market level swung from ~3% to ~7%, so a model trained on the raw rate scores whole vintages
as safer or riskier for reasons unrelated to the borrower. SATO = note rate - market rate
in the origination month removes the market level.

Download: fred.stlouisfed.org -> search MORTGAGE30US -> Download -> CSV -> save as
data/raw/MORTGAGE30US.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PMMS_FILE = "MORTGAGE30US.csv"


def load_pmms(raw_dir: Path) -> pd.Series:
    """Monthly average PMMS rate indexed by YYYYMM (int)."""
    path = Path(raw_dir) / PMMS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the FRED series MORTGAGE30US as CSV and save it there "
            "(needed for the SATO feature)."
        )
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("observation_date", "date"))
    rate = pd.to_numeric(df["MORTGAGE30US"], errors="coerce")
    d = pd.to_datetime(df[date_col])
    s = pd.Series(rate.to_numpy(), index=d.dt.year * 100 + d.dt.month).dropna()
    return s.groupby(level=0).mean()


def origination_month(first_payment_date: pd.Series) -> pd.Series:
    """First payment is normally due in the second month after the note date."""
    idx = (first_payment_date // 100) * 12 + (first_payment_date % 100 - 1) - 2
    return (idx // 12) * 100 + idx % 12 + 1


def add_sato(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    pmms = load_pmms(raw_dir)
    month = origination_month(df["first_payment_date"].astype("int64"))
    market = month.map(pmms)
    missing = int(market.isna().sum())
    if missing:
        months = sorted(month[market.isna()].unique())[:5]
        raise ValueError(f"PMMS rate missing for {missing} loans (e.g. months {months}); "
                         "download the full MORTGAGE30US history.")
    out = df.copy()
    out["pmms_rate"] = market.to_numpy()
    out["sato"] = np.round(out["orig_rate"] - out["pmms_rate"], 4)
    return out
