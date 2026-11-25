"""Read Freddie Mac sample files, one vintage at a time, into a loan-level Parquet table.

Usage:
    python -m pdmodel.freddie --raw data/raw --out data/processed

Put the downloaded files in data/raw/ either as the zip (sample_2014.zip) or extracted
(sample_orig_2014.txt, sample_perf_2014.txt). Only the needed performance columns and
the first 12 months of each loan are kept, so memory stays small.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path

import polars as pl

from pdmodel.freddie.labels import DEFAULT_POLICY, LabelPolicy, build_labels
from pdmodel.freddie.layout import (
    MISSING_CODES,
    MISSING_TEXT,
    ORIG_COLUMNS,
    PERF_COLUMNS,
    PERF_NEEDED,
)

NUMERIC_ORIG = ["fico", "mi_pct", "num_units", "cltv", "dti", "orig_upb", "ltv", "orig_rate",
                "orig_term", "num_borrowers", "vantage_score"]
KEEP_ORIG = ["loan_id", "vintage_year", "orig_quarter", "first_payment_date", "state", "msa",
             "fico", "dti", "ltv", "cltv", "orig_upb", "orig_rate", "orig_term", "mi_pct",
             "num_units", "num_borrowers", "occupancy", "loan_purpose", "first_time_buyer",
             "property_type", "channel", "harp_indicator"]


class LayoutError(ValueError):
    pass


def check_layout(path: Path, expected: list[str], kind: str) -> None:
    """Fail loudly if the file does not look like the layout we expect."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        first = fh.readline().rstrip("\n\r")
    n = first.count("|") + 1
    if n != len(expected):
        raise LayoutError(
            f"{path.name}: {kind} file has {n} columns, layout expects {len(expected)}. "
            "Freddie Mac may have changed the layout: update pdmodel/freddie/layout.py "
            "from the File Layout document."
        )
    fields = dict(zip(expected, first.split("|"), strict=True))
    if not re.fullmatch(r"F\d{2}Q[1-4]\w+", fields["loan_id"]):
        raise LayoutError(f"{path.name}: loan_id column holds {fields['loan_id']!r}, expected like F14Q1...")
    date_col = "first_payment_date" if kind == "origination" else "period"
    if not re.fullmatch(r"(19|20)\d{2}(0[1-9]|1[0-2])", fields[date_col]):
        raise LayoutError(f"{path.name}: {date_col} holds {fields[date_col]!r}, expected YYYYMM")


def _locate(raw: Path, year: int, tmp: Path) -> tuple[Path, Path]:
    orig, perf = raw / f"sample_orig_{year}.txt", raw / f"sample_perf_{year}.txt"
    if orig.exists() and perf.exists():
        return orig, perf
    z = raw / f"sample_{year}.zip"
    if not z.exists():
        raise FileNotFoundError(f"Neither {z.name} nor extracted sample_orig/perf_{year}.txt in {raw}")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
    return tmp / orig.name, tmp / perf.name


def _scan(path: Path, columns: list[str]) -> pl.LazyFrame:
    return pl.scan_csv(path, separator="|", has_header=False, new_columns=columns,
                       infer_schema=False, quote_char=None)


def clean_orig(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(pl.col(c).str.strip_chars().cast(pl.Float64, strict=False) for c in NUMERIC_ORIG)
    lf = lf.with_columns(
        pl.when(pl.col(c) == v).then(None).otherwise(pl.col(c)).alias(c) for c, v in MISSING_CODES.items()
    )
    lf = lf.with_columns(
        pl.when(pl.col(c).str.strip_chars() == v).then(None).otherwise(pl.col(c).str.strip_chars()).alias(c)
        for c, v in MISSING_TEXT.items()
    )
    return lf.with_columns(
        vintage_year=(2000 + pl.col("loan_id").str.slice(1, 2).cast(pl.Int32)),
        orig_quarter=pl.format("20{}Q{}", pl.col("loan_id").str.slice(1, 2), pl.col("loan_id").str.slice(4, 1)),
        first_payment_date=pl.col("first_payment_date").cast(pl.Int64),
    )


def ingest_year(raw: Path, out: Path, year: int, policy: LabelPolicy = DEFAULT_POLICY) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        orig_p, perf_p = _locate(raw, year, Path(tmp))
        check_layout(orig_p, ORIG_COLUMNS, "origination")
        check_layout(perf_p, PERF_COLUMNS, "performance")

        orig = clean_orig(_scan(orig_p, ORIG_COLUMNS)).select(KEEP_ORIG)
        perf = _scan(perf_p, PERF_COLUMNS).select(PERF_NEEDED)
        cutoff = int(perf.select(pl.col("period").cast(pl.Int64).max()).collect().item())
        labels = build_labels(orig, perf, cutoff, policy)
        loans = orig.join(labels, on="loan_id", how="left").collect()

    # HARP relief refinances are a policy programme (LTV can exceed 100%), not ordinary new
    # originations, so they are out of the model's scope.
    loans = loans.with_columns(harp=(pl.col("harp_indicator").fill_null("").str.strip_chars() == "Y"))
    kept = loans.filter(~pl.col("dropped") & ~pl.col("harp")).drop("dropped", "harp", "harp_indicator")
    kept = kept.sort("loan_id")  # polars joins/group-bys run in parallel: fix row order for reproducibility
    kept.write_parquet(out / f"loans_{year}.parquet")
    summary = {
        "vintage": year,
        "performance_cutoff": cutoff,
        "loans_in_file": loans.height,
        "dropped_censored": int(loans["censored"].sum()),
        "dropped_defect": int(loans["defect"].sum()),
        "dropped_harp": int(loans["harp"].sum()),
        "prepaid_in_window": int(loans["prepaid"].sum()),
        "forbearance_dq_not_counted": int(loans["forbearance_dq_excluded"].sum()),
        "loans_kept": kept.height,
        "defaults": int(kept["default_12m"].sum()),
        "default_rate": float(kept["default_12m"].mean()) if kept.height else None,
        "policy": policy.__dict__,
    }
    (out / f"summary_{year}.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_processed(path: Path, years: list[int] | None = None):
    """Load processed vintages as one pandas DataFrame for modelling."""
    files = sorted(path.glob("loans_*.parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in years]
    if not files:
        raise FileNotFoundError(f"No processed vintages in {path}. Run: python -m pdmodel.freddie")
    # Sorting makes the random train/test split independent of file and row order.
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").sort("loan_id")
    return df.to_pandas()
