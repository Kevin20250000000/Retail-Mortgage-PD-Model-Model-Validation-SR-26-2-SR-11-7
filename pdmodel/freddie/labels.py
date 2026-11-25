"""12-month default label for newly originated mortgages.

Window: the first 12 monthly reporting periods starting at the first payment date.

Default (any month in the window):
  * 90+ days past due (delinquency status >= 03) or REO acquisition (RA), or
  * a credit-loss zero balance: 02 third-party sale, 03 short sale / charge-off, 09 REO.

Policies (set in LabelPolicy, documented in the validation report):
  * forbearance: 90+ DPD reported while in forbearance (assistance plan F) or flagged as
    disaster-related is NOT counted as default unless a credit-loss zero balance follows.
  * prepayment: loans paid off (01) inside the window without a prior default are kept as
    non-defaults (default) or dropped.
  * right-censoring: loans whose 12-month window is not fully observed are dropped.
  * defects: loans with zero balance code 96 are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from pdmodel.freddie.layout import CREDIT_EVENT_ZB, EXCLUDE_ZB, PREPAY_ZB

WINDOW = 12


@dataclass(frozen=True)
class LabelPolicy:
    exclude_forbearance_dq: bool = True
    keep_prepaid: bool = True
    window_months: int = WINDOW


DEFAULT_POLICY = LabelPolicy()


def _month_index(col: str) -> pl.Expr:
    v = pl.col(col).cast(pl.Int64)
    return (v // 100) * 12 + (v % 100)


def build_labels(orig: pl.LazyFrame, perf: pl.LazyFrame, data_cutoff: int,
                 policy: LabelPolicy = DEFAULT_POLICY) -> pl.LazyFrame:
    """Return one row per loan_id with default_12m and audit flags.

    orig needs loan_id, first_payment_date; perf needs the PERF_NEEDED columns as strings.
    data_cutoff is the last reporting period in the performance file (YYYYMM).
    """
    w = policy.window_months
    cutoff_idx = (data_cutoff // 100) * 12 + data_cutoff % 100

    p = (
        perf.join(orig.select("loan_id", "first_payment_date"), on="loan_id", how="inner")
        .with_columns(m=_month_index("period") - _month_index("first_payment_date"))
        .filter((pl.col("m") >= 0) & (pl.col("m") < w))
        .with_columns(
            dq_num=pl.col("dq_status").cast(pl.Int64, strict=False),
            zb=pl.col("zero_balance_code").fill_null(""),
            in_forb=(pl.col("assistance_plan") == "F").fill_null(False)
            | (pl.col("dq_disaster") == "Y").fill_null(False),
        )
        .with_columns(
            dq90=((pl.col("dq_num") >= 3).fill_null(False) | (pl.col("dq_status") == "RA")),
            credit_zb=pl.col("zb").is_in(list(CREDIT_EVENT_ZB)),
            prepay_zb=pl.col("zb").is_in(list(PREPAY_ZB)),
            defect_zb=pl.col("zb").is_in(list(EXCLUDE_ZB)),
        )
    )
    dq_counts = pl.col("dq90") & ~pl.col("in_forb") if policy.exclude_forbearance_dq else pl.col("dq90")
    p = p.with_columns(default_evt=dq_counts | pl.col("credit_zb"))

    agg = p.group_by("loan_id").agg(
        default_m=pl.col("m").filter(pl.col("default_evt")).min(),
        prepay_m=pl.col("m").filter(pl.col("prepay_zb")).min(),
        defect=pl.col("defect_zb").any(),
        forb_dq=(pl.col("dq90") & pl.col("in_forb")).any(),
        max_m=pl.col("m").max(),
    )

    out = (
        orig.select("loan_id", "first_payment_date")
        .join(agg, on="loan_id", how="left")
        .with_columns(
            window_end_idx=_month_index("first_payment_date") + (w - 1),
            defaulted=pl.col("default_m").is_not_null()
            & (pl.col("prepay_m").is_null() | (pl.col("default_m") <= pl.col("prepay_m"))),
        )
        .with_columns(
            prepaid=pl.col("prepay_m").is_not_null() & ~pl.col("defaulted"),
            terminated=pl.col("defaulted") | pl.col("prepay_m").is_not_null()
            | pl.col("defect").fill_null(False),
        )
        .with_columns(
            censored=(pl.col("window_end_idx") > cutoff_idx)
            | (~pl.col("terminated") & (pl.col("max_m").fill_null(-1) < w - 1)),
        )
        .with_columns(
            default_12m=pl.col("defaulted").cast(pl.Int8),
            forbearance_dq_excluded=(pl.col("forb_dq") & ~pl.col("defaulted")).fill_null(False),
            defect=pl.col("defect").fill_null(False),
        )
    )
    drop = pl.col("censored") | pl.col("defect")
    if not policy.keep_prepaid:
        drop = drop | pl.col("prepaid")
    return out.with_columns(dropped=drop).select(
        "loan_id", "default_12m", "prepaid", "forbearance_dq_excluded", "censored", "defect", "dropped"
    )
