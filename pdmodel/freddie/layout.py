"""Freddie Mac Single-Family Loan-Level Dataset file layout (Release 47, July 2026).

Source: Freddie Mac SFLLD General User Guide, July 2026. Files are pipe-delimited with no
header. If Freddie Mac changes the layout, edit the two name lists below; `check_layout`
will refuse to run on a file whose column count or content does not match.
"""

from __future__ import annotations

ORIG_COLUMNS: list[str] = [
    "fico",                 # 1  Classic FICO (9999 = not available)
    "first_payment_date",   # 2  YYYYMM
    "first_time_buyer",     # 3  Y / N / 9
    "maturity_date",        # 4
    "msa",                  # 5
    "mi_pct",               # 6  999 = not available, 0 = no MI
    "num_units",            # 7
    "occupancy",            # 8  P / S / I / 9
    "cltv",                 # 9  999 = not available
    "dti",                  # 10 999 = not available (incl. > 65%)
    "orig_upb",             # 11
    "ltv",                  # 12 999 = not available
    "orig_rate",            # 13
    "channel",              # 14
    "ppm_flag",             # 15
    "amortization_type",    # 16
    "state",                # 17
    "property_type",        # 18
    "postal_code",          # 19
    "loan_id",              # 20 e.g. F14Q1xxxxxxx
    "loan_purpose",         # 21 P / C / N / R / 9
    "orig_term",            # 22
    "num_borrowers",        # 23
    "seller_name",          # 24
    "super_conforming",     # 25
    "pre_harp_loan_id",     # 26
    "program_indicator",    # 27
    "harp_indicator",       # 28
    "valuation_method",     # 29
    "io_indicator",         # 30
    "vantage_score",        # 31 VantageScore 4.0 (9999 = not available)
]

PERF_COLUMNS: list[str] = [
    "loan_id",              # 1
    "period",               # 2  YYYYMM
    "current_upb",          # 3
    "dq_status",            # 4  00, 01, 02, 03 ... 99, RA = REO acquisition, XX = n/a
    "loan_age",             # 5
    "remaining_months",     # 6
    "defect_settlement_date",  # 7
    "mod_flag",             # 8
    "zero_balance_code",    # 9  01 prepaid, 02 3rd-party sale, 03 short sale, 09 REO, 96 defect ...
    "zero_balance_date",    # 10
    "current_rate",         # 11
    "non_int_upb",          # 12
    "ddlpi",                # 13
    "mi_recoveries",        # 14
    "net_sales_proceeds",   # 15
    "non_mi_recoveries",    # 16
    "total_expenses",       # 17
    "legal_costs",          # 18
    "maint_costs",          # 19
    "taxes_insurance",      # 20
    "misc_expenses",        # 21
    "actual_loss",          # 22
    "cum_mod_cost",         # 23
    "step_mod_flag",        # 24
    "payment_deferral",     # 25
    "eltv",                 # 26
    "zb_removal_upb",       # 27
    "dq_accrued_interest",  # 28
    "dq_disaster",          # 29 Y = delinquency due to disaster
    "assistance_plan",      # 30 F forbearance, R repayment, T trial
    "period_mod_cost",      # 31
    "int_bearing_upb",      # 32
    "mi_cancel",            # 33
    "servicer_name",        # 34
    "cramdown_costs",       # 35
]

# Only these columns are read from the (large) performance file.
PERF_NEEDED = ["loan_id", "period", "dq_status", "zero_balance_code", "dq_disaster", "assistance_plan"]

MISSING_CODES = {"fico": 9999, "vantage_score": 9999, "dti": 999, "ltv": 999, "cltv": 999,
                 "mi_pct": 999, "num_borrowers": 99, "num_units": 99}
MISSING_TEXT = {"first_time_buyer": "9", "occupancy": "9", "loan_purpose": "9",
                "property_type": "99", "channel": "9"}

# Zero balance codes that represent a credit loss event
CREDIT_EVENT_ZB = {"02", "03", "09"}
PREPAY_ZB = {"01"}
EXCLUDE_ZB = {"96"}  # confirmed defect: loan repurchased, outcome not informative
