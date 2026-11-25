"""Freddie Mac ingestion: label logic on known cases, layout self-check, end-to-end run."""

import json
import zipfile

import pandas as pd
import pytest

from pdmodel.config import FREDDIE_SETTINGS
from pdmodel.freddie.fake import edge_cases, write_fake_vintages
from pdmodel.freddie.ingest import LayoutError, check_layout, ingest_year, load_processed
from pdmodel.freddie.labels import LabelPolicy, build_labels
from pdmodel.freddie.layout import ORIG_COLUMNS, PERF_COLUMNS
from pdmodel.freddie.macro import add_sato
from pdmodel.validation import run_validation


@pytest.fixture(scope="module")
def labels():
    orig, perf, cutoff, expected = edge_cases()
    out = build_labels(orig, perf, cutoff).collect()
    return {r["loan_id"]: r for r in out.iter_rows(named=True)}, expected


@pytest.mark.parametrize("loan_id", list(edge_cases()[3]))
def test_label_edge_case(loan_id, labels):
    got, expected = labels
    for field, value in expected[loan_id].items():
        assert got[loan_id][field] == value, f"{loan_id}.{field}: got {got[loan_id][field]}, expected {value}"


def test_forbearance_policy_switch():
    orig, perf, cutoff, _ = edge_cases()
    out = build_labels(orig, perf, cutoff, LabelPolicy(exclude_forbearance_dq=False)).collect()
    row = out.filter(out["loan_id"] == "F20Q1FORBEAR").row(0, named=True)
    assert row["default_12m"] == 1


def test_window_beyond_cutoff_is_censored():
    orig, perf, _, _ = edge_cases()
    out = build_labels(orig, perf, data_cutoff=202006).collect()  # window ends 202012
    assert out["censored"].all()


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("freddie")
    raw, processed = root / "raw", root / "processed"
    write_fake_vintages(raw, n_per_year=1_500)
    # Exercise the zip path for one vintage, as downloaded from Freddie Mac
    with zipfile.ZipFile(raw / "sample_2014.zip", "w") as zf:
        for kind in ("orig", "perf"):
            f = raw / f"sample_{kind}_2014.txt"
            zf.write(f, f.name)
            f.unlink()
    for year in range(2014, 2024):
        ingest_year(raw, processed, year)
    return raw, processed


def test_layout_check_rejects_wrong_columns(fake_data, tmp_path):
    bad = tmp_path / "sample_orig_2099.txt"
    bad.write_text("700|201401|N\n")
    with pytest.raises(LayoutError, match="columns"):
        check_layout(bad, ORIG_COLUMNS, "origination")


def test_layout_check_accepts_real_layout(fake_data):
    raw, _ = fake_data
    check_layout(raw / "sample_orig_2015.txt", ORIG_COLUMNS, "origination")
    check_layout(raw / "sample_perf_2015.txt", PERF_COLUMNS, "performance")


def test_processed_table(fake_data):
    _, processed = fake_data
    df = load_processed(processed)
    assert set(df["vintage_year"]) == set(range(2014, 2024))
    assert df["loan_id"].is_unique
    assert df["default_12m"].between(0, 1).all() and df["default_12m"].sum() > 0
    assert df["fico"].max() <= 850  # 9999 converted to missing


def test_end_to_end_freddie_validation(fake_data):
    _, processed = fake_data
    res = run_validation(FREDDIE_SETTINGS, data_dir=processed)  # raw dir defaults to processed/../raw
    ids = res.by_id()
    assert ids["DQ-01"].status == "GREEN"
    assert "sato" in res.artefacts["samples"]["train"].columns
    for seg in ("calib", "oot1_2019", "oot2_2020_21", "oot3_2022_23"):
        for check in ("OA-AUC", "CAL-OE", "CAL-JEF"):
            assert f"{check}-{seg}" in ids


def test_harp_loans_excluded(fake_data):
    _, processed = fake_data
    df = load_processed(processed)
    assert df["ltv"].max() <= 200 and "harp_indicator" not in df.columns
    summary = json.loads((processed / "summary_2015.json").read_text())
    assert summary["dropped_harp"] > 0


def test_sato_removes_market_level(tmp_path):
    raw = tmp_path
    (raw / "MORTGAGE30US.csv").write_text("observation_date,MORTGAGE30US\n2022-01-06,3.22\n2022-01-13,3.45\n")
    df = pd.DataFrame({"first_payment_date": [202203], "orig_rate": [3.75]})  # originated Jan 2022
    out = add_sato(df, raw)
    assert out["pmms_rate"].iloc[0] == pytest.approx(3.335)
    assert out["sato"].iloc[0] == pytest.approx(0.415)


def test_missing_pmms_month_is_an_error(tmp_path):
    (tmp_path / "MORTGAGE30US.csv").write_text("DATE,MORTGAGE30US\n2022-01-06,3.22\n")
    with pytest.raises(ValueError, match="PMMS rate missing"):
        add_sato(pd.DataFrame({"first_payment_date": [202403], "orig_rate": [7.0]}), tmp_path)


def test_validation_is_independent_of_row_order(fake_data, tmp_path):
    """Same loans in a different file/row order must give identical results."""
    import polars as pl
    _, processed = fake_data
    shuffled = tmp_path / "processed"
    shuffled.mkdir()
    for f in processed.glob("loans_*.parquet"):
        pl.read_parquet(f).sample(fraction=1.0, shuffle=True, seed=3).write_parquet(shuffled / f.name)
    raw = processed.parent / "raw"
    a = run_validation(FREDDIE_SETTINGS, data_dir=processed, raw_dir=raw).key_metrics()
    b = run_validation(FREDDIE_SETTINGS, data_dir=shuffled, raw_dir=raw).key_metrics()
    assert a == b
