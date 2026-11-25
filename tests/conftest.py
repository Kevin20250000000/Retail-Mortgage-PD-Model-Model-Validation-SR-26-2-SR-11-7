import json
from pathlib import Path

import pytest

from pdmodel.validation import run_validation

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def result():
    """One full validation run shared by every test (about 3 seconds)."""
    return run_validation()


@pytest.fixture(scope="session")
def checks(result):
    return result.by_id()


@pytest.fixture(scope="session")
def findings():
    path = ROOT / "baseline" / "synthetic" / "accepted_findings.json"
    return json.loads(path.read_text()) if path.exists() else {}


@pytest.fixture(scope="session")
def baseline():
    path = ROOT / "baseline" / "synthetic" / "metrics_baseline.json"
    return json.loads(path.read_text()) if path.exists() else None
