"""Merge gate: no RED check may pass CI unless it is a documented, human-approved finding."""

import warnings
from datetime import date

import pytest

from pdmodel.validation import run_validation

CHECK_IDS = [c.id for c in run_validation().checks]
REQUIRED_FINDING_FIELDS = {"severity", "rationale", "remediation", "owner", "approved_by", "review_by"}


@pytest.mark.parametrize("check_id", CHECK_IDS)
def test_no_unapproved_red(check_id, checks, findings):
    c = checks[check_id]
    if c.status == "AMBER":
        warnings.warn(f"{c.id} AMBER: {c.description} = {c.value:.4f} ({c.rule})", stacklevel=1)
    if c.status == "RED":
        assert c.id in findings, (
            f"{c.id} is RED: {c.description} = {c.value:.4f}, rule {c.rule}. "
            "Fix the model, or document it in baseline/<source>/accepted_findings.json for human sign-off."
        )


def test_findings_are_complete_and_current(findings):
    for fid, f in findings.items():
        missing = REQUIRED_FINDING_FIELDS - set(f)
        assert not missing, f"Finding {fid} missing fields: {missing}"
        assert date.fromisoformat(f["review_by"]) >= date.today(), f"Finding {fid} is past review date"


def test_no_stale_findings(checks, findings):
    """A finding that is no longer RED must be closed, so the register stays truthful."""
    for fid in findings:
        assert fid in checks, f"Finding {fid} refers to a check that no longer exists"
        assert checks[fid].status == "RED", f"Finding {fid} is now {checks[fid].status}: close it"
