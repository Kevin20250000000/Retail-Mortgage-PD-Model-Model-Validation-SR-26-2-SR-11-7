"""Reproducibility, regression against the approved baseline, and repo hygiene."""

import re
from pathlib import Path

import pytest

from pdmodel.validation import run_validation

ROOT = Path(__file__).resolve().parent.parent


def test_same_seed_same_numbers(result):
    assert run_validation().key_metrics() == result.key_metrics()


def test_different_seed_different_numbers(result):
    assert run_validation(seed=7).key_metrics() != result.key_metrics()


def test_metrics_match_approved_baseline(result, baseline):
    """Any silent change in model output fails CI. Update with --update-baseline (human review)."""
    if baseline is None:
        pytest.skip("No baseline yet: run `python -m pdmodel.run --update-baseline`")
    current = result.key_metrics()
    assert set(current) == set(baseline), "Set of checks changed: update baseline after review"
    drift = {k: (baseline[k], current[k]) for k in current if abs(current[k] - baseline[k]) > 0.01}
    assert not drift, f"Metrics moved beyond tolerance vs baseline: {drift}"


def test_no_global_rng_in_source():
    """Belt and braces for the ruff ban: randomness only via np.random.Generator."""
    pattern = re.compile(r"np\.random\.(?!default_rng|Generator)\w+")
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in (ROOT / "pdmodel").rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, f"Global RNG used at {offenders}"
