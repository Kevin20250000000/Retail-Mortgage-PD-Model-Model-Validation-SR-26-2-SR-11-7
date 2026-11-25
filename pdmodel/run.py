"""CLI: python -m pdmodel.run [--source synthetic|freddie] [--data data/processed] [--update-baseline]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdmodel.config import FREDDIE_SETTINGS, SETTINGS
from pdmodel.report import write_report
from pdmodel.validation import run_validation

ROOT = Path(__file__).resolve().parent.parent


def baseline_dir(source: str) -> Path:
    return ROOT / "baseline" / source


def load_findings(source: str) -> dict:
    f = baseline_dir(source) / "accepted_findings.json"
    return json.loads(f.read_text()) if f.exists() else {}


def load_conclusion(source: str) -> dict | None:
    f = baseline_dir(source) / "conclusion.json"
    return json.loads(f.read_text()) if f.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["synthetic", "freddie"], default="synthetic")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"),
                    help="Folder of processed Freddie Mac vintages (python -m pdmodel.freddie)")
    ap.add_argument("--raw", default=None, help="Folder with MORTGAGE30US.csv (default: <data>/../raw)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--update-baseline", action="store_true",
                    help="Overwrite the approved baseline. Material change: needs human review.")
    args = ap.parse_args()

    settings = FREDDIE_SETTINGS if args.source == "freddie" else SETTINGS
    data_dir = Path(args.data)
    res = run_validation(settings, data_dir=data_dir, raw_dir=Path(args.raw) if args.raw else None)
    out = Path(args.out) if args.out else ROOT / "reports" / args.source
    path = write_report(res, out, load_findings(args.source), data_dir if args.source == "freddie" else None,
                        load_conclusion(args.source))
    print(res.summary()[["id", "description", "value", "status"]].to_string(index=False))
    print(f"\nReport: {path}")
    if args.update_baseline:
        bd = baseline_dir(args.source)
        bd.mkdir(parents=True, exist_ok=True)
        (bd / "metrics_baseline.json").write_text(json.dumps(res.key_metrics(), indent=2) + "\n")
        print(f"Baseline updated: {bd / 'metrics_baseline.json'}")


if __name__ == "__main__":
    main()
