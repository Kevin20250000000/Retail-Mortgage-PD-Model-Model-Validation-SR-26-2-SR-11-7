"""python -m pdmodel.freddie --raw data/raw --out data/processed [--years 2014 2015 ...]"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from pdmodel.freddie.ingest import ingest_year
from pdmodel.freddie.labels import LabelPolicy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--years", type=int, nargs="*", help="Default: every vintage found in --raw")
    ap.add_argument("--drop-prepaid", action="store_true")
    ap.add_argument("--count-forbearance-dq", action="store_true",
                    help="Count 90+ DPD during forbearance as default (default: not counted)")
    args = ap.parse_args()

    raw = Path(args.raw)
    years = args.years or sorted({int(m.group(1)) for f in raw.iterdir()
                                  if (m := re.search(r"sample_(?:orig_)?(\d{4})", f.name))})
    if not years:
        raise SystemExit(f"No Freddie Mac sample files found in {raw}")
    policy = LabelPolicy(exclude_forbearance_dq=not args.count_forbearance_dq,
                         keep_prepaid=not args.drop_prepaid)
    print(f"{'vintage':>7} {'loans':>8} {'kept':>8} {'defaults':>9} {'rate':>7} "
          f"{'censored':>9} {'harp':>6} {'prepaid':>8} {'forb.excl':>9}")
    for y in years:
        s = ingest_year(raw, Path(args.out), y, policy)
        rate = f"{s['default_rate']:.2%}" if s["default_rate"] is not None else "-"
        print(f"{y:>7} {s['loans_in_file']:>8} {s['loans_kept']:>8} {s['defaults']:>9} {rate:>7} "
              f"{s['dropped_censored']:>9} {s['dropped_harp']:>6} {s['prepaid_in_window']:>8} "
              f"{s['forbearance_dq_not_counted']:>9}")


if __name__ == "__main__":
    main()
