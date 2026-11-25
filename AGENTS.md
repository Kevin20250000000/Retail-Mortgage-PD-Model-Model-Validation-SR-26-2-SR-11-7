# AGENTS.md

Rules for any coding agent (or person) changing this repo. Kept short on purpose:
anything that can be enforced by code is enforced by ruff or a test, not written here.

## Definition of done
`ruff check .` and `pytest -q` both pass. CI attaches the regenerated validation report to every PR.
Reviewers read the report and check table, not the diff line by line.

## Change classes (from model risk "material change" practice)
| Change | Examples | Merge rule |
|---|---|---|
| Non-material | refactor, performance, docs, charts | Auto-merge when CI is green and baseline unchanged |
| Material | features, binning, target definition, sample windows or thresholds in `config.py`, `--update-baseline`, edits to `baseline/<source>/` | Human review required |

## Judgement calls (cannot be linted)
- PD target is 12-month default from origination. Do not mix horizons.
- Development and OOT windows must not overlap. Never tune on OOT.
- A RED check is fixed or documented as a finding with rationale, remediation, owner and review date. Never loosen a threshold to make CI pass.
- Real data lives in `data/` and is never committed (Freddie Mac licence). Label policy changes in `pdmodel/freddie/labels.py` are material.

## When the agent makes a mistake
Add a test or a lint rule that would have caught it. Do not add another line to this file.
