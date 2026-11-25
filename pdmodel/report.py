"""Builds the validation report: one PNG per chart (light background) plus an HTML page."""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from pdmodel.validation import ValidationResult  # noqa: E402

plt.style.use("default")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"
SERIES = ["#2b6cb0", "#dd6b20", "#2f855a", "#805ad5", "#b7791f"]
STATUS_COLOR = {"GREEN": "#2f855a", "AMBER": "#b7791f", "RED": "#c53030"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path.name


def make_charts(res: ValidationResult, out: Path) -> dict[str, str]:
    y, p = res.artefacts["y"], res.artefacts["p"]
    names = res.artefacts["eval_names"]
    charts = {}

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(*roc_curve(y["train"], p["train"])[:2], color=GREY, label="train")
    for k, c in zip(names, SERIES, strict=False):
        fpr, tpr, _ = roc_curve(y[k], p[k])
        ax.plot(fpr, tpr, color=c, label=f"{k} (AUC {res.tables['discrimination'].loc[k, 'AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#a0aec0", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    charts["roc"] = _save(fig, out / "roc.png")

    for k in names:
        hl = res.tables[f"hl_{k}"]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        x = np.arange(1, len(hl) + 1)
        ax.plot(x, hl["pred_rate"] * 100, "o-", color=BLUE, label="Predicted PD")
        ax.plot(x, hl["obs_rate"] * 100, "s-", color=ORANGE, label="Observed default rate")
        ax.set(xlabel="PD decile (1 = lowest risk)", ylabel="%", title=f"Calibration by decile ({k})")
        ax.legend()
        charts[f"calibration_{k}"] = _save(fig, out / f"calibration_{k}.png")

    gb = res.tables["grade_binomial_oot"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(gb.index - 0.2, gb["avg_pd"] * 100, 0.4, color=BLUE, label="Avg PD")
    ax.bar(gb.index + 0.2, gb["odr"] * 100, 0.4, color=ORANGE, label="Observed default rate")
    ax.set(xlabel="Rating grade (1 = best)", ylabel="%", title="PD vs observed default by grade (all OOT)")
    ax.set_xticks(gb.index)
    ax.legend()
    charts["grades"] = _save(fig, out / "grades_oot.png")

    disc = res.tables["discrimination"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    xs = np.arange(len(disc))
    ax.bar(xs - 0.2, disc["avg_pd"] * 100, 0.4, color=BLUE, label="Avg PD")
    ax.bar(xs + 0.2, disc["default_rate"] * 100, 0.4, color=ORANGE, label="Observed default rate")
    ax.set_xticks(xs, disc.index, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="%", title="Predicted vs observed default rate by sample")
    ax.legend()
    charts["by_sample"] = _save(fig, out / "pd_vs_observed_by_sample.png")

    s_dev, s_oot = res.artefacts["score_dev"], res.artefacts["score_oot"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bins = np.linspace(min(s_dev.min(), s_oot.min()), max(s_dev.max(), s_oot.max()), 40)
    ax.hist(s_dev, bins, density=True, alpha=0.55, color=BLUE, label="Development")
    ax.hist(s_oot, bins, density=True, alpha=0.55, color=ORANGE, label="All OOT")
    ax.set(xlabel="Score", ylabel="Density", title="Score distribution")
    ax.legend()
    charts["score_dist"] = _save(fig, out / "score_distribution.png")

    iv = res.tables["iv"]["IV"].sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.barh(iv.index, iv.values, color=BLUE)
    ax.axvline(0.02, color=ORANGE, ls="--", lw=1, label="Retention threshold 0.02")
    ax.set(xlabel="Information value", title="Feature information value")
    ax.legend(loc="lower right")
    charts["iv"] = _save(fig, out / "information_value.png")
    return charts


def _table(df, fmt="{:.4f}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return fmt.format(v)
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = "".join(
        "<tr>" + f"<td>{cell(i)}</td>" + "".join(f"<td>{cell(v)}</td>" for v in r) + "</tr>"
        for i, r in zip(df.index, df.to_numpy(), strict=True)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _data_section(res: ValidationResult, data_dir: Path | None) -> str:
    cfg = res.artefacts["champion"].cfg
    windows = [("Development (train/test)", *cfg.dev)]
    if cfg.calib:
        windows.append(("Validation & calibration", *cfg.calib))
    windows += [(n, lo, hi) for n, lo, hi in cfg.oot]
    wt = pd.DataFrame(windows, columns=["sample", "from", "to"]).set_index("sample")
    if cfg.source == "synthetic":
        intro = ("<p>Synthetic retail loan book (framework demonstration). The development window "
                 "contains the 2020 unemployment spike.</p>")
    else:
        intro = ("<p>Freddie Mac Single-Family Loan-Level Dataset (sample files), newly originated "
                 "fixed-rate mortgages. Default = 90+ days past due, REO acquisition, or credit-loss "
                 "zero balance (02, 03, 09) within the first 12 months after first payment. "
                 "90+ DPD while in forbearance or flagged as disaster-related is not counted. "
                 "Loans without a fully observed 12-month window, and repurchased defects (96), "
                 "are excluded.</p>")
        if data_dir and list(Path(data_dir).glob("summary_*.json")):
            rows = [json.loads(f.read_text()) for f in sorted(Path(data_dir).glob("summary_*.json"))]
            cols = ["loans_in_file", "dropped_censored", "dropped_defect", "dropped_harp",
                    "prepaid_in_window", "forbearance_dq_not_counted", "loans_kept", "defaults", "default_rate"]
            st = pd.DataFrame(rows).set_index("vintage")
            st = st[[c for c in cols if c in st.columns]]
            intro += "<h3>Ingestion summary by vintage</h3>" + _table(st)
    problems = res.artefacts.get("problems") or []
    dq = ("<h3>Data quality problems</h3><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in problems)
          + "</ul>") if problems else "<p>Data quality checks: no problems found.</p>"
    return intro + dq + "<h3>Sample windows</h3>" + _table(wt) + _table(
        res.tables["discrimination"][["n", "default_rate"]])


def _findings_section(res: ValidationResult, findings: dict, conclusion: dict | None) -> str:
    html_out = ""
    if conclusion:
        conds = "".join(f"<li>{html.escape(c)}</li>" for c in conclusion.get("conditions", []))
        html_out += (f"<p><b>Decision:</b> {html.escape(conclusion.get('decision', ''))}</p>"
                     f"<p><b>Intended use:</b> {html.escape(conclusion.get('intended_use', ''))}<br>"
                     f"<b>Out of scope:</b> {html.escape(conclusion.get('out_of_scope', ''))}</p>"
                     f"<p><b>Conditions:</b></p><ul>{conds}</ul>")
    if findings:
        ids = res.by_id()
        rows = [{"finding": k, "current_status": ids[k].status if k in ids else "n/a",
                 "severity": f.get("severity", ""), "rationale": f.get("rationale", ""),
                 "remediation": f.get("remediation", ""), "owner": f.get("owner", ""),
                 "review_by": f.get("review_by", "")} for k, f in findings.items()]
        html_out += "<h3>Findings register</h3>" + _table(pd.DataFrame(rows).set_index("finding"))
    else:
        html_out += "<p>No open findings.</p>"
    return html_out


def write_report(res: ValidationResult, out: Path, findings: dict, data_dir: Path | None = None,
                 conclusion: dict | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    charts = make_charts(res, out)
    summ = res.summary()
    counts = summ["status"].value_counts().reindex(["GREEN", "AMBER", "RED"], fill_value=0)
    source = res.artefacts["source"]

    rows = []
    for c in res.checks:
        note = findings.get(c.id, {}).get("rationale", "")
        badge = f'<span class="b" style="background:{STATUS_COLOR[c.status]}">{c.status}</span>'
        rows.append(f"<tr><td>{c.id}</td><td>{html.escape(c.area)}</td><td>{html.escape(c.description)}"
                    f"</td><td>{c.value:.4f}</td><td>{html.escape(c.rule)}</td><td>{badge}</td>"
                    f"<td>{html.escape(note)}</td></tr>")

    def fig(k, cap):
        return f'<figure><img src="{charts[k]}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'

    recal = ""
    if "recalibration" in res.tables:
        a, b = res.artefacts["champion"].calibration
        recal = (f"<h3>PD recalibration on the calibration sample</h3><p>PD = sigmoid({a:.3f} + {b:.3f} "
                 "&times; logit(raw PD)), fitted on the calibration window only. Scores and grades are "
                 "unchanged (ranking); only the PD level is mapped. O/E after recalibration on the "
                 "calibration sample is in-sample by construction.</p>"
                 + _table(res.tables["recalibration"]))
    defdef = ""
    if "default_definition" in res.tables:
        defdef = ("<h3>Default definition sensitivity: counting forbearance delinquencies</h3>"
                  "<p>Same model, evaluated against an alternative label that also counts 90+ DPD reported "
                  "during forbearance or disaster as default. A large rise in default rate with O/E far "
                  "above 1 and no gain in AUC indicates those delinquencies reflect an external shock "
                  "rather than borrower credit risk, which supports excluding them.</p>"
                  + _table(res.tables["default_definition"]))
    cal_figs = "".join(fig(f"calibration_{k}", f"Calibration, {k}") for k in res.artefacts["eval_names"])
    limitations = (
        "<li>Synthetic data: results demonstrate the framework, not a real portfolio.</li>"
        if source == "synthetic" else
        "<li>Sample files (50,000 loans per vintage), not the full Freddie Mac population.</li>"
        "<li>Agency-conforming, fixed-rate mortgages only; not representative of non-agency loans.</li>")
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>PD Model Validation Report</title><style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:2rem auto;
padding:0 1rem;color:#1a202c;background:#fff;line-height:1.5}}
h1{{margin-bottom:.2rem}} h2{{border-bottom:1px solid #e2e8f0;padding-bottom:.3rem;margin-top:2rem}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;margin:.6rem 0}}
th,td{{border:1px solid #e2e8f0;padding:.35rem .5rem;text-align:left;vertical-align:top}}
th{{background:#f7fafc}} .b{{color:#fff;padding:.1rem .45rem;border-radius:4px;font-size:.78rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem}}
figure{{margin:0}} img{{width:100%;border:1px solid #e2e8f0}} figcaption{{font-size:.8rem;color:#4a5568}}
.kpi{{display:flex;gap:1rem}} .kpi div{{border:1px solid #e2e8f0;border-radius:6px;padding:.5rem 1rem}}
</style></head><body>
<h1>PD Model Validation Report</h1>
<p>12-month Probability of Default model for newly originated loans &middot; data source: <b>{source}</b>
&middot; generated {date.today().isoformat()} by <code>python -m pdmodel.run</code>. Structure follows
SR 26-2 (formerly SR 11-7): data, conceptual soundness, outcomes analysis, ongoing monitoring, benchmarking.</p>
<div class="kpi"><div><b>{counts['GREEN']}</b> green</div><div><b>{counts['AMBER']}</b> amber</div>
<div><b>{counts['RED']}</b> red (accepted findings: {len(findings)})</div></div>

<h2>1. Check summary</h2>
<table><thead><tr><th>ID</th><th>Area</th><th>Check</th><th>Value</th><th>Rule</th><th>Status</th>
<th>Finding / rationale</th></tr></thead><tbody>{''.join(rows)}</tbody></table>

<h2>2. Data</h2>{_data_section(res, data_dir)}

<h2>3. Conceptual soundness</h2>
<div class="grid">{fig('iv', 'Information value by feature')}
<div><h3>Coefficients (WoE logistic)</h3>{_table(res.tables['coefficients'])}
<h3>Variables removed for counter-intuitive sign</h3>{_table(res.tables['dropped_for_sign']) if len(res.tables['dropped_for_sign']) else '<p>None.</p>'}
<h3>Variance inflation factors (WoE inputs)</h3>{_table(res.tables['vif'], '{:.2f}')}
<h3>Sensitivity to shocks (test sample)</h3>{_table(res.tables['sensitivity'], '{:.1f}')}</div></div>

<h2>4. Outcomes analysis</h2>
<h3>Discrimination and portfolio calibration by sample</h3>{_table(res.tables['discrimination'])}
<div class="grid">{fig('roc', 'ROC curve by sample')}{fig('by_sample', 'Predicted vs observed default rate by sample')}
{fig('grades', 'PD vs observed default rate by grade, all OOT')}{cal_figs}</div>
<h3>Grade-level binomial test (all OOT)</h3>{_table(res.tables['grade_binomial_oot'])}
{recal}
{defdef}

<h2>5. Ongoing monitoring</h2>
<div class="grid">{fig('score_dist', 'Score distribution, development vs OOT')}
<div><h3>Characteristic stability index by OOT segment</h3>{_table(res.tables['csi'])}</div></div>

<h2>6. Benchmarking</h2>{_table(res.tables['benchmark'])}

<h2>7. Findings and validation conclusion</h2>{_findings_section(res, findings, conclusion)}

<h2>8. Limitations</h2><ul>{limitations}
<li>Application (origination) scorecard without macro inputs: point-in-time PD drifts with the
economy. Stress testing needs a macro overlay.</li>
<li>Binomial and Jeffreys tests assume independent defaults; correlated defaults make them over-reject.</li>
<li>Hosmer-Lemeshow rejects almost any model in samples of this size; O/E and the Jeffreys test are the primary calibration evidence.</li>
<li>12-month default horizon only; no lifetime PD, LGD or EAD.</li></ul>
</body></html>"""
    path = out / "validation_report.html"
    path.write_text(doc, encoding="utf-8")
    (out / "checks.json").write_text(json.dumps(res.key_metrics(), indent=2), encoding="utf-8")
    return path
