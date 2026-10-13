"""Dashboard: every number is read from the models' metadata.json files (or from this session's
tallies). `collect()` is pure and testable; `render()` draws it with Streamlit and Altair.

Session activity stores only COUNTS per (detector, verdict) in the browser session's memory. The
text a user analyses is never part of it.
"""

import json
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from config import settings
from src.ui import components, theme

FOLDERS = {
    "SMS": "sms_spam",
    "Email": "email_spam",
    "Comments": "toxicity",
    "URL": "phishing",
}
FIGURES = [
    ("sms_confusion_test.png", "SMS: confusion matrix on the test set"),
    ("sms_pr_curves.png", "SMS: precision-recall curves of the candidate models"),
    ("email_pr_curves.png", "Email: precision-recall curves"),
    ("phishing_confusion_test.png", "URL: confusion matrix on newer (temporal) test URLs"),
    ("toxicity_f1_per_label.png", "Comments: F1 per label"),
    ("email_model_comparison.png", "Email: validation PR-AUC by model"),
]


def _load(models_dir: Path) -> dict[str, dict]:
    out = {}
    for module, folder in FOLDERS.items():
        try:
            out[module] = json.loads((models_dir / folder / "metadata.json").read_text("utf-8"))
        except (OSError, ValueError):
            continue
    return out


def collect(models_dir: Path | None = None) -> dict:
    """Build every table the dashboard shows. Missing models are simply absent."""
    meta = _load(Path(models_dir) if models_dir is not None else settings.MODELS_DIR)
    out: dict = {"meta": meta}

    sizes, latency, gap_recall, gap_false = [], [], [], []
    for module, m in meta.items():
        d = m.get("data_sizes", {})
        for split in ("train", "val", "test"):
            if split in d:
                sizes.append({"Detector": module, "Split": split, "Examples": int(d[split])})
        ms = next((m[k] for k in m if k.startswith("inference_ms_per_")), None)
        latency.append(
            {
                "Detector": module,
                "Milliseconds per item": ms,
                "Model size (MB)": m.get("model_file_mb"),
                "Algorithm": m.get("model"),
            }
        )
    out["sizes"] = pd.DataFrame(sizes)
    out["latency"] = pd.DataFrame(latency)

    # Same-source test vs different-source test: how much does quality drop on new data?
    try:
        s = meta["SMS"]["metrics"]
        gap_recall += [
            ("SMS", "Same-source test", s["test"]["recall"]),
            ("SMS", "Different source", s["external_india"]["recall"]),
        ]
        gap_false += [
            ("SMS", "Same-source test", s["test"]["false_positive_rate"]),
            ("SMS", "Different source", s["external_india"]["false_positive_rate"]),
        ]
    except KeyError:
        pass
    try:
        e = meta["Email"]
        ch = e["metrics"]["challenge_modern"]
        gap_recall += [
            ("Email", "Same-source test", e["metrics"]["test"]["recall"]),
            ("Email", "Different source", e["external_2026"]["detection_rate"]),
        ]
        gap_false += [
            ("Email", "Same-source test", e["metrics"]["test"]["false_positive_rate"]),
            ("Email", "Different source", ch["fp"] / max(ch["fp"] + ch["tn"], 1)),
        ]
    except KeyError:
        pass
    try:
        u = meta["URL"]["metrics"]
        comp = u["external"]["compphish"]
        gap_recall += [
            ("URL", "Same-source test", u["test_temporal"]["recall"]),
            ("URL", "Different source", comp["recall"]),
        ]
        gap_false += [
            ("URL", "Same-source test", u["test_temporal"]["false_positive_rate"]),
            ("URL", "Different source", comp["false_positive_rate"]),
        ]
    except KeyError:
        pass
    out["gap_recall"] = pd.DataFrame(gap_recall, columns=["Detector", "Test", "Value"])
    out["gap_false"] = pd.DataFrame(gap_false, columns=["Detector", "Test", "Value"])

    try:
        per = meta["Comments"]["metrics"]["official_test"]["per_label"]
        out["toxicity_labels"] = pd.DataFrame(
            [
                {
                    "Label": k,
                    "Precision": v["precision"],
                    "Recall": v["recall"],
                    "F1": v["f1"],
                    "Positive examples": v["positives"],
                }
                for k, v in per.items()
            ]
        )
    except KeyError:
        pass

    try:
        tm = meta["Comments"]["metrics"]
        out["toxicity_macro"] = (
            tm["test"]["summary"]["macro_f1"],
            tm["official_test"]["summary"]["macro_f1"],
        )
    except KeyError:
        pass

    weak = []
    if "Email" in meta and "challenge_modern" in meta["Email"]["metrics"]:
        ch = meta["Email"]["metrics"]["challenge_modern"]
        weak.append(
            (
                "Email",
                "Ordinary modern emails wrongly flagged",
                f"{ch['fp']} of {ch['fp'] + ch['tn']}",
            )
        )
    if "Comments" in meta and "bias_probe" in meta["Comments"]:
        b = meta["Comments"]["bias_probe"]
        weak.append(
            (
                "Comments",
                f"Neutral sentences flagged for the worst identity term ('{b['worst_term']}')",
                f"{b['max_flagged_share']:.0%}",
            )
        )
    if "URL" in meta and "challenge" in meta["URL"]["metrics"]:
        c = meta["URL"]["metrics"]["challenge"]
        weak.append(
            (
                "URL",
                "Ordinary links wrongly flagged (hand-written set)",
                f"{c['legit_flagged']} of {c['legit_n']}",
            )
        )
    if "SMS" in meta:
        p = meta["SMS"]["metrics"].get("private_heldout")
        if p:
            weak.append(
                (
                    "SMS",
                    "Your own held-out messages wrongly flagged",
                    f"{p['legit_flagged']} of {p['legit']}",
                )
            )
    out["weaknesses"] = pd.DataFrame(weak, columns=["Detector", "Known weakness", "Measured"])

    totals = {
        "ready": len(meta),
        "examples": int(sum(m.get("data_sizes", {}).get("train", 0) for m in meta.values())),
        "ms": float(
            pd.to_numeric(out["latency"].get("Milliseconds per item"), errors="coerce").mean()
        )
        if len(out["latency"])
        else 0.0,
        "mb": float(pd.to_numeric(out["latency"].get("Model size (MB)"), errors="coerce").sum())
        if len(out["latency"])
        else 0.0,
    }
    out["totals"] = totals
    return out


def record_activity(detector: str, verdict: str) -> None:
    """Count one analysis for this browser session (detector + verdict only, never the text)."""
    counter = st.session_state.setdefault("activity", Counter())
    counter[(detector, verdict)] += 1


def activity_frame() -> pd.DataFrame:
    counter = st.session_state.get("activity", Counter())
    return pd.DataFrame(
        [{"Detector": d, "Verdict": v, "Count": n} for (d, v), n in sorted(counter.items())],
        columns=["Detector", "Verdict", "Count"],
    )


def _colors() -> dict:
    p = theme.active_palette()
    return {"cyan": p["a1"], "violet": p["a2"], "pink": p["a3"]}


def _style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_axis(
            labelColor="#93a0c4",
            titleColor="#93a0c4",
            gridColor="rgba(255,255,255,.07)",
            domainColor="rgba(255,255,255,.15)",
        )
        .configure_legend(
            labelColor=theme.active_palette()["muted"],
            titleColor=theme.active_palette()["muted"],
            orient="bottom",
        )
        .configure_view(strokeWidth=0)
    )


def _grouped(frame: pd.DataFrame, title: str, percent: bool = True) -> alt.Chart:
    fmt = ".0%" if percent else ".2f"
    base = alt.Chart(frame).encode(
        x=alt.X("Detector:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="Test:N",
        y=alt.Y("Value:Q", title=title, axis=alt.Axis(format=fmt), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Test:N",
            scale=alt.Scale(
                domain=["Same-source test", "Different source"],
                range=[_colors()["cyan"], _colors()["violet"]],
            ),
        ),
        tooltip=["Detector", "Test", alt.Tooltip("Value:Q", format=fmt)],
    )
    return _style(
        (
            base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            + base.mark_text(dy=-8, color="#e8eefc", fontSize=12).encode(
                text=alt.Text("Value:Q", format=fmt)
            )
        ).properties(height=300)
    )


def render(models_dir: Path | None = None) -> None:
    data = collect(models_dir)
    totals = data["totals"]
    st.html(
        components.section("Overview")
        + components.kpis(
            [
                (f"{totals['ready']} / 4", "detectors ready"),
                (f"{totals['examples']:,}", "training examples"),
                (f"{totals['ms']:.1f} ms", "average time per check"),
                (f"{totals['mb']:.0f} MB", "all models together"),
            ]
        )
    )
    if not data["meta"]:
        st.info("No trained models were found, so there is nothing to chart yet.")

    if len(data["gap_recall"]):
        st.html(components.section("How well do the models generalise?"))
        st.caption(
            "A good score on data like the training data is easy. The honest question is what "
            "happens on data from a different source. Left: how many bad items are caught. "
            "Right: how many ordinary items are wrongly flagged (lower is better)."
        )
        left, right = st.columns(2)
        left.altair_chart(_grouped(data["gap_recall"], "Caught (recall)"), width="stretch")
        right.altair_chart(_grouped(data["gap_false"], "Wrongly flagged"), width="stretch")
        st.caption(
            "SMS: different source = Indian-region messages. Email: caught = real June-October "
            "2026 spam; wrongly flagged = 24 hand-written modern legitimate emails. URL: "
            "different source = the CompPhish collection."
        )

    if "toxicity_labels" in data:
        st.html(components.section("Comments: six labels, six different difficulties"))
        tox = data["toxicity_labels"].melt(
            id_vars=["Label", "Positive examples"],
            value_vars=["Precision", "Recall", "F1"],
            var_name="Metric",
            value_name="Value",
        )
        chart = (
            alt.Chart(tox)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Label:N", title=None, sort=None, axis=alt.Axis(labelAngle=-20)),
                xOffset="Metric:N",
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title="Official test set"),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(
                        range=[_colors()["violet"], _colors()["cyan"], _colors()["pink"]]
                    ),
                ),
                tooltip=[
                    "Label",
                    "Metric",
                    alt.Tooltip("Value:Q", format=".2f"),
                    "Positive examples",
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(_style(chart), width="stretch")
        if "toxicity_macro" in data:
            inner, official = data["toxicity_macro"]
            st.caption(
                f"Macro-F1 is {inner:.2f} on a held-out slice of the training data but "
                f"{official:.2f} on the official test set, which is harder. Rare labels (threat, "
                "severe_toxic, identity_hate) have few examples, so they are weaker."
            )

    if len(data["sizes"]):
        st.html(components.section("Data behind each detector"))
        sizes = (
            alt.Chart(data["sizes"])
            .mark_bar(cornerRadiusEnd=5, size=24)
            .encode(
                y=alt.Y("Detector:N", title=None, axis=alt.Axis(labelOverlap=False)),
                x=alt.X("Examples:Q", stack="zero", title="Examples"),
                color=alt.Color(
                    "Split:N",
                    scale=alt.Scale(
                        domain=["train", "val", "test"],
                        range=[_colors()["violet"], _colors()["cyan"], _colors()["pink"]],
                    ),
                ),
                tooltip=["Detector", "Split", alt.Tooltip("Examples:Q", format=",")],
            )
            .properties(height=alt.Step(40))
        )
        st.altair_chart(_style(sizes), width="stretch")

    if len(data["latency"]):
        st.html(components.section("Speed and size"))
        bench = settings.REPORTS_DIR / "metrics" / "performance.csv"
        if bench.exists():
            table = (
                pd.read_csv(bench)
                .rename(
                    columns={
                        "detector": "Detector",
                        "model_files_mb": "Model size (MB)",
                        "load_seconds": "Load time (s)",
                        "median_ms": "Median (ms)",
                        "p95_ms": "Slowest 5% (ms)",
                        "items_per_second": "Items per second",
                    }
                )
                .drop(columns=["samples"], errors="ignore")
            )
            note = "Measured by scripts/benchmark_models.py: one item at a time, including the explanation."
        else:
            table = data["latency"].copy()
            table["Milliseconds per item"] = pd.to_numeric(table["Milliseconds per item"]).round(1)
            note = "Measured at training time, one item at a time, without the explanation step."
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(note + " Small enough to run on a laptop or a free hosting tier.")

    if len(data["weaknesses"]):
        st.html(components.section("Known weaknesses, measured"))
        st.dataframe(data["weaknesses"], hide_index=True, width="stretch")
        st.caption("Shown on purpose: a project that hides its failures cannot be trusted.")

    st.html(components.section("This session"))
    activity = activity_frame()
    if activity.empty:
        st.caption(
            "Analyse something in another tab and the tally appears here. Only counts are "
            "kept, only for this browser session, and never the text you analysed."
        )
    else:
        chart = (
            alt.Chart(activity)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                y=alt.Y("Detector:N", title=None),
                x=alt.X("Count:Q", title="Analyses", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Verdict:N"),
                tooltip=["Detector", "Verdict", "Count"],
            )
            .properties(height=180)
        )
        st.altair_chart(_style(chart), width="stretch")

    images = [(settings.REPORTS_DIR / "figures" / name, caption) for name, caption in FIGURES]
    images = [(p, c) for p, c in images if p.exists()]
    if images:
        st.html(components.section("Evidence from training"))
        with st.expander("Charts saved during training (confusion matrices, curves)"):
            columns = st.columns(2)
            for i, (path, caption) in enumerate(images):
                columns[i % 2].image(str(path), caption=caption, width="stretch")
