"""Phase 5: prepare the email data (NO training here).

Run from the project root:
    python -m scripts.prepare_email

Pipeline:
    Enron-Spam: load -> inspect -> clean -> deduplicate -> near-duplicate groups
                -> grouped stratified split -> leakage checks -> save
    2026 spam (untroubled.org): parse safely -> visible text -> deduplicate -> save as an
                EXTERNAL set (spam only: it can measure detection rate, not false alarms)
    -> reports/data_quality/email.md  (every number computed here)

The Enron `Date` column is kept for analysis only. It is never a model feature: in this dataset
the date almost identifies the class (ham 1999-2002, spam 2001-2005).
"""

import json

import pandas as pd

from config import settings
from scripts.prepare_sms import table
from src.data.dedup import deduplicate
from src.data.email_parse import parse_email_bytes
from src.data.splitting import check_no_overlap, grouped_stratified_split
from src.data.text_utils import mask_key, normalize_key
from src.features.email_text import join_email

ENRON_FILE = settings.DATA_RAW / "email_enron" / "extracted" / "enron_spam_data.csv"
UNTROUBLED_DIR = settings.DATA_RAW / "email_untroubled"
OUT_DIR = settings.DATA_PROCESSED / "email"
EXTERNAL_DIR = settings.DATA_EXTERNAL_TEST / "email"
REPORT_PATH = settings.REPORTS_DIR / "data_quality" / "email.md"
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]


def load_enron() -> pd.DataFrame:
    raw = pd.read_csv(ENRON_FILE, keep_default_na=False)
    labels = raw["Spam/Ham"].str.strip().str.lower()
    unknown = sorted(set(labels) - {"spam", "ham"})
    if unknown:
        raise ValueError(f"Enron: unexpected labels {unknown}")
    return pd.DataFrame(
        {
            "msg_id": raw["Message ID"].to_numpy(),
            "subject": raw["Subject"].str.strip().to_numpy(),
            "body": raw["Message"].str.strip().to_numpy(),
            "label": (labels == "spam").astype(int).to_numpy(),  # 1 = spam (bad)
            "date": raw["Date"].to_numpy(),  # analysis only, never a feature
        }
    )


def load_2026_spam() -> pd.DataFrame:
    rows = []
    for month in MONTHS:
        for path in sorted((UNTROUBLED_DIR / month).rglob("*")):
            if not path.is_file():
                continue
            try:
                parsed = parse_email_bytes(path.read_bytes())
            except Exception:  # noqa: BLE001 - one broken file must not stop the run
                rows.append({"month": month, "parse_error": True})
                continue
            text = parsed.visible_text
            sample = text[:2000]
            ascii_share = sum(c.isascii() for c in sample) / max(1, len(sample))
            rows.append(
                {
                    "month": month,
                    "subject": parsed.subject,
                    "body": text,
                    "n_attachments": parsed.n_attachments,
                    "hidden_chars": parsed.hidden_chars,
                    "had_html": bool(parsed.html_text),
                    "had_plain": bool(parsed.plain_text),
                    "mostly_non_ascii": ascii_share < 0.8,
                    "parse_error": False,
                }
            )
    df = pd.DataFrame(rows)
    df["label"] = 1
    return df


def facts_enron(df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df["date"], errors="coerce")
    by_class = dates.groupby(df["label"]).agg(["min", "max"])
    return {
        "rows": len(df),
        "ham": int((df["label"] == 0).sum()),
        "spam": int((df["label"] == 1).sum()),
        "empty_subject": int((df["subject"] == "").sum()),
        "empty_body": int((df["body"] == "").sum()),
        "both_empty": int(((df["subject"] == "") & (df["body"] == "")).sum()),
        "ham_dates": f"{by_class.loc[0, 'min']:%Y-%m-%d} to {by_class.loc[0, 'max']:%Y-%m-%d}",
        "spam_dates": f"{by_class.loc[1, 'min']:%Y-%m-%d} to {by_class.loc[1, 'max']:%Y-%m-%d}",
        "spam_after_2002": int(((dates.dt.year >= 2003) & (df["label"] == 1)).sum()),
        "ham_after_2002": int(((dates.dt.year >= 2003) & (df["label"] == 0)).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Enron: clean, deduplicate, group, split ----
    enron = load_enron()
    facts = facts_enron(enron)
    steps = [["loaded", len(enron), "", "", len(enron)]]

    enron = enron[~((enron["subject"] == "") & (enron["body"] == ""))].reset_index(drop=True)
    steps.append(
        [
            "dropped rows with empty subject AND body",
            facts["rows"],
            facts["rows"] - len(enron),
            "",
            len(enron),
        ]
    )

    enron["text"] = [join_email(s, b) for s, b in zip(enron["subject"], enron["body"], strict=True)]
    enron["key"] = enron["text"].map(normalize_key)
    enron, counts = deduplicate(enron)
    steps.append(
        [
            "deduplicated (subject + body, normalized)",
            counts["rows_in"],
            counts["duplicate_rows_removed"],
            counts["conflicting_rows_removed"],
            counts["rows_out"],
        ]
    )

    enron["group"] = enron["text"].str[:400].map(mask_key)  # near-duplicate campaigns
    group_sizes = enron.groupby("group").size()
    multi = group_sizes[group_sizes > 1]
    enron["id"] = range(len(enron))
    enron["split"] = grouped_stratified_split(
        enron, label_col="label", group_col="group", seed=settings.RANDOM_SEED
    )
    splits = {
        s: enron[enron["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")
    }
    key_overlap = check_no_overlap(splits, "key")
    group_overlap = check_no_overlap(splits, "group")
    assert all(v == 0 for v in key_overlap.values()), key_overlap
    assert all(v == 0 for v in group_overlap.values()), group_overlap

    columns = ["id", "msg_id", "subject", "body", "label", "date", "group"]
    for name, part in splits.items():
        part[columns].to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8")
    (OUT_DIR / "split_summary.json").write_text(
        json.dumps(
            {
                "seed": settings.RANDOM_SEED,
                "sizes": {n: len(d) for n, d in splits.items()},
                "spam_share": {n: round(float(d["label"].mean()), 4) for n, d in splits.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- 2026 spam: external set ----
    spam26 = load_2026_spam()
    loaded = len(spam26)
    parse_errors = int(spam26["parse_error"].sum())
    spam26 = spam26[~spam26["parse_error"]].copy()
    spam26["text"] = [
        join_email(s, b) for s, b in zip(spam26["subject"], spam26["body"], strict=True)
    ]
    spam26 = spam26[spam26["text"].str.strip(" .") != ""].copy()
    spam26["key"] = spam26["text"].map(normalize_key)
    before_dedup = len(spam26)
    spam26 = spam26.drop_duplicates("key").reset_index(drop=True)
    after_dedup = len(spam26)
    enron_keys = set(enron["key"])
    overlap_with_enron = int(spam26["key"].isin(enron_keys).sum())
    spam26 = spam26[~spam26["key"].isin(enron_keys)].reset_index(drop=True)
    spam26["id"] = range(len(spam26))
    spam26[
        [
            "id",
            "month",
            "subject",
            "body",
            "label",
            "mostly_non_ascii",
            "n_attachments",
            "hidden_chars",
        ]
    ].to_csv(EXTERNAL_DIR / "untroubled_2026.csv", index=False, encoding="utf-8")

    write_report(
        facts,
        steps,
        multi,
        splits,
        key_overlap,
        group_overlap,
        spam26,
        loaded,
        parse_errors,
        before_dedup,
        after_dedup,
        overlap_with_enron,
    )
    print(f"Saved splits to {OUT_DIR} and external set to {EXTERNAL_DIR}\nWrote {REPORT_PATH}")
    print(
        table(
            ["split", "rows", "spam share"],
            [[n, len(d), f"{d['label'].mean():.3f}"] for n, d in splits.items()],
        )
    )
    print(f"2026 spam kept: {len(spam26)}")


def write_report(
    facts,
    steps,
    multi,
    splits,
    key_overlap,
    group_overlap,
    spam26,
    loaded,
    parse_errors,
    before_dedup,
    after_dedup,
    overlap_with_enron,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    split_tbl = table(
        ["split", "rows", "ham", "spam", "spam share"],
        [
            [
                n,
                len(d),
                int((d["label"] == 0).sum()),
                int((d["label"] == 1).sum()),
                f"{d['label'].mean():.3f}",
            ]
            for n, d in splits.items()
        ],
    )
    leak_tbl = table(
        ["pair", "shared texts", "shared near-duplicate groups"],
        [[k, key_overlap[k], group_overlap[k]] for k in key_overlap],
    )
    month_tbl = table(
        [
            "month",
            "emails kept",
            "mostly non-ASCII text",
            "had HTML part",
            "with attachments",
            "with CSS-hidden text",
        ],
        [
            [
                m,
                len(g),
                int(g["mostly_non_ascii"].sum()),
                int(g["had_html"].sum()),
                int((g["n_attachments"] > 0).sum()),
                int((g["hidden_chars"] > 0).sum()),
            ]
            for m, g in spam26.groupby("month")
        ],
    )
    text = f"""# Email data-quality report

Generated by `python -m scripts.prepare_email` (seed {settings.RANDOM_SEED}). Every number below was computed from the raw files.
Label convention: **1 = spam (bad), 0 = ham (good)**.

## 1. Enron-Spam (training pool)

Source: Metsis, Androutsopoulos and Paliouras, "Spam Filtering with Naive Bayes - Which Naive Bayes?", redistributed as a single CSV by M. Wiechmann (repo licence: GPL-3.0). Columns: `Message ID`, `Subject`, `Message`, `Spam/Ham`, `Date`.

- Rows: **{facts["rows"]:,}** ({facts["ham"]:,} ham, {facts["spam"]:,} spam).
- Empty subject: {facts["empty_subject"]}; empty body: {facts["empty_body"]}; both empty: {facts["both_empty"]}.
- **Dates give the label away:** ham {facts["ham_dates"]}; spam {facts["spam_dates"]}. {facts["spam_after_2002"]:,} spam emails are dated 2003 or later; {facts["ham_after_2002"]} ham emails are. The date is never used as a feature.
- Ham is internal mail of Enron employees (energy trading, 1999-2002); spam is collected later from other sources. The model can therefore learn "is this Enron-style mail?" instead of "is this spam?" (checked in Phase 5 training).
- The text is already **lowercased and tokenized** ("vince , thank you"); real pasted emails are not, so the same normalization is applied at training and prediction time.

## 2. Cleaning and deduplication

{table(["step", "rows in", "removed", "conflicting removed", "rows out"], steps)}

Near-duplicate campaigns (first 400 characters, digits and links masked): {len(multi):,} groups contain more than one email ({int(multi.sum()):,} rows; largest = {int(multi.max())}). They are kept but assigned to one split only.

## 3. Split

Method: StratifiedGroupKFold, 20 folds (3 test, 3 validation, 14 train), seed {settings.RANDOM_SEED}.

{split_tbl}

Leakage checks (must be 0):

{leak_tbl}

## 4. 2026 spam (external test, spam only)

Source: the untroubled.org spam archive (monthly files Jun-Oct 2026; use permitted "without restriction", citation requested). Headers are largely forged and the origin cannot be trusted. Each file is parsed as text only: attachments are counted, never opened; HTML is reduced to visible text; nothing is rendered or fetched. When an email has an HTML part, its visible text is used because many 2026 spam emails put a harmless filler paragraph in the plain-text part and the real lure in the HTML.

- Files read: {loaded:,}; parse errors: {parse_errors}; emails with some text: {before_dedup:,}; after removing exact duplicates (spam campaigns repeat): {after_dedup:,}; identical to an Enron text (removed): {overlap_with_enron}; **kept: {len(spam26):,}**.

{month_tbl}

**Hidden text:** a large share of these emails contain text hidden with inline CSS (`display:none`, `font-size:1px`, `opacity:0`, `mso-hide:all`, zero-size boxes). It is usually a harmless-looking filler paragraph meant to fool word-based filters. The parser drops it, because a reader never sees it (and a user who copies an email copies only what they see). Hidden-text detection covers inline styles only, not external stylesheets.

**This set contains spam only.** It can measure what share of 2026 spam the model catches, but it cannot measure false alarms on modern legitimate email. No public modern legitimate email dataset was found; the owner's own legitimate emails (kept private) are the only realistic source.

## 5. Known limitations

- Training data is 1999-2005; the model has never seen a modern legitimate email.
- Part of the 2026 spam is not English (see "mostly non-ASCII").
- Enron ham and spam come from different sources and years (source bias).
- Spam is labelled by the dataset authors' collection process, not individually verified.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
