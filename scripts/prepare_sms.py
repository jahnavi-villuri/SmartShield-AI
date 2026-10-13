"""Phase 2: prepare the SMS data (NO training here).

Run from the project root:
    python -m scripts.prepare_sms

Pipeline:
    load -> inspect -> map labels -> dedup within each dataset -> combine
    -> dedup across datasets -> near-duplicate groups -> grouped stratified split
    -> leakage checks -> save splits + write reports/data_quality/sms.md

Every number in the report is computed here; nothing is typed by hand.
"""

import json
import re

import pandas as pd

from config import settings
from src.data.dedup import deduplicate
from src.data.inspect_data import inspect_text_frame, print_inspection
from src.data.sms_load import (
    load_india_external,
    load_mendeley_sms,
    load_super_sms,
    load_uci_sms,
)
from src.data.splitting import check_no_overlap, grouped_stratified_split
from src.data.text_utils import mask_key, normalize_key

OUT_DIR = settings.DATA_PROCESSED / "sms"
EXTERNAL_DIR = settings.DATA_EXTERNAL_TEST / "sms"
REPORT_PATH = settings.REPORTS_DIR / "data_quality" / "sms.md"
LINK_PATTERN = re.compile(r"https?://|www\.|\b\w+\.(?:com|io|ly|net|info|org)/", re.IGNORECASE)


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LOAD + INSPECT (raw, before any removal) ---------------------------------------
    uci = load_uci_sms()
    mendeley = load_mendeley_sms()
    super_sms = load_super_sms()
    facts = {}
    for name, df in (
        ("UCI SMS Spam Collection", uci),
        ("Mendeley SMS Phishing", mendeley),
        ("ScamShield Smishing_Dataset (English)", super_sms),
    ):
        facts[name] = inspect_text_frame(df, name)
        print_inspection(facts[name], sample=df)

    # Mendeley's own URL column vs. a simple link check on the text (reliability test)
    mendeley_raw = pd.read_csv(
        settings.DATA_RAW / "sms_mendeley" / "extracted" / "Dataset_5971.csv",
        keep_default_na=False,
    )
    has_link = mendeley_raw["TEXT"].str.contains(LINK_PATTERN)
    says_url = mendeley_raw["URL"].str.strip().str.lower() == "yes"
    url_col_missed = int((has_link & ~says_url).sum())
    url_col_false_alarm = int((~has_link & says_url).sum())

    # 2. KEYS + dedup within each dataset ---------------------------------------------
    steps = []
    cleaned = {}
    for name, df in (("uci", uci), ("mendeley", mendeley), ("super", super_sms)):
        df = df.copy()
        df["key"] = df["text"].map(normalize_key)
        deduped, counts = deduplicate(df)
        cleaned[name] = deduped
        steps.append(
            [
                f"{name}: dedup within dataset",
                counts["rows_in"],
                counts["duplicate_rows_removed"],
                counts["conflicting_rows_removed"],
                counts["rows_out"],
            ]
        )

    # 3. COMBINE + dedup across datasets ------------------------------------------------
    combined_in = pd.concat(cleaned.values(), ignore_index=True)
    # Pairwise overlap, measured on the deduplicated datasets (before pooling).
    keys = {name: set(df["key"]) for name, df in cleaned.items()}
    overlaps = {
        "mendeley in uci": (len(keys["mendeley"] & keys["uci"]), len(keys["mendeley"])),
        "super in uci": (len(keys["super"] & keys["uci"]), len(keys["super"])),
        "super in mendeley": (len(keys["super"] & keys["mendeley"]), len(keys["super"])),
    }
    combined, counts = deduplicate(combined_in)
    steps.append(
        [
            "combined: dedup across datasets",
            counts["rows_in"],
            counts["duplicate_rows_removed"],
            counts["conflicting_rows_removed"],
            counts["rows_out"],
        ]
    )

    # 4. Near-duplicate groups (templates that differ only in numbers/links) ----------
    combined["group"] = combined["text"].map(mask_key)
    group_sizes = combined.groupby("group").size()
    multi = group_sizes[group_sizes > 1]
    combined["id"] = range(len(combined))

    # 5. SPLIT --------------------------------------------------------------------------
    combined["split"] = grouped_stratified_split(
        combined, label_col="label", group_col="group", seed=settings.RANDOM_SEED
    )
    splits = {
        s: combined[combined["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")
    }

    # 6. LEAKAGE CHECKS (must all be zero) ---------------------------------------------
    key_overlap = check_no_overlap(splits, "key")
    group_overlap = check_no_overlap(splits, "group")
    assert all(v == 0 for v in key_overlap.values()), key_overlap
    assert all(v == 0 for v in group_overlap.values()), group_overlap

    # 7. SAVE ---------------------------------------------------------------------------
    columns = ["id", "text", "label", "source", "original_label", "group"]
    for name, df in splits.items():
        df[columns].to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8")
    summary = {
        "seed": settings.RANDOM_SEED,
        "split_method": "StratifiedGroupKFold(20 folds): 3 test, 3 val, 14 train; "
        "groups = masked text (digits/URLs replaced)",
        "sizes": {n: len(d) for n, d in splits.items()},
        "bad_share": {n: round(float(d["label"].mean()), 4) for n, d in splits.items()},
    }
    (OUT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 7b. EXTERNAL TEST SET (different source/region/period; never trained on) ------------
    external = load_india_external()
    external["key"] = external["text"].map(normalize_key)
    external["group"] = external["text"].map(mask_key)
    external_rows_in = len(external)
    external, ext_counts = deduplicate(external)
    in_training = external["key"].isin(set(combined["key"])) | external["group"].isin(
        set(combined["group"])
    )
    external_removed_overlap = int(in_training.sum())
    external = external[~in_training].reset_index(drop=True)
    external["id"] = range(len(external))
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    external[columns].to_csv(EXTERNAL_DIR / "india_telecom.csv", index=False, encoding="utf-8")
    external_info = {
        "rows_in": external_rows_in,
        "duplicates_removed": ext_counts["duplicate_rows_removed"],
        "overlap_with_training_removed": external_removed_overlap,
        "ham": int((external["label"] == 0).sum()),
        "spam": int((external["label"] == 1).sum()),
    }

    # 8. REPORT -------------------------------------------------------------------------
    write_report(
        facts,
        steps,
        url_col_missed,
        url_col_false_alarm,
        overlaps,
        external_info,
        combined,
        multi,
        splits,
        key_overlap,
        group_overlap,
    )
    print(f"\nSaved splits to {OUT_DIR}\nWrote report {REPORT_PATH}")
    for line in table(
        ["split", "rows", "spam share"],
        [[n, len(d), f"{d['label'].mean():.3f}"] for n, d in splits.items()],
    ).splitlines():
        print(line)


def write_report(
    facts,
    steps,
    url_missed,
    url_false,
    overlaps,
    ext,
    combined,
    multi,
    splits,
    key_overlap,
    group_overlap,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_headers = [
        "dataset",
        "rows",
        "ham",
        "spam+smishing",
        "bad share",
        "exact dup rows",
        "empty text",
        "rows with U+FFFD",
        "rows with HTML entities",
        "length min/median/max",
    ]
    raw_rows = [
        [
            f["name"],
            f["rows"],
            f["class_counts"].get(0, 0),
            f["class_counts"].get(1, 0),
            f["bad_class_share"],
            f["exact_duplicate_rows"],
            f["empty_text"],
            f["replacement_char_rows"],
            f["html_entity_rows"],
            f"{f['length_min']}/{f['length_median']:.0f}/{f['length_max']}",
        ]
        for f in facts.values()
    ]
    steps_tbl = table(
        ["step", "rows in", "duplicate rows removed", "conflicting rows removed", "rows out"], steps
    )
    overlap_tbl = table(
        ["pair", "shared messages", "size of first dataset", "share"],
        [[k, a, b, f"{a / b:.0%}"] for k, (a, b) in overlaps.items()],
    )
    comp = combined.groupby("source")["label"].agg(["count", "sum"]).reset_index()
    comp_tbl = table(
        ["source", "rows", "ham", "spam"],
        [
            [r["source"], r["count"], int(r["count"] - r["sum"]), int(r["sum"])]
            for _, r in comp.iterrows()
        ],
    )
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
    sources = sorted(combined["source"].unique())
    src_tbl = table(
        ["split"] + sources,
        [
            [n] + [f"{d['source'].value_counts(normalize=True).get(s, 0):.3f}" for s in sources]
            for n, d in splits.items()
        ],
    )
    leak_tbl = table(
        ["pair", "shared normalized texts", "shared near-duplicate groups"],
        [[k, key_overlap[k], group_overlap[k]] for k in key_overlap],
    )
    ext_tbl = table(
        ["rows loaded", "duplicates removed", "removed (also in training pool)", "ham", "spam"],
        [
            [
                ext["rows_in"],
                ext["duplicates_removed"],
                ext["overlap_with_training_removed"],
                ext["ham"],
                ext["spam"],
            ]
        ],
    )

    text = f"""# SMS data-quality report

Generated by `python -m scripts.prepare_sms` (seed {settings.RANDOM_SEED}). Every number below was computed from the raw files.
Label convention: **1 = spam/smishing (bad), 0 = ham (good)**.

## 1. Raw datasets

{table(raw_headers, raw_rows)}

Dataset notes (from the dataset documentation and our inspection):
- **UCI** (~2011, CC BY 4.0): spam comes mostly from a UK forum; ham comes largely from a Singapore university student corpus plus a PhD thesis. Spam and ham therefore differ in **region and writing style**, not just in intent.
- **Mendeley** (published 2022): collected from images converted to text. Label spelling is inconsistent (`Smishing`/`smishing`, `Spam`/`spam`); we lowercase it. `U+FFFD` is the Unicode replacement character: the original symbol (often `£`) was lost and cannot be recovered. Its own `URL` column is unreliable (misses {url_missed} messages that contain a link, flags {url_false} that do not), so it is not used.
- **ScamShield `Smishing_Dataset`** (Hugging Face, MIT licence on the card; internal source name `super`): English rows only, synthetic rows excluded. Its card says it merges earlier collections; it may be the "Super SMS Dataset" (about 2012-2024) described in research papers, but that is **not confirmed**: it shares almost no messages with UCI. The per-row origin and dates are not given, so ham and spam may come from different sources. Underlying source licences were not verified.

## 2. Cleaning and deduplication steps

{steps_tbl}

Overlap between the (individually deduplicated) datasets:

{overlap_tbl}

- A high share means the second dataset largely re-uses the first; it is **not independent**.
- HTML entities such as `&lt;#&gt;` (= `<#>`) are leftovers of an export step. They are kept unchanged; whether they act as a shortcut is checked in Phase 3.
- Conflicting rows = same normalized text with different binary labels; they are removed entirely.
- Spam vs. smishing disagreements are not conflicts for us because both map to label 1.

Final pooled dataset (after cross-dataset dedup; `+` marks messages found in several sources):

{comp_tbl}

## 3. Near-duplicate templates (grouping)

Messages that differ only in numbers or links share one *group* (`mask_key`). {len(multi)} groups contain more than one message ({int(multi.sum())} rows; largest group = {int(multi.max())}). They are kept, but each group is assigned to **one split only**.

## 4. Split

Method: StratifiedGroupKFold with 20 folds (3 test, 3 validation, 14 train), seed {settings.RANDOM_SEED}.

{split_tbl}

Share of each source inside each split:

{src_tbl}

## 5. Leakage checks (all must be 0)

{leak_tbl}

## 6. External test set (never used for training)

English messages from the `Indian_Telecom_SMS` part of ScamShield plus a 27-row benchmark part. Messages (or near-duplicate templates) that also occur in the training pool were removed.

{ext_tbl}

Caveats: this set comes from a different region and period (about 2021 by its content). Its "spam" is largely telecom promotional alerts, which some people would call legitimate marketing, and its "ham" includes news snippets and short support chats. Treat results on it as a **stress test of generalization**, not as a verdict.

## 7. Known limitations

- No dataset here is from 2025-2026. Results must not be presented as current performance.
- The pooled spam share is much higher than in real inboxes, so precision measured on the test split overstates real-world precision.
- Region/style differences between ham and spam sources may inflate scores.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
