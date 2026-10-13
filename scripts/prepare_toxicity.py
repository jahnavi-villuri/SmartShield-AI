"""Phase 6: prepare the Jigsaw toxic-comment data (NO training here).

Run from the project root:
    python -m scripts.prepare_toxicity

Pipeline:
    load train -> inspect -> clean -> deduplicate (conflicting multi-label rows dropped)
    -> multi-label stratified, grouped split (train / val / internal test)
    official test: keep ONLY the scored rows of the Hugging Face "test" split, drop anything
    already in train -> save; write reports/data_quality/toxicity.md (all numbers computed).

The report never prints offensive comments: only counts and label statistics.

Data notes this script verifies (see the report):
  - the "test" split holds every id twice (once with labels, once without) and uses -1 for
    rows the competition did not score; only labelled, scored rows are a usable test set;
  - severe_toxic is always also toxic; threat has under 500 training examples.
"""

import json

import pandas as pd

from config import settings
from scripts.prepare_sms import table
from src.data.dedup import deduplicate
from src.data.splitting import check_no_overlap, grouped_stratified_split
from src.data.text_utils import mask_key, normalize_key

LABELS = settings.TOXICITY_LABELS
RAW_DIR = settings.DATA_RAW / "toxicity_jigsaw"
OUT_DIR = settings.DATA_PROCESSED / "toxicity"
EXTERNAL_DIR = settings.DATA_EXTERNAL_TEST / "toxicity"
REPORT_PATH = settings.REPORTS_DIR / "data_quality" / "toxicity.md"
IDENTITY_TERMS = [
    "gay", "mexican", "deaf", "blind", "black", "white", "women", "muslim", "asian", "straight",
    "men", "jewish", "american", "christian", "catholic", "hindu", "indian", "transgender",
]  # fmt: skip


def stratum(frame: pd.DataFrame) -> pd.Series:
    """One stratum per label pattern, so rare labels are spread over every split.

    Key = any_toxic*8 + threat*4 + identity_hate*2 + severe_toxic. Strata with fewer than
    60 members are merged into one bucket so the splitter can always make 20 folds.
    """
    any_toxic = frame[LABELS].max(axis=1)
    key = any_toxic * 8 + frame["threat"] * 4 + frame["identity_hate"] * 2 + frame["severe_toxic"]
    small = key.map(key.value_counts()) < 60
    return key.where(~small, 99)


def label_key(frame: pd.DataFrame) -> pd.Series:
    return frame[LABELS].astype(int).astype(str).agg("".join, axis=1)


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_train() -> pd.DataFrame:
    raw = pd.read_parquet(RAW_DIR / "train.parquet")
    frame = raw[["id", "comment_text", *LABELS]].copy()
    frame["comment_text"] = clean_text(frame["comment_text"])
    return frame


def load_official_test() -> tuple[pd.DataFrame, dict]:
    """Scored test rows, built by JOINING two blocks of the Hugging Face "test" split by id.

    The split stacks two blocks instead of merging them: rows that hold only the comment text
    (labels empty) and rows that hold only the labels (text empty), each with the same ids.
    Rows the competition did not score carry -1 in every label and are dropped.
    """
    raw = pd.read_parquet(RAW_DIR / "test.parquet")
    info = {"rows": len(raw)}
    texts = raw[raw["comment_text"].notna()][["id", "comment_text"]]
    labels = raw[raw[LABELS].notna().all(axis=1)][["id", *LABELS]]
    info["text_only_rows"] = len(texts)
    info["label_only_rows"] = len(labels)
    info["text_ids_unique"] = bool(texts["id"].is_unique)
    info["label_ids_unique"] = bool(labels["id"].is_unique)
    joined = texts.merge(labels, on="id", how="inner")
    info["joined"] = len(joined)
    scored = joined[(joined[LABELS] >= 0).all(axis=1)].copy()
    info["scored"] = len(scored)
    scored[LABELS] = scored[LABELS].astype(int)
    scored["comment_text"] = clean_text(scored["comment_text"])
    scored = scored[scored["comment_text"] != ""]
    return scored[["id", "comment_text", *LABELS]].reset_index(drop=True), info


def label_counts(frame: pd.DataFrame) -> list:
    return [int(frame[label].sum()) for label in LABELS]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- train: inspect, clean, deduplicate ----
    train_raw = load_train()
    facts = {
        "rows": len(train_raw),
        "any_toxic": int((train_raw[LABELS].max(axis=1) == 1).sum()),
        "severe_without_toxic": int(
            ((train_raw["severe_toxic"] == 1) & (train_raw["toxic"] == 0)).sum()
        ),
        "non_ascii_share": float(train_raw["comment_text"].map(lambda s: not s.isascii()).mean()),
        "newline_share": float(train_raw["comment_text"].str.contains("\n").mean()),
        "url_share": float(
            train_raw["comment_text"].str.contains(r"https?://|www\.", regex=True).mean()
        ),
        "median_chars": int(train_raw["comment_text"].str.len().median()),
        "p99_chars": int(train_raw["comment_text"].str.len().quantile(0.99)),
    }
    cooc = pd.DataFrame(
        {b: {a: float(train_raw.loc[train_raw[a] == 1, b].mean()) for a in LABELS} for b in LABELS}
    )
    identity_rows = []
    base_toxic = float(train_raw["toxic"].mean())
    for term in IDENTITY_TERMS:
        mask = train_raw["comment_text"].str.contains(rf"\b{term}\b", case=False, regex=True)
        if mask.sum() >= 30:
            identity_rows.append(
                [term, int(mask.sum()), f"{train_raw.loc[mask, 'toxic'].mean():.3f}"]
            )
    identity_rows.sort(key=lambda r: -float(r[2]))

    train_raw["key"] = train_raw["comment_text"].map(normalize_key)
    train_raw["label_key"] = label_key(train_raw)
    steps = [["loaded", len(train_raw), "", "", len(train_raw)]]
    clean, counts = deduplicate(
        train_raw[train_raw["key"] != ""], key_col="key", label_col="label_key"
    )
    steps.append(
        [
            "deduplicated (normalized text; conflicting label sets dropped)",
            counts["rows_in"],
            counts["duplicate_rows_removed"],
            counts["conflicting_rows_removed"],
            counts["rows_out"],
        ]
    )

    clean["group"] = clean["comment_text"].str[:300].map(mask_key)
    group_sizes = clean.groupby("group").size()
    multi = group_sizes[group_sizes > 1]
    clean["stratum"] = stratum(clean)
    clean["split"] = grouped_stratified_split(
        clean, label_col="stratum", group_col="group", seed=settings.RANDOM_SEED
    )
    splits = {
        s: clean[clean["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")
    }
    key_overlap = check_no_overlap(splits, "key")
    group_overlap = check_no_overlap(splits, "group")
    assert all(v == 0 for v in key_overlap.values()), key_overlap
    assert all(v == 0 for v in group_overlap.values()), group_overlap
    columns = ["id", "comment_text", *LABELS, "group"]
    for name, part in splits.items():
        part[columns].to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8")

    # ---- official test ----
    official, test_info = load_official_test()
    official["key"] = official["comment_text"].map(normalize_key)
    official["label_key"] = label_key(official)
    test_rows_loaded = len(official)
    official, test_counts = deduplicate(official, key_col="key", label_col="label_key")
    in_train = official["key"].isin(set(clean["key"]))
    overlap_removed = int(in_train.sum())
    official = official[~in_train].reset_index(drop=True)
    official[["id", "comment_text", *LABELS]].to_csv(
        EXTERNAL_DIR / "official_test.csv", index=False, encoding="utf-8"
    )

    summary = {
        "seed": settings.RANDOM_SEED,
        "sizes": {n: len(d) for n, d in splits.items()} | {"official_test": len(official)},
        "label_counts": {
            n: dict(zip(LABELS, label_counts(d), strict=True)) for n, d in splits.items()
        }
        | {"official_test": dict(zip(LABELS, label_counts(official), strict=True))},
    }
    (OUT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        facts,
        cooc,
        identity_rows,
        base_toxic,
        steps,
        multi,
        splits,
        official,
        test_info,
        test_rows_loaded,
        test_counts,
        overlap_removed,
        key_overlap,
        group_overlap,
    )
    print(f"Saved splits to {OUT_DIR}; official test to {EXTERNAL_DIR}\nWrote {REPORT_PATH}")
    print(
        table(
            ["set", "rows", *LABELS],
            [
                [n, len(d), *label_counts(d)]
                for n, d in [*splits.items(), ("official_test", official)]
            ],
        )
    )


def write_report(
    facts,
    cooc,
    identity_rows,
    base_toxic,
    steps,
    multi,
    splits,
    official,
    test_info,
    test_rows_loaded,
    test_counts,
    overlap_removed,
    key_overlap,
    group_overlap,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    label_tbl = table(
        ["set", "rows", *LABELS, "any toxic"],
        [
            [
                n,
                f"{len(d):,}",
                *[f"{c:,}" for c in label_counts(d)],
                f"{(d[LABELS].max(axis=1) == 1).mean():.1%}",
            ]
            for n, d in [*splits.items(), ("official test", official)]
        ],
    )
    cooc_tbl = table(
        ["given row-label \\ also"] + LABELS,
        [[a] + [f"{cooc.loc[a, b]:.2f}" for b in LABELS] for a in LABELS],
    )
    leak_tbl = table(
        ["pair", "shared texts", "shared near-duplicate groups"],
        [[k, key_overlap[k], group_overlap[k]] for k in key_overlap],
    )
    ident_tbl = table(["term", "comments mentioning it", "share labelled toxic"], identity_rows)
    text = f"""# Toxic-comment data-quality report

Generated by `python -m scripts.prepare_toxicity` (seed {settings.RANDOM_SEED}). **This report contains no offensive comments, only counts.**

Dataset: Jigsaw Toxic Comment Classification Challenge, as redistributed on Hugging Face (`thesofakillers/jigsaw-toxic-comment-classification-challenge`). Comments come from Wikipedia talk pages (about 2017) and were labelled by human raters. Licence per the dataset card: CC0 for the labels, with the comment text under Wikipedia's CC BY-SA 3.0. Task: multi-label (one comment can carry several labels).

## 1. Training file

- Rows: **{facts["rows"]:,}**; comments with at least one label: {facts["any_toxic"]:,} ({facts["any_toxic"] / facts["rows"]:.1%}).
- `severe_toxic` without `toxic`: {facts["severe_without_toxic"]} (it is always also toxic).
- Length: median {facts["median_chars"]} characters, 99th percentile {facts["p99_chars"]:,}, maximum 5,000. Contains a line break: {facts["newline_share"]:.0%}; contains a link: {facts["url_share"]:.1%}; contains non-ASCII characters: {facts["non_ascii_share"]:.1%}.

How labels overlap (share of comments with the row label that also have the column label):

{cooc_tbl}

## 2. Cleaning and deduplication

{table(["step", "rows in", "removed", "conflicting removed", "rows out"], steps)}

Near-duplicate groups (first 300 characters, digits and links masked): {len(multi):,} groups have more than one comment ({int(multi.sum()):,} rows; largest {int(multi.max())}). They are assigned to one split only.

## 3. Splits (multi-label stratified and grouped)

Method: StratifiedGroupKFold with 20 folds (14 train, 3 validation, 3 internal test), seed {settings.RANDOM_SEED}. The stratum is the pattern of the rare labels (threat, identity_hate, severe_toxic) plus "any toxic", so each rare label appears in every split.

{label_tbl}

Leakage checks (must be 0):

{leak_tbl}

## 4. Official test set

The Hugging Face "test" split has {test_info["rows"]:,} rows in **two stacked blocks** instead of one merged table: {test_info["text_only_rows"]:,} rows with the comment text but no labels, and {test_info["label_only_rows"]:,} rows with the labels but no text. Reading it naively yields either text without labels or labels without text. The blocks are joined by `id` (ids unique in each block: {test_info["text_ids_unique"]} / {test_info["label_ids_unique"]}), giving {test_info["joined"]:,} rows. The competition marked rows it did not score with -1 in every label.

- Scored rows (labels not -1): **{test_info["scored"]:,}**.
- After removing duplicates ({test_counts["duplicate_rows_removed"]}), conflicting rows ({test_counts["conflicting_rows_removed"]}) and comments that also appear in the training file ({overlap_removed}): **{len(official):,}** kept.
- Treating -1 as a real label would corrupt every metric; only scored rows are used.

## 5. Identity terms in the data (a source of bias)

Overall {base_toxic:.1%} of training comments are labelled toxic. For comments mentioning an identity term:

{ident_tbl}

A term that appears in far more toxic comments than the base rate will be learned as a toxicity signal, even in a harmless sentence. This is checked directly in training with a bias probe (neutral sentences with identity terms).

## 6. Known limitations

- Wikipedia talk-page comments from about 2017: slang, targets and platform style differ from today's social media.
- Labels come from crowd raters and contain disagreement and noise.
- Rare labels have few examples (`threat` {int(splits["train"]["threat"].sum())} in the training split), so their scores are unstable.
- English only.
- The text is offensive. Read error-analysis files in small batches.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
