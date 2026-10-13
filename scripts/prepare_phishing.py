"""Phase 7: prepare the phishing-URL data (NO training here).

Run from the project root:
    python -m scripts.prepare_phishing

Training pool : PhreshPhish URLs (Jul 2024 - Sep 2025). Test: PhreshPhish's own newer split
                (Sep - Dec 2025), a real temporal hold-out.
External sets : PhiUSIIL (labels flipped!), CompPhish, PhishBench, OpenPhish live feed.
All labels use the project convention: 1 = phishing (bad), 0 = legitimate (good).

URLs are only ever treated as text. Nothing is fetched or opened. The query string and fragment
are ignored everywhere (see the data report: no benign URL in PhreshPhish has a query string).
"""

import json

import pandas as pd

from config import settings
from scripts.prepare_sms import table
from src.data.splitting import check_no_overlap, grouped_stratified_split
from src.features.url_features import HOSTING_PLATFORMS, InvalidUrlError, parse_url

RAW = settings.DATA_RAW
OUT_DIR = settings.DATA_PROCESSED / "phishing"
EXTERNAL_DIR = settings.DATA_EXTERNAL_TEST / "phishing"
REPORT_PATH = settings.REPORTS_DIR / "data_quality" / "phishing.md"


def describe(url: str) -> tuple[str, str, str, bool]:
    """(dedup key, group, registered domain, is_hosting_platform); key == '' when invalid.

    key   = host (without www.) + path without trailing slash, lowercase: query, fragment and scheme
            are ignored, because the models ignore them.
    group = the tenant host for free-hosting platforms (anyone can publish there), otherwise the
            registered domain. All URLs of one group land in one split.
    """
    try:
        p = parse_url(url)
    except InvalidUrlError:
        return "", "", "", False
    host = p.host[4:] if p.host.startswith("www.") else p.host
    key = f"{host}{p.path.rstrip('/')}".lower()
    hosting = p.registered_domain in HOSTING_PLATFORMS or any(
        p.host.endswith("." + h) for h in HOSTING_PLATFORMS
    )
    group = p.host if hosting else p.registered_domain
    return key, group, p.registered_domain, hosting


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    info = [describe(u) for u in frame["url"]]
    frame = frame.copy()
    frame["key"] = [i[0] for i in info]
    frame["group"] = [i[1] for i in info]
    frame["registered_domain"] = [i[2] for i in info]
    frame["hosting"] = [i[3] for i in info]
    return frame


def structure(frame: pd.DataFrame, name: str) -> list:
    """Shape of the URLs: the numbers that reveal structural shortcuts."""
    rows = []
    for label, part in frame.groupby("label"):
        urls = part["url"]
        parsed = [parse_url(u) for u in urls.sample(min(len(urls), 20000), random_state=1)]
        n = len(parsed)
        rows.append(
            [
                name,
                "phishing" if label == 1 else "legitimate",
                f"{len(part):,}",
                f"{sum(p.scheme == 'https' for p in parsed) / n:.2f}",
                f"{sum(p.host.startswith('www.') for p in parsed) / n:.2f}",
                f"{sum(len(p.path) > 1 for p in parsed) / n:.2f}",
                f"{sum(len(p.query) > 0 for p in parsed) / n:.2f}",
                f"{int(urls.str.len().median())}",
                f"{part['hosting'].mean():.2f}",
            ]
        )
    return rows


def load_phresh(split: str) -> pd.DataFrame:
    raw = pd.read_parquet(RAW / "phishing_phreshphish" / f"urls_{split}.parquet")
    labels = raw["label"].str.lower().str.strip()
    unknown = sorted(set(labels) - {"benign", "phish"})
    if unknown:
        raise ValueError(f"PhreshPhish: unexpected labels {unknown}")
    return pd.DataFrame(
        {
            "url": raw["url"].fillna("").str.strip(),
            "label": (labels == "phish").astype(int),  # 1 = phishing
            "target": raw["target"],
            "date": raw["date"].astype(str),
            "lang": raw["lang"],
        }
    )


def load_externals() -> dict[str, pd.DataFrame]:
    out = {}
    phi = pd.read_csv(RAW / "phishing_phiusiil/extracted/phiusiil.csv", keep_default_na=False)
    # PhiUSIIL documents label 1 = LEGITIMATE and 0 = PHISHING: the opposite of this project.
    out["phiusiil"] = pd.DataFrame({"url": phi["URL"].str.strip(), "label": 1 - phi["label"]})
    comp = pd.read_excel(RAW / "phishing_compphish/Mapping_File.xlsx")
    # CompPhish documents label 1 = phishing, 0 = legitimate (same as the project).
    out["compphish"] = pd.DataFrame(
        {"url": comp["url"].astype(str).str.strip(), "label": comp["label"]}
    )
    pb = pd.read_csv(RAW / "phishing_phishbench/extracted/phishbench.csv", keep_default_na=False)
    # PhishBench "Results": 1 = phishing (free-hosting / random domains), 0 = bare popular homepages
    # (verified by reading the URLs of each value; the file does not document the convention).
    out["phishbench"] = pd.DataFrame(
        {"url": pb["URL"].str.strip(), "label": pb["Results"].astype(int)}
    )
    feed = pd.read_csv(RAW / "phishing_openphish/feed_2026-10-12.txt", header=None, names=["url"],
                       keep_default_na=False)  # fmt: skip
    out["openphish_2026-10-12"] = pd.DataFrame({"url": feed["url"].str.strip(), "label": 1})
    return out


def rule_accuracy(frame: pd.DataFrame, rule_legit) -> float:
    """Accuracy of a hand-written rule that says 'legitimate' when rule_legit(url_parts) is True."""
    parsed = [parse_url(u) for u in frame["url"].sample(min(len(frame), 30000), random_state=2)]
    labels = frame.loc[frame["url"].sample(min(len(frame), 30000), random_state=2).index, "label"]
    predicted_legit = [rule_legit(p) for p in parsed]
    return float((pd.Series(predicted_legit, index=labels.index) == (labels == 0)).mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    steps, facts = [], {}

    # ---- PhreshPhish: clean -> dedup -> group split ----
    pool = {s: load_phresh(s) for s in ("train", "test")}
    for name, frame in pool.items():
        n0 = len(frame)
        frame = enrich(frame[frame["url"] != ""])
        invalid = int((frame["key"] == "").sum())
        frame = frame[frame["key"] != ""]
        conflict_keys = frame.groupby("key")["label"].nunique()
        conflicts = frame["key"].isin(conflict_keys[conflict_keys > 1].index)
        n_conflict = int(conflicts.sum())
        frame = frame[~conflicts]
        before = len(frame)
        frame = frame.drop_duplicates("key").reset_index(drop=True)
        steps.append([f"PhreshPhish {name}", f"{n0:,}", f"{invalid:,}", f"{n_conflict:,}",
                      f"{before - len(frame):,}", f"{len(frame):,}"])  # fmt: skip
        pool[name] = frame
    # remove from the (newer) test split anything already in the training split
    leaked = pool["test"]["key"].isin(set(pool["train"]["key"]))
    steps.append(["PhreshPhish test: also in train", f"{len(pool['test']):,}", "", "", f"{int(leaked.sum()):,}",
                  f"{int((~leaked).sum()):,}"])  # fmt: skip
    pool["test"] = pool["test"][~leaked].reset_index(drop=True)

    train_all = pool["train"]
    train_all["split"] = grouped_stratified_split(
        train_all, label_col="label", group_col="group", seed=settings.RANDOM_SEED,
        n_folds=20, val_folds=3, test_folds=0,
    )  # fmt: skip
    parts = {"train": train_all[train_all["split"] == "train"].reset_index(drop=True),
             "val": train_all[train_all["split"] == "val"].reset_index(drop=True),
             "test": pool["test"]}  # fmt: skip
    key_overlap = check_no_overlap({k: v for k, v in parts.items() if k != "test"}, "key")
    group_overlap = check_no_overlap({k: v for k, v in parts.items() if k != "test"}, "group")
    assert all(v == 0 for v in key_overlap.values()), key_overlap
    assert all(v == 0 for v in group_overlap.values()), group_overlap
    cols = [
        "url",
        "label",
        "target",
        "date",
        "lang",
        "group",
        "registered_domain",
        "hosting",
        "key",
    ]
    for name, part in parts.items():
        part[cols].to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8")

    seen_groups = set(parts["train"]["group"]) | set(parts["val"]["group"])
    test_seen = parts["test"]["group"].isin(seen_groups)
    facts["test_seen_domain_share"] = float(test_seen.mean())
    facts["test_seen_by_label"] = {
        int(k): float(v) for k, v in test_seen.groupby(parts["test"]["label"]).mean().items()
    }
    facts["date_ranges"] = {
        k: f"{v['date'].min()} to {v['date'].max()}"
        for k, v in {"train+val": train_all, "test": parts["test"]}.items()
    }
    facts["val_in_train_domain_overlap"] = 0

    # ---- external sets: flip labels, clean, dedup, remove anything in the pool ----
    pool_keys = set(train_all["key"]) | set(parts["test"]["key"])
    ext_rows, structure_rows, rule_rows = [], [], []
    structure_rows += structure(parts["train"], "PhreshPhish train")
    structure_rows += structure(parts["test"], "PhreshPhish test (newer)")
    externals = {}
    for name, frame in load_externals().items():
        n0 = len(frame)
        frame = enrich(frame[frame["url"] != ""])
        invalid = int((frame["key"] == "").sum())
        frame = frame[frame["key"] != ""]
        conflict_keys = frame.groupby("key")["label"].nunique()
        n_conflict = int(frame["key"].isin(conflict_keys[conflict_keys > 1].index).sum())
        frame = frame[~frame["key"].isin(conflict_keys[conflict_keys > 1].index)]
        dups = len(frame) - frame["key"].nunique()
        frame = frame.drop_duplicates("key")
        in_pool = frame["key"].isin(pool_keys)
        frame = frame[~in_pool].reset_index(drop=True)
        frame[["url", "label", "group", "registered_domain", "hosting", "key"]].to_csv(
            EXTERNAL_DIR / f"{name}.csv", index=False, encoding="utf-8"
        )
        externals[name] = frame
        ext_rows.append([name, f"{n0:,}", f"{invalid:,}", f"{n_conflict:,}", f"{dups:,}",
                         f"{int(in_pool.sum()):,}", f"{len(frame):,}",
                         f"{int((frame['label'] == 1).sum()):,} / {int((frame['label'] == 0).sum()):,}"])  # fmt: skip
        structure_rows += structure(frame, name)

    # shortcut baselines: how well do trivial structural rules "detect phishing"?
    phi = externals["phiusiil"]
    rule_rows.append(["PhiUSIIL", "legitimate if the URL is a bare https://www.<domain> homepage",
                      f"{rule_accuracy(phi, lambda p: p.scheme == 'https' and p.host.startswith('www.') and len(p.path) <= 1 and not p.query):.4f}"])  # fmt: skip
    rule_rows.append(
        [
            "PhiUSIIL",
            "legitimate if the URL has no path at all",
            f"{rule_accuracy(phi, lambda p: len(p.path) <= 1):.4f}",
        ]
    )
    rule_rows.append(["PhreshPhish train", "legitimate if the URL has NO query string", f"{rule_accuracy(parts['train'], lambda p: not p.query):.4f}"])  # fmt: skip
    rule_rows.append(["PhreshPhish train", "legitimate if the URL starts with www.", f"{rule_accuracy(parts['train'], lambda p: p.host.startswith('www.')):.4f}"])  # fmt: skip
    rule_rows.append(
        [
            "PhreshPhish train",
            "always say legitimate (majority baseline)",
            f"{(parts['train']['label'] == 0).mean():.4f}",
        ]
    )

    (OUT_DIR / "split_summary.json").write_text(
        json.dumps({"sizes": {k: len(v) for k, v in parts.items()},
                    "phishing_share": {k: round(float(v['label'].mean()), 4) for k, v in parts.items()},
                    **facts}, indent=2),
        encoding="utf-8",
    )  # fmt: skip
    write_report(
        steps, parts, key_overlap, group_overlap, facts, ext_rows, structure_rows, rule_rows
    )
    print(f"Saved to {OUT_DIR} and {EXTERNAL_DIR}\nWrote {REPORT_PATH}")
    print(table(["split", "rows", "phishing share"],
                [[k, f"{len(v):,}", f"{v['label'].mean():.3f}"] for k, v in parts.items()]))  # fmt: skip


def write_report(
    steps, parts, key_overlap, group_overlap, facts, ext_rows, structure_rows, rule_rows
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    split_tbl = table(
        ["split", "rows", "legitimate", "phishing", "phishing share", "distinct groups"],
        [[k, f"{len(v):,}", f"{int((v['label'] == 0).sum()):,}", f"{int((v['label'] == 1).sum()):,}",
          f"{v['label'].mean():.3f}", f"{v['group'].nunique():,}"] for k, v in parts.items()],
    )  # fmt: skip
    leak_tbl = table(["pair", "shared URL keys", "shared groups"],
                     [[k, key_overlap[k], group_overlap[k]] for k in key_overlap])  # fmt: skip
    text = f"""# Phishing-URL data-quality report

Generated by `python -m scripts.prepare_phishing` (seed {settings.RANDOM_SEED}). URLs are plain text; none was visited.
Label convention: **1 = phishing (bad), 0 = legitimate (good)**.

## 1. Datasets and their label conventions

| Dataset | Role | Documented label convention | Mapped to the project |
|---|---|---|---|
| PhreshPhish (CC BY 4.0, "for anti-phishing research"; only the url, label, target, date and lang columns were read, never the HTML) | training pool and temporal test | `label` = benign / phish | phish -> 1 |
| PhiUSIIL (UCI 967) | external only | **1 = legitimate, 0 = phishing** | **flipped**: ours = 1 - label |
| CompPhish (Mendeley, Sep 2024 - Aug 2025; only the URL/label sheet was read) | external | 1 = phishing, 0 = legitimate | unchanged |
| PhishBench-External (Mendeley, Jul 2025) | external | `Results` undocumented; verified by reading URLs: 1 = phishing, 0 = bare popular homepages | unchanged |
| OpenPhish community feed (2026-10-12, 300 URLs) | external, phishing only | all phishing | 1 |

## 2. Cleaning and deduplication

A URL's *key* is its host (without `www.`) plus path, lowercase, ignoring scheme, query and fragment.

{table(["step", "rows in", "invalid URLs", "conflicting-label rows", "duplicates removed", "rows out"], steps)}

## 3. Splits

PhreshPhish's own train split ({facts["date_ranges"]["train+val"]}) is divided 85/15 into train and validation with StratifiedGroupKFold (20 folds, seed {settings.RANDOM_SEED}); groups are the *tenant host* for free-hosting platforms (anyone can publish under pages.dev, weebly, gitbook.io, ...) and the registered domain otherwise. PhreshPhish's own **test split is newer** ({facts["date_ranges"]["test"]}), so test measures performance on future data.

{split_tbl}

Leakage checks between train and validation (must be 0):

{leak_tbl}

**Domains seen in training:** {facts["test_seen_domain_share"]:.1%} of test URLs come from a group that also appears in train/validation (legitimate {facts["test_seen_by_label"].get(0, 0):.1%}, phishing {facts["test_seen_by_label"].get(1, 0):.1%}). Test results are therefore reported overall and on unseen groups only.

## 4. External sets (after cleaning, and after removing anything already in the training pool or test)

{table(["set", "rows loaded", "invalid", "conflicting", "duplicates", "also in pool (removed)", "kept", "phishing / legitimate"], ext_rows)}

## 5. URL shape by class (the shortcut check)

Share of URLs that: use https, start with `www.`, have a path beyond `/`, carry a query string; median length; share on a free-hosting platform.

{table(["dataset", "class", "URLs", "https", "www.", "path", "query", "median length", "free hosting"], structure_rows)}

Read this table before any model is trained:
- **PhiUSIIL**: every legitimate URL is a bare `https://www.<domain>` homepage; no legitimate URL has a path or a query. A model trained on it learns "is it a bare https://www homepage?".
- **PhreshPhish**: legitimate URLs have realistic paths (like real links), but **no legitimate URL has a query string** while phishing URLs do. The query string is therefore ignored by every model and by the app.
- **CompPhish and PhishBench**: the legitimate side is mostly bare popular homepages.
- Phishing is often hosted on free platforms (see the free-hosting column).

## 6. How far trivial rules go (accuracy of a one-line rule, no machine learning)

{table(["dataset", "rule", "accuracy"], rule_rows)}

A rule this simple scoring near-perfectly means the dataset cannot tell us whether a model understands phishing.

## 7. Known limitations

- No page content, certificate, domain age or reputation: URL text only, by design.
- Shortened links (bit.ly, ...) cannot be judged without following them, which this project never does.
- Free-hosting platforms host both legitimate and phishing pages; the URL text alone often cannot separate them.
- The legitimate URLs in every public set come from popular sites; unknown small legitimate sites look unusual to a model.
- Labels come from feeds and crawls, not from per-URL human review.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
