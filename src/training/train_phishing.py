"""Phase 7: train and compare phishing-URL models, then save the selected pipeline.

Run from the project root (after `python -m scripts.prepare_phishing`):
    python -m src.training.train_phishing

RULES FIXED BEFORE LOOKING AT ANY RESULT
  1. Candidates are compared by PR-AUC (phishing = positive class) on the VALIDATION split.
  2. Candidates within TIE_TOLERANCE of the best are ordered by PREFERENCE_ORDER (simple first).
  3. GUARD: a candidate that flags more than MAX_LEGIT_FLAGGED_SHARE of the hand-written
     LEGITIMATE URLs (tests/data/url_challenge.csv) is skipped. Those are ordinary links with
     realistic paths, which the public training data contains too few of. The set is a smoke
     test used only for this gate.
  4. Threshold: highest validation recall with at most MAX_FPR of legitimate URLs flagged.
     (Missing phishing is the costly error, but a tool that flags many real links is unusable.)
  5. The temporal test split and every external set are evaluated once, after the choice.
  6. Shortcut ablations (no www/https, host-only) are DIAGNOSTICS and never change the choice.
All features come from src/features/url_features.py: the same code the app runs. The query string
and fragment are ignored everywhere. No URL is ever fetched.
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from scripts.prepare_phishing import describe
from src.evaluation import plots
from src.evaluation.metrics import get_scores, threshold_for_fpr, threshold_metrics
from src.features.url_features import (
    FEATURE_NAMES,
    UrlFeatureTransformer,
    UrlText,
    features_frame,
)

DATA_DIR = settings.DATA_PROCESSED / "phishing"
EXTERNAL_DIR = settings.DATA_EXTERNAL_TEST / "phishing"
CHALLENGE_FILE = settings.PROJECT_ROOT / "tests" / "data" / "url_challenge.csv"
MODEL_DIR = settings.MODELS_DIR / "phishing"
METRICS_DIR = settings.REPORTS_DIR / "metrics"
FIGURES_DIR = settings.REPORTS_DIR / "figures"

MAX_FPR = 0.02
TIE_TOLERANCE = 0.003
MAX_LEGIT_FLAGGED_SHARE = 0.20
PREFERENCE_ORDER = ["logreg_features", "random_forest", "hist_gb", "logreg_char"]
PATH_FEATURE_PREFIXES = (
    "path",
    "entropy_path", "longest_path", "ends_with", "risky", "double", "n_login_words_path", "brand_in_path", "has_path",
)  # fmt: skip
EXTERNAL_SETS = ["phiusiil", "compphish", "phishbench", "openphish_2026-10-12"]
# Round 2 trains on PhiUSIIL too (its legitimate side is bare homepages, the opposite shape of
# PhreshPhish's deep benign pages), so PhiUSIIL stops being an external test set.


def feature_matrix(urls, n_jobs: int = 6) -> np.ndarray:
    """Feature table for many URLs, computed in parallel chunks (identical code to the app)."""
    urls = list(urls)
    chunks = [urls[i : i + 20000] for i in range(0, len(urls), 20000)]
    frames = Parallel(n_jobs=n_jobs)(delayed(features_frame)(c, "nan") for c in chunks)
    return pd.concat(frames, ignore_index=True).fillna(0.0).to_numpy()


def select_url_model(results: pd.DataFrame) -> tuple[str, bool]:
    real = results[results["model"] != "dummy"].set_index("model")
    best = real["val_pr_auc"].max()
    close = [
        m
        for m in PREFERENCE_ORDER
        if m in real.index and real.loc[m, "val_pr_auc"] >= best - TIE_TOLERANCE
    ]
    for model in close:
        if real.loc[model, "challenge_legit_flagged_share"] <= MAX_LEGIT_FLAGGED_SHARE:
            return model, True
    return close[0], False


def evaluate_set(scores, y, threshold, name) -> dict:
    out = threshold_metrics(y, scores, threshold)
    print(f"  {name}: P {out['precision']:.3f} R {out['recall']:.3f} FPR {out['false_positive_rate']:.3f} "
          f"PR-AUC {out['pr_auc']:.3f} (n={out['n']:,})")  # fmt: skip
    return out


def main(with_phiusiil: bool = False) -> None:
    for folder in (MODEL_DIR, METRICS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    val = pd.read_csv(DATA_DIR / "val.csv", keep_default_na=False)
    test = pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False)
    challenge = pd.read_csv(CHALLENGE_FILE, keep_default_na=False)
    train["source"], val["source"] = "phreshphish", "phreshphish"
    external_names = list(EXTERNAL_SETS)
    if with_phiusiil:
        external_names.remove("phiusiil")
        phi_train = pd.read_csv(DATA_DIR / "phiusiil_train.csv", keep_default_na=False)
        phi_val = pd.read_csv(DATA_DIR / "phiusiil_val.csv", keep_default_na=False)
        train = pd.concat([train, phi_train.assign(source="phiusiil")], ignore_index=True)
        val = pd.concat([val, phi_val.assign(source="phiusiil")], ignore_index=True)
    externals = {
        n: pd.read_csv(EXTERNAL_DIR / f"{n}.csv", keep_default_na=False) for n in external_names
    }
    # External sets must share no URL with ANY training/validation/test data. They were cleaned
    # against the PhreshPhish pool only, so with PhiUSIIL in training they are filtered again here.
    training_keys = set(train["key"]) | set(val["key"]) | set(test["key"])
    for name in list(externals):
        overlap_mask = externals[name]["key"].isin(training_keys)
        if overlap_mask.any():
            print(
                f"external {name}: removed {int(overlap_mask.sum()):,} URLs also in training data"
            )
        externals[name] = externals[name][~overlap_mask].reset_index(drop=True)
    assert not set(train["key"]) & set(val["key"]), "URL leakage between train and validation"
    assert not set(train["group"]) & set(val["group"]), (
        "domain-group leakage between train and validation"
    )
    assert not set(test["key"]) & (set(train["key"]) | set(val["key"])), (
        "test URLs found in training"
    )
    # hand-written URLs that are also in the training data would not test generalisation: drop them
    challenge["key"] = [describe(u)[0] for u in challenge["url"]]
    seen_keys = set(train["key"]) | set(val["key"])
    overlap = challenge["key"].isin(seen_keys)
    if overlap.any():
        print(f"challenge URLs also present in training data (excluded): {int(overlap.sum())}")
    challenge = challenge[~overlap].reset_index(drop=True)
    print(
        f"train {len(train):,} | val {len(val):,} | test {len(test):,} | challenge {len(challenge)}"
    )
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()

    print("computing features (same code as the app)...")
    X = {
        n: feature_matrix(f["url"])
        for n, f in {"train": train, "val": val, "test": test, "challenge": challenge}.items()
    }
    print(f"  features done ({time.perf_counter() - started:.0f}s)")
    legit_mask = (challenge["label"] == 0).to_numpy()

    # ---- candidates: tune on validation ----
    scaler = StandardScaler().fit(X["train"])
    models, rows = {}, []

    def register(name, fitted, scorer, params):
        val_scores = scorer(fitted, "val")
        pick = threshold_for_fpr(y_val, val_scores, MAX_FPR)
        at = threshold_metrics(y_val, val_scores, pick["threshold"])
        ch_scores = scorer(fitted, "challenge")
        flagged = ch_scores >= pick["threshold"]
        rows.append({
            "model": name, "params": json.dumps(params),
            "val_pr_auc": average_precision_score(y_val, val_scores),
            "val_recall_at_fpr_budget": at["recall"], "val_fpr": at["false_positive_rate"],
            "threshold": pick["threshold"], "target_reached": pick["reached_target"],
            "challenge_legit_flagged_share": float(flagged[legit_mask].mean()),
            "challenge_phish_caught_share": float(flagged[~legit_mask].mean()),
        })  # fmt: skip
        models[name] = {
            "fitted": fitted,
            "scorer": scorer,
            "params": params,
            "threshold": pick["threshold"],
        }
        r = rows[-1]
        print(f"{name} {params}: val PR-AUC {r['val_pr_auc']:.4f} | recall@FPR<={MAX_FPR:.0%} {r['val_recall_at_fpr_budget']:.3f} "
              f"| legit challenge flagged {r['challenge_legit_flagged_share']:.2f} ({time.perf_counter() - started:.0f}s)")  # fmt: skip

    def feat_scorer(model):
        return lambda m, split: m.predict_proba(X[split] if split in X else split)[:, 1]

    def scaled_scorer(m, split):
        data = X[split] if isinstance(split, str) else split
        return m.predict_proba(scaler.transform(data))[:, 1]

    dummy = DummyClassifier(strategy="prior").fit(X["train"], y_train)
    register("dummy", dummy, feat_scorer(dummy), {})

    best_lr = None
    for c in (0.1, 1.0, 10.0):
        lr = LogisticRegression(C=c, max_iter=3000).fit(scaler.transform(X["train"]), y_train)
        pr = average_precision_score(y_val, scaled_scorer(lr, "val"))
        if best_lr is None or pr > best_lr[0]:
            best_lr = (pr, c, lr)
    register("logreg_features", best_lr[2], scaled_scorer, {"C": best_lr[1]})

    best_rf = None
    for leaf in (1, 5):
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf, n_jobs=-1,
                                    random_state=settings.RANDOM_SEED).fit(X["train"], y_train)  # fmt: skip
        pr = average_precision_score(y_val, rf.predict_proba(X["val"])[:, 1])
        if best_rf is None or pr > best_rf[0]:
            best_rf = (pr, leaf, rf)
    register(
        "random_forest",
        best_rf[2],
        feat_scorer(best_rf[2]),
        {"n_estimators": 200, "min_samples_leaf": best_rf[1]},
    )

    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                         random_state=settings.RANDOM_SEED).fit(X["train"], y_train)  # fmt: skip
    register("hist_gb", hgb, feat_scorer(hgb), {"max_iter": 300, "learning_rate": 0.1})

    vec = TfidfVectorizer(preprocessor=UrlText(), analyzer="char_wb", ngram_range=(2, 5), min_df=5,
                          max_features=300_000, sublinear_tf=True, dtype=np.float32)  # fmt: skip
    Xc_train = vec.fit_transform(train["url"])
    char_cache = {"val": vec.transform(val["url"]), "challenge": vec.transform(challenge["url"])}
    best_char = None
    for c in (1.0, 4.0):
        clf = LogisticRegression(C=c, max_iter=1000, solver="liblinear").fit(Xc_train, y_train)
        pr = average_precision_score(y_val, clf.predict_proba(char_cache["val"])[:, 1])
        if best_char is None or pr > best_char[0]:
            best_char = (pr, c, clf)
    char_pipe = Pipeline([("features", vec), ("clf", best_char[2])])

    def char_scorer(m, split):
        return m.named_steps["clf"].predict_proba(char_cache[split])[:, 1]

    register("logreg_char", char_pipe, char_scorer, {"C": best_char[1]})

    results = pd.DataFrame(rows)
    results.to_csv(METRICS_DIR / "phishing_model_comparison.csv", index=False)
    chosen, guard_ok = select_url_model(results)
    info = models[chosen]
    threshold = info["threshold"]
    print(
        f"\nCHOSEN: {chosen} {info['params']} | threshold {threshold:.4f} | guard passed: {guard_ok}"
    )

    # ---- the deployable pipeline (feature extraction runs INSIDE it) ----
    if chosen == "logreg_features":
        pipeline = Pipeline(
            [("features", UrlFeatureTransformer()), ("scale", scaler), ("clf", info["fitted"])]
        )
    elif chosen in ("random_forest", "hist_gb"):
        pipeline = Pipeline([("features", UrlFeatureTransformer()), ("clf", info["fitted"])])
    else:
        pipeline = info["fitted"]

    def score_urls(urls) -> np.ndarray:
        return get_scores(pipeline, list(urls))

    # ---- evaluate once ----
    print("evaluation:")
    test_scores = score_urls(test["url"])
    metrics = {"validation": evaluate_set(score_urls(val["url"]), y_val, threshold, "validation"),
               "test_temporal": evaluate_set(test_scores, test["label"].to_numpy(), threshold, "test (Sep-Dec 2025)")}  # fmt: skip
    metrics["validation_by_source"] = {}
    val_scores_all = score_urls(val["url"])
    for source, idx in val.groupby("source").groups.items():
        sel = val.index.get_indexer(idx)
        part = evaluate_set(val_scores_all[sel], y_val[sel], threshold, f"validation / {source}")
        metrics["validation_by_source"][source] = part
    seen = set(train["group"]) | set(val["group"])
    unseen = ~test["group"].isin(seen).to_numpy()
    metrics["test_unseen_groups"] = evaluate_set(
        test_scores[unseen], test["label"].to_numpy()[unseen], threshold, "test, unseen groups only"
    )
    metrics["test_hosting_platforms"] = evaluate_set(test_scores[test["hosting"].to_numpy()], test["label"].to_numpy()[test["hosting"].to_numpy()], threshold, "test, free-hosting URLs only")  # fmt: skip
    month = test["date"].astype(str).str[:7].to_numpy()
    by_month = {}
    for m in sorted(set(month)):
        sel = month == m
        by_month[m] = {"n": int(sel.sum()), **{k: threshold_metrics(test["label"].to_numpy()[sel], test_scores[sel], threshold)[k] for k in ("precision", "recall", "false_positive_rate")}}  # fmt: skip
    metrics["test_by_month"] = by_month

    external_report = {}
    for name, frame in externals.items():
        scores = score_urls(frame["url"])
        y = frame["label"].to_numpy()
        if len(set(y)) == 2:
            external_report[name] = evaluate_set(scores, y, threshold, name)
        else:
            external_report[name] = {
                "n": int(len(y)),
                "detection_rate": float((scores >= threshold).mean()),
            }
            print(
                f"  {name}: detection rate {external_report[name]['detection_rate']:.3f} (n={len(y)}, phishing only)"
            )
    metrics["external"] = external_report

    ch_scores = score_urls(challenge["url"])
    flagged = ch_scores >= threshold
    by_kind = {k: {"n": int((challenge['kind'] == k).sum()), "flagged": int(flagged[(challenge['kind'] == k).to_numpy()].sum())}
               for k in challenge["kind"].unique()}  # fmt: skip
    metrics["challenge"] = {"legit_n": int(legit_mask.sum()), "legit_flagged": int(flagged[legit_mask].sum()),
                            "phish_n": int((~legit_mask).sum()), "phish_caught": int(flagged[~legit_mask].sum()),
                            "by_kind": by_kind}  # fmt: skip
    print(f"  challenge: legitimate flagged {metrics['challenge']['legit_flagged']}/{metrics['challenge']['legit_n']} | phishing-style caught "
          f"{metrics['challenge']['phish_caught']}/{metrics['challenge']['phish_n']} | by kind {by_kind}")  # fmt: skip

    # ---- errors (URL text, Git-ignored folder) ----
    pred = test_scores >= threshold
    err = test.assign(score=test_scores)
    err[(err["label"] == 0) & pred].sort_values("score", ascending=False).head(300)[["url", "score", "date"]].to_csv(DATA_DIR / "errors_false_positives.csv", index=False)  # fmt: skip
    err[(err["label"] == 1) & ~pred].sort_values("score").head(300)[["url", "score", "date", "target"]].to_csv(DATA_DIR / "errors_false_negatives.csv", index=False)  # fmt: skip

    # ---- diagnostics: shortcut ablations (fast gradient boosting; never used for selection) ----
    ablations = {}
    names = np.array(FEATURE_NAMES)
    variants = {
        "all features": np.ones(len(names), bool),
        "without free-hosting / shortener flags": ~np.isin(
            names, ["is_hosting_platform", "is_shortener"]
        ),
        "host only (no path features)": np.array(
            [not n.startswith(PATH_FEATURE_PREFIXES) for n in names]
        ),
    }
    for label, keep in variants.items():
        m = HistGradientBoostingClassifier(max_iter=200, random_state=settings.RANDOM_SEED).fit(
            X["train"][:, keep], y_train
        )
        v_scores = m.predict_proba(X["val"][:, keep])[:, 1]
        pick = threshold_for_fpr(y_val, v_scores, MAX_FPR)
        t_scores = m.predict_proba(X["test"][:, keep])[:, 1]
        c_flag = m.predict_proba(X["challenge"][:, keep])[:, 1] >= pick["threshold"]
        ablations[label] = {
            "n_features": int(keep.sum()),
            "test_pr_auc": float(average_precision_score(test["label"], t_scores)),
            "test_recall": threshold_metrics(test["label"], t_scores, pick["threshold"])["recall"],
            "test_fpr": threshold_metrics(test["label"], t_scores, pick["threshold"])[
                "false_positive_rate"
            ],
            "challenge_legit_flagged": int(c_flag[legit_mask].sum()),
        }
        print(f"ablation [{label}]: {ablations[label]}")
    importances = pd.DataFrame({"feature": FEATURE_NAMES, "importance": best_rf[2].feature_importances_}).sort_values("importance", ascending=False)  # fmt: skip
    importances.to_csv(METRICS_DIR / "phishing_feature_importance.csv", index=False)

    sample = test["url"].sample(300, random_state=settings.RANDOM_SEED).tolist()
    t0 = time.perf_counter()
    for u in sample:
        score_urls([u])
    latency_ms = (time.perf_counter() - t0) / len(sample) * 1000

    plots.plot_confusion(
        metrics["test_temporal"],
        f"Temporal test: {chosen}",
        FIGURES_DIR / "phishing_confusion_test.png",
    )
    plots.plot_model_comparison(results["model"].tolist(), results["val_pr_auc"].tolist(), "Phishing: validation PR-AUC by model", "PR-AUC", FIGURES_DIR / "phishing_model_comparison.png")  # fmt: skip

    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    assert np.allclose(get_scores(reloaded, sample), get_scores(pipeline, sample))
    metadata = {
        "module": "phishing", "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "label_convention": "1 = phishing (bad), 0 = legitimate (good)",
        "model": chosen, "model_params": info["params"], "feature_names": FEATURE_NAMES,
        "score_type": "probability", "threshold": threshold,
        "threshold_rule": f"highest validation recall with at most {MAX_FPR:.0%} of legitimate URLs flagged",
        "selection_rule": f"best validation PR-AUC; ties (within {TIE_TOLERANCE}) -> {PREFERENCE_ORDER}; guard: skip candidates flagging > {MAX_LEGIT_FLAGGED_SHARE:.0%} of hand-written legitimate URLs",  # fmt: skip
        "guard_passed": guard_ok, "query_and_fragment_ignored": True, "seed": settings.RANDOM_SEED,
        "trained_on": ["phreshphish", "phiusiil"] if with_phiusiil else ["phreshphish"],
        "data_sizes": {"train": len(train), "val": len(val), "test": len(test),
                       **{n: len(f) for n, f in externals.items()}, "challenge": len(challenge)},  # fmt: skip
        "date_ranges": json.loads((DATA_DIR / "split_summary.json").read_text())["date_ranges"],
        "metrics": metrics, "ablations": ablations,
        "top_features": importances.head(10).to_dict("records"),
        "inference_ms_per_url": latency_ms, "model_file_mb": round(model_path.stat().st_size / 1e6, 2),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "limitations": [
            "Reads only the URL text: no page content, certificate, domain age or reputation data.",
            "The query string and fragment are ignored.",
            "Shortened links cannot be judged without following them, which this project never does.",
            "Free-hosting platforms host both legitimate and phishing pages; the URL text alone often cannot separate them.",
            "Public datasets contain legitimate URLs mostly from popular sites; small unknown legitimate sites may look unusual.",
            "Trained on 2024-2025 data; phishing changes quickly.",
        ],
    }  # fmt: skip
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    print(
        f"\nmodel {metadata['model_file_mb']} MB | {latency_ms:.2f} ms/URL | total {time.perf_counter() - started:.0f}s"
    )
    print(results.drop(columns=["params"]).round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-phiusiil", action="store_true", help="Round 2: add PhiUSIIL to training"
    )
    main(parser.parse_args().with_phiusiil)
