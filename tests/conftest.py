"""Shared fixtures: tiny synthetic models so tests never need the real data."""

import hashlib
import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SPAM = [
    "win free prize now",
    "free cash claim now",
    "urgent win cash prize call",
    "claim your free prize today",
    "free entry win cash",
    "congratulations you won a prize call now",
    "verify your account now click link",
    "your account is blocked click to verify",
    "sale discount offer buy now",
    "big sale flat discount off",
]
HAM = [
    "see you at lunch",
    "call me later please",
    "running late for dinner",
    "thanks for the help today",
    "are we meeting tomorrow",
    "happy birthday see you soon",
]
SCAM_ONLY = [
    "verify your account now click link",
    "your account is blocked click to verify",
    "urgent win cash prize call",
    "congratulations you won a prize call now",
    "claim your free prize today",
]
PROMO_ONLY = [
    "sale discount offer buy now",
    "big sale flat discount off",
    "free entry win cash",
    "free cash claim now",
    "win free prize now",
]
EMAIL_SPAM = [
    "win a free prize . click the link now to claim your cash prize",
    "cheap pills discount . order now no prescription needed click here",
    "your account is suspended . verify your password at the link immediately",
    "make money fast . guaranteed income work from home sign up today",
    "urgent invoice overdue . pay now using the attached link or face action",
    "limited offer . buy cheap watches online free shipping click here",
]
EMAIL_HAM = [
    "meeting on thursday . hi team the design review moves to thursday at three please bring notes",
    "lunch tomorrow . are you free for lunch tomorrow with the project group",
    "report draft . i attached the draft report please send comments by friday",
    "family dinner . mom is cooking on sunday please let her know if you can come",
    "schedule update . the schedule for next week is attached thanks for your help",
    "course reading . please read chapter four before class and prepare questions",
]

# Toy harmful-comment data: made-up trigger words, one group per label pattern.
# (labels in order: toxic, severe_toxic, obscene, threat, insult, identity_hate)
TOX_GROUPS = [
    (
        [
            "you are an idiot",
            "what a stupid fool",
            "such an idiot honestly",
            "stupid idiot person",
            "you stupid fool",
        ],
        [1, 0, 0, 0, 1, 0],
    ),
    (
        [
            "i will kill you",
            "i will hurt you badly",
            "going to kill you tonight",
            "i will hurt you",
            "kill you soon",
        ],
        [1, 0, 0, 1, 0, 0],
    ),
    (
        [
            "this damn crap",
            "damn this crap thing",
            "what crap damn it",
            "crap and damn",
            "damn crap",
        ],
        [1, 0, 1, 0, 0, 0],
    ),
    (
        [
            "zorbs are vermin",
            "all zorbs vermin disgusting",
            "zorbs vermin go away",
            "vermin zorbs",
            "zorbs disgusting vermin",
        ],
        [1, 1, 0, 0, 0, 1],
    ),
]
TOX_CLEAN = [
    "thanks for the helpful edit",
    "the article needs a better source",
    "see you at the meeting",
    "please add the census figures",
    "good work on the references",
    "i agree with the proposal",
    "the weather was nice today",
    "can you explain this paragraph",
    "lunch is at noon",
    "the schedule is attached",
    "happy to help with that",
    "this page looks much better now",
    "welcome to the project",
    "let us discuss on the talk page",
    "the match ended in a draw",
    "i enjoyed reading the report",
    "the library opens at nine",
    "thank you for the update",
    "please review the draft",
    "that seems reasonable to me",
]


def _fit_toxicity():
    from src.models.toxicity_model import PerLabelLogReg

    texts, labels = [], []
    for group_texts, label_row in TOX_GROUPS:
        texts += group_texts
        labels += [label_row] * len(group_texts)
    texts += TOX_CLEAN
    labels += [[0] * 6] * len(TOX_CLEAN)
    features = TfidfVectorizer().fit(texts)
    return (
        features,
        PerLabelLogReg(C=10, n_jobs=2).fit(features.transform(texts), labels),
        len(texts),
    )


def _toy_urls():
    """Deterministic legitimate-looking and phishing-looking URLs (all on made-up names)."""
    import random

    rng = random.Random(7)
    words = ["garden", "recipes", "news", "travel", "books", "music", "school", "weather", "sports"]
    brands = ["paypal", "apple", "amazon", "netflix", "microsoft", "chase"]
    legit, phish = [], []
    for i in range(120):
        legit.append(
            f"https://www.{rng.choice(words)}{rng.choice(words)}.com/{rng.choice(words)}/"
            f"{rng.choice(words)}-{i}"
        )
        phish.append(
            f"http://{rng.choice(brands)}-secure-login-{rng.randint(10, 99)}."
            f"{rng.choice(['xyz', 'top', 'tk'])}/verify/account/login{i}.php"
        )
    return legit, phish


def _fit_phishing():
    from sklearn.ensemble import RandomForestClassifier

    from src.features.url_features import UrlFeatureTransformer

    legit, phish = _toy_urls()
    urls, labels = legit + phish, [0] * len(legit) + [1] * len(phish)
    features = UrlFeatureTransformer()
    forest = RandomForestClassifier(n_estimators=40, random_state=0).fit(
        features.transform(urls), labels
    )
    return Pipeline([("features", UrlFeatureTransformer()), ("clf", forest)]), len(urls)


def _fit(texts, labels, c=10):
    pipe = Pipeline([("features", TfidfVectorizer()), ("clf", LogisticRegression(C=c))])
    return pipe.fit(texts, labels)


def _save(folder, pipe, meta, *, corrupt=False):
    folder.mkdir(parents=True)
    model_path = folder / "model.joblib"
    joblib.dump(pipe, model_path)
    meta["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if corrupt:
        model_path.write_bytes(model_path.read_bytes() + b"tampered")
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def write_fake_model(
    models_dir,
    *,
    corrupt=False,
    with_type=True,
    corrupt_type=False,
    with_email=True,
    with_toxicity=True,
    with_phishing=True,
):
    metrics = {"precision": 0.9, "recall": 0.9, "accuracy": 0.9}
    main = _fit(SPAM + HAM, [1] * len(SPAM) + [0] * len(HAM))
    _save(
        models_dir / "sms_spam",
        main,
        {
            "model": "toy",
            "threshold": 0.5,
            "score_type": "probability",
            "created_utc": "2026-01-01T00:00:00+00:00",
            "data_sizes": {"train": 16},
            "metrics": {"test": metrics, "external_india": metrics},
            "limitations": ["toy model for tests"],
        },
        corrupt=corrupt,
    )
    if with_type:
        kind = _fit(SCAM_ONLY + PROMO_ONLY, [1] * len(SCAM_ONLY) + [0] * len(PROMO_ONLY))
        _save(
            models_dir / "sms_type",
            kind,
            {
                "model": "toy_type",
                "scam_threshold": 0.5,
                "score_type": "probability",
                "data_sizes": {"dev": 10},
                "metrics": {"cross_validated_dev": metrics},
                "limitations": ["toy type model"],
            },
            corrupt=corrupt_type,
        )
    if with_email:
        mail = _fit(EMAIL_SPAM + EMAIL_HAM, [1] * len(EMAIL_SPAM) + [0] * len(EMAIL_HAM))
        _save(
            models_dir / "email_spam",
            mail,
            {
                "model": "toy_email",
                "threshold": 0.5,
                "score_type": "probability",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "data_sizes": {"train": 12},
                "metrics": {"test": metrics},
                "external_2026": {"detection_rate": 0.9, "n": 100},
                "guard_passed": True,
                "limitations": ["toy email model"],
            },
        )

    if with_toxicity:
        from src.models.toxicity_model import LABELS, ToxicityModel

        features, classifier, n_rows = _fit_toxicity()
        thresholds = {label: 0.5 for label in LABELS}
        per_label = {
            label: {"precision": 0.9, "recall": 0.8, "f1": 0.85, "positives": 10}
            for label in LABELS
        }
        _save(
            models_dir / "toxicity",
            ToxicityModel(features, classifier, thresholds),
            {
                "model": "toy_toxicity",
                "thresholds": thresholds,
                "score_type": "probability",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "data_sizes": {"train": n_rows},
                "metrics": {
                    "official_test": {
                        "summary": {"macro_f1": 0.85, "micro_f1": 0.86},
                        "per_label": per_label,
                    }
                },
                "bias_probe": {"acceptable": True, "max_flagged_share": 0.0, "worst_term": "none"},
                "limitations": ["toy comment model"],
            },
        )

    if with_phishing:
        pipe, n_rows = _fit_phishing()
        _save(
            models_dir / "phishing",
            pipe,
            {
                "model": "toy_phishing",
                "threshold": 0.5,
                "score_type": "probability",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "guard_passed": True,
                "data_sizes": {"train": n_rows},
                "date_ranges": {
                    "train+val": "2024-07-01 to 2025-09-01",
                    "test": "2025-09-01 to 2025-12-01",
                },
                "metrics": {
                    "test_temporal": {"recall": 0.9, "false_positive_rate": 0.02},
                    "challenge": {"legit_flagged": 0, "legit_n": 52},
                },
                "limitations": ["toy URL model"],
            },
        )


@pytest.fixture
def fake_models_dir(tmp_path):
    write_fake_model(tmp_path)
    return tmp_path


@pytest.fixture
def fake_models_no_type(tmp_path):
    write_fake_model(tmp_path, with_type=False)
    return tmp_path


@pytest.fixture
def fake_models_no_email(tmp_path):
    write_fake_model(tmp_path, with_email=False)
    return tmp_path


@pytest.fixture
def corrupt_models_dir(tmp_path):
    write_fake_model(tmp_path, corrupt=True)
    return tmp_path


@pytest.fixture
def corrupt_type_models_dir(tmp_path):
    write_fake_model(tmp_path, corrupt_type=True)
    return tmp_path
