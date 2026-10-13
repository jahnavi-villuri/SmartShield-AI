"""Saved models must load in a FRESH process, as the app loads them.

Bug this guards against: pickle stores the module path of every custom class. A class defined in a
script that was run as `python -m ...` is saved as `__main__.ThatClass`, which no other program can
load. (This happened once with the phishing model's text helper.)
"""

import subprocess
import sys

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import settings
from src.features.url_features import UrlFeatureTransformer, UrlText

LOADER = (
    "import sys, joblib; m = joblib.load(sys.argv[1]); "
    "print(type(m).__module__ + '.' + type(m).__name__)"
)


def _load_in_fresh_process(path):
    return subprocess.run(
        [sys.executable, "-c", LOADER, str(path)],
        capture_output=True,
        text=True,
        cwd=settings.PROJECT_ROOT,
        timeout=120,
    )


def test_url_char_pipeline_loads_in_a_fresh_process(tmp_path):
    urls = [
        "https://www.a.com/x",
        "http://b.example/login.php",
        "https://c.org/news",
        "http://d.xyz/y",
    ]
    pipe = Pipeline(
        [
            (
                "features",
                TfidfVectorizer(preprocessor=UrlText(), analyzer="char_wb", ngram_range=(2, 3)),
            ),
            ("clf", LogisticRegression()),
        ]
    ).fit(urls, [0, 1, 0, 1])
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    result = _load_in_fresh_process(path)
    assert result.returncode == 0, result.stderr[-500:]


def test_url_feature_pipeline_loads_in_a_fresh_process(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    urls = [
        "https://www.a.com/x",
        "http://b.example/login.php",
        "https://c.org/news",
        "http://d.xyz/y",
    ]
    transformer = UrlFeatureTransformer()
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(
        transformer.transform(urls), [0, 1, 0, 1]
    )
    path = tmp_path / "m.joblib"
    joblib.dump(Pipeline([("features", UrlFeatureTransformer()), ("clf", forest)]), path)
    assert _load_in_fresh_process(path).returncode == 0


@pytest.mark.parametrize("module", ["sms_spam", "sms_type", "email_spam", "toxicity", "phishing"])
def test_every_real_saved_model_loads_in_a_fresh_process(module):
    path = settings.MODELS_DIR / module / "model.joblib"
    if not path.exists():
        pytest.skip(f"{module} model not trained")
    result = _load_in_fresh_process(path)
    assert result.returncode == 0, f"{module}: {result.stderr[-400:]}"
