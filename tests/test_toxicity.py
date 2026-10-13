"""Toxicity module tests: model class, thresholds, augmentation, detector, Comments tab."""

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import settings
from src.data.augment_toxicity import (
    PROBE_ONLY_IDENTITIES,
    PROBE_TEMPLATES,
    TRAIN_IDENTITIES,
    TRAIN_TEMPLATES,
    generate_identity_augmentation,
)
from src.detection.toxicity_detector import (
    MAX_COMMENT_CHARS,
    InvalidInputError,
    ModelNotAvailableError,
    load_toxicity_detector,
    validate_comment,
)
from src.features.comment_text import ClippedLower
from src.models.toxicity_model import LABELS
from src.training.train_toxicity import best_f1_threshold, select_toxicity_model

# ---------- building blocks ----------


def test_six_labels_in_the_project_order():
    assert LABELS == ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def test_clipped_lower_lowercases_and_clips():
    assert ClippedLower(5)("ABCDEFGH") == "abcde"


def test_best_f1_threshold_separates_a_clean_ranking():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = best_f1_threshold(y, scores)
    assert 0.3 < threshold <= 0.8


@pytest.mark.filterwarnings("ignore:No positive class")
def test_best_f1_threshold_handles_a_label_with_no_positives():
    assert best_f1_threshold(np.zeros(5), np.linspace(0, 1, 5)) >= 0


def test_selection_prefers_the_smaller_model_within_tolerance():
    assert select_toxicity_model({"logreg_word": 0.700, "logreg_wordchar": 0.702}) == "logreg_word"
    assert (
        select_toxicity_model({"logreg_word": 0.700, "logreg_wordchar": 0.720}) == "logreg_wordchar"
    )


# ---------- augmentation (bias) ----------


def test_augmentation_is_clean_synthetic_and_reproducible():
    a = generate_identity_augmentation(10, seed=1)
    b = generate_identity_augmentation(10, seed=1)
    assert a.equals(b)
    assert (a[LABELS].to_numpy() == 0).all()
    assert len(a) == 10 * len(TRAIN_IDENTITIES)


def test_probe_uses_different_templates_and_identities_than_training():
    assert not set(PROBE_TEMPLATES) & set(TRAIN_TEMPLATES)
    assert not {w.lower() for w in PROBE_ONLY_IDENTITIES} & {w.lower() for w in TRAIN_IDENTITIES}


# ---------- detector ----------


@pytest.mark.parametrize("bad", ["", "   ", "\n", "\x00"])
def test_empty_comment_is_rejected(bad):
    with pytest.raises(InvalidInputError):
        validate_comment(bad)


def test_non_text_and_overlong_comments_are_rejected():
    with pytest.raises(InvalidInputError):
        validate_comment(None)
    assert validate_comment("a" * MAX_COMMENT_CHARS)
    with pytest.raises(InvalidInputError, match="too long"):
        validate_comment("a" * (MAX_COMMENT_CHARS + 1))


def test_missing_and_tampered_models_are_refused(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="not been trained"):
        load_toxicity_detector(tmp_path)
    from tests.conftest import write_fake_model

    write_fake_model(tmp_path, with_type=False, with_email=False)
    model = tmp_path / "toxicity" / "model.joblib"
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ModelNotAvailableError, match="checksum"):
        load_toxicity_detector(tmp_path)


def test_result_has_six_scores_between_zero_and_one(fake_models_dir):
    result = load_toxicity_detector(fake_models_dir).predict("thanks for the helpful edit")
    assert list(result.scores) == LABELS
    assert all(0.0 <= s <= 1.0 for s in result.scores.values())
    assert not result.any_flagged and result.top_terms == {}


def test_labels_are_flagged_independently_and_can_overlap(fake_models_dir):
    detector = load_toxicity_detector(fake_models_dir)
    insult = detector.predict("you are an idiot")
    threat = detector.predict("i will kill you")
    detector.thresholds["insult"] = 0.4  # per-label thresholds: lower only this one
    both = detector.predict("you stupid idiot i will kill you")
    assert {"toxic", "insult"} <= set(insult.flagged) and "threat" not in insult.flagged
    assert {"toxic", "threat"} <= set(threat.flagged)
    assert {"insult", "threat"} <= set(both.flagged)  # one comment, several labels


def test_each_label_uses_its_own_threshold(fake_models_dir):
    detector = load_toxicity_detector(fake_models_dir)
    detector.thresholds["insult"] = 0.999999
    result = detector.predict("you are an idiot")
    assert "insult" not in result.flagged and "toxic" in result.flagged


def test_flagged_labels_come_with_explanatory_words(fake_models_dir):
    result = load_toxicity_detector(fake_models_dir).predict("hello you idiot")
    assert "idiot" in result.top_terms.get("insult", [])


def test_unknown_words_do_not_crash(fake_models_dir):
    assert load_toxicity_detector(fake_models_dir).predict("zzz qqq").scores


# ---------- the Comments tab ----------


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run(models_dir, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", models_dir)
    return AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=30).run()


def _analyze(app, text):
    app.text_area(key="comment_text").set_value(text).run()
    app.button(key="comment_analyze").click().run()
    return app


def test_page_has_all_tabs(fake_models_dir, monkeypatch):
    app = _run(fake_models_dir, monkeypatch)
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Home",
        "SMS",
        "Email",
        "Comments",
        "URL",
        "Dashboard",
        "About",
    ]


def test_comments_tab_flags_labels_and_shows_all_six_scores(fake_models_dir, monkeypatch):
    app = _analyze(_run(fake_models_dir, monkeypatch), "you are an idiot")
    assert not app.exception
    assert any("Flagged" in w.value and "insult" in w.value for w in app.warning)
    assert len(app.get("progress")) == 6  # one bar per label


def test_comments_tab_accepts_a_normal_comment(fake_models_dir, monkeypatch):
    app = _analyze(_run(fake_models_dir, monkeypatch), "thanks for the helpful edit")
    assert any("No label reached its threshold" in s.value for s in app.success)


def test_comments_tab_rejects_empty_and_overlong_input(fake_models_dir, monkeypatch):
    app = _run(fake_models_dir, monkeypatch)
    app.button(key="comment_analyze").click().run()
    assert any("enter a comment" in e.value for e in app.error)
    app = _analyze(app, "a" * (MAX_COMMENT_CHARS + 1))
    assert any("too long" in e.value for e in app.error)


def test_missing_comment_model_does_not_break_other_tabs(tmp_path, monkeypatch):
    from tests.conftest import write_fake_model

    write_fake_model(tmp_path, with_toxicity=False)
    app = _run(tmp_path, monkeypatch)
    assert not app.exception
    assert any("comment model has not been trained" in e.value for e in app.error)
    assert "comment_analyze" not in [b.key for b in app.button]
    assert "sms_analyze" in [b.key for b in app.button]


def test_bias_warning_appears_only_when_the_probe_failed(fake_models_dir, monkeypatch):
    import json

    app = _run(fake_models_dir, monkeypatch)
    assert not any("Known weakness" in w.value for w in app.warning)
    meta_path = fake_models_dir / "toxicity" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["bias_probe"] = {"acceptable": False, "max_flagged_share": 0.5, "worst_term": "foo"}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    st.cache_resource.clear()
    app = _run(fake_models_dir, monkeypatch)
    text = next(w.value for w in app.warning if "Known weakness" in w.value)
    assert "50%" in text and "foo" in text
