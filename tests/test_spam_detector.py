"""Detector tests: input checks, loading, the two stages, explanation."""

import pytest

from src.detection.spam_detector import (
    LABEL_OK,
    LABEL_PROMO,
    LABEL_SCAM,
    LABEL_SPAM,
    MAX_SMS_CHARS,
    InvalidInputError,
    ModelNotAvailableError,
    load_sms_detector,
    validate_message,
)

# ---------- input validation ----------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "\x00"])
def test_empty_or_blank_input_is_rejected(bad):
    with pytest.raises(InvalidInputError):
        validate_message(bad)


def test_non_text_input_is_rejected():
    with pytest.raises(InvalidInputError):
        validate_message(None)
    with pytest.raises(InvalidInputError):
        validate_message(12345)


def test_too_long_input_is_rejected_but_limit_is_allowed():
    assert validate_message("a" * MAX_SMS_CHARS) == "a" * MAX_SMS_CHARS
    with pytest.raises(InvalidInputError):
        validate_message("a" * (MAX_SMS_CHARS + 1))


def test_input_is_stripped():
    assert validate_message("  hello \n") == "hello"


# ---------- loading ----------


def test_missing_model_raises_friendly_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="not been trained"):
        load_sms_detector(tmp_path)


def test_tampered_main_model_is_refused(corrupt_models_dir):
    with pytest.raises(ModelNotAvailableError, match="checksum"):
        load_sms_detector(corrupt_models_dir)


def test_tampered_type_model_is_refused_not_ignored(corrupt_type_models_dir):
    with pytest.raises(ModelNotAvailableError, match="checksum"):
        load_sms_detector(corrupt_type_models_dir)


def test_broken_metadata_is_refused(fake_models_dir):
    (fake_models_dir / "sms_spam" / "metadata.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelNotAvailableError, match="metadata"):
        load_sms_detector(fake_models_dir)


def test_missing_type_model_is_allowed_but_warned(fake_models_no_type):
    detector = load_sms_detector(fake_models_no_type)
    assert detector.type_pipeline is None
    assert any("not available" in w for w in detector.warnings)


# ---------- stage 1 ----------


def test_ordinary_message_is_not_spam(fake_models_dir):
    result = load_sms_detector(fake_models_dir).predict("see you at lunch")
    assert not result.is_spam and result.label == LABEL_OK
    assert result.scam_score is None and result.influential_words == []


def test_unwanted_message_without_type_model_is_plain_spam(fake_models_no_type):
    result = load_sms_detector(fake_models_no_type).predict("win a free prize now")
    assert result.is_spam and result.label == LABEL_SPAM and result.scam_score is None


def test_result_reports_threshold_and_score_type(fake_models_dir):
    result = load_sms_detector(fake_models_dir).predict("free cash prize")
    assert result.threshold == 0.5 and result.score_type == "probability"


def test_prediction_is_deterministic(fake_models_dir):
    detector = load_sms_detector(fake_models_dir)
    assert detector.predict("free prize").score == detector.predict("free prize").score


def test_unknown_words_do_not_crash(fake_models_dir):
    assert 0.0 <= load_sms_detector(fake_models_dir).predict("zzzz qqqq").score <= 1.0


# ---------- stage 2 ----------


def test_promotional_and_scam_are_told_apart(fake_models_dir):
    detector = load_sms_detector(fake_models_dir)
    promo = detector.predict("big sale discount offer buy now")
    scam = detector.predict("your account is blocked verify click link")
    assert promo.is_spam and promo.label == LABEL_PROMO
    assert scam.is_spam and scam.label == LABEL_SCAM
    assert promo.scam_score < scam.scam_score


def test_scam_score_only_present_for_unwanted_messages(fake_models_dir):
    detector = load_sms_detector(fake_models_dir)
    assert detector.predict("thanks for the help today").scam_score is None
    assert detector.predict("big sale discount offer").scam_score is not None


# ---------- explanation ----------


def test_explanation_names_words_that_raise_the_score(fake_models_dir):
    result = load_sms_detector(fake_models_dir).predict("hello friend win free prize now")
    assert result.is_spam
    assert set(result.influential_words) & {"win", "free", "prize"}
    assert "hello" not in result.influential_words


def test_no_explanation_when_disabled(fake_models_dir):
    result = load_sms_detector(fake_models_dir).predict("win free prize now", explain=False)
    assert result.influential_words == []


def test_explanation_is_skipped_for_very_long_messages(fake_models_dir):
    long_spam = "win free prize " * 30  # 90 words, above the explanation limit
    assert load_sms_detector(fake_models_dir).predict(long_spam).influential_words == []


def test_version_mismatch_produces_warning(fake_models_dir):
    detector = load_sms_detector(fake_models_dir)
    assert detector.warnings == []
    detector.metadata["scikit_learn"] = "0.0.1"
    assert "scikit-learn" in detector.warnings[0]
