"""Email detector tests: validation, loading, prediction, explanation."""

import pytest

from src.data.email_parse import html_to_text, parse_email_bytes
from src.detection.email_detector import (
    MAX_BODY_INPUT_CHARS,
    MAX_SUBJECT_CHARS,
    InvalidInputError,
    ModelNotAvailableError,
    load_email_detector,
    validate_email,
)
from src.features.email_text import SignatureMasker, join_email, normalize_email

# ---------- input validation ----------


def test_empty_subject_and_body_is_rejected():
    for subject, body in [("", ""), ("   ", "\n"), ("\x00", "\x00")]:
        with pytest.raises(InvalidInputError):
            validate_email(subject, body)


def test_subject_only_or_body_only_is_accepted():
    assert validate_email("Hello", "") == ("Hello", "")
    assert validate_email("", "Some text") == ("", "Some text")


def test_non_text_is_rejected():
    with pytest.raises(InvalidInputError):
        validate_email(None, "x")
    with pytest.raises(InvalidInputError):
        validate_email("x", 5)


def test_overlong_subject_and_body_are_rejected_at_the_limit():
    assert validate_email("a" * MAX_SUBJECT_CHARS, "b")[0] == "a" * MAX_SUBJECT_CHARS
    with pytest.raises(InvalidInputError, match="subject"):
        validate_email("a" * (MAX_SUBJECT_CHARS + 1), "b")
    with pytest.raises(InvalidInputError, match="body"):
        validate_email("s", "b" * (MAX_BODY_INPUT_CHARS + 1))


def test_pasted_html_is_reduced_to_visible_text_without_scripts():
    _, body = validate_email(
        "s",
        "<html><head><style>p{color:red}</style></head><body><p>Hello</p>"
        "<script>alert(1)</script><a href='http://x.example'>click</a></body></html>",
    )
    assert "Hello" in body and "click" in body
    assert "alert" not in body and "color" not in body and "<" not in body


# ---------- text preparation ----------


def test_normalize_email_undoes_tokenizer_spacing_and_digits():
    assert normalize_email("Vince , thank you ( see 2001 )") == "vince, thank you (see 0)"


def test_join_email_is_one_function_for_training_and_app():
    assert join_email(" Hi ", " body ") == "Hi . body"


def test_signature_masker_removes_listed_words_only():
    assert SignatureMasker({"enron", "ect"})("ECT and Enron plan: win cash") == "and plan win cash"


# ---------- safe parsing ----------


def test_html_to_text_drops_script_and_style():
    assert html_to_text("<style>x{}</style><script>bad()</script><p>Visible</p>") == "Visible"


def test_parse_prefers_html_text_when_plain_part_is_filler():
    raw = (
        b"Subject: Prize\r\nMIME-Version: 1.0\r\n"
        b"Content-Type: multipart/alternative; boundary=B\r\n\r\n"
        b"--B\r\nContent-Type: text/plain\r\n\r\nWe should shift the patio grill.\r\n"
        b"--B\r\nContent-Type: text/html\r\n\r\n<p>Claim your prize now</p>\r\n--B--\r\n"
    )
    parsed = parse_email_bytes(raw)
    assert parsed.subject == "Prize"
    assert "patio" in parsed.plain_text
    assert parsed.visible_text == "Claim your prize now"


def test_parse_counts_attachments_but_never_reads_them():
    raw = (
        b"Subject: s\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=B\r\n\r\n"
        b"--B\r\nContent-Type: text/plain\r\n\r\nhello\r\n"
        b"--B\r\nContent-Type: application/pdf\r\nContent-Disposition: attachment; "
        b'filename="a.pdf"\r\n\r\nSECRET-BYTES\r\n--B--\r\n'
    )
    parsed = parse_email_bytes(raw)
    assert parsed.n_attachments == 1
    assert "SECRET" not in parsed.visible_text


def test_parse_survives_garbage_bytes():
    parsed = parse_email_bytes(b"\xff\xfe\x00 not an email at all")
    assert isinstance(parsed.visible_text, str)


# ---------- loading ----------


def test_missing_email_model_raises_friendly_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="not been trained"):
        load_email_detector(tmp_path)


def test_tampered_email_model_is_refused(tmp_path):
    from tests.conftest import write_fake_model

    write_fake_model(tmp_path, with_type=False)
    model = tmp_path / "email_spam" / "model.joblib"
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ModelNotAvailableError, match="checksum"):
        load_email_detector(tmp_path)


# ---------- prediction ----------


def test_spam_and_normal_email_are_told_apart(fake_models_dir):
    detector = load_email_detector(fake_models_dir)
    spam = detector.predict("Win a prize", "Click the link now to claim your free cash prize")
    ham = detector.predict("Report draft", "I attached the draft report, send comments by Friday")
    assert spam.is_spam and spam.label == "Spam"
    assert not ham.is_spam and ham.label == "Not spam"
    assert ham.score < spam.score


def test_result_reports_threshold_and_truncation_flag(fake_models_dir):
    detector = load_email_detector(fake_models_dir)
    short = detector.predict("hi", "hello there")
    long = detector.predict("hi", "word " * 3000)  # 15,000 characters: allowed but truncated
    assert short.threshold == 0.5 and not short.truncated
    assert long.truncated


def test_unknown_words_do_not_crash(fake_models_dir):
    assert 0 <= load_email_detector(fake_models_dir).predict("zzz", "qqq www").score <= 1


def test_spam_explanation_lists_terms_that_raise_the_score(fake_models_dir):
    result = load_email_detector(fake_models_dir).predict(
        "Cheap offer", "order cheap pills now with no prescription, click here"
    )
    assert result.is_spam
    assert result.influential_terms
    assert set(result.influential_terms) & {"cheap", "click", "pills", "order", "prescription"}


def test_ham_gets_no_explanation(fake_models_dir):
    result = load_email_detector(fake_models_dir).predict(
        "Lunch", "are you free for lunch tomorrow"
    )
    assert result.influential_terms == []


# ---------- hidden text (spam evasion) ----------


@pytest.mark.parametrize(
    "markup",
    [
        '<p>Shown</p><div style="display:none">HIDDEN</div>',
        '<p>Shown</p><span style="opacity:0;color:#fff"><b>HIDDEN <i>deeper</i></b></span>',
        '<p>Shown</p><div style="font-size:1px;line-height:1px;mso-hide:all">HIDDEN</div>',
        '<p>Shown</p><div style="width:1px; line-height:0; overflow:hidden">HIDDEN</div>',
        "<p>Shown</p><p hidden>HIDDEN</p>",
    ],
)
def test_css_hidden_text_is_dropped(markup):
    assert html_to_text(markup) == "Shown"


@pytest.mark.parametrize(
    "markup",
    [
        '<a style="display:inline-block;color:#ffffff;font-size:16px">Claim now</a>',
        '<p style="font-size:12px;color:#333">small but visible</p>',
    ],
)
def test_ordinary_styled_text_is_kept(markup):
    assert html_to_text(markup) != ""


def test_hidden_filler_is_counted_not_read():
    raw = (
        b"Subject: Points expire\r\nMIME-Version: 1.0\r\nContent-Type: text/html\r\n\r\n"
        b'<div style="display:none;font-size:1px">I tried that ramen place</div>'
        b"<p>Redeem your points now</p>"
    )
    parsed = parse_email_bytes(raw)
    assert parsed.visible_text == "Redeem your points now"
    assert parsed.hidden_chars > 0
