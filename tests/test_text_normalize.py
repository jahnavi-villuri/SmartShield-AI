"""Tests for the artifact-removing normalization used by the SMS pipelines."""

import pytest

from src.features.text_normalize import normalize_sms


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Call 09061701461 NOW", "call 0 now"),
        ("Rs XXXX valid till 20/01/2018", "rs 0 valid till 0/0/0"),
        ("Rs XX off", "rs 0 off"),
    ],
)
def test_numbers_and_masked_numbers_become_zero(raw, expected):
    assert normalize_sms(raw) == expected


def test_html_codes_are_decoded_and_anonymisation_marker_becomes_zero():
    assert "&lt;" not in normalize_sms("leave at &lt;#&gt; ok")
    assert "<#>" not in normalize_sms("leave at <#> ok")


def test_triple_x_is_kept_because_it_is_a_real_signal():
    assert "xxx" in normalize_sms("love u xxx")


def test_html_entity_and_plain_marker_give_the_same_result():
    assert normalize_sms("see you at &lt;#&gt; pm") == normalize_sms("see you at <#> pm")


def test_normalization_is_idempotent():
    once = normalize_sms("WIN £500 at 0800 123456! XXXX &lt;#&gt;")
    assert normalize_sms(once) == once
