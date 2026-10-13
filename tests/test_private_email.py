"""Tests for the private-email loader (temporary files only; never the real private data)."""

import pandas as pd

from src.data.private_email import assign_split, load_private_email


def _write(tmp_path, rows):
    path = tmp_path / "emails.csv"
    pd.DataFrame(rows, columns=["subject", "body", "expected_category", "note"]).to_csv(
        path, index=False, encoding="utf-8"
    )
    return path


def test_missing_file_returns_none(tmp_path):
    assert load_private_email(tmp_path / "nope.csv") is None


def test_header_only_file_is_empty_not_an_error(tmp_path):
    path = tmp_path / "emails.csv"
    path.write_text("subject,body,expected_category,note\n", encoding="utf-8")
    assert len(load_private_email(path)) == 0


def test_categories_map_to_labels_promotional_is_not_spam(tmp_path):
    path = _write(
        tmp_path,
        [
            ("Your OTP", "code 123456", "legitimate", ""),
            ("Big sale", "50% off today", "promotional", ""),
            ("You won", "send your bank details", "spam", ""),
            ("Hmm", "who knows", "unclear", ""),
        ],
    )
    data = load_private_email(path).set_index("subject")
    assert len(data) == 3
    assert data.loc["Your OTP", "label"] == 0
    assert data.loc["Big sale", "label"] == 0
    assert data.loc["You won", "label"] == 1


def test_html_in_body_is_reduced_to_visible_text(tmp_path):
    path = _write(
        tmp_path,
        [("s", "<p>Hello</p><div style='display:none'>hidden</div>", "legitimate", "")],
    )
    assert load_private_email(path).loc[0, "body"] == "Hello"


def test_split_is_deterministic_and_stable_when_data_grows(tmp_path):
    assert assign_split("Hi", "there") == assign_split(" hi ", " THERE ")
    first = load_private_email(_write(tmp_path, [("alpha", "one", "legitimate", "")]))
    path2 = tmp_path / "second.csv"
    pd.DataFrame(
        [("beta", "two", "spam", ""), ("alpha", "one", "legitimate", "")],
        columns=["subject", "body", "expected_category", "note"],
    ).to_csv(path2, index=False)
    second = load_private_email(path2).set_index("subject")
    assert second.loc["alpha", "split"] == first.loc[0, "split"]


def test_duplicates_are_removed(tmp_path):
    path = _write(tmp_path, [("a", "b", "legitimate", ""), ("a", "b", "legitimate", "")])
    assert len(load_private_email(path)) == 1
