"""Tests for the private-message loader (uses a temporary file, never the real private data)."""

import pandas as pd

from src.data.private_sms import assign_split, load_private_sms


def _write(tmp_path, rows):
    path = tmp_path / "messages.csv"
    pd.DataFrame(rows, columns=["id", "text", "expected_category", "note"]).to_csv(
        path, index=False, encoding="utf-8"
    )
    return path


def test_missing_file_returns_none(tmp_path):
    assert load_private_sms(tmp_path / "nope.csv") is None


def test_categories_map_to_stage1_labels_and_unclear_is_skipped(tmp_path):
    path = _write(
        tmp_path,
        [
            (1, "your otp is 123456", "legitimate", ""),
            (2, "50% off today", "promotional", ""),
            (3, "verify your account now", "scam", ""),
            (4, "who knows", "unclear", ""),
        ],
    )
    data = load_private_sms(path).set_index("text")
    assert len(data) == 3
    assert data.loc["your otp is 123456", "label"] == 0
    assert data.loc["50% off today", "label"] == 1
    assert data.loc["verify your account now", "label"] == 1


def test_split_is_deterministic_and_independent_of_other_rows():
    assert assign_split("hello there") == assign_split("  Hello There ")
    assert assign_split("a message") in {"train", "test"}


def test_adding_messages_never_moves_existing_ones(tmp_path):
    first = load_private_sms(_write(tmp_path, [(1, "alpha message", "legitimate", "")]))
    path2 = tmp_path / "second.csv"
    pd.DataFrame(
        [(1, "beta message", "promotional", ""), (2, "alpha message", "legitimate", "")],
        columns=["id", "text", "expected_category", "note"],
    ).to_csv(path2, index=False)
    second = load_private_sms(path2).set_index("text")
    assert second.loc["alpha message", "split"] == first.loc[0, "split"]


def test_duplicates_and_blank_text_are_removed(tmp_path):
    path = _write(
        tmp_path,
        [(1, "same", "legitimate", ""), (2, "same", "legitimate", ""), (3, "  ", "legitimate", "")],
    )
    assert len(load_private_sms(path)) == 1
