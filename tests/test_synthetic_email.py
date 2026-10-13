"""Loader for the synthetic modern-email dataset (darkknight25)."""

import pandas as pd

from src.data.synthetic_email import load_synthetic_email


def _write(tmp_path, rows):
    path = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=["subject", "body", "label"]).to_csv(path, index=False)
    return path


def test_missing_file_gives_none(tmp_path):
    assert load_synthetic_email(tmp_path / "nope.csv") is None


def test_labels_mapped_duplicates_dropped_and_split_is_stable(tmp_path):
    rows = [
        ("Your receipt", "Thanks for your order.", "benign"),
        ("Your receipt", "Thanks for your order.", "benign"),  # duplicate
        ("Verify now", "Click https://x.example", "phishing"),
        ("Odd", "unknown label", "other"),
    ]
    frame = load_synthetic_email(_write(tmp_path, rows))
    assert sorted(frame["label"]) == [0, 1]
    again = load_synthetic_email(_write(tmp_path, rows + [("New", "A later addition.", "benign")]))
    old = frame.set_index("text")["split"]
    assert all(again.set_index("text").loc[t, "split"] == s for t, s in old.items())
