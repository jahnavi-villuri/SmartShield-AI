"""Tests for the template-written augmentation data (training only, never test)."""

import pandas as pd

from src.data.augment_sms import generate_augmentation
from src.data.text_utils import normalize_key

CHALLENGE = "tests/data/sms_challenge.csv"


def test_generation_is_reproducible_and_balanced():
    a = generate_augmentation(200, seed=1)
    b = generate_augmentation(200, seed=1)
    assert a.equals(b)
    assert a["label"].value_counts().to_dict() == {0: 200, 1: 200}
    assert not a["text"].map(normalize_key).duplicated().any()


def test_rows_are_marked_synthetic_with_group_column():
    df = generate_augmentation(50, seed=2)
    assert set(df["source"]) == {"synthetic_aug"}
    assert {"text", "label", "source", "original_label", "group"} <= set(df.columns)


def test_both_classes_use_the_same_currency_vocabulary():
    """The point of the augmentation: 'Rs' must not identify a class."""
    df = generate_augmentation(300, seed=3)
    has_money = df["text"].str.contains(r"Rs|INR|₹", regex=True)
    for label in (0, 1):
        assert has_money[df["label"] == label].mean() > 0.5


def test_scam_links_use_reserved_example_domains_only():
    df = generate_augmentation(300, seed=4)
    links = df["text"].str.extractall(r"https?://([^/\s]+)")[0]
    assert len(links) > 0
    assert links.str.endswith((".example", ".test")).all()


def test_no_challenge_message_appears_verbatim_in_augmentation():
    challenge = pd.read_csv(CHALLENGE, keep_default_na=False)
    aug = generate_augmentation(1500, seed=42)
    overlap = set(challenge["text"].map(normalize_key)) & set(aug["text"].map(normalize_key))
    assert overlap == set()


def test_challenge_file_is_well_formed():
    challenge = pd.read_csv(CHALLENGE, keep_default_na=False)
    assert set(challenge["label"]) == {0, 1}
    assert challenge["text"].str.strip().ne("").all()
