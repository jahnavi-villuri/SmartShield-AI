"""Phase 2 tests: text keys, label mapping, deduplication, splitting.

All data here is tiny and synthetic, so the tests run without the real datasets.
"""

import pandas as pd
import pytest

from src.data.dedup import deduplicate
from src.data.inspect_data import inspect_text_frame
from src.data.sms_load import _map_labels
from src.data.splitting import check_no_overlap, grouped_stratified_split
from src.data.text_utils import mask_key, normalize_key, strip_text

# ---------- text keys ----------


def test_normalize_key_ignores_case_and_spacing():
    assert normalize_key("  Call me   NOW ") == normalize_key("call me now")


def test_mask_key_groups_templates_with_different_numbers():
    a = "WINNER! Claim 900 now, call 09061701461"
    b = "WINNER! Claim 500 now, call 09061701939"
    assert mask_key(a) == mask_key(b)


def test_mask_key_replaces_urls():
    assert mask_key("visit http://a.com/x now") == mask_key("visit https://b.org/y now")


def test_strip_text_only_trims_outer_whitespace():
    assert strip_text("\tHello  world \n") == "Hello  world"


# ---------- label mapping ----------


def test_label_mapping_is_one_equals_bad():
    labels = pd.Series(["ham", "spam", "ham"])
    assert _map_labels(labels, {"ham": 0, "spam": 1}, "test").tolist() == [0, 1, 0]


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        _map_labels(pd.Series(["ham", "junk"]), {"ham": 0, "spam": 1}, "test")


# ---------- deduplication ----------


def _frame(rows):
    df = pd.DataFrame(rows, columns=["text", "label", "source"])
    df["key"] = df["text"].map(normalize_key)
    return df


def test_dedup_removes_exact_and_normalized_duplicates():
    df = _frame([("Hello there", 0, "a"), ("hello  THERE", 0, "a"), ("Other", 1, "a")])
    out, counts = deduplicate(df)
    assert len(out) == 2
    assert counts["duplicate_rows_removed"] == 1


def test_dedup_drops_all_copies_of_conflicting_labels():
    df = _frame([("Same text", 0, "a"), ("same text", 1, "b"), ("Unique", 1, "a")])
    out, counts = deduplicate(df)
    assert out["text"].tolist() == ["Unique"]
    assert counts["conflicting_rows_removed"] == 2


def test_dedup_records_every_source_of_a_message():
    df = _frame([("Shared msg", 0, "uci"), ("shared msg", 0, "mendeley")])
    out, _ = deduplicate(df)
    assert out.loc[0, "source"] == "mendeley+uci"


# ---------- splitting ----------


def _synthetic(n_groups=200):
    rows = []
    for g in range(n_groups):
        label = 1 if g % 5 == 0 else 0  # 20% bad
        for member in range(1 + g % 3):  # groups of size 1..3
            rows.append((f"text {g} {member}", label, f"group{g}"))
    return pd.DataFrame(rows, columns=["text", "label", "group"])


def test_split_has_no_group_leakage_and_all_rows_assigned():
    df = _synthetic()
    df["split"] = grouped_stratified_split(df, "label", "group", seed=42)
    assert set(df["split"]) == {"train", "val", "test"}
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    assert all(v == 0 for v in check_no_overlap(parts, "group").values())
    assert sum(len(p) for p in parts.values()) == len(df)


def test_split_is_reproducible_and_stratified():
    df = _synthetic()
    first = grouped_stratified_split(df, "label", "group", seed=42)
    second = grouped_stratified_split(df, "label", "group", seed=42)
    assert first.equals(second)
    df["split"] = first
    shares = df.groupby("split")["label"].mean()
    assert shares.max() - shares.min() < 0.1


def test_check_no_overlap_detects_leakage():
    a = pd.DataFrame({"key": ["x", "y"]})
    b = pd.DataFrame({"key": ["y", "z"]})
    assert check_no_overlap({"a": a, "b": b}, "key") == {"a & b": 1}


# ---------- inspection ----------


def test_inspect_text_frame_counts_problems():
    df = pd.DataFrame({"text": ["ok", "ok", "  ", "bad �"], "label": [0, 0, 1, 1]})
    facts = inspect_text_frame(df, "toy")
    assert facts["rows"] == 4
    assert facts["empty_text"] == 1
    assert facts["exact_duplicate_rows"] == 1
    assert facts["replacement_char_rows"] == 1
    assert facts["class_counts"] == {0: 2, 1: 2}


# ---------- processed files (only when Phase 2 has been run locally) ----------


def test_saved_splits_have_no_overlap_and_binary_labels():
    from config import settings

    folder = settings.DATA_PROCESSED / "sms"
    if not (folder / "train.csv").exists():
        pytest.skip("run `python -m scripts.prepare_sms` first")
    parts = {n: pd.read_csv(folder / f"{n}.csv") for n in ("train", "val", "test")}
    for df in parts.values():
        assert set(df["label"].unique()) <= {0, 1}
        assert df["text"].notna().all()
    for df in parts.values():
        df["key"] = df["text"].map(normalize_key)
    assert all(v == 0 for v in check_no_overlap(parts, "key").values())
    assert all(v == 0 for v in check_no_overlap(parts, "group").values())
