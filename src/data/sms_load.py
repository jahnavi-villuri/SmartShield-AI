"""Loaders for the raw SMS datasets.

Each loader returns the SAME columns, so later code never cares which file a row came from:

    text            the message (outer whitespace stripped, nothing else changed)
    label           project convention: 1 = spam/smishing (bad), 0 = ham (good)
    source          "uci" or "mendeley"
    original_label  the dataset's own label, kept for reporting only
"""

from pathlib import Path

import pandas as pd

from config import settings
from src.data.text_utils import strip_text

UCI_FILE = settings.DATA_RAW / "sms_uci" / "extracted" / "SMSSpamCollection"
MENDELEY_FILE = settings.DATA_RAW / "sms_mendeley" / "extracted" / "Dataset_5971.csv"
SCAMSHIELD_DIR = settings.DATA_RAW / "sms_scamshield"
SCAMSHIELD_PARTS = ("train", "validation", "test")


def _map_labels(original: pd.Series, mapping: dict[str, int], dataset: str) -> pd.Series:
    """Map dataset labels to 0/1 and fail loudly on anything unexpected."""
    unknown = sorted(set(original) - set(mapping))
    if unknown:
        raise ValueError(f"{dataset}: unexpected label values {unknown}")
    return original.map(mapping).astype("int64")


def load_uci_sms(path: Path = UCI_FILE) -> pd.DataFrame:
    """UCI SMS Spam Collection: tab-separated, no header, labels `ham` / `spam`."""
    raw = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["original_label", "text"],
        quoting=3,  # csv.QUOTE_NONE: quote characters inside messages are just text
        keep_default_na=False,  # the message "NA" must not become a missing value
        encoding="utf-8",
    )
    label = _map_labels(raw["original_label"], {"ham": 0, "spam": 1}, "UCI")
    return pd.DataFrame(
        {
            "text": raw["text"].map(strip_text),
            "label": label,
            "source": "uci",
            "original_label": raw["original_label"],
        }
    )


def _load_scamshield(path_dir: Path = SCAMSHIELD_DIR) -> pd.DataFrame:
    """All three ScamShield parquet files (train/validation/test) stacked together.

    We ignore ScamShield's own split: we make our own so every dataset is split the same way.
    """
    parts = [pd.read_parquet(path_dir / f"{name}.parquet") for name in SCAMSHIELD_PARTS]
    return pd.concat(parts, ignore_index=True)


def _scamshield_subset(source_dataset: str, source_name: str, languages=("English",)):
    raw = _load_scamshield()
    keep = (raw["source_dataset"] == source_dataset) & raw["language"].isin(languages)
    # Rows whose id starts with "synth" are machine-generated: never used.
    keep &= ~raw["id"].str.startswith("synth")
    raw = raw[keep]
    label = _map_labels(raw["is_scam"].astype(int).astype(str), {"0": 0, "1": 1}, source_name)
    return pd.DataFrame(
        {
            "text": raw["text"].map(strip_text).to_numpy(),
            "label": label.to_numpy(),
            "source": source_name,
            "original_label": raw["head2_scam_intent"].to_numpy(),
        }
    )


def load_super_sms() -> pd.DataFrame:
    """ScamShield's `Smishing_Dataset` part (English): a large multi-source collection.

    The ScamShield authors say it merges earlier SMS collections (about 2012-2024).
    The per-row origin is not given, so ham/spam may come from different sources.
    """
    return _scamshield_subset("Smishing_Dataset", "super")


def load_india_external() -> pd.DataFrame:
    """Held-out external test material: English Indian telecom messages + a tiny benchmark.

    Never used for training. Spam here is largely telecom promotional alerts.
    """
    telecom = _scamshield_subset("Indian_Telecom_SMS", "india_telecom")
    bench = _scamshield_subset("Benchmark_Ground_Truth", "benchmark")
    return pd.concat([telecom, bench], ignore_index=True)


def load_mendeley_sms(path: Path = MENDELEY_FILE) -> pd.DataFrame:
    """Mendeley SMS Phishing Dataset: columns LABEL, TEXT, URL, EMAIL, PHONE.

    LABEL spelling is inconsistent (`Smishing` / `smishing`, `Spam` / `spam`), so we
    lowercase it first. Its URL/EMAIL/PHONE columns are NOT used: they are unreliable
    (see the data-quality report) and the app could not reproduce them.
    """
    raw = pd.read_csv(path, keep_default_na=False, encoding="utf-8")
    original = raw["LABEL"].str.strip().str.lower()
    label = _map_labels(original, {"ham": 0, "spam": 1, "smishing": 1}, "Mendeley")
    return pd.DataFrame(
        {
            "text": raw["TEXT"].map(strip_text),
            "label": label,
            "source": "mendeley",
            "original_label": original,
        }
    )
