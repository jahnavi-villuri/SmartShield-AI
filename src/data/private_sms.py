"""Loader for the owner's private SMS samples (data/private/my_sms/messages.csv).

These are real messages from the owner's own phone. The folder is ignored by Git.
Columns used: text, expected_category (legitimate / promotional / scam / unclear).

Mapping for the main (stage 1) model:  legitimate -> 0 (ordinary),  promotional or scam -> 1.
Rows marked `unclear` are skipped.

Split: each message goes to "train" or "test" by a hash of its text. Using a hash (not row order)
means adding more messages later never moves the ones already assigned, so a message that was
held out for testing is never silently promoted into training.
"""

import hashlib

import pandas as pd

from config import settings

PRIVATE_FILE = settings.DATA_PRIVATE / "my_sms" / "messages.csv"
_LABELS = {"legitimate": 0, "promotional": 1, "scam": 1}


def assign_split(text: str) -> str:
    digest = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()
    return "test" if int(digest, 16) % 2 == 0 else "train"


def load_private_sms(path=PRIVATE_FILE) -> pd.DataFrame | None:
    """Return the private messages (or None when the file does not exist)."""
    if not path.exists():
        return None
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    raw = raw[raw["expected_category"].isin(_LABELS)].copy()
    raw["text"] = raw["text"].str.strip()
    raw = raw[raw["text"] != ""].drop_duplicates("text")
    frame = pd.DataFrame(
        {
            "text": raw["text"].to_numpy(),
            "label": raw["expected_category"].map(_LABELS).astype(int).to_numpy(),
            "category": raw["expected_category"].to_numpy(),
            "source": "private",
            "original_label": raw["expected_category"].to_numpy(),
        }
    )
    frame["group"] = frame["text"].str.lower()
    frame["split"] = frame["text"].map(assign_split)
    return frame.reset_index(drop=True)
