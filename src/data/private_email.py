"""Loader for the owner's private email samples (data/private/my_email/emails.csv).

These are real emails from the owner's own mailbox. The folder is ignored by Git.
Columns: subject, body, expected_category (legitimate / promotional / spam), optional note.

Mapping for the email model, which flags spam and phishing only:
    legitimate -> 0,  promotional -> 0 (marketing is not spam),  spam -> 1.
Rows with any other category (for example `unclear`) are skipped.

Split: a hash of subject + body decides "train" or "test", so adding emails later never moves an
email that was already assigned (see src/data/private_sms.py for the same idea).
Only the visible text is kept: HTML in the body is reduced to text, nothing is fetched.
"""

import hashlib

import pandas as pd

from config import settings
from src.data.email_parse import html_to_text
from src.features.email_text import join_email

PRIVATE_FILE = settings.DATA_PRIVATE / "my_email" / "emails.csv"
_LABELS = {"legitimate": 0, "promotional": 0, "spam": 1}


def assign_split(subject: str, body: str) -> str:
    digest = hashlib.sha1(f"{subject.strip()}\n{body.strip()}".lower().encode("utf-8")).hexdigest()
    return "test" if int(digest, 16) % 2 == 0 else "train"


def load_private_email(path=PRIVATE_FILE) -> pd.DataFrame | None:
    """Return the private emails (or None when the file does not exist)."""
    if not path.exists():
        return None
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    raw = raw[raw["expected_category"].isin(_LABELS)].copy()
    raw["subject"] = raw["subject"].str.strip()
    raw["body"] = [
        html_to_text(b) if "<" in b and ">" in b else b.strip() for b in raw["body"].astype(str)
    ]
    raw = raw[(raw["subject"] != "") | (raw["body"] != "")]
    raw = raw.drop_duplicates(["subject", "body"])
    frame = pd.DataFrame(
        {
            "subject": raw["subject"].to_numpy(),
            "body": raw["body"].to_numpy(),
            "label": raw["expected_category"].map(_LABELS).astype(int).to_numpy(),
            "category": raw["expected_category"].to_numpy(),
        }
    )
    frame["text"] = [join_email(s, b) for s, b in zip(frame["subject"], frame["body"], strict=True)]
    frame["group"] = frame["text"].astype(str).str.lower()
    frame["split"] = [
        assign_split(s, b) for s, b in zip(frame["subject"], frame["body"], strict=True)
    ]
    return frame.reset_index(drop=True)
