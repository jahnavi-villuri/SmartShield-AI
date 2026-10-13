"""Loader for the darkknight25/phishing_benign_email_dataset (200 short, template-like emails).

Source: https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset (MIT licence),
saved to data/external_test/email/darkknight25.csv. It is SYNTHETIC-looking: every body is one
sentence of ~60 characters. It is useful because it contains modern legitimate automated mail
(invoices, receipts, 2FA notices) that the Enron data lacks, but it must not be trusted as real
email. Half goes to training (by hash, so assignment never changes) and half stays untouched as a
test; the 2026 spam set and the hand-written challenge set remain the separate checks.
"""

import pandas as pd

from config import settings
from src.data.private_email import assign_split
from src.features.email_text import join_email

SYNTHETIC_FILE = settings.DATA_EXTERNAL_TEST / "email" / "darkknight25.csv"
_LABELS = {"benign": 0, "phishing": 1}


def load_synthetic_email(path=SYNTHETIC_FILE) -> pd.DataFrame | None:
    if not path.exists():
        return None
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    raw = raw[raw["label"].isin(_LABELS)].drop_duplicates(["subject", "body"])
    frame = pd.DataFrame(
        {
            "subject": raw["subject"].str.strip().to_numpy(),
            "body": raw["body"].str.strip().to_numpy(),
            "label": raw["label"].map(_LABELS).astype(int).to_numpy(),
        }
    )
    frame["text"] = [join_email(s, b) for s, b in zip(frame["subject"], frame["body"], strict=True)]
    frame["group"] = frame["text"].str.lower()
    frame["split"] = [
        assign_split(s, b) for s, b in zip(frame["subject"], frame["body"], strict=True)
    ]
    return frame.reset_index(drop=True)
