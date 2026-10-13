"""Small text helpers used for deduplication and grouping.

IMPORTANT: these functions build *keys* for comparing messages. The text that the
model trains on is never rewritten by them (only stripped of outer whitespace).
"""

import re

_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")


def strip_text(text: str) -> str:
    """Remove leading/trailing whitespace (some raw rows start with a tab)."""
    return text.strip()


def normalize_key(text: str) -> str:
    """Dedup key: lowercase and collapse runs of whitespace.

    "Call me  NOW " and "call me now" get the same key, so they count as duplicates.
    """
    return _WHITESPACE.sub(" ", text.lower()).strip()


def mask_key(text: str) -> str:
    """Near-duplicate key: normalize, then replace URLs and numbers by placeholders.

    Spam templates often differ only in the phone number or link:
    "Call 09061701461 now" and "Call 09061701939 now" share one mask key.
    """
    key = normalize_key(text)
    key = _URL.sub(" <url> ", key)
    key = _DIGITS.sub("0", key)
    return _WHITESPACE.sub(" ", key).strip()
