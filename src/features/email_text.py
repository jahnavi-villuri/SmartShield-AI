"""Text preparation shared by email training and the app (train/serve parity).

Two things matter here:
1. `join_email` builds the single string the model sees from a subject and a body. Training and
   the app both call it, so they can never disagree.
2. `normalize_email` makes raw modern text look like the Enron training text. That dataset is
   already lowercased and tokenized ("vince , thank you") and its digits are real; a pasted email
   is not. Without this step the same email would look different at training and prediction time.
"""

import html
import re

MAX_BODY_CHARS = 6_000  # the model reads only the first 6,000 characters of the body
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")
_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


def join_email(subject: str, body: str) -> str:
    """The text the email model reads: subject, a separator, then the body."""
    return f"{(subject or '').strip()} . {(body or '').strip()[:MAX_BODY_CHARS]}"


def normalize_email(text: str) -> str:
    """Lowercase, decode HTML codes, undo tokenizer spacing, map digit runs to `0`."""
    text = html.unescape(text).lower()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _DIGITS.sub("0", text)
    return _WHITESPACE.sub(" ", text).strip()


class SignatureMasker:
    """Normalize, then drop words that identify the DATA SOURCE rather than spam.

    The words are learned from the training set only (words common in ham, almost absent in
    spam, such as "enron", "ect", "hou", "vince"). A real user's email contains none of them
    in the same way, so a model must not rely on them. The list travels inside the saved
    pipeline, so prediction masks exactly the same words. A class (not a lambda) so joblib can
    save it.
    """

    def __init__(self, words=()):
        self.words = frozenset(words)

    def __call__(self, text: str) -> str:
        tokens = re.findall(r"\w\w+", normalize_email(text))
        return " ".join(t for t in tokens if t not in self.words)
