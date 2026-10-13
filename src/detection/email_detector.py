"""Inference for the email spam model. No Streamlit, no training code in here.

Input is a subject and a body, typed or pasted as text. Nothing is fetched, rendered or opened:
links stay plain text. HTML pasted into the body is reduced to its visible text first, the same
way the 2026 test emails were parsed.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn

from config import settings
from src.data.email_parse import html_to_text
from src.detection.common import (
    DetectorError,
    InvalidInputError,
    ModelNotAvailableError,
    load_verified,
)
from src.evaluation.metrics import get_scores
from src.features.email_text import join_email

MAX_SUBJECT_CHARS = 300
MAX_BODY_INPUT_CHARS = 20_000  # the model itself reads only the first 6,000 characters
EMAIL_MODULE_DIR = "email_spam"
_LOOKS_LIKE_HTML = re.compile(r"<\s*(html|body|div|p|table|a|br|span)\b", re.IGNORECASE)

__all__ = [
    "DetectorError",
    "EmailDetector",
    "EmailResult",
    "InvalidInputError",
    "ModelNotAvailableError",
    "load_email_detector",
    "validate_email",
]


@dataclass(frozen=True)
class EmailResult:
    is_spam: bool
    label: str  # "Spam" or "Not spam"
    score: float
    threshold: float
    score_type: str
    influential_terms: list[str] = field(default_factory=list)
    truncated: bool = False  # True when the body was longer than the model reads


class EmailDetector:
    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])
        self.score_type = metadata.get("score_type", "probability")

    @property
    def warnings(self) -> list[str]:
        recorded = self.metadata.get("scikit_learn")
        if recorded and recorded != sklearn.__version__:
            return [
                f"The email model was trained with scikit-learn {recorded}; this environment has "
                f"{sklearn.__version__}. Results may differ."
            ]
        return []

    def predict(self, subject: str, body: str, explain: bool = True) -> EmailResult:
        subject, body = validate_email(subject, body)
        text = join_email(subject, body)
        score = float(get_scores(self.pipeline, [text])[0])
        is_spam = score >= self.threshold
        terms = self._influential_terms(text) if (explain and is_spam) else []
        from src.features.email_text import MAX_BODY_CHARS

        return EmailResult(
            is_spam=is_spam,
            label="Spam" if is_spam else "Not spam",
            score=score,
            threshold=self.threshold,
            score_type=self.score_type,
            influential_terms=terms,
            truncated=len(body) > MAX_BODY_CHARS,
        )

    def _influential_terms(self, text: str, top_k: int = 5) -> list[str]:
        """Terms in this email that push the score toward spam (linear models only).

        contribution = the term's weight x its value in this email. Reads the weights directly,
        so it is exact for the model, unlike a leave-one-word-out approximation.
        """
        try:
            features = self.pipeline.named_steps["features"]
            clf = self.pipeline.named_steps["clf"]
            row = features.transform([text])
            if hasattr(clf, "coef_"):
                weights = np.asarray(clf.coef_).ravel()
            elif hasattr(clf, "feature_log_prob_"):
                weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
            else:
                return []
            contribution = row.multiply(weights).tocsr()
            indices = contribution.indices
            values = contribution.data
            order = np.argsort(values)[::-1][:top_k]
            names = features.get_feature_names_out()
            return [str(names[indices[i]]) for i in order if values[i] > 0]
        except Exception:  # noqa: BLE001 - an explanation must never break a prediction
            return []


def validate_email(subject, body) -> tuple[str, str]:
    """Return (subject, body) cleaned, or raise InvalidInputError."""
    if not isinstance(subject, str) or not isinstance(body, str):
        raise InvalidInputError("The subject and body must be text.")
    subject = subject.replace("\x00", "").strip()
    body = body.replace("\x00", "").strip()
    if not body and not subject:
        raise InvalidInputError("Please enter an email subject or body to analyse.")
    if len(subject) > MAX_SUBJECT_CHARS:
        raise InvalidInputError(
            f"The subject is too long ({len(subject)} characters). The limit is "
            f"{MAX_SUBJECT_CHARS} characters."
        )
    if len(body) > MAX_BODY_INPUT_CHARS:
        raise InvalidInputError(
            f"The email body is too long ({len(body):,} characters). The limit is "
            f"{MAX_BODY_INPUT_CHARS:,} characters."
        )
    if _LOOKS_LIKE_HTML.search(body):
        body = html_to_text(body) or body  # visible text only; nothing is rendered or fetched
    return subject, body


def load_email_detector(models_dir: Path | None = None) -> EmailDetector:
    root = models_dir or settings.MODELS_DIR
    hint = "Run `python -m scripts.prepare_email` and then `python -m src.training.train_email`."
    try:
        pipeline, metadata = load_verified(
            root / EMAIL_MODULE_DIR, "threshold", "email model", hint
        )
    except FileNotFoundError as exc:
        raise ModelNotAvailableError(f"The email model has not been trained yet. {hint}") from exc
    return EmailDetector(pipeline, metadata)
