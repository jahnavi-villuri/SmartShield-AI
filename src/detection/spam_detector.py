"""Inference for the SMS detector. No Streamlit, no training code in here.

Two stages:
  1. the main model scores "unwanted" (promotional or scam) vs legitimate;
  2. if unwanted, a small second model says promotional vs scam.
If the second model has not been trained, an unwanted message is simply labelled "Spam".

The UI calls `load_sms_detector()` once and `SpamDetector.predict()` per message.
All failure modes raise a `DetectorError` subclass with a message that is safe to show.
"""

from dataclasses import dataclass, field
from pathlib import Path

import sklearn

from config import settings
from src.detection.common import (  # noqa: F401 - re-exported for callers
    DetectorError,
    InvalidInputError,
    ModelNotAvailableError,
    load_verified,
)
from src.evaluation.metrics import get_scores

MAX_SMS_CHARS = 1000  # the longest messages in the training data are ~900 characters
MAX_EXPLAIN_WORDS = 40  # explanation costs one prediction per word; keep it cheap
MIN_WORD_EFFECT = 0.02  # a word must move the score by at least this to be reported
SMS_MODULE_DIR = "sms_spam"
TYPE_MODULE_DIR = "sms_type"

LABEL_OK = "Not spam"
LABEL_PROMO = "Promotional"
LABEL_SCAM = "Scam"
LABEL_SPAM = "Spam"  # unwanted, type unknown (second model not available)


@dataclass(frozen=True)
class SpamResult:
    is_spam: bool  # True for anything unwanted: promotional, scam or "Spam"
    label: str  # "Not spam", "Promotional", "Scam" or "Spam"
    score: float  # main model's score for "unwanted"
    threshold: float  # scores at or above this are called unwanted
    score_type: str  # "probability" or "decision_score"
    scam_score: float | None = None  # second model's score for "scam" (unwanted messages only)
    influential_words: list[str] = field(default_factory=list)


class SpamDetector:
    """Loaded models plus the metadata that describes them."""

    def __init__(self, pipeline, metadata: dict, type_pipeline=None, type_metadata=None):
        self.pipeline = pipeline
        self.metadata = metadata
        self.type_pipeline = type_pipeline
        self.type_metadata = type_metadata
        self.threshold = float(metadata["threshold"])
        self.score_type = metadata.get("score_type", "probability")
        self.scam_threshold = float((type_metadata or {}).get("scam_threshold", 0.5))

    @property
    def warnings(self) -> list[str]:
        """Non-fatal problems worth showing (version mismatch, missing second model)."""
        notes = []
        for meta in (self.metadata, self.type_metadata):
            recorded = (meta or {}).get("scikit_learn")
            if recorded and recorded != sklearn.__version__:
                notes.append(
                    f"A model was trained with scikit-learn {recorded}; this environment has "
                    f"{sklearn.__version__}. Results may differ."
                )
        if self.type_pipeline is None:
            notes.append(
                "The promotional-vs-scam model is not available, so unwanted messages are "
                "shown as 'Spam' without a type."
            )
        return notes

    def predict(self, text: str, explain: bool = True) -> SpamResult:
        cleaned = validate_message(text)
        score = float(get_scores(self.pipeline, [cleaned])[0])
        unwanted = score >= self.threshold
        if not unwanted:
            return SpamResult(False, LABEL_OK, score, self.threshold, self.score_type)

        scam_score, label = None, LABEL_SPAM
        if self.type_pipeline is not None:
            scam_score = float(get_scores(self.type_pipeline, [cleaned])[0])
            label = LABEL_SCAM if scam_score >= self.scam_threshold else LABEL_PROMO
        words = self._influential_words(cleaned, score) if explain else []
        return SpamResult(True, label, score, self.threshold, self.score_type, scam_score, words)

    def _influential_words(self, text: str, base_score: float, top_k: int = 3) -> list[str]:
        """Occlusion explanation: remove each word and see how much the score drops.

        Works for any model (no access to weights needed). It shows what the score is
        sensitive to; it is an approximation, not a proof of why the model decided.
        """
        words = text.split()
        if not words or len(words) > MAX_EXPLAIN_WORDS:
            return []
        variants = [" ".join(words[:i] + words[i + 1 :]) or " " for i in range(len(words))]
        drops = base_score - get_scores(self.pipeline, variants)
        ranked = sorted(zip(drops, words, strict=True), key=lambda pair: pair[0], reverse=True)
        picked = []
        for drop, word in ranked:
            if drop < MIN_WORD_EFFECT or len(picked) == top_k:
                break
            if word not in picked:
                picked.append(word[:40])
        return picked


def validate_message(text) -> str:
    """Return the cleaned message or raise InvalidInputError."""
    if not isinstance(text, str):
        raise InvalidInputError("The message must be text.")
    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        raise InvalidInputError("Please enter a message to analyse.")
    if len(cleaned) > MAX_SMS_CHARS:
        raise InvalidInputError(
            f"The message is too long ({len(cleaned)} characters). "
            f"The limit is {MAX_SMS_CHARS} characters."
        )
    return cleaned


def load_sms_detector(models_dir: Path | None = None) -> SpamDetector:
    """Load the main SMS model (required) and the promotional-vs-scam model (optional)."""
    root = models_dir or settings.MODELS_DIR
    hint = "Run `python -m scripts.prepare_sms` and then `python -m src.training.train_sms`."
    try:
        pipeline, metadata = load_verified(root / SMS_MODULE_DIR, "threshold", "SMS model", hint)
    except FileNotFoundError as exc:
        raise ModelNotAvailableError(f"The SMS model has not been trained yet. {hint}") from exc
    try:
        type_pipeline, type_metadata = load_verified(
            root / TYPE_MODULE_DIR,
            "scam_threshold",
            "promotional-vs-scam model",
            "Run `python -m src.training.train_sms_type`.",
        )
    except FileNotFoundError:
        type_pipeline, type_metadata = None, None  # optional: fall back to plain "Spam"
    return SpamDetector(pipeline, metadata, type_pipeline, type_metadata)
