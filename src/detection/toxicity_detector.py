"""Inference for the harmful-comment model. No Streamlit, no training code in here.

Returns six scores (one per label) and the labels whose score reaches that label's own threshold.
Nothing is stored; the comment is analysed in memory.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn

from config import settings
from src.detection.common import (
    DetectorError,
    InvalidInputError,
    ModelNotAvailableError,
    load_verified,
)
from src.models.toxicity_model import LABELS

MAX_COMMENT_CHARS = 5_000  # the longest comments in the training data are 5,000 characters
TOXICITY_MODULE_DIR = "toxicity"

__all__ = [
    "DetectorError",
    "InvalidInputError",
    "ModelNotAvailableError",
    "ToxicityDetector",
    "ToxicityResult",
    "load_toxicity_detector",
    "validate_comment",
]


@dataclass(frozen=True)
class ToxicityResult:
    scores: dict[str, float]  # all six labels, in LABELS order
    thresholds: dict[str, float]
    flagged: list[str]  # labels whose score reached their own threshold
    top_terms: dict[str, list[str]] = field(default_factory=dict)  # per flagged label
    truncated: bool = False  # True when the comment was longer than the model reads

    @property
    def any_flagged(self) -> bool:
        return bool(self.flagged)


class ToxicityDetector:
    def __init__(self, model, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.thresholds = {label: float(metadata["thresholds"][label]) for label in LABELS}

    @property
    def warnings(self) -> list[str]:
        recorded = self.metadata.get("scikit_learn")
        if recorded and recorded != sklearn.__version__:
            return [
                f"The comment model was trained with scikit-learn {recorded}; this environment "
                f"has {sklearn.__version__}. Results may differ."
            ]
        return []

    def predict(self, text: str, explain: bool = True) -> ToxicityResult:
        cleaned = validate_comment(text)
        row = self.model.scores([cleaned])[0]
        scores = {label: float(score) for label, score in zip(LABELS, row, strict=True)}
        flagged = [label for label in LABELS if scores[label] >= self.thresholds[label]]
        terms = self._top_terms(cleaned, flagged) if (explain and flagged) else {}
        return ToxicityResult(scores, dict(self.thresholds), flagged, terms, len(cleaned) > 1500)

    def _top_terms(self, text: str, flagged: list[str], top_k: int = 4) -> dict[str, list[str]]:
        """Words in this comment that push each flagged label up (word features only).

        contribution = the word's weight for that label x its value in this comment.
        """
        try:
            features = self.model.features
            word_vec = (
                features.transformer_list[0][1]
                if hasattr(features, "transformer_list")
                else features
            )
            row = word_vec.transform([text])
            names = word_vec.get_feature_names_out()
            result = {}
            for label in flagged:
                est = self.model.classifier.estimators_[LABELS.index(label)]
                weights = est.coef_.ravel()[: row.shape[1]]
                contribution = row.multiply(weights).tocsr()
                order = np.argsort(contribution.data)[::-1][:top_k]
                result[label] = [
                    str(names[contribution.indices[i]]) for i in order if contribution.data[i] > 0
                ]
            return result
        except Exception:  # noqa: BLE001 - an explanation must never break a prediction
            return {}


def validate_comment(text) -> str:
    if not isinstance(text, str):
        raise InvalidInputError("The comment must be text.")
    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        raise InvalidInputError("Please enter a comment to analyse.")
    if len(cleaned) > MAX_COMMENT_CHARS:
        raise InvalidInputError(
            f"The comment is too long ({len(cleaned):,} characters). The limit is "
            f"{MAX_COMMENT_CHARS:,} characters."
        )
    return cleaned


def load_toxicity_detector(models_dir: Path | None = None) -> ToxicityDetector:
    root = models_dir or settings.MODELS_DIR
    hint = (
        "Run `python -m scripts.prepare_toxicity` and then `python -m src.training.train_toxicity`."
    )
    try:
        model, metadata = load_verified(
            root / TOXICITY_MODULE_DIR, "thresholds", "comment model", hint
        )
    except FileNotFoundError as exc:
        raise ModelNotAvailableError(f"The comment model has not been trained yet. {hint}") from exc
    return ToxicityDetector(model, metadata)
