"""Multi-label toxicity model: one binary Logistic Regression per label, shared features.

Why one classifier per label: the six labels overlap (a comment can be toxic, obscene and an
insult at once), so each question ("is it an insult?") is answered separately and each label gets
its own decision threshold. This keeps the original multi-label meaning instead of forcing one
category per comment.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import settings

LABELS = list(settings.TOXICITY_LABELS)


class PerLabelLogReg:
    """Six independent binary logistic regressions fitted on the same feature matrix."""

    def __init__(self, C: float = 4.0, n_jobs: int = 6):
        self.C = C
        self.n_jobs = n_jobs
        self.estimators_: list[LogisticRegression] = []

    def fit(self, X, Y) -> "PerLabelLogReg":
        Y = np.asarray(Y)

        def fit_one(j: int) -> LogisticRegression:
            clf = LogisticRegression(C=self.C, solver="liblinear", max_iter=200)
            return clf.fit(X, Y[:, j])

        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            self.estimators_ = list(pool.map(fit_one, range(Y.shape[1])))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probability of each label, shape (n_comments, n_labels)."""
        return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])


class ToxicityModel:
    """Fitted feature extractor + per-label classifiers + per-label thresholds."""

    def __init__(self, features, classifier: PerLabelLogReg, thresholds: dict[str, float]):
        self.features = features
        self.classifier = classifier
        self.thresholds = dict(thresholds)

    def scores(self, texts) -> np.ndarray:
        """(n, 6) probabilities in the order of LABELS."""
        return self.classifier.predict_proba(self.features.transform(list(texts)))

    def flags(self, texts) -> np.ndarray:
        """(n, 6) booleans: each label compared with its own threshold."""
        cut = np.array([self.thresholds[label] for label in LABELS])
        return self.scores(texts) >= cut

    def predict_one(self, text: str) -> dict[str, float]:
        row = self.scores([text])[0]
        return {label: float(score) for label, score in zip(LABELS, row, strict=True)}
