"""Read-only summary of every saved model, taken from its metadata.json.

The Home and About pages use this so that no number is ever typed into the UI by hand: if a
model is retrained, the pages change with it. Nothing here loads a .joblib file.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from config import settings


@dataclass(frozen=True)
class ModelSummary:
    key: str
    title: str
    available: bool
    model: str = ""
    trained_on: str = ""
    train_size: int = 0
    headline: str = ""
    limitations: list[str] = field(default_factory=list)


MODEL_DIRS = {
    "sms": ("SMS messages", "sms_spam"),
    "email": ("Emails", "email_spam"),
    "comments": ("Comments", "toxicity"),
    "url": ("URLs", "phishing"),
}


def _headline(key: str, meta: dict) -> str:
    metrics = meta["metrics"]
    if key == "sms":
        t = metrics["test"]
        return f"precision {t['precision']:.2f}, recall {t['recall']:.2f} on held-out test data"
    if key == "email":
        t = metrics["test"]
        return f"precision {t['precision']:.2f}, recall {t['recall']:.2f} on held-out test data"
    if key == "comments":
        s = metrics["official_test"]["summary"]
        return (
            f"macro-F1 {s['macro_f1']:.2f}, micro-F1 {s['micro_f1']:.2f} on the official test set"
        )
    t = metrics["test_temporal"]
    return (
        f"caught {t['recall']:.0%} of newer phishing URLs, flagged "
        f"{t['false_positive_rate']:.1%} of newer legitimate ones"
    )


def summarize_models(models_dir: Path | None = None) -> list[ModelSummary]:
    """One summary per detector; a missing or unreadable model is reported, never raised."""
    root = Path(models_dir) if models_dir is not None else settings.MODELS_DIR
    summaries = []
    for key, (title, folder) in MODEL_DIRS.items():
        path = root / folder / "metadata.json"
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(
                ModelSummary(
                    key=key,
                    title=title,
                    available=True,
                    model=str(meta["model"]),
                    trained_on=str(meta["created_utc"])[:10],
                    train_size=int(meta["data_sizes"]["train"]),
                    headline=_headline(key, meta),
                    limitations=list(meta.get("limitations", [])),
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            summaries.append(ModelSummary(key=key, title=title, available=False))
    return summaries
