"""Pieces shared by every detector: user-safe errors and the checksum-verified model loader."""

import hashlib
import json
from pathlib import Path

import joblib


class DetectorError(Exception):
    """Base class: the message is written for end users."""


class InvalidInputError(DetectorError):
    """The user's input cannot be analysed (empty, too long, wrong type)."""


class ModelNotAvailableError(DetectorError):
    """The model file is missing, unreadable or does not match its recorded checksum."""


def load_verified(folder: Path, required_key: str, what: str, hint: str):
    """Load (pipeline, metadata) from `folder` after the SHA-256 check.

    Joblib files can run code when loaded, so we only load a file whose SHA-256 matches
    the checksum training recorded in metadata.json. Raises FileNotFoundError(what) when the
    files are absent, so the caller can decide whether the model is required.
    """
    model_path, meta_path = folder / "model.joblib", folder / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(what)
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = metadata["model_sha256"]
        _ = metadata[required_key]
    except (OSError, ValueError, KeyError) as exc:
        raise ModelNotAvailableError(f"The {what} metadata is unreadable. {hint}") from exc
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected:
        raise ModelNotAvailableError(
            f"The {what} file does not match its recorded checksum, so it was not loaded. {hint}"
        )
    try:
        return joblib.load(model_path), metadata
    except Exception as exc:  # noqa: BLE001 - any load failure becomes a friendly error
        raise ModelNotAvailableError(f"The {what} could not be loaded. {hint}") from exc
