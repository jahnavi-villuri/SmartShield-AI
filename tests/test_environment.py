"""Phase 1 smoke tests: the environment and settings are sane."""

import importlib

import pytest

from config import settings


@pytest.mark.parametrize(
    "package",
    ["streamlit", "pandas", "numpy", "sklearn", "joblib", "matplotlib", "seaborn"],
)
def test_core_packages_import(package):
    importlib.import_module(package)


def test_label_convention_is_one_equals_bad():
    assert settings.LABEL_BAD == 1
    assert settings.LABEL_GOOD == 0


def test_toxicity_has_six_labels():
    assert len(settings.TOXICITY_LABELS) == 6


def test_project_folders_exist():
    for folder in (settings.DATA_RAW, settings.MODELS_DIR, settings.REPORTS_DIR):
        assert folder.exists(), f"missing folder: {folder}"
