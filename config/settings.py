"""Single place for project-wide settings.

Every script imports paths, the random seed and the label convention from here,
so a value is never typed in two places.
"""

from pathlib import Path

# Project root = the folder that contains `config/`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL_TEST = PROJECT_ROOT / "data" / "external_test"
DATA_PRIVATE = PROJECT_ROOT / "data" / "private"

# Model and report folders
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Reproducibility: every random operation (splits, models) uses this seed.
RANDOM_SEED = 42

# Project-wide label convention: 1 = bad (spam / phishing / toxic), 0 = good.
LABEL_BAD = 1
LABEL_GOOD = 0

# Toxicity labels (Jigsaw), kept in one fixed order.
TOXICITY_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Per-session limit on Analyze clicks (see src/detection/rate_limit.py)
RATE_LIMIT_CALLS = 30
RATE_LIMIT_WINDOW_S = 60
