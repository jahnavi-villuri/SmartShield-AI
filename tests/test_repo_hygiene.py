"""The repository-hygiene rules (what must never be committed)."""

from scripts.check_repo_hygiene import problems


def _size(_path: str) -> int:
    return 10


def test_allowed_files_pass():
    files = [
        "app.py",
        "data/raw/.gitkeep",
        "models/sms_spam/metadata.json",
        "reports/metrics/sms_final_metrics.csv",
        "reports/figures/x.png",
    ]
    assert problems(files, _size) == []


def test_models_data_and_secrets_are_rejected():
    files = [
        "models/sms_spam/model.joblib",
        "data/raw/sms.csv",
        "data/private/my_sms/messages.csv",
        "reports/metrics/round2/sms_test_false_negatives.csv",
        "reports/metrics/email_2026_missed_spam.csv",
        "data/processed/toxicity/errors_val.csv",
        ".env",
        "deploy.key",
        ".coverage",
        "htmlcov/index.html",
    ]
    flagged = {line.split(":")[0] for line in problems(files, _size)}
    assert flagged == set(files)


def test_large_files_are_rejected():
    assert problems(["notes/big.txt"], lambda _p: 5_000_000)
