"""Fail if files that must never be published are about to be committed.

Lists what Git would add (tracked + untracked, honouring .gitignore) and rejects model binaries,
dataset files, private data, quoted third-party text, secrets and very large files.

    python -m scripts.check_repo_hygiene
"""

import re
import subprocess
import sys

MAX_BYTES = 2_000_000
FORBIDDEN = [
    (r"\.(joblib|pkl|pickle|parquet|zip|gz)$", "model or data binary"),
    (r"^data/(?!.*\.gitkeep$).+", "dataset or private data (only .gitkeep may live in data/)"),
    (r"(false_negatives|false_positives|missed_spam)\.csv$", "quotes third-party messages"),
    (r"(^|/)errors_.*\.csv$", "quotes third-party comments"),
    (r"(^|/)\.env$|\.key$|\.pem$", "secret"),
    (r"(^|/)\.coverage(\..*)?$|(^|/)htmlcov/", "test-tool output"),
]


def candidate_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def problems(files: list[str], size_of) -> list[str]:
    found = []
    for path in files:
        for pattern, why in FORBIDDEN:
            if re.search(pattern, path):
                found.append(f"{path}: {why}")
        if size_of(path) > MAX_BYTES:
            found.append(f"{path}: larger than {MAX_BYTES / 1e6:.0f} MB")
    return found


def main() -> int:
    from pathlib import Path

    def size_of(path: str) -> int:
        p = Path(path)
        return p.stat().st_size if p.is_file() else 0

    found = problems(candidate_files(), size_of)
    for line in found:
        print("FORBIDDEN:", line)
    print(f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
