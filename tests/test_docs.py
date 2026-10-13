"""Documentation guard rails: links resolve, model cards match the model files, licence present."""

import re
from pathlib import Path

import pytest

from config import settings
from scripts.make_model_cards import OUT, build_cards

ROOT = settings.PROJECT_ROOT
DOC_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _relative_targets(path: Path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # code blocks may hold example links
    for target in LINK.findall(text):
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        yield target.split("#")[0]


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_and_images_resolve(path):
    missing = [t for t in _relative_targets(path) if t and not (path.parent / t).exists()]
    assert not missing, f"{path.relative_to(ROOT)} links to missing files: {missing}"


def test_model_cards_match_the_model_metadata():
    cards = build_cards()
    assert {"sms.md", "email.md", "comments.md", "url.md", "README.md"} <= set(cards)
    for name, text in cards.items():
        written = (OUT / name).read_text(encoding="utf-8")
        assert written.replace("\r\n", "\n") == text, (
            f"docs/model_cards/{name} is stale: run python -m scripts.make_model_cards"
        )


def test_model_cards_state_limitations_and_ethics():
    for name in ("sms.md", "email.md", "comments.md", "url.md"):
        text = (OUT / name).read_text(encoding="utf-8")
        for heading in ("## Intended use", "## Known limitations", "## Ethical considerations"):
            assert heading in text, (name, heading)
        assert "Out of scope" in text


def test_licence_exists_and_separates_code_from_data():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License") and "datasets" in text.lower()


def test_dataset_cards_cover_every_data_source():
    cards = "".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "docs" / "dataset_cards").glob("*.md")
    )
    for source in (
        "UCI SMS Spam Collection", "Mendeley", "ScamShield", "Enron", "untroubled", "Jigsaw",
        "PhreshPhish", "PhiUSIIL", "CompPhish", "PhishBench", "OpenPhish", "darkknight25",
    ):  # fmt: skip
        assert source in cards, f"no dataset card mentions {source}"


def test_readme_indexes_the_main_documents():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for doc in (
        "docs/concepts.md",
        "docs/security.md",
        "docs/model_cards",
        "docs/dataset_cards",
        "LICENSE",
    ):
        assert doc in readme, f"README does not mention {doc}"


def test_every_phase_has_a_readme_in_order():
    numbers = sorted(
        {int(m.group(1)) for p in (ROOT / "docs" / "phases").glob("phase-*-readme.md")
        if (m := re.match(r"phase-(\d+)", p.name))}
    )  # fmt: skip
    assert numbers == list(range(numbers[0], numbers[-1] + 1))
