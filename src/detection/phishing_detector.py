"""Inference for the phishing-URL model. No Streamlit, no training code in here.

SAFETY RULES (enforced by tests):
  - The URL is treated as text only. It is never fetched, resolved, opened or turned into a link.
  - Only http:// and https:// addresses are accepted; javascript:, data:, file: and the like are
    refused with a message.
  - The result is never "safe": the strongest positive wording is "Likely legitimate".
  - Shortened links get "Cannot judge": their destination is unknowable without opening them.
  - The query string and fragment are ignored (the model was trained that way).
"""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import sklearn

from config import settings
from src.detection.common import (
    DetectorError,
    InvalidInputError,
    ModelNotAvailableError,
    load_verified,
)
from src.evaluation.metrics import get_scores
from src.features.url_features import (
    MAX_URL_CHARS,
    URL_SHORTENERS,
    InvalidUrlError,
    clean_url,
    extract_features,
    parse_url,
)

PHISHING_MODULE_DIR = "phishing"
LABEL_PHISHING = "Potentially phishing"
LABEL_LEGIT = "Likely legitimate"
LABEL_UNKNOWN = "Cannot judge"

__all__ = [
    "DetectorError",
    "InvalidInputError",
    "LABEL_LEGIT",
    "LABEL_PHISHING",
    "LABEL_UNKNOWN",
    "MAX_URL_CHARS",
    "ModelNotAvailableError",
    "PhishingDetector",
    "PhishingResult",
    "load_phishing_detector",
    "observations",
]


@dataclass(frozen=True)
class PhishingResult:
    label: str  # "Potentially phishing", "Likely legitimate" or "Cannot judge"
    is_phishing: bool
    judged: bool  # False for shortened links: no verdict is made
    score: float
    threshold: float
    url: str  # the cleaned URL, for display as PLAIN TEXT only
    ignored_query: bool = False  # True when a ?query or #fragment was present and ignored
    assumed_scheme: bool = False  # True when the user typed no http(s):// and http was assumed
    observations: list[str] = field(default_factory=list)  # plain-language things worth noticing
    features: dict[str, float] = field(default_factory=dict)


def observations(
    features: dict[str, float], registered_domain: str, explicit_http: bool = False
) -> list[str]:
    """Plain-language notes computed from the URL itself. They describe the address; they are not
    the model's reasoning, and a legitimate URL can trigger several of them."""
    notes = []
    if features["is_ip_host"]:
        notes.append("The address uses a numeric IP instead of a domain name.")
    if features["has_at"]:
        notes.append("The address contains '@' (text before it can disguise the real host).")
    if features["has_punycode"]:
        notes.append("The domain uses punycode (xn--), which can imitate other letters.")
    if features["brand_not_registered_domain"]:
        notes.append("A well-known brand name appears in the address but is not the real domain.")
    if features["n_login_words_host"]:
        notes.append("The domain name contains words like login, verify, secure or account.")
    if features["n_subdomains"] >= 3:
        notes.append(f"It has {features['n_subdomains']:.0f} subdomain levels.")
    if features["host_hyphens"] >= 3:
        notes.append("The domain contains many hyphens.")
    if features["tld_suspicious"]:
        notes.append("The ending (top-level domain) is one often used for throw-away sites.")
    if features["is_hosting_platform"]:
        notes.append(
            f"It is a page on a free hosting platform ({registered_domain}); anyone can publish "
            "there, so the address alone says little about who is behind it."
        )
    if features["url_len"] > 90:
        notes.append("The address is unusually long.")
    if explicit_http:
        notes.append("It uses http, not https (https alone does not make a site safe either).")
    if features["risky_extension"]:
        notes.append("The path ends in a file type that can carry malware (such as .exe or .zip).")
    return notes


class PhishingDetector:
    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])

    @property
    def warnings(self) -> list[str]:
        recorded = self.metadata.get("scikit_learn")
        if recorded and recorded != sklearn.__version__:
            return [
                f"The URL model was trained with scikit-learn {recorded}; this environment has "
                f"{sklearn.__version__}. Results may differ."
            ]
        return []

    def predict(self, raw_url: str) -> PhishingResult:
        try:
            cleaned = clean_url(raw_url)
            parsed = parse_url(cleaned)
            features = extract_features(cleaned)
        except InvalidUrlError as exc:
            raise InvalidInputError(str(exc)) from exc
        split = urlsplit(cleaned)
        ignored = bool(split.query or split.fragment)
        assumed = not raw_url.strip().lower().startswith(("http://", "https://"))
        explicit_http = cleaned.lower().startswith("http://") and not assumed
        notes = observations(features, parsed.registered_domain, explicit_http)
        score = float(get_scores(self.pipeline, [cleaned])[0])
        shortened = parsed.registered_domain in URL_SHORTENERS
        if shortened:
            notes.insert(
                0,
                "This is a shortened link. Its real destination cannot be known without opening "
                "it, which this app never does, so no verdict is given. Be extra careful.",
            )
            return PhishingResult(
                LABEL_UNKNOWN,
                False,
                False,
                score,
                self.threshold,
                cleaned,
                ignored,
                assumed,
                notes,
                features,
            )
        is_phishing = score >= self.threshold
        return PhishingResult(
            LABEL_PHISHING if is_phishing else LABEL_LEGIT,
            is_phishing,
            True,
            score,
            self.threshold,
            cleaned,
            ignored,
            assumed,
            notes,
            features,
        )


def load_phishing_detector(models_dir: Path | None = None) -> PhishingDetector:
    root = models_dir or settings.MODELS_DIR
    hint = (
        "Run `python -m scripts.prepare_phishing` and then `python -m src.training.train_phishing`."
    )
    try:
        pipeline, metadata = load_verified(
            root / PHISHING_MODULE_DIR, "threshold", "URL model", hint
        )
    except FileNotFoundError as exc:
        raise ModelNotAvailableError(f"The URL model has not been trained yet. {hint}") from exc
    return PhishingDetector(pipeline, metadata)
