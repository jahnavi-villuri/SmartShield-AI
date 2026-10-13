"""URL feature extraction: the ONE function used by training and by the app.

Everything here works on the URL *string*. Nothing is fetched, resolved or opened: no DNS, no
WHOIS, no page content. That is a hard rule of this project, so features such as domain age,
certificates or page forms (which other datasets include) are deliberately unavailable.

Because training and the app call the same code, feature names, order and types always match.
"""

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd

MAX_URL_CHARS = 2_000

# Two-part public suffixes (a small approximation of the Public Suffix List, no downloads).
_TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.in", "net.in", "org.in", "ac.in", "gov.in", "co.jp", "ne.jp", "or.jp", "ac.jp",
    "com.br", "com.mx", "com.ar", "com.co", "com.cn", "net.cn", "org.cn", "co.za", "co.nz",
    "co.kr", "com.sg", "com.hk", "com.tw", "com.tr", "com.ua", "com.ng", "com.pk", "com.my",
    "com.ph", "com.vn", "co.id", "co.th", "com.eg", "com.sa", "co.il",
}  # fmt: skip

# Free hosting / site-builder / tunnelling platforms: anyone can publish a page under these names.
HOSTING_PLATFORMS = {
    "pages.dev", "workers.dev", "firebaseapp.com", "web.app", "weebly.com", "wixsite.com",
    "blogspot.com", "github.io", "gitlab.io", "herokuapp.com", "vercel.app", "netlify.app",
    "ipfs.io", "cloudflare-ipfs.com", "000webhostapp.com", "glitch.me", "godaddysites.com",
    "sites.google.com", "docs.google.com", "forms.gle", "myshopify.com", "webflow.io",
    "framer.app", "carrd.co", "my.canva.site", "notion.site", "ngrok.io", "trycloudflare.com",
    "azurewebsites.net", "appspot.com", "onrender.com", "repl.co", "fleek.co", "pinata.cloud",
    "wordpress.com", "square.site", "strikingly.com", "yolasite.com", "webcindario.com",
}  # fmt: skip
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly",
    "rebrand.ly", "shorturl.at", "t.ly", "tiny.cc", "lnkd.in", "rb.gy", "s.id", "qr.ae",
}  # fmt: skip
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "shop", "click", "icu", "site", "online", "live",
    "vip", "cc", "buzz", "cyou", "monster", "rest", "work", "support", "life", "fit", "bond",
}  # fmt: skip
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "io", "co", "uk", "de", "in", "ca", "au", "fr"}
LOGIN_WORDS = (
    "login", "signin", "sign-in", "verify", "verification", "secure", "account", "update",
    "confirm", "password", "passwd", "credential", "wallet", "billing", "invoice", "payment",
    "suspend", "unlock", "recover", "authenticate", "validate", "alert", "support", "helpdesk",
)  # fmt: skip
BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google", "gmail", "amazon",
    "netflix", "facebook", "instagram", "whatsapp", "linkedin", "dropbox", "docusign", "adobe",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "barclays", "sbi", "hdfc", "icici",
    "paytm", "phonepe", "binance", "coinbase", "metamask", "ledger", "trustwallet", "dhl", "fedex",
    "usps", "ups", "irs", "steam", "roblox", "spotify", "airtel", "jio", "zomato",
)  # fmt: skip
RISKY_EXTENSIONS = (".exe", ".zip", ".scr", ".apk", ".js", ".jar", ".iso", ".rar", ".bat")
SCRIPT_EXTENSIONS = (".php", ".asp", ".aspx", ".jsp", ".cgi", ".html", ".htm")

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_LONG_HEX = re.compile(r"[0-9a-f]{16,}", re.IGNORECASE)
_CTRL_OR_SPACE = re.compile(r"[\s\x00-\x1f\x7f]")


class InvalidUrlError(ValueError):
    """The text cannot be treated as an http(s) URL. The message is safe to show a user."""


@dataclass(frozen=True)
class ParsedUrl:
    original: str
    scheme: str
    host: str
    port: int | None
    path: str
    query: str
    registered_domain: str
    subdomain_labels: tuple[str, ...]
    tld: str


def clean_url(raw: str) -> str:
    """Validate and normalise user input. Raises InvalidUrlError; never raises anything else.

    Accepts `example.com/path` (assumes http, because the text gives no scheme) and refuses
    other schemes (javascript:, data:, file:, ftp:...), spaces, control characters and
    over-long input. Nothing is opened or resolved.
    """
    if not isinstance(raw, str):
        raise InvalidUrlError("The URL must be text.")
    text = raw.strip()
    if not text:
        raise InvalidUrlError("Please enter a URL.")
    if len(text) > MAX_URL_CHARS:
        raise InvalidUrlError(f"The URL is too long (limit {MAX_URL_CHARS:,} characters).")
    if _CTRL_OR_SPACE.search(text):
        raise InvalidUrlError("A URL cannot contain spaces or control characters.")
    if _SCHEME.match(text):
        if not text.lower().startswith(("http://", "https://")):
            raise InvalidUrlError("Only http:// and https:// addresses can be analysed.")
    elif ":" in text.split("/")[0] and not re.match(r"^[^/]*:\d+(/|$)", text):
        raise InvalidUrlError("Only http:// and https:// addresses can be analysed.")
    else:
        text = "http://" + text
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
        _ = parts.port  # validates the port number
    except ValueError as exc:
        raise InvalidUrlError("That does not look like a valid URL.") from exc
    if not host or "." not in host and not _is_ip(host):
        raise InvalidUrlError("That does not look like a valid web address (no domain found).")
    if not _is_ip(host) and not re.fullmatch(r"[a-z0-9._~-]+|xn--[a-z0-9.-]+|[^\s/]+", host):
        raise InvalidUrlError("That does not look like a valid web address.")
    return text


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    return bool(_IPV4.match(host)) or bool(re.fullmatch(r"0x[0-9a-f]+|\d{8,10}", host))


def parse_url(raw: str) -> ParsedUrl:
    text = clean_url(raw)
    parts = urlsplit(text)
    host = (parts.hostname or "").lower().rstrip(".")
    labels = host.split(".") if not _is_ip(host) else [host]
    if _is_ip(host) or len(labels) < 2:
        registered, subs, tld = host, (), ""
    else:
        suffix_len = 2 if ".".join(labels[-2:]) in _TWO_PART_SUFFIXES and len(labels) >= 3 else 1
        registered = ".".join(labels[-(suffix_len + 1) :])
        subs = tuple(labels[: -(suffix_len + 1)])
        tld = ".".join(labels[-suffix_len:])
    return ParsedUrl(
        text, parts.scheme.lower(), host, parts.port, parts.path, parts.query, registered, subs, tld
    )


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = np.array(list({c: text.count(c) for c in set(text)}.values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _count(text: str, chars: str) -> int:
    return sum(text.count(c) for c in chars)


def _contains_any(text: str, words) -> int:
    return sum(1 for w in words if w in text)


def extract_features(raw: str) -> dict[str, float]:
    """All features of one URL, in a fixed order. Raises InvalidUrlError for unusable input."""
    p = parse_url(raw)
    # The scheme (http/https) is NOT a feature either: it is unreliable (most phishing uses
    # https today), users often paste addresses without it, and a model that leaned on it
    # called a bare 'dell.com' phishing. The detector reports it separately as a note.
    # The query string and fragment are IGNORED on purpose: in the training data no legitimate URL
    # carries a query string, so a model that saw it would learn "has a ?" = phishing, which is
    # false for real links. Only scheme, host, port and path are analysed.
    # "www." is not part of the address: www.example.com and example.com are the same site, and in
    # the training data "has www" separated the classes by accident (legitimate sites in the
    # public sets nearly always have it, phishing rarely does), which would flag github.com.
    netloc = urlsplit(p.original).netloc
    if netloc.lower().startswith("www."):
        netloc = netloc[4:]
    url_no_scheme = netloc + p.path
    host = p.host[4:] if p.host.startswith("www.") else p.host
    path = unquote(p.path).lower()
    rd_name = p.registered_domain.split(".")[0] if p.registered_domain else ""
    host_labels = host.split(".")
    subdomain_text = ".".join(p.subdomain_labels)
    n_sub = len([s for s in p.subdomain_labels if s != "www"])
    lowered = url_no_scheme.lower()
    path_segments = [s for s in path.split("/") if s]

    brands_in_host = [b for b in BRANDS if b in host]
    brand_in_sub = [b for b in BRANDS if b in subdomain_text or (b in rd_name and rd_name != b)]
    return {
        # --- whole URL ---
        "url_len": len(url_no_scheme),
        "n_dots": url_no_scheme.count("."),
        "n_hyphens": url_no_scheme.count("-"),
        "n_underscores": url_no_scheme.count("_"),
        "n_digits": sum(c.isdigit() for c in url_no_scheme),
        "digit_ratio": sum(c.isdigit() for c in url_no_scheme) / max(len(url_no_scheme), 1),
        "n_special": sum((not c.isalnum()) and c not in "./-:" for c in url_no_scheme),
        "has_at": float("@" in url_no_scheme),
        "n_percent": url_no_scheme.count("%"),
        "has_long_hex": float(bool(_LONG_HEX.search(url_no_scheme))),
        "entropy_url": _entropy(lowered),
        # --- host ---
        "host_len": len(host),
        "is_ip_host": float(_is_ip(host)),
        "has_port": float(p.port is not None),
        "n_host_labels": len(host_labels),
        "n_subdomains": n_sub,
        "host_hyphens": host.count("-"),
        "host_digits": sum(c.isdigit() for c in host),
        "host_digit_ratio": sum(c.isdigit() for c in host) / max(len(host), 1),
        "longest_host_label": max(len(label) for label in host_labels),
        "entropy_host": _entropy(host),
        "has_punycode": float("xn--" in host),
        "rd_len": len(rd_name),
        "rd_hyphens": rd_name.count("-"),
        "rd_digits": sum(c.isdigit() for c in rd_name),
        "tld_len": len(p.tld.split(".")[-1]) if p.tld else 0,
        "tld_common": float(p.tld.split(".")[-1] in COMMON_TLDS),
        "tld_suspicious": float(p.tld.split(".")[-1] in SUSPICIOUS_TLDS),
        "is_hosting_platform": float(
            p.registered_domain in HOSTING_PLATFORMS
            or any(host.endswith("." + h) for h in HOSTING_PLATFORMS)
        ),
        "is_shortener": float(p.registered_domain in URL_SHORTENERS),
        "n_login_words_host": _contains_any(host, LOGIN_WORDS),
        "n_brands_in_host": len(brands_in_host),
        "brand_not_registered_domain": float(len(brand_in_sub) > 0),
        # --- path and query ---
        "path_len": len(path),
        "path_depth": len(path_segments),
        "longest_path_segment": max((len(s) for s in path_segments), default=0),
        "path_digits": sum(c.isdigit() for c in path),
        "entropy_path": _entropy(path),
        "has_path": float(len(path) > 1),
        "ends_with_script_ext": float(path.endswith(SCRIPT_EXTENSIONS)),
        "risky_extension": float(path.endswith(RISKY_EXTENSIONS)),
        "double_slash_in_path": float("//" in path),
        "n_login_words_path": _contains_any(path, LOGIN_WORDS),
        "brand_in_path": float(
            any(b in path for b in BRANDS) and not any(b in rd_name for b in BRANDS)
        ),
    }


FEATURE_NAMES = list(extract_features("https://www.example.com/a?b=1").keys())


def features_frame(urls, on_error: str = "raise") -> pd.DataFrame:
    """Feature table (one row per URL, columns in FEATURE_NAMES order).

    on_error="nan" returns NaN rows for unusable URLs (for bulk dataset processing);
    "raise" propagates InvalidUrlError (for the app).
    """
    rows = []
    for u in urls:
        try:
            rows.append(extract_features(u))
        except InvalidUrlError:
            if on_error == "raise":
                raise
            rows.append(dict.fromkeys(FEATURE_NAMES, np.nan))
    return pd.DataFrame(rows, columns=FEATURE_NAMES).astype("float64")


class UrlFeatureTransformer:
    """sklearn-compatible transformer: list of URLs -> feature matrix. Picklable class."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return features_frame(X, on_error="nan").fillna(0.0).to_numpy()

    def get_feature_names_out(self, *args, **kwargs):
        return np.array(FEATURE_NAMES)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class UrlText:
    """Char-model input: host (no www.) + path, lowercase. Query, fragment and scheme are dropped.

    A class (not a lambda) and defined HERE (not in a training script) so that a saved pipeline
    can be loaded from any program: pickle stores the module path of every class it saves.
    """

    def __call__(self, url: str) -> str:
        try:
            p = parse_url(url)
        except InvalidUrlError:
            return ""
        host = p.host[4:] if p.host.startswith("www.") else p.host
        return (host + p.path).lower()[:300]
