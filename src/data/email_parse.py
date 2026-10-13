"""Safe parsing of raw emails into plain text. Treats every message as hostile input.

What this does NOT do: open or save attachments, follow links, load images, run scripts, or
render HTML. It only reads text. Attachments are counted, never decoded. HTML is reduced to its
visible text with the standard-library parser (script/style/head content is dropped).

Why we need the HTML text: many 2026 spam emails carry a harmless filler paragraph in the
text/plain part (to fool word-based filters) while the real lure is in the HTML part, which is
what mail clients show. So "visible text" prefers the HTML part when there is one.
"""

import email
import email.policy
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser

MAX_TEXT_CHARS = 20_000  # long newsletters are cut; training and the app apply the same cap
_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6", "hr"}
_SKIP_TAGS = {"script", "style", "head", "title", "noscript"}
_WHITESPACE = re.compile(r"[ \t\r\f\v ​‌‍⁠﻿]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


# Inline-CSS tricks that hide text from the reader. Spam uses them to hide a harmless "filler"
# paragraph that fools word-based filters. Only inline styles are inspected (no stylesheet parsing).
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none"
    r"|visibility\s*:\s*hidden"
    r"|mso-hide\s*:\s*all"
    r"|opacity\s*:\s*0(?![.\d])"
    r"|font-size\s*:\s*0(?:px|pt|em|rem|%)?\s*(?:;|$|\")"
    r"|font-size\s*:\s*1(?:px|pt)\s*(?:;|$)"
    r"|(?:max-)?height\s*:\s*0(?:px|pt)?\s*(?:;|$)[^\"]*overflow\s*:\s*hidden"
    r"|overflow\s*:\s*hidden[^\"]*(?:max-)?height\s*:\s*0(?:px|pt)?\s*(?:;|$)"
    r"|(?:max-)?width\s*:\s*[01]px[^\"]*overflow\s*:\s*hidden"
    r"|overflow\s*:\s*hidden[^\"]*(?:max-)?width\s*:\s*[01]px",
    re.IGNORECASE,
)
_VOID_TAGS = {"br", "hr", "img", "meta", "link", "input", "area", "base", "col", "wbr", "source"}


def _is_hidden(attrs) -> bool:
    values = dict(attrs)
    if "hidden" in values:
        return True
    return bool(_HIDDEN_STYLE.search((values.get("style") or "").replace("\n", " ") + ";"))


class _TextExtractor(HTMLParser):
    """Collects visible text. Skips script/style/head and elements hidden by inline CSS."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._stack: list[tuple[str, bool]] = []  # (tag, makes_hidden)
        self._hidden_depth = 0
        self.hidden_chars = 0  # how much text was dropped as hidden (for reporting)

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and not self._hidden_depth:
            self.parts.append("\n")
        if tag not in _VOID_TAGS:
            hides = _is_hidden(attrs)
            self._stack.append((tag, hides))
            if hides:
                self._hidden_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS and not self._hidden_depth:
            self.parts.append("\n")
        # close the most recent matching open tag (tolerates sloppy HTML)
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                for _, hides in self._stack[i:]:
                    if hides:
                        self._hidden_depth -= 1
                del self._stack[i:]
                break

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._hidden_depth:
            self.hidden_chars += len(data.strip())
        else:
            self.parts.append(data)


def clean_whitespace(text: str) -> str:
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", text).strip()


def html_to_text_and_hidden(markup: str) -> tuple[str, int]:
    """(visible text, number of characters dropped because CSS hides them)."""
    parser = _TextExtractor()
    try:
        parser.feed(markup[: MAX_TEXT_CHARS * 20])
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML must never crash a pipeline
        return "", 0
    return clean_whitespace(html.unescape("".join(parser.parts))), parser.hidden_chars


def html_to_text(markup: str) -> str:
    """Visible text of an HTML string (no network access, no rendering)."""
    return html_to_text_and_hidden(markup)[0]


@dataclass(frozen=True)
class ParsedEmail:
    subject: str
    plain_text: str
    html_text: str
    n_attachments: int
    content_types: tuple[str, ...]
    hidden_chars: int = 0  # text dropped because inline CSS hides it from the reader

    @property
    def visible_text(self) -> str:
        """What a person reading the email sees: the HTML text if present, else plain text."""
        return (self.html_text or self.plain_text)[:MAX_TEXT_CHARS]


def _part_text(msg, kind: str) -> str:
    try:
        part = msg.get_body(preferencelist=(kind,))
        return part.get_content() if part is not None else ""
    except Exception:  # noqa: BLE001 - bad charsets etc.
        return ""


def parse_email_bytes(raw: bytes) -> ParsedEmail:
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    try:
        subject = clean_whitespace(str(msg.get("Subject", "") or ""))
    except Exception:  # noqa: BLE001 - malformed encoded headers
        subject = ""
    plain = clean_whitespace(_part_text(msg, "plain"))
    html_text, hidden_chars = html_to_text_and_hidden(_part_text(msg, "html"))
    try:
        n_attachments = sum(1 for _ in msg.iter_attachments())
    except Exception:  # noqa: BLE001
        n_attachments = 0
    types = tuple(sorted({p.get_content_type() for p in msg.walk() if not p.is_multipart()}))
    return ParsedEmail(
        subject, plain[:MAX_TEXT_CHARS], html_text, n_attachments, types, hidden_chars
    )
