"""Text normalization that removes dataset artifacts (not meaning).

Why: diagnostics on the Round 1 SMS model showed it learning how the datasets were produced:
  - HTML codes such as `&lt;#&gt;` appear only in ham (0% spam in training);
  - `<#>` and `XXXX` are anonymisation markers that replaced real numbers, and were applied
    differently in different sources (rows with `XX` masks were 98% spam).
A spam filter must not rely on such leftovers, because real users' messages contain neither.

This function is passed to the vectorizers as `preprocessor=`, so the SAME normalization runs
during training and inside the saved pipeline at prediction time.
"""

import html
import re

# Masks "xx", "xxxx", ... but NOT "xxx" (kisses in chat, adult spam).
_MASKED_NUMBER = re.compile(r"\bx{2}\b|\bx{4,}\b")
_ANON_MARKER = re.compile(r"<#>")
_DIGITS = re.compile(r"\d+")


def normalize_sms(text: str) -> str:
    """Lowercase, decode HTML codes, and map every number-like token to the digit `0`."""
    text = html.unescape(text).lower()
    text = _ANON_MARKER.sub(" 0 ", text)
    text = _MASKED_NUMBER.sub("0", text)
    return _DIGITS.sub("0", text)
