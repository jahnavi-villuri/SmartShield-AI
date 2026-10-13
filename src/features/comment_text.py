"""Text preparation for comment models. A class (not a lambda) so joblib can save it."""


class ClippedLower:
    """Lowercase and keep only the first `max_chars` characters.

    Long comments are rare (99% are under 3,500 characters) and cost memory in character
    n-gram models; the start of a comment carries most of the signal. The same clipping runs
    at training and prediction time because it lives inside the saved pipeline.
    """

    def __init__(self, max_chars: int):
        self.max_chars = max_chars

    def __call__(self, text: str) -> str:
        return text[: self.max_chars].lower()
