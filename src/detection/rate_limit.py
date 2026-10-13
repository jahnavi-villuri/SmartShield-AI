"""A small sliding-window rate limiter (one instance per browser session).

It slows down a person or script hammering the Analyze buttons in one session. It is NOT a defence
against a determined attacker, who can open many sessions: real protection needs the hosting
layer (reverse proxy or platform limits). See docs/security.md.
"""

import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self, max_calls: int, window_seconds: float, clock: Callable[[], float] = time.monotonic
    ):
        if max_calls < 1 or window_seconds <= 0:
            raise ValueError("max_calls must be >= 1 and window_seconds > 0")
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._clock = clock
        self._calls: deque[float] = deque()

    def allow(self) -> bool:
        """Record one call and return True, or return False when the window is full."""
        now = self._clock()
        while self._calls and now - self._calls[0] >= self.window_seconds:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            return False
        self._calls.append(now)
        return True

    def retry_after(self) -> float:
        """Seconds until the next call would be allowed (0 when allowed now)."""
        if len(self._calls) < self.max_calls:
            return 0.0
        return max(0.0, self.window_seconds - (self._clock() - self._calls[0]))
