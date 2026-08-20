import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Header, HTTPException

from app.config import RATE_LIMIT_REQUESTS_PER_MINUTE


class SlidingWindowRateLimiter:
    # tracks request timestamps per key in a deque and evicts anything
    # older than the window on each call. simple, correct, and fine at the
    # request volumes a single-process demo service sees — swap for a
    # Redis-backed limiter before running more than one worker process

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS_PER_MINUTE, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True


rate_limiter = SlidingWindowRateLimiter()


def enforce_rate_limit(x_api_key: str = Header(...)) -> None:
    if not rate_limiter.allow(x_api_key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
