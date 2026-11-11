"""Retry with exponential backoff for HTTP calls to rate-limited providers."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from whisper_brief.errors import APIError

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before retry number `attempt + 1`.

        A server-provided Retry-After wins. Otherwise the delay doubles each attempt,
        with jitter so parallel workers do not retry in lockstep.
        """
        if retry_after is not None:
            return min(self.max_delay, max(0.0, retry_after))
        ceiling = min(self.max_delay, self.base_delay * (2**attempt))
        return random.uniform(ceiling / 2, ceiling)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _body_snippet(response: httpx.Response, limit: int = 500) -> str:
    try:
        response.read()
        return response.text[:limit]
    except httpx.HTTPError:
        return ""


def send_with_retry(
    send: Callable[[], httpx.Response],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    what: str = "request",
) -> httpx.Response:
    """Call `send` until it returns a non-error response or the retries run out.

    Retries on timeouts, connection errors and the status codes in RETRYABLE_STATUS.
    Other 4xx responses fail straight away, since sending the same request again
    will not change the answer.
    """
    attempt = 0
    while True:
        try:
            response = send()
        except httpx.TransportError as exc:
            if attempt >= policy.max_retries:
                raise APIError(
                    f"{what} failed after {attempt + 1} attempts: {type(exc).__name__}: {exc}"
                ) from exc
            sleep(policy.delay_for(attempt))
            attempt += 1
            continue

        if response.status_code < 400:
            return response

        status = response.status_code
        if status in RETRYABLE_STATUS and attempt < policy.max_retries:
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            response.close()
            sleep(policy.delay_for(attempt, retry_after))
            attempt += 1
            continue

        body = _body_snippet(response)
        response.close()
        tries = f" after {attempt + 1} attempts" if attempt else ""
        raise APIError(
            f"{what} failed with HTTP {status}{tries}: {body}", status_code=status, body=body
        )
