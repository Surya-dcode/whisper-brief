"""Chunk transcription against an OpenAI-compatible /audio/transcriptions endpoint."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

import httpx

from whisper_brief.audio import AudioChunk
from whisper_brief.errors import APIError
from whisper_brief.models import ChunkTranscript, Segment
from whisper_brief.retry import RetryPolicy, send_with_retry


class Transcriber(Protocol):
    def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscript: ...


class APITranscriber:
    """Sends one chunk per request. Safe to share across threads."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        language: str | None = None,
        timeout: float = 120.0,
        policy: RetryPolicy | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.model = model
        self.language = language
        self.policy = policy or RetryPolicy()
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._sleep = sleep

    def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscript:
        # Read once so every retry uploads the same bytes from the start.
        audio = chunk.path.read_bytes()
        form = {"model": self.model, "response_format": "verbose_json"}
        if self.language:
            form["language"] = self.language

        def send() -> httpx.Response:
            return self._client.post(
                self.url,
                headers=self._headers,
                data=form,
                files={"file": (chunk.path.name, audio, "audio/wav")},
            )

        response = send_with_retry(
            send, self.policy, sleep=self._sleep, what=f"transcription of chunk {chunk.index}"
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise APIError(f"transcription of chunk {chunk.index} returned invalid JSON") from exc
        return parse_transcription(payload, chunk)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def parse_transcription(payload: dict[str, Any], chunk: AudioChunk) -> ChunkTranscript:
    """Turn a verbose_json response into a ChunkTranscript with chunk-relative times."""
    text = str(payload.get("text") or "").strip()
    segments = [
        Segment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]).strip())
        for s in payload.get("segments") or []
        if str(s.get("text", "")).strip()
    ]
    if not segments and text:
        # Some servers ignore verbose_json and return only text. Keep it as one segment.
        segments = [Segment(start=0.0, end=chunk.duration, text=text)]
    return ChunkTranscript(
        index=chunk.index,
        offset=chunk.offset,
        duration=chunk.duration,
        text=text,
        language=payload.get("language") or None,
        segments=segments,
    )


def transcribe_chunks(
    chunks: Sequence[AudioChunk],
    transcriber: Transcriber,
    *,
    concurrency: int = 3,
    on_done: Callable[[int, int], None] | None = None,
) -> list[ChunkTranscript]:
    """Transcribe chunks with at most `concurrency` requests in flight.

    Results come back sorted by chunk index. If any chunk fails for good, pending
    chunks are cancelled and the error is raised.
    """
    results: list[ChunkTranscript] = []
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="transcribe")
    try:
        futures = [pool.submit(transcriber.transcribe_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            results.append(future.result())
            if on_done is not None:
                on_done(len(results), len(chunks))
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return sorted(results, key=lambda r: r.index)
