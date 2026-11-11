from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest
import respx

from tests.conftest import GROQ, transcription_payload
from whisper_brief.audio import AudioChunk
from whisper_brief.errors import APIError
from whisper_brief.models import ChunkTranscript
from whisper_brief.retry import RetryPolicy
from whisper_brief.transcribe import APITranscriber, transcribe_chunks

URL = f"{GROQ}/audio/transcriptions"


@pytest.fixture
def chunk(tmp_path: Path) -> AudioChunk:
    path = tmp_path / "chunk_0000.wav"
    path.write_bytes(b"RIFF fake wav bytes")
    return AudioChunk(index=0, path=path, offset=0.0, duration=5.0)


class Sleeps(list):
    def __call__(self, seconds: float) -> None:
        self.append(seconds)


def make_client(max_retries: int = 3) -> tuple[APITranscriber, Sleeps]:
    sleeps = Sleeps()
    client = APITranscriber(
        base_url=GROQ,
        api_key="test-key",
        model="whisper-large-v3-turbo",
        policy=RetryPolicy(max_retries=max_retries),
        sleep=sleeps,
    )
    return client, sleeps


def test_success_parses_segments(api: respx.MockRouter, chunk: AudioChunk) -> None:
    route = api.post(URL).mock(return_value=httpx.Response(200, json=transcription_payload()))
    client, sleeps = make_client()

    result = client.transcribe_chunk(chunk)

    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer test-key"
    body = request.content
    assert b'name="model"' in body and b"whisper-large-v3-turbo" in body
    assert b"verbose_json" in body
    assert b"RIFF fake wav bytes" in body
    assert result.text == "hello there"
    assert result.language == "english"
    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (0.0, 2.0, "hello there"),
        (2.5, 4.0, "second part"),
    ]
    assert sleeps == []


def test_retries_on_429_then_succeeds(api: respx.MockRouter, chunk: AudioChunk) -> None:
    route = api.post(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "2"}, json={"error": "rate limited"}),
            httpx.Response(200, json=transcription_payload()),
        ]
    )
    client, sleeps = make_client()

    result = client.transcribe_chunk(chunk)

    assert route.call_count == 2
    assert sleeps == [2.0]
    assert result.text == "hello there"


def test_retries_on_timeout_and_5xx(api: respx.MockRouter, chunk: AudioChunk) -> None:
    route = api.post(URL).mock(
        side_effect=[
            httpx.ReadTimeout("slow"),
            httpx.Response(503),
            httpx.Response(200, json=transcription_payload()),
        ]
    )
    client, sleeps = make_client()

    client.transcribe_chunk(chunk)

    assert route.call_count == 3
    assert len(sleeps) == 2
    # Backoff grows: the second wait is drawn from a higher range than the first.
    assert 0.5 <= sleeps[0] <= 1.0
    assert 1.0 <= sleeps[1] <= 2.0


def test_gives_up_after_max_retries(api: respx.MockRouter, chunk: AudioChunk) -> None:
    route = api.post(URL).mock(return_value=httpx.Response(500, text="upstream broke"))
    client, sleeps = make_client(max_retries=2)

    with pytest.raises(APIError) as info:
        client.transcribe_chunk(chunk)

    assert route.call_count == 3
    assert len(sleeps) == 2
    assert info.value.status_code == 500
    assert "upstream broke" in str(info.value)


def test_does_not_retry_auth_errors(api: respx.MockRouter, chunk: AudioChunk) -> None:
    route = api.post(URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    client, sleeps = make_client()

    with pytest.raises(APIError, match="HTTP 401"):
        client.transcribe_chunk(chunk)

    assert route.call_count == 1
    assert sleeps == []


def test_text_only_response_becomes_one_segment(api: respx.MockRouter, chunk: AudioChunk) -> None:
    api.post(URL).mock(return_value=httpx.Response(200, json={"text": "just text"}))
    client, _ = make_client()

    result = client.transcribe_chunk(chunk)

    assert [(s.start, s.end, s.text) for s in result.segments] == [(0.0, 5.0, "just text")]


class SlowFake:
    """Finishes chunks out of order and records how many run at once."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscript:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05 * (5 - chunk.index))
        with self.lock:
            self.active -= 1
        return ChunkTranscript(
            index=chunk.index, offset=chunk.offset, duration=10, text=f"chunk {chunk.index}"
        )


def test_transcribe_chunks_keeps_order_and_limits_concurrency(tmp_path: Path) -> None:
    chunks = [AudioChunk(i, tmp_path / f"{i}.wav", i * 10.0, 10.0) for i in range(5)]
    fake = SlowFake()
    progress: list[tuple[int, int]] = []

    results = transcribe_chunks(
        chunks, fake, concurrency=2, on_done=lambda d, t: progress.append((d, t))
    )

    assert [r.index for r in results] == [0, 1, 2, 3, 4]
    assert fake.peak <= 2
    assert progress[-1] == (5, 5)


def test_transcribe_chunks_raises_first_failure(tmp_path: Path) -> None:
    class Broken:
        def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscript:
            raise APIError("boom")

    chunks = [AudioChunk(i, tmp_path / f"{i}.wav", 0.0, 1.0) for i in range(3)]
    with pytest.raises(APIError, match="boom"):
        transcribe_chunks(chunks, Broken(), concurrency=2)
