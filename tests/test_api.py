from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient

from tests.conftest import VALID_BRIEF
from whisper_brief.api import create_app
from whisper_brief.config import Settings
from whisper_brief.errors import FFmpegError
from whisper_brief.models import Brief, Transcript
from whisper_brief.pipeline import RunResult


def settings() -> Settings:
    return Settings(api_key="test-key")


def test_post_briefs_returns_brief() -> None:
    seen: dict = {}

    def fake_run(src: Path, out: Path, s: Settings) -> RunResult:
        seen["suffix"] = src.suffix
        seen["bytes"] = src.read_bytes()
        return RunResult(transcript=Transcript(text="x"), brief=Brief(**VALID_BRIEF))

    client = TestClient(create_app(settings, fake_run))
    response = client.post("/briefs", files={"file": ("call.m4a", b"audio-bytes", "audio/mp4")})

    assert response.status_code == 200
    assert response.json() == VALID_BRIEF
    assert seen == {"suffix": ".m4a", "bytes": b"audio-bytes"}


def test_bad_audio_is_a_400() -> None:
    def fake_run(*_: object) -> RunResult:
        raise FFmpegError("no audio stream")

    client = TestClient(create_app(settings, fake_run))
    response = client.post("/briefs", files={"file": ("x.mp3", b"junk", "audio/mpeg")})

    assert response.status_code == 400
    assert "no audio stream" in response.json()["detail"]


def test_health() -> None:
    assert TestClient(create_app(settings)).get("/health").json() == {"status": "ok"}
