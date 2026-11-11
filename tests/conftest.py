from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import respx

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg and ffprobe are not installed")

GROQ = "https://api.groq.com/openai/v1"
OPENAI = "https://api.openai.com/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Start every test with no provider keys or overrides from the real environment,
    and in an empty working directory so no local .env file is picked up."""
    import os

    for name in list(os.environ):
        if name.startswith("WHISPER_BRIEF_") or name in {"GROQ_API_KEY", "OPENAI_API_KEY"}:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def api() -> Iterator[respx.MockRouter]:
    """Intercept every httpx request. Any request without a matching route fails the test,
    so nothing in the suite can reach the network."""
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        yield router


def make_tone(path: Path, seconds: float, *, rate: int = 44_100, channels: int = 2) -> Path:
    """Write a sine tone with ffmpeg. The defaults differ from 16 kHz mono on purpose."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}:sample_rate={rate}",
            "-ac",
            str(channels),
            str(path),
        ],
        check=True,
    )
    return path


def transcription_payload(text: str = "hello there", language: str = "english") -> dict:
    return {
        "text": text,
        "language": language,
        "duration": 5.0,
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": f" {text}"},
            {"id": 1, "start": 2.5, "end": 4.0, "text": " second part"},
        ],
    }


def chat_payload(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


VALID_BRIEF = {
    "title": "Weekly sync",
    "summary": "The team reviewed the release and agreed on next steps.",
    "key_points": ["Release is on track"],
    "decisions": ["Ship on Friday"],
    "action_items": [
        {"task": "Write release notes", "owner": "Priya"},
        {"task": "Tag the build", "owner": None},
    ],
    "open_questions": ["Who covers support next week?"],
    "language": "en",
}
