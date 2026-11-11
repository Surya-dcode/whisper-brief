from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from whisper_brief.audio import AudioChunk
from whisper_brief.errors import LocalBackendUnavailable
from whisper_brief.local import LocalTranscriber


def test_missing_faster_whisper_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(LocalBackendUnavailable, match=r"whisper-brief\[local\]"):
        LocalTranscriber()


def test_uses_faster_whisper_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created: dict = {}

    class FakeModel:
        def __init__(self, size: str, **kwargs: object) -> None:
            created["size"] = size

        def transcribe(self, path: str, language: str | None = None):
            segs = [
                types.SimpleNamespace(start=0.0, end=1.0, text=" hi "),
                types.SimpleNamespace(start=1.0, end=1.2, text="  "),
            ]
            return iter(segs), types.SimpleNamespace(language="en")

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel)
    )

    result = LocalTranscriber("tiny").transcribe_chunk(
        AudioChunk(index=1, path=tmp_path / "c.wav", offset=600.0, duration=600.0)
    )

    assert created["size"] == "tiny"
    assert result.language == "en"
    assert result.offset == 600.0
    assert [(s.start, s.text) for s in result.segments] == [(0.0, "hi")]
