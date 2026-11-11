from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_tone, requires_ffmpeg
from whisper_brief import audio
from whisper_brief.errors import FFmpegError

pytestmark = [requires_ffmpeg, pytest.mark.ffmpeg]


@pytest.mark.parametrize("ext", ["m4a", "mp3", "wav"])
def test_normalize_produces_16k_mono_pcm(tmp_path: Path, ext: str) -> None:
    src = make_tone(tmp_path / f"tone.{ext}", 2)
    before = audio.probe(src)
    assert (before.sample_rate, before.channels) == (44_100, 2)

    out = audio.normalize(src, tmp_path / "norm" / "out.wav")

    info = audio.probe(out)
    assert info.sample_rate == 16_000
    assert info.channels == 1
    assert info.codec == "pcm_s16le"
    assert info.duration == pytest.approx(2.0, abs=0.1)


def test_normalize_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError, match="not found"):
        audio.normalize(tmp_path / "missing.m4a", tmp_path / "out.wav")


def test_normalize_rejects_non_audio(tmp_path: Path) -> None:
    junk = tmp_path / "notes.m4a"
    junk.write_text("this is not audio")
    with pytest.raises(FFmpegError):
        audio.normalize(junk, tmp_path / "out.wav")


def test_split_25s_into_10s_chunks(tmp_path: Path) -> None:
    wav = audio.normalize(make_tone(tmp_path / "long.m4a", 25), tmp_path / "long.wav")

    chunks = audio.split(wav, tmp_path / "chunks", 10)

    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.offset for c in chunks] == pytest.approx([0.0, 10.0, 20.0], abs=0.1)
    assert [c.duration for c in chunks] == pytest.approx([10.0, 10.0, 5.0], abs=0.1)
    for chunk in chunks:
        assert chunk.path.is_file()
        info = audio.probe(chunk.path)
        assert (info.sample_rate, info.channels) == (16_000, 1)


def test_split_short_file_gives_one_chunk(tmp_path: Path) -> None:
    wav = audio.normalize(make_tone(tmp_path / "short.wav", 3), tmp_path / "short16.wav")
    chunks = audio.split(wav, tmp_path / "chunks", 600)
    assert len(chunks) == 1
    assert chunks[0].offset == 0.0
    assert chunks[0].duration == pytest.approx(3.0, abs=0.1)


def test_split_rejects_bad_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        audio.split(tmp_path / "x.wav", tmp_path / "chunks", 0)
