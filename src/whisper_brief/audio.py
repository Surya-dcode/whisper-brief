"""Audio preparation with ffmpeg: normalize to 16 kHz mono WAV, then split into chunks."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from whisper_brief.errors import FFmpegError

SAMPLE_RATE = 16_000
CHANNELS = 1

# Segments shorter than this are dropped. The segment muxer can leave a near-empty
# last file when the input length is an exact multiple of the chunk length.
_MIN_CHUNK_SECONDS = 0.05


@dataclass(frozen=True)
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str


@dataclass(frozen=True)
class AudioChunk:
    index: int
    path: Path
    offset: float
    duration: float


def find_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(
            f"{name} was not found on PATH. Install ffmpeg (for example `brew install ffmpeg` "
            "or `apt-get install ffmpeg`) and try again."
        )
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise FFmpegError(f"{Path(cmd[0]).name} exited with code {result.returncode}: {tail}")
    return result


def probe(path: Path) -> AudioInfo:
    """Read duration, sample rate, channel count and codec of the first audio stream."""
    cmd = [
        find_binary("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(_run(cmd).stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise FFmpegError(f"no audio stream found in {path}")
    stream = streams[0]
    return AudioInfo(
        duration=float(data.get("format", {}).get("duration") or 0.0),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        codec=str(stream.get("codec_name") or ""),
    )


def normalize(src: Path, dst: Path) -> Path:
    """Convert any audio or video file ffmpeg can read into 16 kHz mono 16-bit PCM WAV."""
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FFmpegError(f"input file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_binary("ffmpeg"),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    return dst


def split(src: Path, out_dir: Path, chunk_seconds: float) -> list[AudioChunk]:
    """Split a WAV file into chunks of `chunk_seconds` with ffmpeg's segment muxer.

    Offsets come from the measured length of each chunk rather than index * chunk_seconds,
    so any rounding at the cut points does not add up over a long recording.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than 0")
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.wav"):
        old.unlink()

    _run(
        [
            find_binary("ffmpeg"),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-f",
            "segment",
            "-segment_time",
            f"{chunk_seconds:g}",
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            str(out_dir / "chunk_%04d.wav"),
        ]
    )

    chunks: list[AudioChunk] = []
    offset = 0.0
    for path in sorted(out_dir.glob("chunk_*.wav")):
        duration = probe(path).duration
        if duration < _MIN_CHUNK_SECONDS:
            path.unlink()
            continue
        chunks.append(AudioChunk(index=len(chunks), path=path, offset=offset, duration=duration))
        offset += duration
    if not chunks:
        raise FFmpegError(f"splitting {src} produced no audio chunks; is the file empty?")
    return chunks
