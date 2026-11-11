"""Optional on-device transcription with faster-whisper (`pip install whisper-brief[local]`)."""

from __future__ import annotations

from typing import Any

from whisper_brief.audio import AudioChunk
from whisper_brief.errors import LocalBackendUnavailable
from whisper_brief.models import ChunkTranscript, Segment


class LocalTranscriber:
    """Runs a faster-whisper model in this process. The import happens on construction,
    so the package works without faster-whisper installed until --local is used."""

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise LocalBackendUnavailable(
                "the local backend needs faster-whisper. Install it with "
                "`pip install 'whisper-brief[local]'`."
            ) from exc
        self.language = language
        self._model: Any = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscript:
        segments, info = self._model.transcribe(str(chunk.path), language=self.language)
        parsed = [
            Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]
        return ChunkTranscript(
            index=chunk.index,
            offset=chunk.offset,
            duration=chunk.duration,
            text=" ".join(s.text for s in parsed),
            language=getattr(info, "language", None),
            segments=parsed,
        )
