"""Merge per-chunk transcripts into one transcript on the recording's timeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from whisper_brief.models import ChunkInfo, ChunkTranscript, Segment, Transcript


def stitch(chunks: Sequence[ChunkTranscript]) -> Transcript:
    """Order chunks by index and shift every segment by its chunk's offset."""
    ordered = sorted(chunks, key=lambda c: c.index)
    indices = [c.index for c in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate chunk indices: {indices}")

    segments = [
        Segment(
            start=round(chunk.offset + seg.start, 3),
            end=round(chunk.offset + seg.end, 3),
            text=seg.text,
        )
        for chunk in ordered
        for seg in chunk.segments
    ]
    text = " ".join(c.text.strip() for c in ordered if c.text.strip())
    duration = max((c.offset + c.duration for c in ordered), default=0.0)

    return Transcript(
        language=_dominant_language(ordered),
        duration=round(duration, 3),
        text=text,
        segments=segments,
        chunks=[ChunkInfo(index=c.index, offset=c.offset, duration=c.duration) for c in ordered],
    )


def _dominant_language(chunks: Sequence[ChunkTranscript]) -> str | None:
    """The language that covers the most audio. A short chunk of silence or music
    can be detected as something else, so a plain majority vote is not enough."""
    weights: Counter[str] = Counter()
    for chunk in chunks:
        if chunk.language:
            weights[chunk.language.lower()] += max(chunk.duration, 0.0) or 1.0
    if not weights:
        return None
    return weights.most_common(1)[0][0]
