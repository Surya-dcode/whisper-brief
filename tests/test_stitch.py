from __future__ import annotations

import pytest

from whisper_brief.models import ChunkTranscript, Segment
from whisper_brief.stitch import stitch


def chunk(
    index: int, offset: float, duration: float, language: str | None = "en"
) -> ChunkTranscript:
    return ChunkTranscript(
        index=index,
        offset=offset,
        duration=duration,
        text=f"part {index}",
        language=language,
        segments=[
            Segment(start=0.0, end=1.5, text=f"part {index} a"),
            Segment(start=2.0, end=3.25, text=f"part {index} b"),
        ],
    )


def test_shifts_segments_by_chunk_offset_and_orders_chunks() -> None:
    transcript = stitch([chunk(2, 20.0, 5.0), chunk(0, 0.0, 10.0), chunk(1, 10.0, 10.0)])

    assert [(s.start, s.end, s.text) for s in transcript.segments] == [
        (0.0, 1.5, "part 0 a"),
        (2.0, 3.25, "part 0 b"),
        (10.0, 11.5, "part 1 a"),
        (12.0, 13.25, "part 1 b"),
        (20.0, 21.5, "part 2 a"),
        (22.0, 23.25, "part 2 b"),
    ]
    assert transcript.text == "part 0 part 1 part 2"
    assert transcript.duration == 25.0
    assert [c.offset for c in transcript.chunks] == [0.0, 10.0, 20.0]


def test_segment_starts_never_go_backwards() -> None:
    transcript = stitch([chunk(i, i * 600.0, 600.0) for i in range(4)])
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)


def test_keeps_language_that_covers_most_audio() -> None:
    transcript = stitch(
        [
            chunk(0, 0.0, 600.0, "Spanish"),
            chunk(1, 600.0, 600.0, "spanish"),
            chunk(2, 1200.0, 4.0, "english"),
        ]
    )
    assert transcript.language == "spanish"


def test_language_is_none_when_not_detected() -> None:
    assert stitch([chunk(0, 0.0, 5.0, None)]).language is None


def test_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        stitch([chunk(0, 0.0, 5.0), chunk(0, 5.0, 5.0)])


def test_empty_input_gives_empty_transcript() -> None:
    transcript = stitch([])
    assert transcript.text == ""
    assert transcript.segments == []
    assert transcript.duration == 0.0
