"""Data models shared by the pipeline stages and written to disk."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Segment(BaseModel):
    """A piece of transcript with start and end times in seconds."""

    start: float
    end: float
    text: str


class ChunkTranscript(BaseModel):
    """The transcript of one chunk. Segment times are relative to the chunk start."""

    index: int
    offset: float
    duration: float
    text: str
    language: str | None = None
    segments: list[Segment] = Field(default_factory=list)


class ChunkInfo(BaseModel):
    index: int
    offset: float
    duration: float


class Transcript(BaseModel):
    """The full transcript. Segment times are relative to the start of the recording."""

    language: str | None = None
    duration: float = 0.0
    text: str
    segments: list[Segment] = Field(default_factory=list)
    chunks: list[ChunkInfo] = Field(default_factory=list)


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: str = Field(min_length=1)
    owner: str | None = None


class Brief(BaseModel):
    """The structured brief the chat model must return."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    language: str = Field(min_length=1)
