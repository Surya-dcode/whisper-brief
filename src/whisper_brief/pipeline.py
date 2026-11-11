"""The stages wired together: normalize, chunk, transcribe, stitch, brief, write."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from whisper_brief import audio
from whisper_brief.brief import BriefGenerator
from whisper_brief.config import Settings
from whisper_brief.models import Brief, Transcript
from whisper_brief.render import brief_to_markdown
from whisper_brief.retry import RetryPolicy
from whisper_brief.stitch import stitch
from whisper_brief.transcribe import APITranscriber, Transcriber, transcribe_chunks

Progress = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


@dataclass
class RunResult:
    transcript: Transcript
    brief: Brief | None = None
    files: list[Path] = field(default_factory=list)


def build_transcriber(settings: Settings) -> Transcriber:
    if settings.local:
        from whisper_brief.local import LocalTranscriber

        return LocalTranscriber(settings.local_model, language=settings.language)
    return APITranscriber(
        base_url=settings.base_url,
        api_key=settings.require_api_key(),
        model=settings.transcribe_model,
        language=settings.language,
        timeout=settings.timeout,
        policy=RetryPolicy(max_retries=settings.max_retries),
    )


def transcribe_file(
    input_path: Path,
    settings: Settings,
    *,
    progress: Progress = _quiet,
    transcriber: Transcriber | None = None,
) -> Transcript:
    input_path = Path(input_path)
    owned = transcriber is None
    if transcriber is None:
        transcriber = build_transcriber(settings)
    try:
        with tempfile.TemporaryDirectory(prefix="whisper-brief-") as tmp:
            work = Path(tmp)
            progress(f"Normalizing {input_path.name} to 16 kHz mono WAV")
            wav = audio.normalize(input_path, work / "normalized.wav")

            progress(f"Splitting into {settings.chunk_minutes:g}-minute chunks")
            chunks = audio.split(wav, work / "chunks", settings.chunk_minutes * 60)

            # A local model already uses every core, so running it in parallel only adds memory.
            concurrency = 1 if settings.local else settings.concurrency
            progress(f"Transcribing {len(chunks)} chunk(s), up to {concurrency} at a time")
            results = transcribe_chunks(
                chunks,
                transcriber,
                concurrency=concurrency,
                on_done=lambda done, total: progress(f"  chunk {done}/{total} done"),
            )
    finally:
        if owned and hasattr(transcriber, "close"):
            transcriber.close()
    return stitch(results)


def generate_brief(
    transcript: Transcript,
    settings: Settings,
    *,
    on_token: Callable[[str], None] | None = None,
) -> Brief:
    generator = BriefGenerator(
        base_url=settings.base_url,
        api_key=settings.require_api_key(),
        model=settings.chat_model,
        json_mode=settings.json_mode,
        timeout=settings.timeout,
        policy=RetryPolicy(max_retries=settings.max_retries),
    )
    try:
        return generator.generate(transcript, on_token=on_token)
    finally:
        generator.close()


def write_outputs(
    out_dir: Path,
    *,
    transcript: Transcript | None = None,
    brief: Brief | None = None,
    markdown: bool = False,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if transcript is not None:
        path = out_dir / "transcript.json"
        path.write_text(transcript.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(path)
    if brief is not None:
        path = out_dir / "brief.json"
        path.write_text(brief.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(path)
        if markdown:
            md = out_dir / "brief.md"
            md.write_text(brief_to_markdown(brief), encoding="utf-8")
            written.append(md)
    return written


def run(
    input_path: Path,
    out_dir: Path,
    settings: Settings,
    *,
    markdown: bool = False,
    on_token: Callable[[str], None] | None = None,
    progress: Progress = _quiet,
) -> RunResult:
    """Full pipeline. Checks for an API key first so a missing key fails before any work."""
    settings.require_api_key()
    transcript = transcribe_file(input_path, settings, progress=progress)
    # Write the transcript before the brief step, so a failed brief does not lose it.
    files = write_outputs(out_dir, transcript=transcript)

    progress(f"Writing brief with {settings.chat_model}")
    brief = generate_brief(transcript, settings, on_token=on_token)
    files += write_outputs(out_dir, brief=brief, markdown=markdown)
    return RunResult(transcript=transcript, brief=brief, files=files)
