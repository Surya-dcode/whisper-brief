"""Command line interface: `whisper-brief run | transcribe | brief | serve`."""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from whisper_brief import __version__, pipeline
from whisper_brief.config import Settings
from whisper_brief.errors import WhisperBriefError
from whisper_brief.models import Transcript

app = typer.Typer(
    name="whisper-brief",
    help="Audio file in, structured brief out.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)


class Provider(StrEnum):
    groq = "groq"
    openai = "openai"


InputFile = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, readable=True, help="Audio or video file."),
]
OutDir = Annotated[Path, typer.Option("--out", "-o", help="Directory for output files.")]
ChunkMinutes = Annotated[
    float | None, typer.Option(help="Chunk length in minutes. Default 10.", min=0.01)
]
ProviderOpt = Annotated[Provider | None, typer.Option(help="Provider preset. Default groq.")]
LocalOpt = Annotated[bool, typer.Option("--local", help="Transcribe on this machine.")]
StreamOpt = Annotated[bool, typer.Option("--stream", help="Print brief tokens as they arrive.")]
MarkdownOpt = Annotated[bool, typer.Option("--markdown", help="Also write brief.md.")]
BaseUrl = Annotated[str | None, typer.Option(help="OpenAI-compatible API base URL.")]
ApiKey = Annotated[str | None, typer.Option(help="API key. Prefer the env var.")]
Model = Annotated[str | None, typer.Option("--model", help="Transcription model.")]
ChatModel = Annotated[str | None, typer.Option(help="Chat model used for the brief.")]
Concurrency = Annotated[int | None, typer.Option(help="Chunks transcribed at once.", min=1)]
Language = Annotated[
    str | None, typer.Option(help="Spoken language as an ISO 639-1 code. Default: detect.")
]


def _say(message: str) -> None:
    typer.echo(message, err=True)


def _write_token(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


def _settings(**overrides: object) -> Settings:
    return Settings.from_env(**overrides)


def _fail(exc: Exception) -> typer.Exit:
    _say(f"error: {exc}")
    return typer.Exit(code=1)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"whisper-brief {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Audio file in, structured brief out."""
    # Only the current directory's .env, and never over variables already set.
    load_dotenv(Path.cwd() / ".env", override=False)


@app.command()
def run(
    input_file: InputFile,
    out: OutDir = Path("out"),
    chunk_minutes: ChunkMinutes = None,
    provider: ProviderOpt = None,
    local: LocalOpt = False,
    stream: StreamOpt = False,
    markdown: MarkdownOpt = False,
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    model: Model = None,
    chat_model: ChatModel = None,
    concurrency: Concurrency = None,
    language: Language = None,
) -> None:
    """Transcribe a recording, then write the transcript and the brief."""
    try:
        settings = _settings(
            provider=provider.value if provider else None,
            base_url=base_url,
            api_key=api_key,
            transcribe_model=model,
            chat_model=chat_model,
            chunk_minutes=chunk_minutes,
            concurrency=concurrency,
            language=language,
            local=local,
        )
        result = pipeline.run(
            input_file,
            out,
            settings,
            markdown=markdown,
            on_token=_write_token if stream else None,
            progress=_say,
        )
    except WhisperBriefError as exc:
        raise _fail(exc) from exc
    if stream:
        _write_token("\n")
    for path in result.files:
        _say(f"wrote {path}")


@app.command()
def transcribe(
    input_file: InputFile,
    out: OutDir = Path("out"),
    chunk_minutes: ChunkMinutes = None,
    provider: ProviderOpt = None,
    local: LocalOpt = False,
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    model: Model = None,
    concurrency: Concurrency = None,
    language: Language = None,
) -> None:
    """Transcribe a recording and write transcript.json only."""
    try:
        settings = _settings(
            provider=provider.value if provider else None,
            base_url=base_url,
            api_key=api_key,
            transcribe_model=model,
            chunk_minutes=chunk_minutes,
            concurrency=concurrency,
            language=language,
            local=local,
        )
        transcript = pipeline.transcribe_file(input_file, settings, progress=_say)
        files = pipeline.write_outputs(out, transcript=transcript)
    except WhisperBriefError as exc:
        raise _fail(exc) from exc
    for path in files:
        _say(f"wrote {path}")


@app.command()
def brief(
    transcript_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="A transcript.json."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output directory. Default: next to the transcript."),
    ] = None,
    provider: ProviderOpt = None,
    stream: StreamOpt = False,
    markdown: MarkdownOpt = False,
    base_url: BaseUrl = None,
    api_key: ApiKey = None,
    chat_model: ChatModel = None,
) -> None:
    """Write brief.json from an existing transcript.json."""
    try:
        transcript = Transcript.model_validate_json(transcript_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        _say(f"error: {transcript_file} is not a valid transcript file: {exc}")
        raise typer.Exit(code=1) from exc
    try:
        settings = _settings(
            provider=provider.value if provider else None,
            base_url=base_url,
            api_key=api_key,
            chat_model=chat_model,
        )
        result = pipeline.generate_brief(
            transcript, settings, on_token=_write_token if stream else None
        )
        files = pipeline.write_outputs(
            out or transcript_file.parent, brief=result, markdown=markdown
        )
    except WhisperBriefError as exc:
        raise _fail(exc) from exc
    if stream:
        _write_token("\n")
    for path in files:
        _say(f"wrote {path}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Run the HTTP API (needs `pip install 'whisper-brief[api]'`)."""
    try:
        import uvicorn

        from whisper_brief.api import create_app
    except ImportError as exc:
        _say("error: the API server needs extra packages: pip install 'whisper-brief[api]'")
        raise typer.Exit(code=1) from exc
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
