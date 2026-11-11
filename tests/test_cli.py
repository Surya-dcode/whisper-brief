from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from tests.conftest import (
    GROQ,
    OPENAI,
    VALID_BRIEF,
    chat_payload,
    make_tone,
    requires_ffmpeg,
    transcription_payload,
)
from whisper_brief.cli import app

runner = CliRunner()


def mock_provider(api: respx.MockRouter, base: str) -> tuple[respx.Route, respx.Route]:
    stt = api.post(f"{base}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=transcription_payload())
    )
    chat = api.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_payload(json.dumps(VALID_BRIEF)))
    )
    return stt, chat


@requires_ffmpeg
def test_run_end_to_end(
    api: respx.MockRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    stt, chat = mock_provider(api, GROQ)
    src = make_tone(tmp_path / "meeting.m4a", 15)
    out = tmp_path / "out"

    result = runner.invoke(
        app, ["run", str(src), "--out", str(out), "--chunk-minutes", "0.1", "--markdown"]
    )

    assert result.exit_code == 0, result.output
    assert stt.call_count == 3
    assert chat.call_count == 1

    transcript = json.loads((out / "transcript.json").read_text())
    assert [c["offset"] for c in transcript["chunks"]] == pytest.approx([0, 6, 12], abs=0.1)
    starts = [s["start"] for s in transcript["segments"]]
    assert starts == pytest.approx([0, 2.5, 6, 8.5, 12, 14.5], abs=0.1)
    assert transcript["language"] == "english"

    brief = json.loads((out / "brief.json").read_text())
    assert brief == VALID_BRIEF
    assert (out / "brief.md").read_text().startswith("# Weekly sync")


@requires_ffmpeg
def test_run_with_openai_provider_and_stream(
    api: respx.MockRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    stt, _ = mock_provider(api, OPENAI)
    sse = (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": json.dumps(VALID_BRIEF)}}]})
        + "\n\ndata: [DONE]\n\n"
    )
    api.post(f"{OPENAI}/chat/completions").mock(
        return_value=httpx.Response(200, content=sse.encode())
    )
    src = make_tone(tmp_path / "call.wav", 3)

    result = runner.invoke(
        app, ["run", str(src), "--out", str(tmp_path / "o"), "--provider", "openai", "--stream"]
    )

    assert result.exit_code == 0, result.output
    assert b"whisper-1" in stt.calls.last.request.content
    assert stt.calls.last.request.headers["authorization"] == "Bearer openai-key"
    assert '"Weekly sync"' in result.stdout
    assert not (tmp_path / "o" / "brief.md").exists()


@requires_ffmpeg
def test_transcribe_writes_transcript_only(
    api: respx.MockRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _, chat = mock_provider(api, GROQ)
    src = make_tone(tmp_path / "memo.mp3", 2)

    result = runner.invoke(app, ["transcribe", str(src), "--out", str(tmp_path / "t")])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "t" / "transcript.json").is_file()
    assert not (tmp_path / "t" / "brief.json").exists()
    assert chat.call_count == 0


def test_brief_from_transcript_file(
    api: respx.MockRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    stt, chat = mock_provider(api, GROQ)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "language": "english",
                "duration": 4.0,
                "text": "We ship Friday.",
                "segments": [{"start": 0.0, "end": 4.0, "text": "We ship Friday."}],
            }
        )
    )

    result = runner.invoke(app, ["brief", str(transcript), "--markdown"])

    assert result.exit_code == 0, result.output
    assert stt.call_count == 0
    assert chat.call_count == 1
    assert json.loads((tmp_path / "brief.json").read_text())["title"] == "Weekly sync"
    assert (tmp_path / "brief.md").is_file()


def test_brief_rejects_bad_transcript_file(tmp_path: Path) -> None:
    bad = tmp_path / "transcript.json"
    bad.write_text('{"segments": "nope"}')
    result = runner.invoke(app, ["brief", str(bad)])
    assert result.exit_code == 1
    assert "not a valid transcript" in result.output


def test_missing_api_key_fails_before_any_work(tmp_path: Path) -> None:
    src = tmp_path / "a.wav"
    src.write_bytes(b"not used")
    result = runner.invoke(app, ["run", str(src)])
    assert result.exit_code == 1
    assert "GROQ_API_KEY" in result.output


def test_dotenv_in_working_directory_is_loaded(
    api: respx.MockRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, chat = mock_provider(api, GROQ)
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"text": "hello"}))

    result = runner.invoke(app, ["brief", str(transcript)])

    assert result.exit_code == 0, result.output
    assert chat.calls.last.request.headers["authorization"] == "Bearer from-dotenv"


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "whisper-brief" in result.output
