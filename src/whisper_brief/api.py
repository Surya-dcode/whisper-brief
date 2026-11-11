"""A small HTTP API: POST an audio file to /briefs and get the brief back as JSON."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from whisper_brief import __version__, pipeline
from whisper_brief.config import Settings
from whisper_brief.errors import APIError, BriefValidationError, ConfigError, FFmpegError

Runner = Callable[..., pipeline.RunResult]


def create_app(
    settings_factory: Callable[[], Settings] = Settings.from_env,
    runner: Runner = pipeline.run,
) -> FastAPI:
    app = FastAPI(title="whisper-brief", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # A plain `def` so FastAPI runs the blocking pipeline in its thread pool.
    @app.post("/briefs")
    def create_brief(file: Annotated[UploadFile, File(description="Audio file")]) -> dict[str, Any]:
        try:
            settings = settings_factory()
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        suffix = Path(file.filename or "").suffix or ".bin"
        with tempfile.TemporaryDirectory(prefix="whisper-brief-api-") as tmp:
            src = Path(tmp) / f"upload{suffix}"
            with src.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            try:
                result = runner(src, Path(tmp) / "out", settings)
            except ConfigError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except FFmpegError as exc:
                raise HTTPException(status_code=400, detail=f"could not read audio: {exc}") from exc
            except (APIError, BriefValidationError) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        if result.brief is None:
            raise HTTPException(status_code=500, detail="pipeline returned no brief")
        return result.brief.model_dump()

    return app
