"""Settings, resolved from CLI flags first, then environment variables, then provider defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from whisper_brief.errors import ConfigError

JsonMode = Literal["json_object", "json_schema", "off"]
JSON_MODES: tuple[str, ...] = ("json_object", "json_schema", "off")

ENV_PREFIX = "WHISPER_BRIEF_"


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    transcribe_model: str
    chat_model: str
    key_env: str


PROVIDERS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        transcribe_model="whisper-large-v3-turbo",
        chat_model="llama-3.3-70b-versatile",
        key_env="GROQ_API_KEY",
    ),
    "openai": ProviderPreset(
        name="openai",
        base_url="https://api.openai.com/v1",
        transcribe_model="whisper-1",
        chat_model="gpt-4o-mini",
        key_env="OPENAI_API_KEY",
    ),
}


@dataclass(frozen=True)
class Settings:
    provider: str = "groq"
    base_url: str = PROVIDERS["groq"].base_url
    api_key: str | None = None
    transcribe_model: str = PROVIDERS["groq"].transcribe_model
    chat_model: str = PROVIDERS["groq"].chat_model
    chunk_minutes: float = 10.0
    concurrency: int = 3
    max_retries: int = 4
    timeout: float = 120.0
    json_mode: JsonMode = "json_object"
    language: str | None = None
    local: bool = False
    local_model: str = "small"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> Settings:
        """Build settings. Overrides set to None are ignored, so CLI flags can be passed as-is."""
        env = os.environ if env is None else env
        given = {key: value for key, value in overrides.items() if value is not None}

        def pick(key: str, default: Any = None) -> Any:
            if key in given:
                return given[key]
            raw = env.get(ENV_PREFIX + key.upper())
            return raw if raw not in (None, "") else default

        provider = str(pick("provider", "groq")).lower()
        if provider not in PROVIDERS:
            choices = ", ".join(sorted(PROVIDERS))
            raise ConfigError(f"unknown provider {provider!r}; choose one of: {choices}")
        preset = PROVIDERS[provider]

        api_key = pick("api_key") or env.get(preset.key_env) or None
        json_mode = str(pick("json_mode", "json_object"))
        if json_mode not in JSON_MODES:
            raise ConfigError(
                f"json mode must be one of {', '.join(JSON_MODES)}, got {json_mode!r}"
            )

        settings = cls(
            provider=provider,
            base_url=str(pick("base_url", preset.base_url)).rstrip("/"),
            api_key=api_key,
            transcribe_model=str(pick("transcribe_model", preset.transcribe_model)),
            chat_model=str(pick("chat_model", preset.chat_model)),
            chunk_minutes=_number(pick("chunk_minutes", 10.0), "chunk minutes", float),
            concurrency=_number(pick("concurrency", 3), "concurrency", int),
            max_retries=_number(pick("max_retries", 4), "max retries", int),
            timeout=_number(pick("timeout", 120.0), "timeout", float),
            json_mode=json_mode,  # type: ignore[arg-type]
            language=pick("language"),
            local=bool(given.get("local", False)),
            local_model=str(pick("local_model", "small")),
        )
        settings.validate()
        return settings

    @property
    def key_env(self) -> str:
        return PROVIDERS[self.provider].key_env

    def validate(self) -> None:
        if self.chunk_minutes <= 0:
            raise ConfigError("chunk minutes must be greater than 0")
        if self.concurrency < 1:
            raise ConfigError("concurrency must be at least 1")
        if self.max_retries < 0:
            raise ConfigError("max retries cannot be negative")
        if self.timeout <= 0:
            raise ConfigError("timeout must be greater than 0")

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                f"no API key found. Set {self.key_env} (or {ENV_PREFIX}API_KEY) in your "
                "environment or .env file, or pass --api-key."
            )
        return self.api_key


def _number(value: Any, label: str, kind: type) -> Any:
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number, got {value!r}") from exc
