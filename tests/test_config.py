from __future__ import annotations

import pytest

from whisper_brief.config import Settings
from whisper_brief.errors import ConfigError


def test_groq_is_the_default() -> None:
    s = Settings.from_env({"GROQ_API_KEY": "g"})
    assert s.provider == "groq"
    assert s.base_url == "https://api.groq.com/openai/v1"
    assert s.transcribe_model == "whisper-large-v3-turbo"
    assert s.api_key == "g"
    assert s.chunk_minutes == 10.0


def test_openai_preset_uses_its_own_key() -> None:
    s = Settings.from_env({"GROQ_API_KEY": "g", "OPENAI_API_KEY": "o"}, provider="openai")
    assert s.transcribe_model == "whisper-1"
    assert s.base_url == "https://api.openai.com/v1"
    assert s.api_key == "o"


def test_env_overrides_and_flags_win_over_env() -> None:
    env = {
        "WHISPER_BRIEF_BASE_URL": "http://localhost:9000/v1/",
        "WHISPER_BRIEF_API_KEY": "generic",
        "WHISPER_BRIEF_CHUNK_MINUTES": "5",
        "WHISPER_BRIEF_CONCURRENCY": "8",
        "WHISPER_BRIEF_TRANSCRIBE_MODEL": "env-model",
    }
    s = Settings.from_env(env, transcribe_model="flag-model", concurrency=None)
    assert s.base_url == "http://localhost:9000/v1"
    assert s.api_key == "generic"
    assert s.chunk_minutes == 5.0
    assert s.concurrency == 8
    assert s.transcribe_model == "flag-model"


@pytest.mark.parametrize(
    ("env", "overrides", "message"),
    [
        ({}, {"provider": "acme"}, "unknown provider"),
        ({"WHISPER_BRIEF_CHUNK_MINUTES": "ten"}, {}, "must be a number"),
        ({}, {"chunk_minutes": 0}, "greater than 0"),
        ({}, {"concurrency": 0}, "at least 1"),
        ({}, {"json_mode": "yaml"}, "json mode"),
    ],
)
def test_invalid_settings(env: dict, overrides: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env, **overrides)


def test_missing_key_message_names_the_env_var() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        Settings.from_env({}, provider="openai").require_api_key()
