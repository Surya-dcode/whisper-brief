"""Generate a structured brief from a transcript with an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from whisper_brief.config import JsonMode
from whisper_brief.errors import APIError, BriefValidationError, WhisperBriefError
from whisper_brief.models import Brief, Transcript
from whisper_brief.retry import RetryPolicy, send_with_retry

SYSTEM_PROMPT = """\
You turn transcripts of recorded conversations into a short, factual brief.

Reply with one JSON object and nothing else. Use exactly these keys:
{
  "title": "short descriptive title",
  "summary": "two to four sentences on what the recording covers",
  "key_points": ["main points, one sentence each"],
  "decisions": ["decisions that were actually made"],
  "action_items": [{"task": "what needs doing", "owner": "name of the person, or null"}],
  "open_questions": ["questions raised but not answered"],
  "language": "ISO 639-1 code of the transcript language, for example en"
}

Rules:
- Use only what is in the transcript. Do not invent names, dates or numbers.
- Set "owner" to null unless the transcript says who will do the task.
- Use an empty list when a section has nothing in it.
- Write the brief in the same language as the transcript.
"""

RETRY_PROMPT = """\
Your previous reply could not be used: {error}
Reply again with only the corrected JSON object, using the keys described earlier."""

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class BriefParseError(BriefValidationError):
    """One response failed to parse. The generator retries once before giving up."""


def format_transcript(transcript: Transcript) -> str:
    """Render the transcript as timestamped lines so the model can follow the order of events."""
    if not transcript.segments:
        return transcript.text.strip()
    return "\n".join(f"[{_clock(s.start)}] {s.text}" for s in transcript.segments)


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def build_messages(transcript: Transcript) -> list[dict[str, str]]:
    hint = f"Detected language: {transcript.language}\n\n" if transcript.language else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{hint}Transcript:\n{format_transcript(transcript)}"},
    ]


def parse_brief(content: str) -> Brief:
    """Parse and validate a model reply. Tolerates a surrounding ```json fence."""
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BriefParseError(f"the reply is not valid JSON ({exc.msg} at char {exc.pos})") from exc
    if not isinstance(data, dict):
        raise BriefParseError(f"expected a JSON object, got {type(data).__name__}")
    try:
        return Brief.model_validate(data)
    except ValidationError as exc:
        raise BriefParseError(_describe(exc)) from exc


def _describe(exc: ValidationError) -> str:
    problems = []
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "(root)"
        problems.append(f"{where}: {err['msg']}")
    return "the JSON does not match the brief schema: " + "; ".join(problems)


def response_format(mode: JsonMode) -> dict[str, Any] | None:
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "brief", "schema": Brief.model_json_schema()},
        }
    return None


class BriefGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        json_mode: JsonMode = "json_object",
        temperature: float = 0.2,
        timeout: float = 120.0,
        policy: RetryPolicy | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.json_mode = json_mode
        self.temperature = temperature
        self.policy = policy or RetryPolicy()
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._sleep = sleep

    def generate(
        self, transcript: Transcript, *, on_token: Callable[[str], None] | None = None
    ) -> Brief:
        """Ask for a brief. If the reply does not validate, send the error back once."""
        if not transcript.text.strip() and not transcript.segments:
            raise WhisperBriefError("the transcript is empty, so there is nothing to summarize")

        messages = build_messages(transcript)
        content = self._complete(messages, on_token)
        try:
            return parse_brief(content)
        except BriefParseError as first:
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": RETRY_PROMPT.format(error=first)},
            ]
            if on_token is not None:
                on_token("\n")
            content = self._complete(messages, on_token)
            try:
                return parse_brief(content)
            except BriefParseError as second:
                raise BriefValidationError(
                    f"the model returned an unusable brief twice. Last problem: {second}"
                ) from second

    def _complete(
        self, messages: list[dict[str, str]], on_token: Callable[[str], None] | None
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        fmt = response_format(self.json_mode)
        if fmt is not None:
            payload["response_format"] = fmt
        if on_token is not None:
            payload["stream"] = True
            return self._stream(payload, on_token)

        response = send_with_retry(
            lambda: self._client.post(self.url, headers=self._headers, json=payload),
            self.policy,
            sleep=self._sleep,
            what="brief request",
        )
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise APIError("brief request returned an unexpected response shape") from exc

    def _stream(self, payload: dict[str, Any], on_token: Callable[[str], None]) -> str:
        def send() -> httpx.Response:
            request = self._client.build_request(
                "POST", self.url, headers=self._headers, json=payload
            )
            return self._client.send(request, stream=True)

        response = send_with_retry(send, self.policy, sleep=self._sleep, what="brief request")
        parts: list[str] = []
        try:
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    piece = event["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError, AttributeError):
                    continue
                if piece:
                    parts.append(piece)
                    on_token(piece)
        finally:
            response.close()
        return "".join(parts)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
