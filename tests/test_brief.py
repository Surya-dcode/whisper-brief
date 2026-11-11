from __future__ import annotations

import json

import httpx
import pytest
import respx

from tests.conftest import GROQ, VALID_BRIEF, chat_payload
from whisper_brief.brief import BriefGenerator, format_transcript, parse_brief
from whisper_brief.errors import BriefValidationError, WhisperBriefError
from whisper_brief.models import Segment, Transcript
from whisper_brief.render import brief_to_markdown

URL = f"{GROQ}/chat/completions"


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        language="english",
        duration=75.0,
        text="We ship Friday. Priya writes the notes.",
        segments=[
            Segment(start=0.0, end=3.0, text="We ship Friday."),
            Segment(start=65.0, end=70.0, text="Priya writes the notes."),
        ],
    )


def make_generator(**kwargs) -> BriefGenerator:
    return BriefGenerator(
        base_url=GROQ, api_key="test-key", model="test-model", sleep=lambda _: None, **kwargs
    )


def sent_json(route: respx.Route, call: int = -1) -> dict:
    return json.loads(route.calls[call].request.content)


def test_valid_json_passes(api: respx.MockRouter, transcript: Transcript) -> None:
    route = api.post(URL).mock(
        return_value=httpx.Response(200, json=chat_payload(json.dumps(VALID_BRIEF)))
    )

    brief = make_generator().generate(transcript)

    assert route.call_count == 1
    assert brief.title == "Weekly sync"
    assert brief.action_items[0].owner == "Priya"
    assert brief.action_items[1].owner is None
    body = sent_json(route)
    assert body["model"] == "test-model"
    assert body["response_format"] == {"type": "json_object"}
    assert "stream" not in body
    assert "[00:01:05] Priya writes the notes." in body["messages"][1]["content"]
    assert "Detected language: english" in body["messages"][1]["content"]


def test_invalid_json_triggers_exactly_one_retry(
    api: respx.MockRouter, transcript: Transcript
) -> None:
    route = api.post(URL).mock(
        side_effect=[
            httpx.Response(200, json=chat_payload("Sure! Here is the brief: {title: oops")),
            httpx.Response(200, json=chat_payload(json.dumps(VALID_BRIEF))),
        ]
    )

    brief = make_generator().generate(transcript)

    assert route.call_count == 2
    assert brief.decisions == ["Ship on Friday"]
    retry_messages = sent_json(route, 1)["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert retry_messages[-1]["role"] == "user"
    assert "not valid JSON" in retry_messages[-1]["content"]


def test_schema_violation_raises_clear_error(api: respx.MockRouter, transcript: Transcript) -> None:
    bad = json.dumps({"title": "Weekly sync", "key_points": "not a list"})
    route = api.post(URL).mock(return_value=httpx.Response(200, json=chat_payload(bad)))

    with pytest.raises(BriefValidationError) as info:
        make_generator().generate(transcript)

    assert route.call_count == 2
    message = str(info.value)
    assert "twice" in message
    assert "summary: Field required" in message
    assert "key_points" in message
    assert "language: Field required" in message


def test_streaming_forwards_tokens(api: respx.MockRouter, transcript: Transcript) -> None:
    text = json.dumps(VALID_BRIEF)
    pieces = [text[i : i + 20] for i in range(0, len(text), 20)]
    events = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": p}}]}) for p in pieces
    ]
    sse = "\n\n".join([": keep-alive", *events, "data: [DONE]"]) + "\n\n"
    route = api.post(URL).mock(
        return_value=httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    tokens: list[str] = []

    brief = make_generator().generate(transcript, on_token=tokens.append)

    assert brief.title == "Weekly sync"
    assert tokens == pieces
    assert sent_json(route)["stream"] is True


def test_json_schema_mode_sends_schema(api: respx.MockRouter, transcript: Transcript) -> None:
    route = api.post(URL).mock(
        return_value=httpx.Response(200, json=chat_payload(json.dumps(VALID_BRIEF)))
    )

    make_generator(json_mode="json_schema").generate(transcript)

    fmt = sent_json(route)["response_format"]
    assert fmt["type"] == "json_schema"
    assert "action_items" in fmt["json_schema"]["schema"]["properties"]


def test_json_mode_off_omits_response_format(api: respx.MockRouter, transcript: Transcript) -> None:
    route = api.post(URL).mock(
        return_value=httpx.Response(200, json=chat_payload(json.dumps(VALID_BRIEF)))
    )
    make_generator(json_mode="off").generate(transcript)
    assert "response_format" not in sent_json(route)


def test_chat_request_retries_on_rate_limit(api: respx.MockRouter, transcript: Transcript) -> None:
    route = api.post(URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=chat_payload(json.dumps(VALID_BRIEF))),
        ]
    )
    make_generator().generate(transcript)
    assert route.call_count == 2


def test_empty_transcript_is_rejected_without_a_request(api: respx.MockRouter) -> None:
    route = api.post(URL)
    with pytest.raises(WhisperBriefError, match="empty"):
        make_generator().generate(Transcript(text="   "))
    assert route.call_count == 0


def test_parse_brief_accepts_code_fence() -> None:
    brief = parse_brief("```json\n" + json.dumps(VALID_BRIEF) + "\n```")
    assert brief.language == "en"


def test_parse_brief_rejects_json_array() -> None:
    with pytest.raises(BriefValidationError, match="JSON object"):
        parse_brief("[1, 2]")


def test_format_transcript_without_segments_uses_text() -> None:
    assert format_transcript(Transcript(text=" plain text ")) == "plain text"


def test_markdown_rendering() -> None:
    md = brief_to_markdown(parse_brief(json.dumps(VALID_BRIEF)))
    assert md.startswith("# Weekly sync\n")
    assert "## Action items" in md
    assert "- Write release notes (Priya)" in md
    assert "- Tag the build\n" in md
