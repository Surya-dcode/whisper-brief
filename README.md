# whisper-brief

[![CI](https://github.com/Surya-dcode/whisper-brief/actions/workflows/ci.yml/badge.svg)](https://github.com/Surya-dcode/whisper-brief/actions/workflows/ci.yml)

whisper-brief turns an audio recording into a timestamped transcript and a structured brief: a title, a summary, key points, decisions, action items with owners, and open questions. It prepares the audio with ffmpeg, transcribes it with any OpenAI-compatible Whisper endpoint (Groq by default, or a local faster-whisper model), and asks a chat model for a brief that is checked against a schema before it is written to disk.

## Pipeline

```
 input.m4a / .mp3 / .wav / .webm
          |
          v
 +-------------------+   ffmpeg: 16 kHz, mono, 16-bit PCM WAV
 |   1. normalize    |
 +-------------------+
          |
          v
 +-------------------+   ffmpeg segment muxer, N-minute chunks (default 10)
 |   2. chunk        |   chunk_0000.wav  chunk_0001.wav  chunk_0002.wav ...
 +-------------------+
          |
          v
 +-------------------+   POST /audio/transcriptions per chunk
 |   3. transcribe   |   limited concurrency, retry with backoff on 429/5xx/timeouts
 +-------------------+   (or faster-whisper on this machine with --local)
          |
          v
 +-------------------+   order by chunk index, shift timestamps by chunk offset,
 |   4. stitch       |   keep the detected language       --> transcript.json
 +-------------------+
          |
          v
 +-------------------+   POST /chat/completions in JSON mode, pydantic validation,
 |   5. brief        |   one retry with the validation error  --> brief.json
 +-------------------+                                       --> brief.md (optional)
```

## Quick start

You need Python 3.11 or newer and ffmpeg on your `PATH` (`brew install ffmpeg` on macOS, `sudo apt-get install ffmpeg` on Debian or Ubuntu).

```bash
git clone https://github.com/Surya-dcode/whisper-brief.git
cd whisper-brief
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy the example env file and add a Groq API key:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=...
```

Run it:

```bash
whisper-brief run meeting.m4a --out ./out --markdown
```

This writes `out/transcript.json`, `out/brief.json` and `out/brief.md`. Progress messages go to stderr.

Other commands:

```bash
# Transcript only
whisper-brief transcribe meeting.m4a --out ./out

# Brief from a transcript you already have, printing tokens as they arrive
whisper-brief brief out/transcript.json --stream

# Use OpenAI instead of Groq (reads OPENAI_API_KEY, uses whisper-1)
whisper-brief run meeting.m4a --provider openai

# Any other OpenAI-compatible server
whisper-brief run meeting.m4a --base-url http://localhost:8080/v1 --model my-whisper --chat-model my-llm

# Transcribe on this machine (the brief still uses the chat endpoint)
pip install -e ".[local]"
whisper-brief run meeting.m4a --local
```

`--stream` prints the brief's tokens to stdout as the model writes them. `--chunk-minutes`, `--concurrency` and `--language` adjust the transcription step. Run `whisper-brief <command> --help` to see every flag.

### HTTP API

A small FastAPI app runs the same pipeline as a service:

```bash
pip install -e ".[api]"
whisper-brief serve --port 8000

curl -F "file=@meeting.m4a" http://127.0.0.1:8000/briefs
```

`POST /briefs` takes a multipart upload in the `file` field and returns the brief JSON. `GET /health` returns `{"status": "ok"}`. Each request is handled synchronously and there is no authentication, so put it behind your own auth before exposing it.

## Example output

This is a made-up example to show the shape of `brief.json`. It is not output from a real recording.

```json
{
  "title": "Mobile app release planning",
  "summary": "The team reviewed the open bugs for the 2.4 release and agreed to ship once the login crash is fixed. Testing on older Android devices is still unassigned.",
  "key_points": [
    "The login crash only affects users who signed up with email.",
    "Push notification changes are done and tested."
  ],
  "decisions": [
    "Release 2.4 waits for the login crash fix."
  ],
  "action_items": [
    { "task": "Fix the login crash for email sign-ups", "owner": "Maya" },
    { "task": "Write the release notes", "owner": "Jordan" },
    { "task": "Test on older Android devices", "owner": null }
  ],
  "open_questions": [
    "Should the release go out on a Friday?"
  ],
  "language": "en"
}
```

`transcript.json` (also a made-up example) holds the full text, the detected language, the total duration, the chunk offsets, and every segment with `start` and `end` in seconds from the start of the recording:

```json
{
  "language": "english",
  "duration": 1834.2,
  "text": "Okay, let's start with the open bugs ...",
  "segments": [
    { "start": 0.0, "end": 3.1, "text": "Okay, let's start with the open bugs." },
    { "start": 3.1, "end": 7.8, "text": "The login crash is still the big one." }
  ],
  "chunks": [
    { "index": 0, "offset": 0.0, "duration": 600.0 },
    { "index": 1, "offset": 600.0, "duration": 600.0 },
    { "index": 2, "offset": 1200.0, "duration": 634.2 }
  ]
}
```

## Design notes

**Why 16 kHz mono.** Whisper models work on 16 kHz mono audio, so the server would resample anyway. Doing it locally first means every provider gets the same input, uploads are smaller than a stereo 44.1 kHz or 48 kHz original, and any container ffmpeg can read (m4a, mp3, webm, even a video file) becomes one known format. Uncompressed PCM also lets the chunking step cut the file without re-encoding.

**Why chunking.** Hosted transcription endpoints limit upload size (OpenAI documents 25 MB per file). Ten minutes of 16 kHz mono 16-bit PCM is about 19 MB, which fits under that limit. Chunks also let several requests run at once. Offsets come from the measured length of each chunk rather than `index * chunk_length`, so small differences at the cut points do not add up over a long recording.

**Why retries and limited concurrency.** Transcription APIs rate-limit by requests and by audio length, and they return 429 or 5xx when busy. Each request retries on 408, 429, 500, 502, 503 and 504 responses, timeouts and connection errors, waiting longer each time (exponential backoff with jitter) and honoring a `Retry-After` header when the server sends one. Other 4xx errors, such as a bad key, fail straight away because sending the same request again will not help. Concurrency is capped (3 by default) so a long recording does not open dozens of requests at once and hit the rate limit it is trying to avoid. Results are put back in chunk order no matter which request finishes first.

**Why validate the LLM output.** A chat model can return invalid JSON, wrap it in a code fence, leave out a field or use the wrong type. The reply is parsed into a pydantic model, so code that reads `brief.json` can rely on its shape. When a reply fails, the exact error (for example `summary: Field required`) goes back to the model once and it gets a second try. If that also fails, the run stops with an error that names the problem fields instead of writing a broken file. The transcript is written before the brief step, so it is kept even when the brief fails.

## Configuration

Command line flags take priority over environment variables, which take priority over the provider defaults. A `.env` file in the current directory is loaded at startup and never overrides variables that are already set.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | none | API key for the `groq` provider. |
| `OPENAI_API_KEY` | none | API key for the `openai` provider. |
| `WHISPER_BRIEF_PROVIDER` | `groq` | Provider preset: `groq` or `openai`. Same as `--provider`. |
| `WHISPER_BRIEF_API_KEY` | none | Key for any provider, used ahead of the provider-specific variable. Same as `--api-key`. |
| `WHISPER_BRIEF_BASE_URL` | provider default | Base URL of an OpenAI-compatible API. Same as `--base-url`. |
| `WHISPER_BRIEF_TRANSCRIBE_MODEL` | `whisper-large-v3-turbo` (groq), `whisper-1` (openai) | Transcription model. Same as `--model`. |
| `WHISPER_BRIEF_CHAT_MODEL` | `llama-3.3-70b-versatile` (groq), `gpt-4o-mini` (openai) | Chat model for the brief. Same as `--chat-model`. |
| `WHISPER_BRIEF_CHUNK_MINUTES` | `10` | Chunk length in minutes. Same as `--chunk-minutes`. |
| `WHISPER_BRIEF_CONCURRENCY` | `3` | Chunks transcribed at the same time. Same as `--concurrency`. |
| `WHISPER_BRIEF_MAX_RETRIES` | `4` | Retries per request after the first attempt. |
| `WHISPER_BRIEF_TIMEOUT` | `120` | Per-request timeout in seconds. |
| `WHISPER_BRIEF_JSON_MODE` | `json_object` | How to ask for JSON: `json_object`, `json_schema` (sends the brief's JSON schema), or `off` for servers that support neither. |
| `WHISPER_BRIEF_LANGUAGE` | detect | Spoken language as an ISO 639-1 code, passed to the transcription model. Same as `--language`. |
| `WHISPER_BRIEF_LOCAL_MODEL` | `small` | faster-whisper model size or path used with `--local`. |

## Running tests

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

The tests make no network calls. Every HTTP request goes through [respx](https://lundberg.github.io/respx/), and a request without a mocked route fails the test. The audio tests generate short sine-wave files with ffmpeg in a temporary directory and check the results with ffprobe; they are skipped if ffmpeg is not installed. CI runs the same steps on Python 3.11 and 3.12.

## Limitations

- No speaker diarization. The transcript does not say who said what, so action item owners only appear when a name is spoken.
- Chunks are cut at fixed times, so a word that falls on a boundary can be split or lost. Each chunk is transcribed without the text of the chunk before it.
- The whole transcript goes to the chat model in one request. A very long recording can exceed the model's context window; there is no map-reduce summarization step yet.
- Language detection is per chunk. The transcript keeps the language that covers the most audio, so a recording that switches languages gets a single label.
- Some servers do not accept a JSON `response_format` together with streaming. If `--stream` fails with a 400, set `WHISPER_BRIEF_JSON_MODE=off`.
- The local backend runs one chunk at a time, and its speed depends on your hardware and the model size.
- The HTTP API is synchronous and has no authentication or upload size limit.

## License

MIT. See [LICENSE](LICENSE).
