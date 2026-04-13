
# audio-diarization-service

Minimal GPU-first HTTP wrapper for offline speaker diarization with `BUT-FIT/diarizen-wavlm-large-s80-md-v2`.

## What It Does

- Starts a single FastAPI service around the upstream `DiariZenPipeline`
- Loads the model on container startup and only reports ready after load + warmup finish
- Exposes `POST /v1/diarize` for multipart audio upload
- Persists the Hugging Face cache in the repo-local `./.cache/models/huggingface` directory so subsequent runs can stay offline with `DIARIZEN_OFFLINE=true`
- Uses one process and one request at a time to keep the local GPU dedicated to diarization

## Runtime Config

All runtime settings live in `.env`, including `HF_TOKEN`.

Key settings:

- `HF_TOKEN`: Hugging Face token when needed
- `HOST_PORT`: published HTTP port
- `HF_HOME`: persistent model cache inside the container
- `DIARIZEN_MODEL_REPO`: defaults to `BUT-FIT/diarizen-wavlm-large-s80-md-v2`
- `DIARIZEN_OFFLINE`: set to `true` after the first successful pull if you want cache-only startup
- `DIARIZEN_WARMUP_AUDIO`: defaults to the upstream bundled speech sample so readiness only flips after a real diarization pass
- `REQUIRE_CUDA`: fail fast when no GPU is available
- `ENABLE_TF32`: defaults to `true` for faster dedicated-GPU inference on supported NVIDIA hardware

## Start

```bash
docker compose up -d --build
```

Readiness:

```bash
curl http://127.0.0.1:8000/health/ready
```

Inside this devcontainer, use:

```bash
curl http://host.docker.internal:8000/health/ready
```

## API

Request:

```bash
curl -X POST http://127.0.0.1:8000/v1/diarize \
  -F "file=@test.opus;type=audio/ogg"
```

Response shape:

```json
{
  "model": {
    "repo_id": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
    "upstream_ref": "d52b8d5e3d96632b1a8a0dc34762bf811471e441",
    "device": "cuda:0"
  },
  "audio": {
    "filename": "test.opus",
    "content_type": "audio/ogg",
    "bytes": 32105
  },
  "summary": {
    "speaker_count": 2,
    "segment_count": 12,
    "max_end_seconds": 29.914,
    "processing_seconds": 4.287
  },
  "speakers": ["speaker_0", "speaker_1"],
  "segments": [
    {
      "start_seconds": 0.0,
      "end_seconds": 2.731,
      "duration_seconds": 2.731,
      "speaker": "speaker_0",
      "track": "_"
    }
  ]
}
```

## Smoke Test

```bash
bash smoke-test.sh
```

The smoke test:

- builds and starts the stack
- waits for `/health/ready` and container health to turn healthy
- submits `test.opus`
- validates the returned JSON
- shuts the stack down cleanly
