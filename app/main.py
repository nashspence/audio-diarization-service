import asyncio
import logging
import os
import subprocess
import tempfile
import time
import wave
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse
from huggingface_hub import hf_hub_download, snapshot_download
from lightning_fabric.utilities import cloud_io as lightning_cloud_io
import torchaudio


if not hasattr(torchaudio, "AudioMetaData"):
    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = AudioMetaData


_lightning_load = lightning_cloud_io._load


def trusted_checkpoint_load(path_or_url, map_location=None, weights_only=None):
    if weights_only is None:
        weights_only = False
    return _lightning_load(path_or_url, map_location=map_location, weights_only=weights_only)


lightning_cloud_io._load = trusted_checkpoint_load


def _decode_with_soundfile(source: Any) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    return audio, sample_rate


def _decode_with_ffmpeg(source_path: str) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as decoded_file:
        decoded_path = decoded_file.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                source_path,
                decoded_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return _decode_with_soundfile(decoded_path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(decoded_path)


def compatible_torchaudio_load(
    uri: Any,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
):
    del format, buffer_size, backend, normalize

    temp_input_path: str | None = None

    try:
        try:
            audio, sample_rate = _decode_with_soundfile(uri)
        except Exception:
            if isinstance(uri, (str, os.PathLike)):
                audio, sample_rate = _decode_with_ffmpeg(os.fspath(uri))
            else:
                suffix = ".wav"
                if hasattr(uri, "name"):
                    suffix = Path(getattr(uri, "name")).suffix or suffix
                current_pos = uri.tell() if hasattr(uri, "tell") else None
                payload = uri.read()
                if current_pos is not None and hasattr(uri, "seek"):
                    uri.seek(current_pos)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
                    temp_input.write(payload)
                    temp_input_path = temp_input.name
                audio, sample_rate = _decode_with_ffmpeg(temp_input_path)

        if frame_offset or num_frames != -1:
            end_frame = None if num_frames == -1 else frame_offset + num_frames
            audio = audio[frame_offset:end_frame]

        waveform = torch.from_numpy(audio.T if channels_first else audio)
        return waveform, sample_rate
    finally:
        if temp_input_path:
            with suppress(FileNotFoundError):
                os.unlink(temp_input_path)


torchaudio.load = compatible_torchaudio_load


from diarizen.pipelines.inference import DiariZenPipeline


LOGGER = logging.getLogger("audio-diarization-service")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def round_seconds(value: float) -> float:
    return round(float(value), 3)


def normalize_speaker_label(label: Any) -> str:
    if isinstance(label, int):
        return f"speaker_{label}"

    text = str(label)
    if text.startswith("speaker_"):
        return text
    if text.isdigit():
        return f"speaker_{text}"
    return text


def make_silent_wav(duration_seconds: int, sample_rate: int) -> BytesIO:
    frames = max(duration_seconds, 1) * sample_rate
    payload = b"\x00\x00" * frames
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)
    buffer.seek(0)
    return buffer


@dataclass
class Settings:
    model_repo: str = field(default_factory=lambda: os.getenv("DIARIZEN_MODEL_REPO", "BUT-FIT/diarizen-wavlm-large-s80-md-v2"))
    diarizen_ref: str = field(default_factory=lambda: os.getenv("DIARIZEN_REF", "d52b8d5e3d96632b1a8a0dc34762bf811471e441"))
    embedding_repo: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_REPO", "pyannote/wespeaker-voxceleb-resnet34-LM"))
    embedding_filename: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_FILENAME", "pytorch_model.bin"))
    hf_home: str = field(default_factory=lambda: os.getenv("HF_HOME", "/models/hf"))
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN") or None)
    offline: bool = field(default_factory=lambda: env_bool("DIARIZEN_OFFLINE", False))
    require_cuda: bool = field(default_factory=lambda: env_bool("REQUIRE_CUDA", True))
    torch_num_threads: int = field(default_factory=lambda: int(os.getenv("TORCH_NUM_THREADS", "1")))
    warmup_duration_seconds: int = field(default_factory=lambda: int(os.getenv("DIARIZEN_WARMUP_SECONDS", "16")))
    warmup_audio_path: str = field(default_factory=lambda: os.getenv("DIARIZEN_WARMUP_AUDIO", "/opt/diarizen-src/example/EN2002a_30s.wav"))
    matmul_precision: str = field(default_factory=lambda: os.getenv("TORCH_MATMUL_PRECISION", "high"))
    enable_tf32: bool = field(default_factory=lambda: env_bool("ENABLE_TF32", True))


@dataclass
class ServiceState:
    settings: Settings
    device: str
    started_at: str = field(default_factory=utc_now)
    phase: str = "starting"
    ready: bool = False
    error: str | None = None
    loaded_at: str | None = None
    warmup_seconds: float | None = None
    pipeline: DiariZenPipeline | None = None
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    load_task: asyncio.Task[None] | None = None


def initialize_pipeline(settings: Settings) -> tuple[DiariZenPipeline, float]:
    if settings.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this service, but no GPU is available.")

    if settings.torch_num_threads > 0:
        torch.set_num_threads(settings.torch_num_threads)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(settings.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = settings.enable_tf32
        torch.backends.cudnn.allow_tf32 = settings.enable_tf32

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

    token = settings.hf_token
    cache_dir = settings.hf_home

    diarizen_hub = snapshot_download(
        repo_id=settings.model_repo,
        cache_dir=cache_dir,
        token=token,
        local_files_only=settings.offline,
    )
    embedding_model = hf_hub_download(
        repo_id=settings.embedding_repo,
        filename=settings.embedding_filename,
        cache_dir=cache_dir,
        token=token,
        local_files_only=settings.offline,
    )

    pipeline = DiariZenPipeline(
        diarizen_hub=Path(diarizen_hub).expanduser().resolve(),
        embedding_model=embedding_model,
    )

    warmup_input: Any
    if Path(settings.warmup_audio_path).is_file():
        warmup_input = settings.warmup_audio_path
    else:
        warmup_input = make_silent_wav(settings.warmup_duration_seconds, sample_rate=16000)

    warmup_started = time.perf_counter()
    with torch.inference_mode():
        pipeline(warmup_input, sess_name="warmup")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return pipeline, time.perf_counter() - warmup_started


async def load_pipeline(state: ServiceState) -> None:
    state.phase = "loading"
    state.ready = False
    state.error = None

    try:
        pipeline, warmup_seconds = await asyncio.to_thread(initialize_pipeline, state.settings)
    except Exception as exc:
        LOGGER.exception("Failed to initialize diarization pipeline.")
        state.phase = "error"
        state.error = str(exc)
        state.ready = False
        return

    state.pipeline = pipeline
    state.warmup_seconds = round_seconds(warmup_seconds)
    state.loaded_at = utc_now()
    state.phase = "ready"
    state.ready = True
    LOGGER.info("Diarization pipeline is ready on %s.", state.device)


def health_payload(state: ServiceState) -> dict[str, Any]:
    return {
        "service": "audio-diarization-service",
        "status": "ready" if state.ready else state.phase,
        "ready": state.ready,
        "model": {
            "repo_id": state.settings.model_repo,
            "upstream_ref": state.settings.diarizen_ref,
            "device": state.device,
            "loaded_at": state.loaded_at,
            "warmup_seconds": state.warmup_seconds,
        },
        "started_at": state.started_at,
        "error": state.error,
    }


def serialize_annotation(annotation: Any) -> tuple[list[dict[str, Any]], list[str]]:
    segments: list[dict[str, Any]] = []
    speakers: set[str] = set()

    for turn, track, speaker in annotation.itertracks(yield_label=True):
        speaker_label = normalize_speaker_label(speaker)
        speakers.add(speaker_label)
        segments.append(
            {
                "start_seconds": round_seconds(turn.start),
                "end_seconds": round_seconds(turn.end),
                "duration_seconds": round_seconds(turn.end - turn.start),
                "speaker": speaker_label,
                "track": str(track),
            }
        )

    segments.sort(key=lambda item: (item["start_seconds"], item["end_seconds"], item["speaker"]))
    return segments, sorted(speakers)


def run_diarization(pipeline: DiariZenPipeline, audio_path: str, session_name: str) -> Any:
    with torch.inference_mode():
        return pipeline(audio_path, sess_name=session_name)


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state = ServiceState(settings=settings, device=device)
    state.load_task = asyncio.create_task(load_pipeline(state))
    app.state.service = state

    try:
        yield
    finally:
        if state.load_task and not state.load_task.done():
            state.load_task.cancel()
            with suppress(asyncio.CancelledError):
                await state.load_task


app = FastAPI(
    title="Audio Diarization Service",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, Any]:
    state: ServiceState = app.state.service
    return {
        "service": "audio-diarization-service",
        "version": app.version,
        "ready": state.ready,
        "endpoints": ["/health/live", "/health/ready", "/v1/diarize"],
    }


@app.get("/health/live")
async def health_live() -> dict[str, Any]:
    state: ServiceState = app.state.service
    payload = health_payload(state)
    payload["status"] = "alive"
    return payload


@app.get("/health/ready")
async def health_ready() -> ORJSONResponse:
    state: ServiceState = app.state.service
    status_code = 200 if state.ready else 503
    return ORJSONResponse(status_code=status_code, content=health_payload(state))


@app.post("/v1/diarize")
async def diarize(file: UploadFile = File(...)) -> dict[str, Any]:
    state: ServiceState = app.state.service
    if not state.ready or state.pipeline is None:
        raise HTTPException(status_code=503, detail=health_payload(state))

    original_name = file.filename or "upload"
    suffix = Path(original_name).suffix or ".bin"
    session_name = Path(original_name).stem or "session"
    temp_path: str | None = None
    size_bytes = 0
    started = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                temporary_file.write(chunk)
            temp_path = temporary_file.name

        async with state.request_lock:
            annotation = await asyncio.to_thread(run_diarization, state.pipeline, temp_path, session_name)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Diarization request failed.")
        raise HTTPException(status_code=422, detail=f"Unable to diarize the uploaded audio: {exc}") from exc
    finally:
        await file.close()
        if temp_path:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)

    segments, speakers = serialize_annotation(annotation)
    max_end = max((segment["end_seconds"] for segment in segments), default=0.0)

    return {
        "model": {
            "repo_id": state.settings.model_repo,
            "upstream_ref": state.settings.diarizen_ref,
            "device": state.device,
        },
        "audio": {
            "filename": original_name,
            "content_type": file.content_type or "application/octet-stream",
            "bytes": size_bytes,
        },
        "summary": {
            "speaker_count": len(speakers),
            "segment_count": len(segments),
            "max_end_seconds": round_seconds(max_end),
            "processing_seconds": round_seconds(time.perf_counter() - started),
        },
        "speakers": speakers,
        "segments": segments,
    }
