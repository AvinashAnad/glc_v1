"""whisper.cpp STT provider.

Local-only demo implementation for routing `/v1/transcribe?prefer=local`
through a locally installed `whisper-cli` and GGML model.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any

from glc.voice.stt.base import STTError, STTProvider, TranscribeResult

DEFAULT_MODEL_PATH = "~/.glc/models/whisper-base/ggml-base.bin"
DEFAULT_BINARY_PATH = "~/.glc/bin/whisper-cli"
DEFAULT_LANGUAGE = "en"


class Provider(STTProvider):
    name = "whisper_cpp"

    async def transcribe(self, audio: bytes, mime: str) -> TranscribeResult:
        if _is_silent(audio):
            return TranscribeResult(
                text="",
                language=DEFAULT_LANGUAGE,
                duration_ms=0,
                provider=self.name,
                cost_usd=0.0,
            )

        mock = self.config.get("mock")
        if mock is not None:
            return await mock.transcribe(audio, mime)

        binary = _binary_path(self.config)
        model = _model_path(self.config)
        suffix = _suffix_for_mime(mime)
        no_gpu = bool(self.config.get("no_gpu", self.config.get("whisper_no_gpu", True)))

        with tempfile.TemporaryDirectory(prefix="glc-whisper-cpp-") as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / f"audio{suffix}"
            output_base = tmp_path / "transcript"
            audio_path.write_bytes(audio)

            argv = [binary, "-m", str(model), "-f", str(audio_path), "-oj", "-of", str(output_base)]
            if no_gpu:
                argv.insert(1, "-ng")

            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                detail = (stderr or stdout).decode(errors="replace").strip()
                raise STTError(f"whisper-cli failed: {detail}", status=502)

            payload = _load_json(output_base.with_suffix(".json"))
            return _result_from_json(payload)


def _binary_path(config: dict[str, Any]) -> str:
    configured = config.get("binary") or config.get("whisper_cli") or os.getenv("WHISPER_CPP_BIN")
    if configured:
        path = Path(str(configured)).expanduser()
        if path.exists():
            return str(path)
        raise STTError(f"whisper-cli not found at {path}", status=500)

    found = shutil.which("whisper-cli")
    if found:
        return found

    fallback = Path(DEFAULT_BINARY_PATH).expanduser()
    if fallback.exists():
        return str(fallback)
    raise STTError("whisper-cli is not installed or is not on PATH", status=500)


def _model_path(config: dict[str, Any]) -> Path:
    configured = config.get("model_path") or os.getenv("WHISPER_CPP_MODEL") or DEFAULT_MODEL_PATH
    path = Path(str(configured)).expanduser()
    if not path.exists():
        raise STTError(f"whisper.cpp model not found at {path}", status=500)
    return path


def _suffix_for_mime(mime: str) -> str:
    if "mpeg" in mime or "mp3" in mime:
        return ".mp3"
    if "ogg" in mime:
        return ".ogg"
    if "flac" in mime:
        return ".flac"
    return ".wav"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise STTError(f"whisper-cli did not write JSON output at {path}", status=502)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise STTError(f"whisper-cli JSON output was invalid: {e}", status=502) from e


def _result_from_json(payload: dict[str, Any]) -> TranscribeResult:
    segments = payload.get("transcription") or []
    text = " ".join(str(seg.get("text", "")).strip() for seg in segments).strip()
    language = str((payload.get("result") or {}).get("language") or DEFAULT_LANGUAGE)
    duration_ms = 0
    for seg in segments:
        offsets = seg.get("offsets") or {}
        try:
            duration_ms = max(duration_ms, int(offsets.get("to") or 0))
        except (TypeError, ValueError):
            continue
    return TranscribeResult(
        text=text,
        language=language,
        duration_ms=duration_ms,
        provider="whisper_cpp",
        cost_usd=0.0,
    )


def _is_silent(audio: bytes) -> bool:
    if not audio:
        return True
    if all(b == 0 for b in audio):
        return True
    try:
        import io

        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getsampwidth() != 2:
                return False
            frames = wav.readframes(wav.getnframes())
    except (EOFError, wave.Error):
        return False
    return _rms_16bit_pcm(frames) == 0.0


def _rms_16bit_pcm(frames: bytes) -> float:
    usable = len(frames) - (len(frames) % 2)
    if usable <= 0:
        return 0.0
    total = 0
    count = 0
    for (sample,) in struct.iter_unpack("<h", frames[:usable]):
        total += sample * sample
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(total / count)
