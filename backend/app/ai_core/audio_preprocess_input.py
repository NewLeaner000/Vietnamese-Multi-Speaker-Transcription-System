from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

import numpy as np
import soundfile as sf


# Upload/UI allow-list. The pipeline still validates by decoding audio content.
# Containers such as mp4/mov/webm are accepted only when they contain an audio stream.
SUPPORTED_AUDIO_EXTENSIONS = [
    "wav", "mp3", "flac", "m4a", "mp4", "aac", "ogg", "opus", "webm",
    "mov", "mkv", "wma", "aiff", "aif", "amr", "3gp", "mpeg", "mpg",
]


_FFMPEG_FIRST_EXTENSIONS = {
    ".mp4", ".m4a", ".mov", ".mkv", ".webm", ".aac", ".wma", ".amr", ".3gp", ".mpeg", ".mpg"
}


def _to_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float32, copy=False)
    if y.dtype == np.int16:
        return (y.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if y.dtype == np.int32:
        return (y.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if y.dtype == np.uint8:
        return ((y.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    return y.astype(np.float32, copy=False)


def _ffmpeg_executable() -> str | None:
    """Find ffmpeg executable. Fallback to hardcoded C:\\ffmpeg if PATH fails."""
    import shutil
    import os
    exe = shutil.which("ffmpeg")
    if exe is not None:
        return exe
    
    # Windows VSCode environment variable cache workaround
    possible_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg-2026-06-10-git-b29bdd3715-full_build\bin\ffmpeg.exe"
    ]
    for p in possible_paths:
        if os.path.isfile(p):
            return p
            
    return None


def _probe_with_ffprobe(audio_path: Path) -> Dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {
            "path": str(audio_path),
            "samplerate": 0,
            "channels": 0,
            "frames": 0,
            "duration_sec": 0.0,
            "format": audio_path.suffix.lower().lstrip("."),
            "subtype": "ffprobe_not_found",
        }

    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,duration:format=duration,format_name",
        "-of", "json",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams") or []
        fmt = payload.get("format") or {}
        stream = streams[0] if streams else {}

        duration_raw = stream.get("duration") or fmt.get("duration") or 0.0
        try:
            duration_sec = float(duration_raw)
        except Exception:
            duration_sec = 0.0

        try:
            samplerate = int(stream.get("sample_rate") or 0)
        except Exception:
            samplerate = 0

        try:
            channels = int(stream.get("channels") or 0)
        except Exception:
            channels = 0

        frames = int(duration_sec * samplerate) if duration_sec > 0 and samplerate > 0 else 0
        return {
            "path": str(audio_path),
            "samplerate": samplerate,
            "channels": channels,
            "frames": frames,
            "duration_sec": duration_sec,
            "format": str(fmt.get("format_name") or audio_path.suffix.lower().lstrip(".")),
            "subtype": "ffprobe",
        }
    except Exception:
        return {
            "path": str(audio_path),
            "samplerate": 0,
            "channels": 0,
            "frames": 0,
            "duration_sec": 0.0,
            "format": audio_path.suffix.lower().lstrip("."),
            "subtype": "ffprobe_error",
        }


def _normalize_with_ffmpeg(input_path: Path, output_path: Path) -> None:
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install ffmpeg or add ffmpeg/bin to PATH to decode this audio format."
        )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-map", "0:a:0",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed to convert audio: {err}")


def probe_audio_info(audio_path: str | Path) -> Dict[str, Any]:
    audio_path = Path(audio_path)
    try:
        info = sf.info(str(audio_path))
        return {
            "path": str(audio_path),
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration_sec": float(info.duration),
            "format": str(info.format),
            "subtype": str(info.subtype),
        }
    except Exception:
        return _probe_with_ffprobe(audio_path)


def _normalize_with_python_decoders(input_path: Path, output_path: Path) -> Dict[str, Any]:
    src_info = {}
    try:
        src_info = probe_audio_info(input_path)
        data, sr = sf.read(str(input_path), always_2d=False)
        original_dtype = str(np.asarray(data).dtype)
    except Exception:
        # Fallback for formats/libs soundfile cannot decode cleanly.
        # This still keeps the same normalization contract: mono / 16 kHz / PCM_16 WAV.
        import librosa
        data, sr = librosa.load(str(input_path), sr=None, mono=False)
        original_dtype = str(np.asarray(data).dtype)
        if isinstance(data, np.ndarray) and data.ndim == 1:
            channels = 1
            frames = int(data.shape[0])
        else:
            channels = int(data.shape[0])
            frames = int(data.shape[-1])
        src_info = {
            "path": str(input_path),
            "samplerate": int(sr),
            "channels": int(channels),
            "frames": int(frames),
            "duration_sec": float(frames / max(sr, 1)),
            "format": input_path.suffix.lower().lstrip("."),
            "subtype": "python_decoder",
        }

    y = np.asarray(data)

    # Convert shape to mono.
    if y.ndim == 2:
        # soundfile -> (samples, channels), librosa mono=False -> (channels, samples)
        if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
            # likely (channels, samples)
            y = np.mean(y, axis=0)
        else:
            # likely (samples, channels)
            y = np.mean(y, axis=1)

    y = _to_float32(y)

    if int(sr) != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=int(sr), target_sr=16000, res_type="kaiser_fast")
        sr = 16000

    # Final safety.
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y = np.clip(y, -1.0, 1.0)

    sf.write(str(output_path), y, 16000, subtype="PCM_16")

    dst_info = probe_audio_info(output_path)
    return {
        "source": src_info,
        "normalized": dst_info,
        "original_dtype": original_dtype,
        "changed_samplerate": int(src_info.get("samplerate", 16000)) != 16000,
        "changed_channels": int(src_info.get("channels", 1)) != 1,
        "output_path": str(output_path),
        "decoder": "python",
    }


def normalize_audio_to_mono16k(input_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    """
    Normalize any supported audio/video-with-audio input into:
    - WAV
    - mono
    - 16 kHz
    - PCM_16

    Returns a metadata dict for logging/UI.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src_info = probe_audio_info(input_path)

    # Prefer ffmpeg when available because it reliably decodes both audio files
    # and video containers that contain an audio stream. This does not change
    # downstream DER/ASR/Qwen logic; it only guarantees a stable input WAV.
    if _ffmpeg_executable():
        try:
            _normalize_with_ffmpeg(input_path, output_path)
            dst_info = probe_audio_info(output_path)
            return {
                "source": src_info,
                "normalized": dst_info,
                "original_dtype": "unknown_ffmpeg_decode",
                "changed_samplerate": int(src_info.get("samplerate", 16000) or 16000) != 16000,
                "changed_channels": int(src_info.get("channels", 1) or 1) != 1,
                "output_path": str(output_path),
                "decoder": "ffmpeg",
            }
        except Exception as ff_exc:
            # Keep a Python fallback for environments where ffmpeg exists but
            # the local build cannot handle a specific codec.
            try:
                meta = _normalize_with_python_decoders(input_path, output_path)
                meta["ffmpeg_error"] = str(ff_exc)
                return meta
            except Exception as py_exc:
                raise RuntimeError(
                    f"Cannot decode audio input '{input_path.name}'. "
                    f"FFmpeg error: {ff_exc}. Python decoder error: {py_exc}"
                ) from py_exc

    # Environment fallback: preserve the previous Python decoder behavior when
    # ffmpeg is not installed.
    try:
        return _normalize_with_python_decoders(input_path, output_path)
    except Exception as py_exc:
        raise RuntimeError(
            f"Cannot decode audio input '{input_path.name}'. "
            f"Python decoder error: {py_exc}. ffmpeg was not found in PATH."
        ) from py_exc
