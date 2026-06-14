"""
run_asr_inference.py  (faster-whisper edition)
────────────────────
Bước 1: Chạy ASR model trên từng RTTM segment.
KHÔNG load STM (ground truth) — tránh data leakage hoàn toàn.

Output: CSV với các cột:
    sample_id, audio_path, start, end, duration,
    speaker, data_name, group_name, bucket, source_type,
    overlap_ratio, predicted_text, was_retried

Engine backend: faster-whisper (CTranslate2) — 2-4× faster than
HuggingFace transformers on the same GPU.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import logging
import math
import os
import time
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

from faster_whisper import WhisperModel

# ── Shared config ────────────────────────────────────────────────────────────
from config import PAD_LEFT_SEC, PAD_RIGHT_SEC
from manifest_builder import (
    AudioParent,
    RTTMSegment,
    build_overlap_regions,
    compute_overlap_ratio,
    get_audio_duration,
    normalize_name,
    parse_metadata_txt,
    parse_rttm,
    stable_id,
    CLASS_TO_BUCKET,
)

# ── Constants ────────────────────────────────────────────────────────────────
UNKNOWN_PREFIX        = "unknown"
MIN_RTTM_DUR_SEC      = 0.5
WHISPER_MAX_SEC       = 29.0
INFERENCE_TRIM_SEC    = min(PAD_LEFT_SEC, PAD_RIGHT_SEC)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("run_asr_inference")


# ── Timing helpers ───────────────────────────────────────────────────────────

class TimingLogger:
    """Collects timing measurements and prints a summary table at the end."""

    def __init__(self):
        self.model_load_sec: float = 0.0
        self.scan_sec: float = 0.0
        self.segment_times: List[float] = []
        self.pipeline_start: float = time.perf_counter()

    def record_segment(self, elapsed: float) -> None:
        self.segment_times.append(elapsed)

    def print_summary(self, total_segments: int, retry_count: int) -> None:
        wall_clock = time.perf_counter() - self.pipeline_start
        total_inference = sum(self.segment_times)
        avg_per_seg = total_inference / max(len(self.segment_times), 1)
        min_seg = min(self.segment_times) if self.segment_times else 0.0
        max_seg = max(self.segment_times) if self.segment_times else 0.0

        LOGGER.info("=" * 65)
        LOGGER.info("  ASR PERFORMANCE SUMMARY (faster-whisper)")
        LOGGER.info("=" * 65)
        LOGGER.info("  %-30s %10.2f s", "Model load time", self.model_load_sec)
        LOGGER.info("  %-30s %10.2f s", "RTTM scan time", self.scan_sec)
        LOGGER.info("  %-30s %10d", "Total segments", total_segments)
        LOGGER.info("  %-30s %10d", "Retried segments", retry_count)
        LOGGER.info("  %-30s %10.2f s", "Total inference time", total_inference)
        LOGGER.info("  %-30s %10.3f s", "Avg per segment", avg_per_seg)
        LOGGER.info("  %-30s %10.3f s", "Min segment time", min_seg)
        LOGGER.info("  %-30s %10.3f s", "Max segment time", max_seg)
        LOGGER.info("  %-30s %10.2f s", "Total wall-clock time", wall_clock)
        LOGGER.info("=" * 65)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio_segment(
    audio_path: str,
    start: float,
    end: float,
    target_sr: int = 16000,
    trim_sec: float = INFERENCE_TRIM_SEC,
) -> np.ndarray:
    min_dur_to_trim = 2 * trim_sec + 0.10
    duration = end - start
    if trim_sec > 0 and duration > min_dur_to_trim:
        start = start + trim_sec
        end   = end   - trim_sec

    info        = sf.info(audio_path)
    start_frame = max(0, int(start * info.samplerate))
    end_frame   = int(end * info.samplerate)
    audio, sr   = sf.read(audio_path, start=start_frame, stop=end_frame)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # Resample using numpy linear interpolation (avoids torch/torchaudio dependency)
        orig_len = len(audio)
        new_len  = int(orig_len * target_sr / sr)
        indices  = np.linspace(0, orig_len - 1, new_len)
        audio    = np.interp(indices, np.arange(orig_len), audio)
    return audio.astype(np.float32)


def split_long_rttm_segments(
    rttm_segments: List[RTTMSegment],
    max_sec: float = WHISPER_MAX_SEC,
) -> List[RTTMSegment]:
    result: List[RTTMSegment] = []
    for seg in rttm_segments:
        dur = seg.end - seg.start
        if dur <= max_sec:
            result.append(seg)
            continue
        n_parts  = math.ceil(dur / max_sec)
        part_dur = dur / n_parts
        for i in range(n_parts):
            s = seg.start + i * part_dur
            e = min(seg.end, s + part_dur)
            if e > s:
                result.append(RTTMSegment(start=s, end=e, speaker=seg.speaker))
    return result


# ── Hallucination / no-speech detection ─────────────────────────────────────

def is_hallucination(text: str, duration_sec: float,
                     wps_threshold: float = 8.0) -> bool:
    words = text.split()
    if not words:
        return False
    if duration_sec > 0 and len(words) / duration_sec > wps_threshold:
        return True
    if len(words) >= 8:
        half = len(words) // 2
        if words[:half] == words[half: half * 2]:
            return True
    return False


def _check_compression_ratio(text: str, threshold: float = 2.4) -> bool:
    """Return True if text looks like a hallucination based on compression ratio."""
    encoded = text.encode("utf-8")
    if not encoded:
        return False
    comp_ratio = len(encoded) / max(len(zlib.compress(encoded)), 1)
    return comp_ratio > threshold


def transcribe_segment(
    model: WhisperModel,
    audio: np.ndarray,
    duration_sec: float,
    *,
    no_speech_threshold: float = 0.6,
    compression_ratio_threshold: float = 2.4,
    wps_threshold: float = 8.0,
) -> Tuple[str, bool]:
    """
    Transcribe a single audio segment using faster-whisper.

    faster-whisper natively supports:
    - no_speech_prob filtering
    - compression_ratio_threshold for hallucination detection
    - temperature fallback (tries greedy first, then sampling on hallucination)

    Returns (text, was_retried).
    """
    # First pass: greedy decoding (temperature=0)
    segments_iter, info = model.transcribe(
        audio,
        language="vi",
        task="transcribe",
        beam_size=5,
        best_of=5,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=0.0,
        vad_filter=False,  # We already have RTTM segments — no need for VAD
    )

    # Collect all sub-segments from the iterator
    text_parts = []
    for seg in segments_iter:
        text_parts.append(seg.text.strip())
    text = " ".join(text_parts).strip()

    # Check for no speech
    if not text:
        return "", False

    # Check for hallucination with our own logic (same as original engine)
    hallucinated = _check_compression_ratio(text, compression_ratio_threshold)
    if not hallucinated:
        hallucinated = is_hallucination(text, duration_sec, wps_threshold)

    if not hallucinated:
        return text, False

    # Retry with sampling (temperature=0.4) — same as original engine's retry_temperature
    segments_iter2, _ = model.transcribe(
        audio,
        language="vi",
        task="transcribe",
        beam_size=1,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=0.4,
        vad_filter=False,
    )

    text_parts2 = []
    for seg in segments_iter2:
        text_parts2.append(seg.text.strip())
    text2 = " ".join(text_parts2).strip()

    if is_hallucination(text2, duration_sec, wps_threshold):
        return "", True
    return text2, True


# ── Scan RTTM-only ───────────────────────────────────────────────────────────

def scan_rttm_only(
    split_root: Path,
    source_type: str = "custom_test",
    skip_unknown_speaker: bool = False,
) -> List[dict]:
    metadata = parse_metadata_txt(split_root / "metadata.txt")
    rows: List[dict] = []

    for data_dir in sorted(split_root.glob("data*")):
        if not data_dir.is_dir():
            continue
        labeled_dir = data_dir / "labeled"
        audio_path  = data_dir / "mixture.wav"
        if not labeled_dir.exists() or not audio_path.exists():
            continue

        data_name      = data_dir.name
        info           = metadata.get(data_name, {})
        raw_class_name = info.get("class_name", data_name)
        episode_index  = info.get("episode_index", "")
        class_name     = normalize_name(raw_class_name)
        bucket         = CLASS_TO_BUCKET.get(class_name, "silver_real")

        rttm_candidates = sorted(labeled_dir.glob("*.rttm"))
        rttm_hyp        = [p for p in rttm_candidates if p.stem.startswith("hyp")]
        rttm_path       = (rttm_hyp or rttm_candidates or [None])[0]
        if rttm_path is None:
            LOGGER.warning("%s: không tìm thấy *.rttm, bỏ qua.", data_dir)
            continue

        rttm_segments = parse_rttm(rttm_path)
        if not rttm_segments:
            continue

        rttm_segments   = split_long_rttm_segments(rttm_segments)
        overlap_regions = build_overlap_regions(rttm_segments)
        full_duration   = get_audio_duration(audio_path)

        parent_audio_id = stable_id(
            source_type, "test", data_name, str(audio_path.resolve())
        )

        for seg in rttm_segments:
            if seg.end - seg.start < MIN_RTTM_DUR_SEC:
                continue
            if skip_unknown_speaker and seg.speaker.startswith(UNKNOWN_PREFIX):
                continue

            start = max(0.0, seg.start - PAD_LEFT_SEC)
            end   = min(full_duration, seg.end + PAD_RIGHT_SEC)
            if end <= start:
                continue

            overlap_ratio = compute_overlap_ratio(start, end, overlap_regions)
            sample_id     = stable_id(
                parent_audio_id, round(start, 2), round(end, 2)
            )

            rows.append({
                "sample_id":       sample_id,
                "parent_audio_id": parent_audio_id,
                "audio_path":      str(audio_path),
                "rttm_path":       str(rttm_path),
                "data_name":       data_name,
                "class_name":      class_name,
                "group_name":      class_name,
                "bucket":          bucket,
                "source_type":     source_type,
                "episode_index":   episode_index,
                "split_name":      "test",
                "speaker":         seg.speaker,
                "rttm_start":      round(seg.start, 4),
                "rttm_end":        round(seg.end,   4),
                "start":           round(start, 4),
                "end":             round(end,   4),
                "duration":        round(end - start, 4),
                "overlap_ratio":   round(overlap_ratio, 4),
            })

    return rows


# ── Export JSON cho chatbot ───────────────────────────────────────────────────

def export_chatbot_json(
    out_rows: list,
    out_csv_path,
    retry_count: int,
    model_name: str,
    split_root,
) -> None:
    """
    Ghi thêm file JSON song song với CSV.
    Format: { "meta": {...}, "segments": [...] }
    """
    json_path = Path(str(out_csv_path)).with_suffix(".json")
    chatbot_segments = [
        {
            "speaker":       row["speaker"],
            "start":         row["start"],
            "end":           row["end"],
            "duration":      row["duration"],
            "text":          row["predicted_text"],
            "overlap_ratio": row["overlap_ratio"],
            "was_retried":   row["was_retried"],
        }
        for row in out_rows
        if row["predicted_text"]
    ]
    chatbot_json = {
        "meta": {
            "total_segments":  len(out_rows),
            "valid_segments":  len(chatbot_segments),
            "retry_count":     retry_count,
            "model":           str(model_name),
            "source_dir":      str(split_root),
            "engine":          "faster-whisper (CTranslate2)",
        },
        "segments": chatbot_segments,
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(chatbot_json, jf, ensure_ascii=False, indent=2)
    LOGGER.info("Chatbot JSON: %s (%d segments)", json_path, len(chatbot_segments))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Chạy ASR inference trên RTTM segments (faster-whisper). KHÔNG dùng STM."
    )
    parser.add_argument("--dir",          required=True)
    parser.add_argument("--model",        required=True)
    parser.add_argument("--out_csv",      required=True)
    parser.add_argument("--skip_unknown", action="store_true")
    parser.add_argument("--trim_sec",     type=float, default=INFERENCE_TRIM_SEC)
    parser.add_argument("--wps_threshold",type=float, default=8.0)
    parser.add_argument("--comp_ratio",   type=float, default=2.4)
    parser.add_argument("--retry_temp",   type=float, default=0.4)
    parser.add_argument("--compute_type", type=str,   default="float16",
                        choices=["float16", "int8_float16", "int8", "float32"],
                        help="CTranslate2 compute type (default: float16)")
    args = parser.parse_args()

    if args.trim_sec > PAD_LEFT_SEC:
        LOGGER.warning(
            "--trim_sec=%.3fs > PAD_LEFT_SEC=%.3fs → giảm về PAD_LEFT_SEC.",
            args.trim_sec, PAD_LEFT_SEC,
        )
        args.trim_sec = PAD_LEFT_SEC

    timer = TimingLogger()

    # ── Load model ───────────────────────────────────────────────────────────
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu"
    # Auto-detect CUDA availability
    try:
        import ctranslate2
        if not ctranslate2.get_cuda_device_count():
            device = "cpu"
    except Exception:
        pass

    LOGGER.info("Load faster-whisper model từ %s lên %s (compute_type=%s) ...",
                args.model, device, args.compute_type)

    t0 = time.perf_counter()
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=args.compute_type,
    )
    timer.model_load_sec = time.perf_counter() - t0
    LOGGER.info("Model loaded in %.2f s", timer.model_load_sec)

    # ── Scan RTTM segments ───────────────────────────────────────────────────
    split_root = Path(args.dir)
    LOGGER.info("Quét RTTM tại: %s", split_root)

    t0 = time.perf_counter()
    rows = scan_rttm_only(split_root, skip_unknown_speaker=args.skip_unknown)
    timer.scan_sec = time.perf_counter() - t0

    if not rows:
        LOGGER.error("Không tìm thấy segment nào. Kiểm tra lại --dir.")
        return
    LOGGER.info("Tìm thấy %d RTTM segments (scan took %.2f s).", len(rows), timer.scan_sec)

    # ── Inference ────────────────────────────────────────────────────────────
    retry_count = 0
    out_rows    = []

    for idx, row in enumerate(tqdm(rows, desc="ASR Inference (faster-whisper)")):
        seg_t0 = time.perf_counter()

        audio = load_audio_segment(
            row["audio_path"],
            row["start"],
            row["end"],
            trim_sec=args.trim_sec,
        )
        duration_sec = row["end"] - row["start"]

        pred_text, was_retried = transcribe_segment(
            model, audio, duration_sec,
            compression_ratio_threshold=args.comp_ratio,
            wps_threshold=args.wps_threshold,
        )
        if was_retried:
            retry_count += 1

        seg_elapsed = time.perf_counter() - seg_t0
        timer.record_segment(seg_elapsed)

        out_rows.append({**row, "predicted_text": pred_text, "was_retried": was_retried})

        # Progress reporting for Streamlit UI
        print(f"PROGRESS:{idx + 1}/{len(rows)}", flush=True)

    # ── Ghi CSV ──────────────────────────────────────────────────────────────
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id", "parent_audio_id", "audio_path", "rttm_path",
        "data_name", "class_name", "group_name", "bucket", "source_type",
        "episode_index", "split_name", "speaker",
        "rttm_start", "rttm_end",
        "start", "end", "duration",
        "overlap_ratio",
        "predicted_text", "was_retried",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    LOGGER.info("Ghi %d dòng vào: %s", len(out_rows), out_path)
    LOGGER.info(
        "Retry: %d / %d (%.1f%%)",
        retry_count, len(out_rows),
        retry_count / max(len(out_rows), 1) * 100,
    )
    LOGGER.info(
        "CONFIG — trim_sec=%.3fs | wps=%.1f | comp_ratio=%.2f | compute_type=%s",
        args.trim_sec, args.wps_threshold, args.comp_ratio, args.compute_type,
    )

    # ── Performance summary ──────────────────────────────────────────────────
    timer.print_summary(len(out_rows), retry_count)

    # ── Ghi JSON cho chatbot ─────────────────────────────────────────────────
    export_chatbot_json(out_rows, out_path, retry_count, args.model, split_root)


if __name__ == "__main__":
    main()