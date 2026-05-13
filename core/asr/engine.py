"""
run_asr_inference.py
────────────────────
Bước 1: Chạy ASR model trên từng RTTM segment.
KHÔNG load STM (ground truth) — tránh data leakage hoàn toàn.

Output: CSV với các cột:
    sample_id, audio_path, start, end, duration,
    speaker, data_name, group_name, bucket, source_type,
    overlap_ratio, predicted_text, was_retried
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import logging
import math
import os
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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
MIN_RTTM_DUR_SEC      = 0.5          # bỏ RTTM segment quá ngắn
WHISPER_MAX_SEC       = 29.0         # cắt segment dài trước khi feed Whisper
INFERENCE_TRIM_SEC    = min(PAD_LEFT_SEC, PAD_RIGHT_SEC)  # trim nhẹ tại inference

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("run_asr_inference")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio_segment(
    audio_path: str,
    start: float,
    end: float,
    target_sr: int = 16000,
    trim_sec: float = INFERENCE_TRIM_SEC,
) -> np.ndarray:
    """
    Đọc đoạn [start, end] từ file audio (start/end đã bao gồm PAD).
    Trim nhẹ vào phía trong để Whisper không nghe ké câu liền kề.
    Bất biến: trim_sec <= min(PAD_LEFT_SEC, PAD_RIGHT_SEC).
    """
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
        wav   = torch.from_numpy(audio).float().unsqueeze(0)
        wav   = torchaudio.functional.resample(wav, sr, target_sr)
        audio = wav.squeeze(0).numpy()
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


def _is_no_speech(ids_list: list, no_speech_token_id: int) -> bool:
    """Check <|nospeech|> chỉ ở vị trí đầu sequence (index 1-3)."""
    return no_speech_token_id in ids_list[1:4]


def decode_with_retry(
    model, processor, input_features, attention_mask,
    duration_sec: float,
    no_speech_token_id: int,
    compression_ratio_threshold: float = 2.4,
    retry_temperature: float = 0.4,
    wps_threshold: float = 8.0,
) -> Tuple[str, bool]:
    """
    Greedy decode → kiểm tra hallucination → retry với temperature nếu cần.
    Trả về (pred_text, was_retried).
    """
    gen_kwargs = dict(
        language="vi",
        task="transcribe",
        max_length=256,
        attention_mask=attention_mask,
    )

    with torch.no_grad():
        ids = model.generate(input_features, **gen_kwargs)
    ids_list = ids[0].tolist()

    if _is_no_speech(ids_list, no_speech_token_id):
        return "", False  # silence thật

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    hallucinated = False
    encoded = text.encode("utf-8")
    if encoded:
        comp_ratio = len(encoded) / max(len(zlib.compress(encoded)), 1)
        if comp_ratio > compression_ratio_threshold:
            hallucinated = True
    if not hallucinated:
        hallucinated = is_hallucination(text, duration_sec, wps_threshold)
    if not hallucinated:
        return text, False

    # Retry
    with torch.no_grad():
        ids2 = model.generate(
            input_features, **gen_kwargs,
            do_sample=True, temperature=retry_temperature,
        )
    text2 = processor.batch_decode(ids2, skip_special_tokens=True)[0]
    if is_hallucination(text2, duration_sec, wps_threshold):
        return "", True  # vẫn hallucinate → bỏ
    return text2, True


# ── Scan RTTM-only (không đụng STM) ─────────────────────────────────────────

def scan_rttm_only(
    split_root: Path,
    source_type: str = "custom_test",
    skip_unknown_speaker: bool = False,
) -> List[dict]:
    """
    Duyệt thư mục data*, đọc RTTM để lấy boundary segment.
    KHÔNG load STM / transcript. Text = "" (sẽ điền sau khi decode ASR).

    Trả về list[dict] với các field đủ để chạy inference và sau đó đánh giá.
    """
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

        # ── Load RTTM (prediction) — không STM ──────────────────────────────
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
                # transcript_path sẽ KHÔNG có ở đây — tránh leakage
                "data_name":       data_name,
                "class_name":      class_name,
                "group_name":      class_name,
                "bucket":          bucket,
                "source_type":     source_type,
                "episode_index":   episode_index,
                "split_name":      "test",
                "speaker":         seg.speaker,
                "rttm_start":      round(seg.start, 4),   # RTTM boundary gốc
                "rttm_end":        round(seg.end,   4),   # (để eval align lại với STM)
                "start":           round(start, 4),        # sau khi pad (dùng để load audio)
                "end":             round(end,   4),
                "duration":        round(end - start, 4),
                "overlap_ratio":   round(overlap_ratio, 4),
                # predicted_text điền bên dưới sau khi inference
            })

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Chạy ASR inference trên RTTM segments. KHÔNG dùng STM."
    )
    parser.add_argument("--dir",           required=True, help="Thư mục chứa data* (có mixture.wav + labeled/*.rttm)")
    parser.add_argument("--model",         required=True, help="Đường dẫn tới Whisper model / adapter")
    parser.add_argument("--out_csv",       required=True, help="Đường dẫn CSV output chứa predictions")
    parser.add_argument("--skip_unknown",  action="store_true", help="Bỏ segment speaker unknown_*")
    parser.add_argument("--trim_sec",      type=float, default=INFERENCE_TRIM_SEC,
                        help=f"Trim mỗi đầu segment tại inference (giây). Mặc định={INFERENCE_TRIM_SEC:.2f}s")
    parser.add_argument("--wps_threshold",    type=float, default=8.0)
    parser.add_argument("--comp_ratio",       type=float, default=2.4)
    parser.add_argument("--retry_temp",       type=float, default=0.4)
    args = parser.parse_args()

    # Validate trim
    if args.trim_sec > PAD_LEFT_SEC:
        LOGGER.warning(
            "--trim_sec=%.3fs > PAD_LEFT_SEC=%.3fs → giảm về PAD_LEFT_SEC.",
            args.trim_sec, PAD_LEFT_SEC,
        )
        args.trim_sec = PAD_LEFT_SEC

    # ── Load model ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Load model từ %s lên %s ...", args.model, device)
    processor = WhisperProcessor.from_pretrained(args.model)
    model     = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()
    no_speech_token_id = processor.tokenizer.convert_tokens_to_ids("<|nospeech|>")
    LOGGER.info("no_speech_token_id = %d", no_speech_token_id)

    # ── Scan RTTM (không STM) ────────────────────────────────────────────────
    split_root = Path(args.dir)
    LOGGER.info("Quét RTTM tại: %s", split_root)
    rows = scan_rttm_only(
        split_root,
        skip_unknown_speaker=args.skip_unknown,
    )
    if not rows:
        LOGGER.error("Không tìm thấy segment nào. Kiểm tra lại --dir.")
        return
    LOGGER.info("Tìm thấy %d RTTM segments.", len(rows))

    # ── Inference ────────────────────────────────────────────────────────────
    retry_count = 0
    out_rows    = []

    for row in tqdm(rows, desc="ASR Inference"):
        audio = load_audio_segment(
            row["audio_path"],
            row["start"],
            row["end"],
            trim_sec=args.trim_sec,
        )
        duration_sec   = row["end"] - row["start"]
        inputs         = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        pred_text, was_retried = decode_with_retry(
            model, processor, input_features, attention_mask,
            duration_sec        = duration_sec,
            no_speech_token_id  = no_speech_token_id,
            compression_ratio_threshold = args.comp_ratio,
            retry_temperature   = args.retry_temp,
            wps_threshold       = args.wps_threshold,
        )
        if was_retried:
            retry_count += 1

        out_rows.append({**row, "predicted_text": pred_text, "was_retried": was_retried})

    # ── Ghi CSV ──────────────────────────────────────────────────────────────
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id", "parent_audio_id", "audio_path", "rttm_path",
        "data_name", "class_name", "group_name", "bucket", "source_type",
        "episode_index", "split_name", "speaker",
        "rttm_start", "rttm_end",    # RTTM boundary gốc — eval script dùng để align STM
        "start", "end", "duration",  # boundary sau pad — dùng để load audio
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
        "CONFIG — trim_sec=%.3fs | wps=%.1f | comp_ratio=%.2f | retry_temp=%.2f",
        args.trim_sec, args.wps_threshold, args.comp_ratio, args.retry_temp,
    )


if __name__ == "__main__":
    main()
