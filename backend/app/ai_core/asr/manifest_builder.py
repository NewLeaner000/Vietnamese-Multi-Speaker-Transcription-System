from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import soundfile as sf

from config import (
    CLASS_TO_BUCKET,
    HARD_MAX_SEC,
    KEY_CANDIDATES,
    MERGE_GAP_SEC,
    MERGE_GAP_SAME_SPK_SEC,
    MIN_KEEP_SEC,
    PAD_LEFT_SEC,
    PAD_RIGHT_SEC,
    PREFERRED_MAX_SEC,
    StageSpec,
)

LOGGER = logging.getLogger("manifest_builder")

@dataclass
class RTTMSegment:
    start: float
    end: float
    speaker: str

@dataclass
class TextSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

@dataclass
class AudioParent:
    parent_audio_id: str
    audio_path: str
    transcript_path: str
    rttm_path: Optional[str]
    split_name: str
    source_type: str
    bucket: str
    class_name: str
    group_name: str
    data_name: str
    episode_index: str
    full_duration: float
    base_utt_ids: List[str] = field(default_factory=list)

@dataclass
class SampleItem:
    sample_id: str
    parent_audio_id: str
    audio_path: str
    transcript_path: str
    split_name: str
    source_type: str
    bucket: str
    class_name: str
    group_name: str
    data_name: str
    episode_index: str
    start: float
    end: float
    duration: float
    text: str
    speaker: Optional[str]
    overlap_ratio: float
    is_robustness: bool
    parent_clean_id: Optional[str] = None
    base_utt_ids: List[str] = field(default_factory=list)

def normalize_name(name: str) -> str:
    s = name.strip().lower()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def stable_id(*parts: Any) -> str:
    joined = "||".join(str(p) for p in parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def get_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)

def overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def temporal_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = overlap_len(a0, a1, b0, b1)
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def parse_metadata_txt(path: Path) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("|")] if "|" in line else re.split(r"\s{2,}|\t+", line)
            parts = [x.strip() for x in parts if x.strip()]
            if len(parts) < 2:
                continue
            data_name = parts[0]
            class_name = parts[1]
            speakers = parts[2] if len(parts) > 2 else ""
            episode = parts[3] if len(parts) > 3 else ""
            result[data_name] = {
                "class_name": class_name,
                "speakers": speakers,
                "episode_index": episode,
            }
    return result

def parse_rttm(path: Optional[Path]) -> List[RTTMSegment]:
    if path is None or not path.exists():
        return []
    segments: List[RTTMSegment] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            segments.append(RTTMSegment(start=start, end=start + dur, speaker=parts[7]))
    return sorted(segments, key=lambda x: (x.start, x.end))

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def first_existing_key(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in row:
            return key
    return None

def parse_stm(path: Path) -> List[TextSegment]:
    out: List[TextSegment] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            speaker = parts[2]
            start = float(parts[3])
            end = float(parts[4])
            text = clean_text(" ".join(parts[6:]))
            if text and end > start:
                out.append(TextSegment(start=start, end=end, text=text, speaker=speaker))
    return out

def parse_transcript_jsonl(path: Path) -> List[TextSegment]:
    out: List[TextSegment] = []
    for row in read_jsonl(path):
        start_key = first_existing_key(row, KEY_CANDIDATES["start"])
        end_key = first_existing_key(row, KEY_CANDIDATES["end"])
        text_key = first_existing_key(row, KEY_CANDIDATES["text"])
        spk_key = first_existing_key(row, KEY_CANDIDATES["speaker"])
        if start_key is None or end_key is None or text_key is None:
            continue
        start = float(row[start_key])
        end = float(row[end_key])
        text = clean_text(row[text_key])
        speaker = str(row[spk_key]) if spk_key is not None and row.get(spk_key) is not None else None
        if text and end > start:
            out.append(TextSegment(start=start, end=end, text=text, speaker=speaker))
    return sorted(out, key=lambda x: (x.start, x.end))

def load_text_segments(labeled_dir: Path) -> Tuple[List[TextSegment], Optional[Path]]:
    candidates = [
        labeled_dir / "transcript_dataset.jsonl",
        labeled_dir / "transcript.jsonl",
        labeled_dir / "mixture.stm",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            segs = parse_transcript_jsonl(path)
        else:
            segs = parse_stm(path)
        if segs:
            return segs, path
    return [], None

def build_speech_regions(rttm_segments: List[RTTMSegment], merge_gap: float = 0.15) -> List[Tuple[float, float]]:
    if not rttm_segments:
        return []
    intervals = sorted((x.start, x.end) for x in rttm_segments)
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + merge_gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def build_overlap_regions(rttm_segments: List[RTTMSegment]) -> List[Tuple[float, float]]:
    events: List[Tuple[float, int]] = []
    for seg in rttm_segments:
        events.append((seg.start, 1))
        events.append((seg.end, -1))
    events.sort(key=lambda x: (x[0], -x[1]))
    active = 0
    prev_t: Optional[float] = None
    regions: List[Tuple[float, float]] = []
    current_start: Optional[float] = None
    for t, delta in events:
        if prev_t is not None and active >= 2 and t > prev_t and current_start is None:
            current_start = prev_t
        if prev_t is not None and active < 2 and current_start is not None and t > prev_t:
            regions.append((current_start, prev_t))
            current_start = None
        active += delta
        prev_t = t
    if current_start is not None and prev_t is not None:
        regions.append((current_start, prev_t))
    return [(s, e) for s, e in regions if e > s]

def compute_overlap_ratio(start: float, end: float, overlap_regions: List[Tuple[float, float]]) -> float:
    dur = max(1e-6, end - start)
    ov = sum(overlap_len(start, end, s, e) for s, e in overlap_regions)
    return ov / dur

def clamp_to_speech(start: float, end: float, speech_regions: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not speech_regions:
        return start, end
    best_pair = (start, end)
    best_overlap = -1.0
    for s, e in speech_regions:
        ov = overlap_len(start, end, s, e)
        if ov > best_overlap:
            best_overlap = ov
            clipped_s = max(start, s)
            clipped_e = min(end, e)
            if clipped_e > clipped_s:
                best_pair = (clipped_s, clipped_e)
    return best_pair

def merge_short_segments(segments: List[TextSegment]) -> List[TextSegment]:
    if not segments:
        return []
    result: List[TextSegment] = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        if (cur.end - cur.start) >= MIN_KEEP_SEC:
            result.append(cur)
            i += 1
            continue

        merged = False
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            gap = nxt.start - cur.end
            same_spk = cur.speaker and nxt.speaker and cur.speaker == nxt.speaker
            if gap <= MERGE_GAP_SEC and (same_spk or gap <= 0.25):
                segments[i + 1] = TextSegment(
                    start=cur.start,
                    end=nxt.end,
                    text=clean_text(cur.text + " " + nxt.text),
                    speaker=cur.speaker if same_spk else cur.speaker or nxt.speaker,
                )
                i += 1
                merged = True
        if merged:
            continue

        if result:
            prev = result[-1]
            gap = cur.start - prev.end
            same_spk = prev.speaker and cur.speaker and prev.speaker == cur.speaker
            if gap <= MERGE_GAP_SEC and (same_spk or gap <= 0.25):
                result[-1] = TextSegment(
                    start=prev.start,
                    end=cur.end,
                    text=clean_text(prev.text + " " + cur.text),
                    speaker=prev.speaker if same_spk else prev.speaker or cur.speaker,
                )
            else:
                result.append(cur)
        else:
            result.append(cur)
        i += 1
    return result

def split_long_segment(seg: TextSegment) -> List[TextSegment]:
    dur = seg.end - seg.start
    if dur <= HARD_MAX_SEC:
        return [seg]

    n_parts = math.ceil(dur / HARD_MAX_SEC)
    part_dur = dur / n_parts
    words = seg.text.split()

    step = max(1, len(words) // n_parts)
    chunks: List[TextSegment] = []
    for idx in range(n_parts):
        s = seg.start + idx * part_dur
        e = min(seg.end, s + part_dur)

        word_slice = words[idx * step:] if idx == n_parts - 1 else words[idx * step:(idx + 1) * step]
        txt = clean_text(" ".join(word_slice) or seg.text)
        if e > s and txt:
            chunks.append(TextSegment(start=s, end=e, text=txt, speaker=seg.speaker))
    return chunks

def _has_other_speaker_in_gap(
    gap_start: float,
    gap_end: float,
    speaker: Optional[str],
    rttm_segments: List[RTTMSegment],
) -> bool:
    """
    Kiểm tra trong vùng gap [gap_start, gap_end] có speaker khác đang nói không.
    Dùng để quyết định có nên merge 2 segment cùng speaker hay không.
    Nếu không có RTTM (rttm_segments rỗng) thì trả về False — cho phép merge.
    """
    if not rttm_segments or not speaker:
        return False
    for seg in rttm_segments:
        if seg.speaker == speaker:
            continue
        # Speaker khác có overlap với vùng gap
        if seg.start < gap_end and seg.end > gap_start:
            return True
    return False


def build_chunks_from_text_segments(
    text_segments: List[TextSegment],
    speech_regions: List[Tuple[float, float]],
    rttm_segments: Optional[List[RTTMSegment]] = None,
) -> List[TextSegment]:
    """
    Build chunks từ text segments.

    Merge logic:
    - Khác speaker hoặc gap ngắn (≤ 0.20s): dùng MERGE_GAP_SEC = 0.50s (giữ nguyên)
    - Cùng speaker, gap lớn hơn (≤ MERGE_GAP_SAME_SPK_SEC = 1.50s): merge có điều kiện
        Điều kiện: (1) duration sau merge ≤ PREFERRED_MAX_SEC
                   (2) không có speaker khác nói trong vùng gap theo RTTM
    """
    if not text_segments:
        return []

    rttm_segs = rttm_segments or []

    clamped: List[TextSegment] = []
    for seg in text_segments:
        s, e = clamp_to_speech(seg.start, seg.end, speech_regions)
        if e <= s:
            s, e = seg.start, seg.end
        clamped.append(TextSegment(start=s, end=e, text=seg.text, speaker=seg.speaker))

    clamped.sort(key=lambda x: (x.start, x.end))
    merged_short = merge_short_segments(clamped)

    merged: List[TextSegment] = []
    cur = merged_short[0]
    for nxt in merged_short[1:]:
        gap = nxt.start - cur.end
        same_spk = bool(cur.speaker and nxt.speaker and cur.speaker == nxt.speaker)
        candidate_dur = nxt.end - cur.start

        # ── Điều kiện merge gốc (không đổi) ──────────────────────────────────
        normal_merge = (
            gap <= MERGE_GAP_SEC
            and candidate_dur <= PREFERRED_MAX_SEC
            and (same_spk or gap <= 0.20)
        )

        # ── Điều kiện merge mở rộng: chỉ cùng speaker, gap lớn hơn ──────────
        # Ba điều kiện phải đồng thời đúng:
        #   1. Cùng speaker
        #   2. Gap trong ngưỡng mở rộng MERGE_GAP_SAME_SPK_SEC
        #   3. Duration sau merge vẫn ≤ PREFERRED_MAX_SEC
        #   4. Không có speaker khác nói trong vùng gap đó (kiểm tra RTTM)
        extended_merge = (
            same_spk
            and MERGE_GAP_SEC < gap <= MERGE_GAP_SAME_SPK_SEC
            and candidate_dur <= PREFERRED_MAX_SEC
            and not _has_other_speaker_in_gap(cur.end, nxt.start, cur.speaker, rttm_segs)
        )

        if normal_merge or extended_merge:
            if extended_merge and not normal_merge:
                LOGGER.debug(
                    "Extended same-spk merge: spk=%s gap=%.2fs [%.2f–%.2f]",
                    cur.speaker, gap, cur.end, nxt.start,
                )
            cur = TextSegment(
                start=cur.start,
                end=nxt.end,
                text=clean_text(cur.text + " " + nxt.text),
                speaker=cur.speaker if same_spk else cur.speaker or nxt.speaker,
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)

    final: List[TextSegment] = []
    for seg in merged:
        if seg.end - seg.start > HARD_MAX_SEC:
            final.extend(split_long_segment(seg))
        else:
            final.append(seg)
    return [x for x in final if x.end > x.start and x.text.strip()]

def dedup_samples(samples: List[SampleItem]) -> List[SampleItem]:

    by_parent: Dict[str, List[SampleItem]] = defaultdict(list)
    for s in samples:
        by_parent[s.parent_audio_id].append(s)

    kept_after_local: List[SampleItem] = []
    for parent_id, group in by_parent.items():
        group = sorted(group, key=lambda x: (x.start, x.end, x.sample_id))
        local_kept: List[SampleItem] = []
        for cand in group:
            is_dup = False
            for prev in local_kept:
                if clean_text(prev.text) != clean_text(cand.text):
                    continue
                if abs(prev.start - cand.start) <= 0.30 and abs(prev.end - cand.end) <= 0.30:
                    is_dup = True
                    break
                if temporal_iou(prev.start, prev.end, cand.start, cand.end) >= 0.85:
                    is_dup = True
                    break
            if not is_dup:
                local_kept.append(cand)
        kept_after_local.extend(local_kept)

    seen_fingerprints: set = set()
    kept: List[SampleItem] = []
    for s in kept_after_local:
        fp = stable_id(clean_text(s.text), round(s.duration, 1))

        fp_strict = stable_id(s.audio_path, round(s.start, 2), round(s.end, 2), clean_text(s.text))
        if fp_strict in seen_fingerprints:
            continue
        seen_fingerprints.add(fp_strict)
        kept.append(s)
    return kept

def limit_vivos_reuse(samples: List[SampleItem], max_reuse_per_base_utt: int = 2) -> List[SampleItem]:
    usage: Dict[str, int] = defaultdict(int)
    kept: List[SampleItem] = []
    for s in samples:
        if s.source_type != "vivos" or not s.base_utt_ids:
            kept.append(s)
            continue
        allow = True
        for base_id in s.base_utt_ids:
            if usage[base_id] >= max_reuse_per_base_utt:
                allow = False
                break
        if allow:
            kept.append(s)
            for base_id in s.base_utt_ids:
                usage[base_id] += 1
    return kept

def scan_split_root(
    split_root: Path,
    split_name: str,
    source_type: str,
    forced_bucket: Optional[str] = None,
    metadata_mode: str = "class",
) -> Tuple[List[AudioParent], List[SampleItem]]:
    metadata = parse_metadata_txt(split_root / "metadata.txt")
    parents: List[AudioParent] = []
    samples: List[SampleItem] = []

    for data_dir in sorted(split_root.glob("data*")):
        if not data_dir.is_dir():
            continue
        labeled_dir = data_dir / "labeled"
        audio_path = data_dir / "mixture.wav"
        if not labeled_dir.exists() or not audio_path.exists():
            continue

        data_name = data_dir.name
        info = metadata.get(data_name, {})
        raw_class_name = info.get("class_name", data_name)
        episode_index = info.get("episode_index", "")

        if metadata_mode == "overlap":
            class_name = "vivos"
            group_name = normalize_name(raw_class_name)
            bucket = forced_bucket or "vivos"
        else:
            class_name = normalize_name(raw_class_name)
            group_name = class_name
            if forced_bucket is not None:
                bucket = forced_bucket
            elif source_type == "dubbed":
                bucket = "silver_dubbed"
            elif source_type == "robustness":
                bucket = "robustness"
            else:
                bucket = CLASS_TO_BUCKET.get(class_name, "silver_real")

        text_segments, transcript_path = load_text_segments(labeled_dir)
        if not text_segments or transcript_path is None:
            LOGGER.warning("Skip %s because transcript file was not found", data_dir)
            continue

        rttm_path = labeled_dir / "mixture.rttm"
        rttm_segments = parse_rttm(rttm_path if rttm_path.exists() else None)
        speech_regions = build_speech_regions(rttm_segments)
        overlap_regions = build_overlap_regions(rttm_segments)

        full_duration = get_audio_duration(audio_path)
        parent_audio_id = stable_id(source_type, split_name, data_name, str(audio_path.resolve()))

        base_utt_ids: List[str] = []
        if source_type == "vivos" and transcript_path.suffix == ".jsonl":
            try:
                for row in read_jsonl(transcript_path):
                    for key in ["base_utt_id", "utt_id", "source_utt_id", "utt"]:
                        if key in row:
                            base_utt_ids.append(str(row[key]))
                base_utt_ids = sorted(set(base_utt_ids))
            except Exception:
                base_utt_ids = []

        parent = AudioParent(
            parent_audio_id=parent_audio_id,
            audio_path=str(audio_path),
            transcript_path=str(transcript_path),
            rttm_path=str(rttm_path) if rttm_path.exists() else None,
            split_name=split_name,
            source_type=source_type,
            bucket=bucket,
            class_name=class_name,
            group_name=group_name,
            data_name=data_name,
            episode_index=episode_index,
            full_duration=full_duration,
            base_utt_ids=base_utt_ids,
        )
        parents.append(parent)

        chunks = build_chunks_from_text_segments(text_segments, speech_regions, rttm_segments)
        for chunk in chunks:
            start = max(0.0, chunk.start - PAD_LEFT_SEC)
            end = min(full_duration, chunk.end + PAD_RIGHT_SEC)
            if end <= start:
                continue
            overlap_ratio = compute_overlap_ratio(start, end, overlap_regions)
            samples.append(
                SampleItem(
                    sample_id=stable_id(parent_audio_id, round(start, 2), round(end, 2), clean_text(chunk.text)),
                    parent_audio_id=parent_audio_id,
                    audio_path=str(audio_path),
                    transcript_path=str(transcript_path),
                    split_name=split_name,
                    source_type=source_type,
                    bucket=bucket,
                    class_name=class_name,
                    group_name=group_name,
                    data_name=data_name,
                    episode_index=episode_index,
                    start=start,
                    end=end,
                    duration=end - start,
                    text=chunk.text,
                    speaker=chunk.speaker,
                    overlap_ratio=overlap_ratio,
                    is_robustness=(source_type == "robustness"),
                    base_utt_ids=base_utt_ids,
                )
            )

    samples = dedup_samples(samples)
    if source_type == "vivos":
        samples = limit_vivos_reuse(samples, max_reuse_per_base_utt=2)
    return parents, samples

def scan_all_sources(
    self_labeled_root: str,
    dubbed_root: str = "",
    vivos_root: str = "",
    robustness_root: str = "",
) -> Dict[str, Dict[str, List[Any]]]:
    outputs: Dict[str, Dict[str, List[Any]]] = {}
    self_root = Path(self_labeled_root)

    p, s = scan_split_root(self_root / "Tong_hop_data_labelled", "train", "self_labeled")
    outputs["self_train"] = {"parents": p, "samples": s}
    p, s = scan_split_root(self_root / "val_labeled", "val", "self_labeled")
    outputs["self_val"] = {"parents": p, "samples": s}
    p, s = scan_split_root(self_root / "test_labeled", "test", "self_labeled")
    outputs["self_test"] = {"parents": p, "samples": s}

    if dubbed_root:
        p, s = scan_split_root(Path(dubbed_root), "train", "dubbed", forced_bucket="silver_dubbed")
    else:
        p, s = [], []
    outputs["dubbed_train"] = {"parents": p, "samples": s}

    if vivos_root:
        p, s = scan_split_root(Path(vivos_root), "train", "vivos", forced_bucket="vivos", metadata_mode="overlap")
    else:
        p, s = [], []
    outputs["vivos_train"] = {"parents": p, "samples": s}

    if robustness_root:
        p, s = scan_split_root(Path(robustness_root), "train", "robustness", forced_bucket="robustness")
    else:
        p, s = [], []
    outputs["robustness_train"] = {"parents": p, "samples": s}

    return outputs

def choose_parent_ids_to_target_hours(parents: List[AudioParent], target_hours: float, seed: int) -> List[str]:
    rng = random.Random(seed)
    shuffled = parents[:]
    rng.shuffle(shuffled)
    chosen: List[str] = []
    acc = 0.0
    for p in shuffled:
        if acc >= target_hours:
            break
        chosen.append(p.parent_audio_id)
        acc += p.full_duration / 3600.0

    if target_hours > 0:
        deviation = abs(acc - target_hours) / target_hours
        if deviation > 0.10:
            LOGGER.warning(
                "choose_parent_ids_to_target_hours: actual=%.2fh target=%.2fh deviation=%.0f%% — "
                "consider stratified sampling for more stable distribution.",
                acc, target_hours, deviation * 100,
            )
    return chosen

def build_stage_manifest(all_data: Dict[str, Dict[str, List[Any]]], stage: StageSpec, seed: int) -> Tuple[List[SampleItem], List[SampleItem]]:
    self_train_parents: List[AudioParent] = all_data["self_train"]["parents"]
    self_train_samples: List[SampleItem] = all_data["self_train"]["samples"]
    dubbed_parents: List[AudioParent] = all_data["dubbed_train"]["parents"]
    dubbed_samples: List[SampleItem] = all_data["dubbed_train"]["samples"]
    vivos_parents: List[AudioParent] = all_data["vivos_train"]["parents"]
    vivos_samples: List[SampleItem] = all_data["vivos_train"]["samples"]
    rob_parents: List[AudioParent] = all_data["robustness_train"]["parents"]
    rob_samples: List[SampleItem] = all_data["robustness_train"]["samples"]

    chosen_parent_ids = {p.parent_audio_id for p in self_train_parents}
    chosen_parent_ids.update(p.parent_audio_id for p in dubbed_parents)

    if stage.name == "stage2":
        chosen_parent_ids.update(p.parent_audio_id for p in rob_parents)

    base_hours = 0.0
    for p in self_train_parents:
        base_hours += p.full_duration / 3600.0
    for p in dubbed_parents:
        base_hours += p.full_duration / 3600.0
    if stage.name == "stage2":
        for p in rob_parents:
            base_hours += p.full_duration / 3600.0

    vivos_ratio = stage.source_ratios.get("vivos", 0.0)
    if vivos_ratio > 0.0 and vivos_parents:
        target = base_hours * vivos_ratio
        chosen_parent_ids.update(choose_parent_ids_to_target_hours(vivos_parents, target, seed + 13))

    merged_samples = self_train_samples + dubbed_samples + vivos_samples + rob_samples
    train_manifest = [
        s for s in merged_samples
        if s.parent_audio_id in chosen_parent_ids and s.overlap_ratio <= stage.allowed_overlap_ratio
    ]
    val_manifest = [
        s for s in all_data["self_val"]["samples"]
        if s.overlap_ratio <= max(0.35, stage.allowed_overlap_ratio)
    ]

    train_manifest = dedup_samples(train_manifest)
    val_manifest = dedup_samples(val_manifest)
    if any(x.source_type == "vivos" for x in train_manifest):
        train_manifest = limit_vivos_reuse(train_manifest, max_reuse_per_base_utt=2)
    return train_manifest, val_manifest

def export_pipeline_manifests(
    output_dir: str,
    all_data: Dict[str, Dict[str, List[Any]]],
    stage1_spec: StageSpec,
    stage2_spec: StageSpec,
    seed: int,
) -> Dict[str, Path]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    unified_rows = []
    for source_name, obj in all_data.items():
        for s in obj.get("samples", []):
            row = dataclasses.asdict(s)
            row["source_partition"] = source_name
            unified_rows.append(row)
    write_jsonl(out_root / "unified_manifest.jsonl", unified_rows)

    summary_rows = []
    for source_name, obj in all_data.items():
        parents: List[AudioParent] = obj.get("parents", [])
        samples: List[SampleItem] = obj.get("samples", [])
        summary_rows.append({
            "source_name": source_name,
            "parents": len(parents),
            "samples": len(samples),
            "hours": round(sum(p.full_duration for p in parents) / 3600.0, 3),
        })
    write_jsonl(out_root / "scan_summary.jsonl", summary_rows)

    stage1_train, stage1_val = build_stage_manifest(all_data, stage1_spec, seed)
    stage2_train, stage2_val = build_stage_manifest(all_data, stage2_spec, seed + 100)

    write_jsonl(out_root / "stage1_train_manifest.jsonl", [dataclasses.asdict(x) for x in stage1_train])
    write_jsonl(out_root / "stage1_val_manifest.jsonl", [dataclasses.asdict(x) for x in stage1_val])
    write_jsonl(out_root / "stage2_train_manifest.jsonl", [dataclasses.asdict(x) for x in stage2_train])
    write_jsonl(out_root / "stage2_val_manifest.jsonl", [dataclasses.asdict(x) for x in stage2_val])

    return {
        "unified_manifest": out_root / "unified_manifest.jsonl",
        "scan_summary": out_root / "scan_summary.jsonl",
        "stage1_train": out_root / "stage1_train_manifest.jsonl",
        "stage1_val": out_root / "stage1_val_manifest.jsonl",
        "stage2_train": out_root / "stage2_train_manifest.jsonl",
        "stage2_val": out_root / "stage2_val_manifest.jsonl",
    }