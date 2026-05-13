#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Qwen pipeline for DER + ASR transcripts.

Goals:
- Reduce Qwen summary latency on RTX 4060/4080 Laptop 8GB
- Preserve the existing file flow and output schema
- Keep output stable for demo / final defense use

Main optimizations:
- Deterministic generation by default
- Rule normalize by default, LLM normalize only for very noisy blocks
- Score-based block routing: only high-value blocks go through LLM
- Dynamic max_new_tokens per block
- Much rarer JSON repair pass
- Rule reducer by default
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.serialization

# ---- PyTorch compatibility -------------------------------------------------
_orig_torch_load = torch.serialization.load
def _safe_load(f, *args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(f, *args, **kwargs)
torch.serialization.load = _safe_load
torch.load = _safe_load

sys.stdout.reconfigure(encoding="utf-8")

try:
    import accelerate.utils.memory  # type: ignore
    if not hasattr(accelerate.utils.memory, "clear_device_cache"):
        accelerate.utils.memory.clear_device_cache = lambda **kwargs: None
except Exception:
    pass

try:
    import transformers  # type: ignore
    if not hasattr(transformers, "EncoderDecoderCache"):
        transformers.EncoderDecoderCache = type("EncoderDecoderCache", (object,), {})
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False

# ---- constants -------------------------------------------------------------
BLOCK_SEGMENT_TYPES = [
    "progress_update",
    "technical_discussion",
    "planning",
    "issue_report",
    "presentation_prep",
    "formatting_review",
    "other",
]

TECH_TERMS = [
    "qwen", "whisper", "phowhisper", "asr", "der", "diarization", "diarizen",
    "streamlit", "website", "web", "venv", "checkpoint", "reference",
    "huggingface", "model", "format", "image", "plot", "build", "pyannote",
    "eend", "rttm", "enrollment", "embedding", "inference", "batch", "gpu",
    "cuda", "vram", "4bit", "lora", "adapter", "pipeline", "json", "csv",
    "jsonl", "transcript", "audio", "wav", "mp3", "segment", "block",
    "overlap", "speaker", "dataset", "data", "loss", "metric", "evaluation",
    "finetune", "training", "epoch", "install", "import", "error", "exception",
    "traceback", "fix", "bug", "obs", "javascript", "js", "react", "fastapi",
    "latex", "citation", "4090", "4080", "4060",
]

ACTION_PATTERNS = [
    r"\bfix\b", r"\bsửa\b", r"\bchỉnh\b", r"\bbuild\b", r"\bnghiên cứu\b",
    r"\bkiểm tra\b", r"\bchạy lại\b", r"\btổng hợp\b", r"\blabel\b",
    r"\bghi âm\b", r"\bthu âm\b", r"\btriển khai\b", r"\bupdate\b",
    r"\bpush\b", r"\bcommit\b", r"\btest\b", r"\breference\b", r"\btrích dẫn\b",
]

ISSUE_PATTERNS = [
    r"\blỗi\b", r"\berror\b", r"\bbug\b", r"\bconflict\b", r"\btraceback\b",
    r"\bkhông chạy\b", r"\bmất code\b", r"\bkhó chịu\b", r"\bbuild lại\b",
    r"\bthiếu\b", r"\bbị đề\b", r"\bbị lặp\b", r"\bchậm\b",
]

FORMAT_PATTERNS = [
    r"\bformat\b", r"\bhình ảnh\b", r"\btiêu đề\b", r"\bxuống hàng\b",
    r"\breference\b", r"\btrích dẫn\b", r"\blatex\b", r"\bslide\b",
    r"\bbản\b", r"\bthông số\b", r"\bplot\b",
]

TRANSITION_PATTERNS = [
    r"quý vị thân mến.*quay trở lại",
    r"chúng tôi sẽ quay trở lại",
    r"nghỉ giải lao",
    r"mời bạn vào",
    r"xin mời.*khách mời",
    r"chúng ta cùng đến với",
    r"trò chơi",
    r"game",
    r"quảng cáo",
]

FILLERS = {"ờ", "ờm", "ừm", "à", "ạ", "dạ", "ha", "hả", "ơ", "ừ"}
HALLUCINATION_EXACT = {"xin chào và hẹn gặp lại", "hẹn gặp lại", "google"}

BRACKET_LINE_RE = re.compile(
    r"^\[(?P<start>[\d\.]+)s\s*-\s*(?P<end>[\d\.]+)s\]\s*(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)

NORMALIZE_SYSTEM_PROMPT = """\
Bạn là bộ chuẩn hóa transcript tiếng Việt cho dữ liệu ASR nhiều người nói.

Mục tiêu:
- Chỉ sửa lỗi nhẹ và bỏ lặp vô nghĩa do ASR.
- Không tóm tắt, không suy diễn, không thêm ý mới.
- Giữ nguyên speaker, start, end.
- Giữ nguyên số dòng.
- Trả về JSON hợp lệ duy nhất.

Schema:
{"normalized_lines":[{"start":0.0,"end":1.0,"speaker":"speaker_01","clean_text":"..."}]}
"""

LOCAL_EXTRACT_SYSTEM_PROMPT = """\
Bạn đang xử lý 1 block transcript kỹ thuật ngắn.

Mục tiêu:
- Trích xuất đúng ý, cụ thể, trung thực
- Không dùng câu generic
- Không bịa
- Chỉ dùng thông tin có trong block

segment_type chỉ được chọn 1 trong:
["progress_update","technical_discussion","planning","issue_report","presentation_prep","formatting_review","other"]

Schema bắt buộc:
{"block_id":0,"segment_type":"technical_discussion","overview":"1-2 câu ngắn, cụ thể.","key_points":["..."],"action_items":[{"owner":"speaker_01|null","task":"...","status":"open|in_progress|done"}],"speaker_insights":[{"speaker":"speaker_01","role_in_block":"reported|asked|proposed|clarified","detail":"..."}],"facts":[{"type":"model|tool|issue|decision|metric|data|ui|reference","value":"..."}]}

Ràng buộc:
- overview luôn là string
- key_points, action_items, speaker_insights, facts luôn là list
- Nếu không có action item thì trả []
- Không trả markdown, không giải thích
"""

REDUCER_SYSTEM_PROMPT = """\
Bạn là bộ gộp block summary đã trích xuất trước đó.

Đầu vào là các block summary JSON. Chỉ dùng thông tin trong các block đó.
Không thêm fact mới.

Schema:
{"meeting_overview":"2-4 câu rõ ràng.","conversation_main_summary":"1-2 câu nói đúng chủ đề chính.","meeting_type":"technical_sync|presentation_prep|debug_discussion|mixed","segments":[{"title":"...","block_ids":[0,1],"summary":"..."}],"action_items":[{"owner":"speaker_01|null","task":"...","status":"open"}],"speaker_insights":[{"speaker":"speaker_01","insight":"...","evidence_spans":[[0.0,1.0]]}],"speaker_main_summaries":[{"speaker":"speaker_01","main_summary":"...","evidence_spans":[[0.0,1.0]],"confidence":"high|medium|low"}],"risk_flags":[],"quality_notes":[]}
"""

REPAIR_JSON_PROMPT = """\
Sửa output sau thành JSON hợp lệ theo schema yêu cầu. Không thêm fact mới. Chỉ trả JSON.
"""

# ---- dataclasses -----------------------------------------------------------
@dataclass
class PipelineConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: Optional[str] = None
    load_in_4bit: bool = True
    enable_thinking: bool = False

    max_input_tokens: int = 2304
    normalize_max_new_tokens: int = 128
    local_summary_max_new_tokens: int = 128
    reducer_max_new_tokens: int = 0
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 4
    do_sample: bool = False

    merge_gap_sec: float = 1.0
    block_max_chars: int = 2600
    block_max_duration_sec: float = 60.0
    block_max_utterances: int = 8
    min_text_chars: int = 2
    keep_single_word_fillers: bool = False

    smart_skip_normalize: bool = True
    smart_skip_summary: bool = True
    prefer_rule_normalize: bool = True
    use_rule_reducer: bool = True

    # new optimization knobs
    max_llm_blocks_stable: int = 12
    max_llm_blocks_fast: int = 8
    min_llm_score_stable: int = 3
    min_llm_score_fast: int = 4
    repair_max_new_tokens_stable: int = 64
    repair_max_new_tokens_fast: int = 48


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str
    source_idx: int


@dataclass
class Block:
    block_id: int
    start: float
    end: float
    segments: List[Segment]
    is_transition_heavy: bool

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def text(self) -> str:
        return "\n".join(
            f"[{s.start:.1f}s - {s.end:.1f}s] {s.speaker}: {s.text}" for s in self.segments
        )

# ---- text utils ------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def cleanup_repeated_sentences(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    # repeated adjacent token
    toks = text.split()
    out = [toks[0]] if toks else []
    for tok in toks[1:]:
        if tok.lower() == out[-1].lower():
            continue
        out.append(tok)
    text = " ".join(out)
    # repeated sentence fragments
    parts = [normalize_whitespace(x) for x in re.split(r"(?<=[\.\?!])\s+", text) if normalize_whitespace(x)]
    seen = set()
    dedup = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return " ".join(dedup).strip()


def clean_segment_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"\b(ờm|ừm|ờ)\b", "", text, flags=re.IGNORECASE)
    text = text.replace(" ,", ",").replace(" .", ".")
    text = re.sub(r"([,\.\?!])\1{1,}", r"\1", text)
    text = cleanup_repeated_sentences(text)
    return normalize_whitespace(text)


def maybe_is_filler(text: str) -> bool:
    txt = normalize_whitespace(text).lower().strip(" .,!?:;-")
    return txt in FILLERS


def looks_like_transition(text: str) -> bool:
    t = normalize_whitespace(text).lower()
    return any(re.search(p, t) for p in TRANSITION_PATTERNS)


def estimate_noise_score(text: str) -> int:
    score = 0
    low = normalize_whitespace(text).lower()
    if re.search(r"\b(ờ|ờm|à|ừm|dạ)\b", low):
        score += 1
    if re.search(r"([a-zA-ZÀ-ỹ]+)(\s+\1){2,}", low, flags=re.IGNORECASE):
        score += 1
    if len(re.findall(r"[!?\.]{2,}", text)) > 0:
        score += 1
    if any(h in low for h in HALLUCINATION_EXACT):
        score += 2
    return score

# ---- JSON utils ------------------------------------------------------------
def safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [x for x in (first_obj, first_arr) if x != -1]
    if starts:
        text = text[min(starts):]
    last_obj = text.rfind("}")
    last_arr = text.rfind("]")
    ends = [x for x in (last_obj, last_arr) if x != -1]
    if ends:
        text = text[: max(ends) + 1]
    return json.loads(text)


def sanitize_text(text: Any) -> str:
    return cleanup_repeated_sentences(normalize_whitespace(str(text or "")))

# ---- transcript loading ----------------------------------------------------
def load_transcript_jsonl(path: str) -> List[Segment]:
    segments: List[Segment] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            speaker = str(item.get("speaker", item.get("label", "Unknown")))
            text = clean_segment_text(item.get("text", item.get("content", "")))
            if text:
                segments.append(Segment(start=start, end=end, speaker=speaker, text=text, source_idx=idx))
    segments.sort(key=lambda s: (s.start, s.end, s.source_idx))
    return segments


def load_transcript_text(path: str) -> List[Segment]:
    segments: List[Segment] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            m = BRACKET_LINE_RE.match(line)
            if not m:
                continue
            segments.append(
                Segment(
                    start=float(m.group("start")),
                    end=float(m.group("end")),
                    speaker=clean_segment_text(m.group("speaker")),
                    text=clean_segment_text(m.group("text")),
                    source_idx=idx,
                )
            )
    segments = [s for s in segments if s.text]
    segments.sort(key=lambda s: (s.start, s.end, s.source_idx))
    return segments


def load_transcript(path: str) -> List[Segment]:
    return load_transcript_jsonl(path) if path.lower().endswith(".jsonl") else load_transcript_text(path)


def filter_segments(segments: List[Segment], cfg: PipelineConfig) -> List[Segment]:
    out: List[Segment] = []
    for s in segments:
        text = clean_segment_text(s.text)
        if len(text) < cfg.min_text_chars:
            continue
        if (not cfg.keep_single_word_fillers) and maybe_is_filler(text):
            continue
        out.append(Segment(s.start, s.end, s.speaker, text, s.source_idx))
    return out


def merge_adjacent_segments(segments: List[Segment], cfg: PipelineConfig) -> List[Segment]:
    if not segments:
        return []
    merged = [segments[0]]
    for s in segments[1:]:
        prev = merged[-1]
        gap = max(0.0, s.start - prev.end)
        same_speaker = s.speaker == prev.speaker
        should_merge = same_speaker and gap <= cfg.merge_gap_sec
        if should_merge:
            merged[-1] = Segment(
                start=prev.start,
                end=max(prev.end, s.end),
                speaker=prev.speaker,
                text=clean_segment_text(prev.text + " " + s.text),
                source_idx=prev.source_idx,
            )
        else:
            merged.append(s)
    return merged


def split_into_blocks(segments: List[Segment], cfg: PipelineConfig) -> List[Block]:
    if not segments:
        return []
    blocks: List[Block] = []
    cur: List[Segment] = []
    cur_chars = 0
    cur_start = segments[0].start
    transition_heavy = False

    def flush(cur_segments: List[Segment], transition_flag: bool) -> None:
        if not cur_segments:
            return
        blocks.append(
            Block(
                block_id=len(blocks),
                start=cur_segments[0].start,
                end=cur_segments[-1].end,
                segments=list(cur_segments),
                is_transition_heavy=transition_flag,
            )
        )

    for s in segments:
        seg_text = f"[{s.start:.1f}s - {s.end:.1f}s] {s.speaker}: {s.text}\n"
        seg_chars = len(seg_text)
        current_duration = 0.0 if not cur else (s.end - cur_start)
        is_transition = looks_like_transition(s.text)

        force_new = False
        if cur and is_transition and current_duration >= 25:
            force_new = True
        if cur and (cur_chars + seg_chars > cfg.block_max_chars):
            force_new = True
        if cur and (current_duration > cfg.block_max_duration_sec):
            force_new = True
        if cur and len(cur) >= cfg.block_max_utterances:
            force_new = True

        if force_new:
            flush(cur, transition_heavy)
            cur = []
            cur_chars = 0
            cur_start = s.start
            transition_heavy = False

        cur.append(s)
        cur_chars += seg_chars
        transition_heavy = transition_heavy or is_transition

    flush(cur, transition_heavy)
    return blocks

# ---- heuristics ------------------------------------------------------------
def detect_tech_tags(lines: List[Dict[str, Any]]) -> List[str]:
    text = " ".join(normalize_whitespace(x.get("clean_text", x.get("text", ""))).lower() for x in lines)
    found: List[str] = []
    for term in TECH_TERMS:
        t = term.lower()
        if len(t) <= 4:
            if re.search(rf"\b{re.escape(t)}\b", text):
                found.append(term)
        elif t in text:
            found.append(term)
    return list(dict.fromkeys(found))


def detect_action_hints(lines: List[Dict[str, Any]]) -> bool:
    text = " ".join(normalize_whitespace(x.get("clean_text", x.get("text", ""))).lower() for x in lines)
    return any(re.search(p, text) for p in ACTION_PATTERNS)


def infer_segment_type_from_text(text_blob: str, is_transition_heavy: bool) -> str:
    low = normalize_whitespace(text_blob).lower()
    if any(re.search(p, low) for p in FORMAT_PATTERNS):
        return "formatting_review"
    if any(re.search(p, low) for p in ISSUE_PATTERNS):
        return "issue_report"
    if any(k in low for k in ["kế hoạch", "triển khai", "sẽ", "cần", "phải", "ghi âm", "thu âm", "label", "tổng hợp lại"]):
        return "planning"
    if any(k in low for k in ["đã làm", "đang làm", "xong rồi", "bữa giờ", "tiến độ", "đánh giá", "checkpoint"]):
        return "progress_update"
    if any(k in low for k in [
        "streamlit", "website", "venv", "qwen", "whisper", "checkpoint", "model",
        "asr", "der", "pyannote", "diarization", "huggingface", "embedding",
        "rttm", "json", "csv", "enrollment", "gpu", "cuda",
    ]):
        return "technical_discussion"
    if is_transition_heavy:
        return "presentation_prep"
    return "other"


def block_to_lines(block: Block) -> List[Dict[str, Any]]:
    return [{"start": s.start, "end": s.end, "speaker": s.speaker, "clean_text": s.text} for s in block.segments]


def compute_block_value_score(block: Block, lines: List[Dict[str, Any]]) -> int:
    text_blob = " ".join(x.get("clean_text", "") for x in lines)
    tech_tags = detect_tech_tags(lines)
    has_actions = detect_action_hints(lines)
    speakers = len({x["speaker"] for x in lines})
    chars = sum(len(x.get("clean_text", "")) for x in lines)

    score = 0
    score += min(len(tech_tags), 4)
    score += 3 if has_actions else 0
    score += 2 if any(re.search(p, text_blob.lower()) for p in ISSUE_PATTERNS) else 0
    score += 2 if any(re.search(p, text_blob.lower()) for p in FORMAT_PATTERNS) else 0
    score += 1 if chars >= 700 else 0
    score += 1 if chars >= 1100 else 0
    score += 1 if speakers >= 2 else 0
    score += 1 if block.duration >= 30 else 0
    score -= 2 if block.is_transition_heavy else 0
    if chars <= 220 and speakers <= 1 and not has_actions and len(tech_tags) <= 1:
        score -= 2
    return score


def build_rule_overview(segment_type: str, lines: List[Dict[str, Any]]) -> str:
    tags = detect_tech_tags(lines)
    tags_text = ", ".join(tags[:4])
    if segment_type == "technical_discussion":
        return f"Đoạn trao đổi kỹ thuật về {tags_text}." if tags_text else "Đoạn trao đổi kỹ thuật."
    if segment_type == "issue_report":
        return f"Đoạn nêu lỗi hoặc vướng mắc liên quan đến {tags_text}." if tags_text else "Đoạn nêu lỗi hoặc vướng mắc."
    if segment_type == "progress_update":
        return f"Đoạn cập nhật tiến độ liên quan đến {tags_text}." if tags_text else "Đoạn cập nhật tiến độ."
    if segment_type == "planning":
        return f"Đoạn bàn kế hoạch triển khai liên quan đến {tags_text}." if tags_text else "Đoạn bàn kế hoạch triển khai."
    if segment_type == "formatting_review":
        return "Đoạn rà soát format, hình ảnh, trình bày hoặc reference."
    if segment_type == "presentation_prep":
        return "Đoạn chuẩn bị thuyết trình hoặc demo."
    return "Đoạn trao đổi khác."


def extract_rule_action_items(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for x in lines:
        text = normalize_whitespace(x.get("clean_text", x.get("text", "")))
        low = text.lower()
        if len(text) < 8:
            continue
        if any(re.search(p, low) for p in ACTION_PATTERNS):
            owner = x.get("speaker") or None
            status = "open"
            if any(k in low for k in ["đang", "đợi", "build lại", "fix lại"]):
                status = "in_progress"
            if any(k in low for k in ["xong", "đã", "ok rồi"]):
                status = "done"
            items.append({
                "owner": owner,
                "task": text,
                "status": status,
                "evidence_span": [x.get("start"), x.get("end")],
            })
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for it in items:
        key = (str(it.get("owner")).lower(), str(it.get("task")).lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped[:6]


def build_rule_speaker_insights(lines: List[Dict[str, Any]], segment_type: str) -> List[Dict[str, Any]]:
    by_speaker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for x in lines:
        by_speaker[str(x.get("speaker", "unknown"))].append(x)

    insights: List[Dict[str, Any]] = []
    for speaker, items in by_speaker.items():
        text_blob = " ".join(normalize_whitespace(i.get("clean_text", i.get("text", ""))) for i in items)
        detail = build_rule_overview(segment_type, [{"clean_text": text_blob}])
        role = "reported"
        if "?" in text_blob or any(k in text_blob.lower() for k in ["ủa", "vậy", "sao", "thế", "đúng không"]):
            role = "asked"
        elif any(k in text_blob.lower() for k in ["cần", "sẽ", "phải", "để", "nên"]):
            role = "proposed"
        insights.append({
            "speaker": speaker,
            "role_in_block": role,
            "detail": detail,
            "evidence_spans": [[items[0]["start"], items[-1]["end"]]],
        })
    return insights[:6]


def summarize_block_rule(normalized_block: Dict[str, Any]) -> Dict[str, Any]:
    lines = normalized_block.get("normalized_lines", [])
    text_blob = " ".join(x.get("clean_text", "") for x in lines)
    seg_type = infer_segment_type_from_text(text_blob, bool(normalized_block.get("is_transition_heavy", False)))
    overview = build_rule_overview(seg_type, lines)
    facts: List[Dict[str, Any]] = [{"type": "tool", "value": tag} for tag in detect_tech_tags(lines)[:6]]
    action_items = extract_rule_action_items(lines)
    speaker_insights = build_rule_speaker_insights(lines, seg_type)
    key_points = [overview]
    if action_items:
        key_points.append(f"Có {len(action_items)} việc cần xử lý hoặc theo dõi.")
    return {
        "block_id": normalized_block["block_id"],
        "segment_type": seg_type,
        "overview": overview,
        "key_points": key_points[:4],
        "action_items": action_items,
        "speaker_insights": speaker_insights,
        "facts": facts,
        "notes": ["rule_summary"],
    }

# ---- model runner ----------------------------------------------------------
class QwenRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_t0 = time.perf_counter()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()
        self.load_time_sec = time.perf_counter() - load_t0
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_reserved() / 1024**2
            print(f"[QWEN-PROFILE] model_load_sec={self.load_time_sec:.2f} | peak_vram_mb={peak:.1f}", flush=True)

    def _load_tokenizer(self):
        tok = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        return tok

    def _load_model(self):
        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True,
        }
        if self.cfg.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            )
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = "sdpa"

        try:
            model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **kwargs)

        if self.cfg.adapter_path:
            if not _HAS_PEFT:
                raise RuntimeError("peft chưa được cài nhưng bạn đã truyền --adapter_path")
            model = PeftModel.from_pretrained(model, self.cfg.adapter_path)
            if hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
        return model

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_input_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.cfg.do_sample,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.cfg.repetition_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
        )
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = self.cfg.temperature
            gen_kwargs["top_p"] = self.cfg.top_p

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)
        elapsed = time.perf_counter() - t0
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_reserved() / 1024**2
            print(f"[QWEN-PROFILE] generate_sec={elapsed:.3f} | max_new_tokens={max_new_tokens} | peak_vram_mb={peak:.1f}", flush=True)

        input_len = inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        try:
            return safe_json_loads(text)
        except Exception as e:
            raise RuntimeError(f"Model không trả về JSON hợp lệ. Raw output:\n{text}\n\nParse error: {e}") from e

# ---- schema validation -----------------------------------------------------
def build_overview_from_fields(segment_type: str, key_points: List[str], facts: List[Dict[str, Any]], action_items: List[Dict[str, Any]]) -> str:
    tags = [sanitize_text(x.get("value", "")) for x in facts if isinstance(x, dict) and x.get("value")]
    tags = [x for x in tags if x]
    if key_points:
        base = sanitize_text(key_points[0])
        if base:
            return base
    if tags:
        return build_rule_overview(segment_type, [{"clean_text": " ".join(tags)}])
    if action_items:
        return build_rule_overview(segment_type, [{"clean_text": " ".join(x.get("task", "") for x in action_items)}])
    return ""


def validate_and_fix_block_summary(data: Any, block_id: int) -> Tuple[Dict[str, Any], bool]:
    severe = False
    if not isinstance(data, dict):
        data = {}
        severe = True

    segment_type = data.get("segment_type")
    if segment_type not in BLOCK_SEGMENT_TYPES:
        segment_type = "other"

    overview = data.get("overview", "")
    if not isinstance(overview, str):
        if isinstance(overview, list):
            overview = " ".join(str(x) for x in overview if isinstance(x, (str, int, float)))
        else:
            overview = ""

    key_points_raw = data.get("key_points", [])
    if not isinstance(key_points_raw, list):
        key_points_raw = [key_points_raw]
    key_points: List[str] = []
    for kp in key_points_raw:
        if isinstance(kp, dict):
            kp = kp.get("text") or kp.get("content") or kp.get("detail") or ""
        kp = sanitize_text(kp)
        if kp:
            key_points.append(kp)
    key_points = key_points[:5]

    actions_raw = data.get("action_items", [])
    if not isinstance(actions_raw, list):
        actions_raw = []
    action_items: List[Dict[str, Any]] = []
    for it in actions_raw:
        if not isinstance(it, dict):
            continue
        task = sanitize_text(it.get("task", ""))
        if not task:
            continue
        owner = it.get("owner")
        status = str(it.get("status", "open"))
        if status not in {"open", "in_progress", "done"}:
            status = "open"
        action_items.append({"owner": owner, "task": task, "status": status})

    insights_raw = data.get("speaker_insights", [])
    if isinstance(insights_raw, dict):
        insights_raw = [{"speaker": k, "detail": v} for k, v in insights_raw.items()]
    if not isinstance(insights_raw, list):
        insights_raw = []
    speaker_insights: List[Dict[str, Any]] = []
    for it in insights_raw:
        if not isinstance(it, dict):
            continue
        speaker = sanitize_text(it.get("speaker", ""))
        detail = sanitize_text(it.get("detail", ""))
        role = str(it.get("role_in_block", "reported"))
        if role not in {"reported", "asked", "proposed", "clarified"}:
            role = "reported"
        if speaker and detail:
            speaker_insights.append({"speaker": speaker, "role_in_block": role, "detail": detail})

    facts_raw = data.get("facts", [])
    if not isinstance(facts_raw, list):
        facts_raw = []
    facts: List[Dict[str, Any]] = []
    for it in facts_raw:
        if isinstance(it, dict):
            fact_type = str(it.get("type", "tool"))
            value = sanitize_text(it.get("value", ""))
        else:
            fact_type = "tool"
            value = sanitize_text(it)
        if value:
            facts.append({"type": fact_type, "value": value})
    facts = facts[:8]

    overview = sanitize_text(overview)
    if not overview:
        overview = build_overview_from_fields(segment_type, key_points, facts, action_items)

    if not overview and not key_points and not facts and not action_items and not speaker_insights:
        severe = True

    fixed = {
        "block_id": block_id,
        "segment_type": segment_type,
        "overview": overview,
        "key_points": key_points,
        "action_items": action_items,
        "speaker_insights": speaker_insights,
        "facts": facts,
    }
    return fixed, severe


def normalized_lines_match_input(clean_lines: List[Dict[str, Any]], block: Block, tol: float = 0.11) -> bool:
    if len(clean_lines) != len(block.segments):
        return False
    for out, inp in zip(clean_lines, block.segments):
        try:
            if abs(float(out["start"]) - float(inp.start)) > tol:
                return False
            if abs(float(out["end"]) - float(inp.end)) > tol:
                return False
            if str(out["speaker"]) != str(inp.speaker):
                return False
        except Exception:
            return False
    return True

# ---- normalize / summarize -------------------------------------------------
def should_skip_llm_normalize(block: Block, cfg: PipelineConfig) -> bool:
    if cfg.prefer_rule_normalize:
        return estimate_noise_score(block.text) < 4
    if not cfg.smart_skip_normalize:
        return False
    return estimate_noise_score(block.text) <= 1 and len(block.text) <= 1400


def normalize_block_rule(block: Block) -> Dict[str, Any]:
    clean_lines: List[Dict[str, Any]] = []
    for s in block.segments:
        text = clean_segment_text(s.text)
        low = text.lower()
        if low in HALLUCINATION_EXACT:
            text = ""
        clean_lines.append({
            "start": s.start,
            "end": s.end,
            "speaker": s.speaker,
            "clean_text": normalize_whitespace(text),
        })
    return {
        "block_id": block.block_id,
        "start": block.start,
        "end": block.end,
        "is_transition_heavy": block.is_transition_heavy,
        "normalized_lines": clean_lines,
        "notes": ["rule_normalized"],
    }


def normalize_block_llm(runner: QwenRunner, block: Block, cfg: PipelineConfig) -> Dict[str, Any]:
    lines = block_to_lines(block)
    user_prompt = json.dumps({"normalized_lines": lines}, ensure_ascii=False)
    try:
        result = runner.generate_json(NORMALIZE_SYSTEM_PROMPT, user_prompt, cfg.normalize_max_new_tokens)
        clean_lines = result.get("normalized_lines", [])
        if not clean_lines or not normalized_lines_match_input(clean_lines, block):
            return normalize_block_rule(block)
        fixed = []
        for row in clean_lines:
            fixed.append({
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": str(row["speaker"]),
                "clean_text": clean_segment_text(str(row.get("clean_text", ""))),
            })
        return {
            "block_id": block.block_id,
            "start": block.start,
            "end": block.end,
            "is_transition_heavy": block.is_transition_heavy,
            "normalized_lines": fixed,
            "notes": ["llm_normalized"],
        }
    except Exception:
        return normalize_block_rule(block)


def normalize_block(runner: Optional[QwenRunner], block: Block, cfg: PipelineConfig) -> Dict[str, Any]:
    if runner is None or should_skip_llm_normalize(block, cfg):
        return normalize_block_rule(block)
    return normalize_block_llm(runner, block, cfg)


def should_skip_llm_summary(normalized_block: Dict[str, Any], cfg: PipelineConfig) -> bool:
    if not cfg.smart_skip_summary:
        return False
    lines = normalized_block.get("normalized_lines", [])
    total_chars = sum(len(x.get("clean_text", "")) for x in lines)
    tech_tags = detect_tech_tags(lines)
    has_actions = detect_action_hints(lines)
    if len(lines) <= 2 and total_chars <= 220 and len(tech_tags) <= 1 and not has_actions:
        return True
    if bool(normalized_block.get("is_transition_heavy", False)) and total_chars <= 420:
        return True
    return False


def dynamic_summary_tokens(normalized_block: Dict[str, Any], cfg: PipelineConfig, qwen_mode: str) -> int:
    lines = normalized_block.get("normalized_lines", [])
    chars = sum(len(x.get("clean_text", "")) for x in lines)
    tech_tags = len(detect_tech_tags(lines))
    has_actions = detect_action_hints(lines)

    if qwen_mode == "fast":
        if chars <= 260 and tech_tags <= 1 and not has_actions:
            return 72
        if chars <= 600 and tech_tags <= 2 and not has_actions:
            return 96
        if chars <= 1100 and tech_tags <= 4:
            return 112
        return min(cfg.local_summary_max_new_tokens, 128)
    else:
        if chars <= 260 and tech_tags <= 1 and not has_actions:
            return 80
        if chars <= 600 and tech_tags <= 2 and not has_actions:
            return 104
        if chars <= 1100 and tech_tags <= 4:
            return 128
        return min(cfg.local_summary_max_new_tokens, 144)


def summarize_block_llm(runner: QwenRunner, normalized_block: Dict[str, Any], cfg: PipelineConfig, qwen_mode: str, perf: Dict[str, Any]) -> Dict[str, Any]:
    lines = normalized_block.get("normalized_lines", [])
    block_id = normalized_block["block_id"]
    text = "\n".join(
        f"[{x['start']:.1f}s - {x['end']:.1f}s] {x['speaker']}: {x['clean_text']}" for x in lines
    )
    user_obj = {
        "block_id": block_id,
        "time": f"{normalized_block['start']:.1f}s - {normalized_block['end']:.1f}s",
        "candidate_tags": detect_tech_tags(lines)[:10],
        "has_action_hints": detect_action_hints(lines),
        "lines": text,
    }
    user_prompt = json.dumps(user_obj, ensure_ascii=False)
    max_new_tokens = dynamic_summary_tokens(normalized_block, cfg, qwen_mode)

    try:
        result = runner.generate_json(LOCAL_EXTRACT_SYSTEM_PROMPT, user_prompt, max_new_tokens)
        fixed, severe = validate_and_fix_block_summary(result, block_id)
        # accept first-pass result if it is usable
        if fixed["overview"] or fixed["key_points"] or fixed["facts"] or fixed["action_items"]:
            fixed["notes"] = ["llm_extract"]
            return fixed
        if not severe:
            fixed["notes"] = ["llm_extract_soft_fixed"]
            return fixed
        raise ValueError("severe invalid llm block summary")
    except Exception as e1:
        perf["summary_repair_calls"] = perf.get("summary_repair_calls", 0) + 1
        try:
            repair_payload = {
                "block_id": block_id,
                "segment_type": infer_segment_type_from_text(text, bool(normalized_block.get("is_transition_heavy", False))),
                "raw_error": str(e1)[:300],
                "lines": text[:1800],
            }
            repair_tokens = cfg.repair_max_new_tokens_fast if qwen_mode == "fast" else cfg.repair_max_new_tokens_stable
            repaired = runner.generate_json(REPAIR_JSON_PROMPT, json.dumps(repair_payload, ensure_ascii=False), repair_tokens)
            fixed2, severe2 = validate_and_fix_block_summary(repaired, block_id)
            if fixed2["overview"] or fixed2["key_points"] or fixed2["facts"] or fixed2["action_items"]:
                fixed2["notes"] = ["llm_extract_repaired"]
                return fixed2
            if not severe2:
                fixed2["notes"] = ["llm_extract_repaired_soft"]
                return fixed2
        except Exception:
            pass
        return summarize_block_rule(normalized_block)


def dedup_action_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("owner", None)).strip().lower(),
            str(item.get("task", None)).strip().lower(),
        )
        if key in seen or not item.get("task"):
            continue
        seen.add(key)
        out.append(item)
    return out


def segment_title_from_type(seg_type: str) -> str:
    mapping = {
        "progress_update": "Cập nhật tiến độ",
        "technical_discussion": "Trao đổi kỹ thuật",
        "planning": "Lập kế hoạch triển khai",
        "issue_report": "Báo lỗi / vướng mắc",
        "presentation_prep": "Chuẩn bị thuyết trình / demo",
        "formatting_review": "Rà soát format / tài liệu",
        "other": "Nội dung khác",
    }
    return mapping.get(seg_type, "Nội dung khác")

# ---- reducer ---------------------------------------------------------------
def reduce_summaries_rule(block_summaries: List[Dict[str, Any]], normalized_blocks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    normalized_blocks = normalized_blocks or []
    grouped_segments: List[Dict[str, Any]] = []
    current_type: Optional[str] = None
    current_block_ids: List[int] = []
    current_overviews: List[str] = []
    all_action_items: List[Dict[str, Any]] = []
    type_counter: Counter[str] = Counter()

    def flush_group():
        nonlocal current_type, current_block_ids, current_overviews
        if not current_block_ids or current_type is None:
            return
        grouped_segments.append({
            "title": segment_title_from_type(current_type),
            "block_ids": current_block_ids[:],
            "summary": cleanup_repeated_sentences(" ".join(current_overviews)),
            "_seg_type": current_type,
        })
        current_type = None
        current_block_ids = []
        current_overviews = []

    for bs in block_summaries:
        bid = int(bs.get("block_id", -1))
        seg_type = bs.get("segment_type") or "other"
        if seg_type not in BLOCK_SEGMENT_TYPES:
            seg_type = "other"
        type_counter[seg_type] += 1
        overview = sanitize_text(bs.get("overview", "")) or build_rule_overview(seg_type, [])
        if current_type is None:
            current_type = seg_type
        if seg_type != current_type or (current_block_ids and bid != current_block_ids[-1] + 1):
            flush_group()
            current_type = seg_type
        current_block_ids.append(bid)
        current_overviews.append(overview)
        all_action_items.extend(bs.get("action_items", []))
    flush_group()

    if type_counter["issue_report"] > 0:
        meeting_type = "debug_discussion"
    elif type_counter["presentation_prep"] > 0 or type_counter["formatting_review"] > 0:
        meeting_type = "presentation_prep"
    elif type_counter["technical_discussion"] > 0 or type_counter["progress_update"] > 0 or type_counter["planning"] > 0:
        meeting_type = "technical_sync"
    else:
        meeting_type = "mixed"

    type_phrase_map = {
        "technical_discussion": "trao đổi kỹ thuật",
        "progress_update": "cập nhật tiến độ",
        "planning": "kế hoạch triển khai",
        "issue_report": "xử lý lỗi và vướng mắc",
        "presentation_prep": "chuẩn bị demo hoặc trình bày",
        "formatting_review": "rà soát tài liệu và format",
        "other": "các trao đổi khác",
    }
    top_phrases = [type_phrase_map.get(t, t) for t, _ in type_counter.most_common(3)]
    meeting_overview = "Cuộc trao đổi tập trung vào " + ", ".join(top_phrases[:3]) + "."
    if grouped_segments:
        details = [g["summary"] for g in grouped_segments[:2] if sanitize_text(g["summary"])]
        if details:
            meeting_overview = cleanup_repeated_sentences(meeting_overview + " " + " ".join(details))
    conversation_main_summary = "Nội dung chính tập trung vào " + ", ".join(top_phrases[:2]) + "."

    per_speaker_lines: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for nb in normalized_blocks:
        for ln in nb.get("normalized_lines", []):
            per_speaker_lines[str(ln.get("speaker", "unknown"))].append(ln)

    speaker_insights: List[Dict[str, Any]] = []
    speaker_main_summaries: List[Dict[str, Any]] = []
    risk_flags: List[str] = []
    if len(type_counter) > 1:
        risk_flags.append("Transcript có nhiều segment/chủ đề; cần chú ý tránh gộp nhầm.")

    for speaker, lines in sorted(per_speaker_lines.items()):
        text_blob = " ".join(normalize_whitespace(x.get("clean_text", "")) for x in lines)
        local_type = infer_segment_type_from_text(text_blob, False)
        tags = detect_tech_tags(lines)
        spans = [[float(lines[0]["start"]), float(lines[-1]["end"])]] if lines else []
        primary = type_phrase_map.get(local_type, "trao đổi chuyên môn")
        if tags:
            main_summary = f"Người này chủ yếu trao đổi về {primary}, nổi bật với các nội dung như {', '.join(tags[:3])}."
            confidence = "high" if len(lines) >= 4 else "medium"
        else:
            main_summary = f"Người này chủ yếu tham gia phần {primary}."
            confidence = "medium" if len(lines) >= 3 else "low"
        insight = f"Tham gia chủ yếu ở phần {primary}."
        if tags:
            insight = f"Tham gia trao đổi về {', '.join(tags[:3])}."
        speaker_insights.append({"speaker": speaker, "insight": insight, "evidence_spans": spans})
        speaker_main_summaries.append({
            "speaker": speaker,
            "main_summary": main_summary,
            "evidence_spans": spans,
            "confidence": confidence,
        })

    return {
        "meeting_overview": meeting_overview,
        "conversation_main_summary": conversation_main_summary,
        "meeting_type": meeting_type,
        "segments": [{"title": g["title"], "block_ids": g["block_ids"], "summary": g["summary"]} for g in grouped_segments],
        "action_items": dedup_action_items(all_action_items),
        "speaker_insights": speaker_insights,
        "speaker_main_summaries": speaker_main_summaries,
        "risk_flags": risk_flags,
        "quality_notes": ["optimized_rule_reducer_v3"],
    }


def reduce_summaries_llm(runner: QwenRunner, block_summaries: List[Dict[str, Any]], cfg: PipelineConfig) -> Dict[str, Any]:
    packed = json.dumps(block_summaries, ensure_ascii=False, indent=2)
    return runner.generate_json(REDUCER_SYSTEM_PROMPT, packed, cfg.reducer_max_new_tokens)


def reduce_summaries(runner: Optional[QwenRunner], block_summaries: List[Dict[str, Any]], normalized_blocks: List[Dict[str, Any]], cfg: PipelineConfig) -> Dict[str, Any]:
    if cfg.use_rule_reducer or runner is None or cfg.reducer_max_new_tokens <= 0:
        return reduce_summaries_rule(block_summaries, normalized_blocks)
    try:
        out = reduce_summaries_llm(runner, block_summaries, cfg)
        if "meeting_overview" not in out or "segments" not in out:
            raise ValueError("bad reducer output")
        return out
    except Exception:
        return reduce_summaries_rule(block_summaries, normalized_blocks)

# ---- final output ----------------------------------------------------------
def build_cleaned_transcript_output(normalized_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for block in normalized_blocks:
        for line in block.get("normalized_lines", []):
            text = sanitize_text(line.get("clean_text", ""))
            if not text:
                continue
            rows.append({
                "block_id": block["block_id"],
                "start": line["start"],
                "end": line["end"],
                "speaker": line["speaker"],
                "text": text,
            })
    rows.sort(key=lambda x: (x["start"], x["end"], x["block_id"]))
    return rows


def build_final_output(source_path: str, blocks: List[Block], normalized_blocks: List[Dict[str, Any]], block_summaries: List[Dict[str, Any]], reduced: Dict[str, Any], perf: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": source_path,
        "stats": {
            "num_blocks": len(blocks),
            "num_normalized_blocks": len(normalized_blocks),
            "num_block_summaries": len(block_summaries),
        },
        "meeting_overview": reduced.get("meeting_overview", ""),
        "conversation_main_summary": reduced.get("conversation_main_summary", reduced.get("meeting_overview", "")),
        "meeting_type": reduced.get("meeting_type", "other"),
        "segments": reduced.get("segments", []),
        "action_items": reduced.get("action_items", []),
        "speaker_insights": reduced.get("speaker_insights", []),
        "speaker_main_summaries": reduced.get("speaker_main_summaries", []),
        "risk_flags": reduced.get("risk_flags", []),
        "quality_notes": reduced.get("quality_notes", []),
        "cleaned_transcript": build_cleaned_transcript_output(normalized_blocks),
        "debug": {
            "blocks": [
                {
                    "block_id": b.block_id,
                    "start": b.start,
                    "end": b.end,
                    "duration": b.duration,
                    "is_transition_heavy": b.is_transition_heavy,
                    "num_segments": len(b.segments),
                }
                for b in blocks
            ],
            "block_summaries": block_summaries,
            "perf": perf,
        },
    }


def export_markdown(final_data: Dict[str, Any], out_path: str) -> None:
    def fmt_span(span: Any) -> str:
        if isinstance(span, list) and len(span) == 2:
            try:
                return f"[{float(span[0]):.1f}s - {float(span[1]):.1f}s]"
            except Exception:
                return str(span)
        return str(span)

    lines: List[str] = []
    lines.append("# 1. TÓM TẮT NỘI DUNG CHÍNH CỦA CUỘC HỘI THOẠI\n")
    lines.append(final_data.get("conversation_main_summary", "") or final_data.get("meeting_overview", "") or "Không đủ bằng chứng để tóm tắt nội dung chính.")

    lines.append("\n# 2. TÓM TẮT THEO TỪNG NGƯỜI NÓI\n")
    speaker_main = final_data.get("speaker_main_summaries", [])
    if not speaker_main:
        lines.append("Không đủ bằng chứng để tóm tắt theo từng người nói.")
    else:
        for item in speaker_main:
            lines.append(f"- **{item.get('speaker', 'unknown')}**: {item.get('main_summary', '')}")
            if item.get("confidence"):
                lines.append(f"  - confidence: {item.get('confidence')}")
            spans = item.get("evidence_spans", [])
            if spans:
                lines.append(f"  - evidence_spans: {', '.join(fmt_span(x) for x in spans)}")

    lines.append("\n# 3. TỔNG QUAN CUỘC HỌP\n")
    lines.append(final_data.get("meeting_overview", "") or "Không đủ bằng chứng để kết luận tổng quan.")

    lines.append("\n# 4. ACTION ITEMS\n")
    action_items = final_data.get("action_items", [])
    if not action_items:
        lines.append("Không có action item rõ ràng.")
    else:
        for item in action_items:
            lines.append(f"- owner: {item.get('owner', None)}")
            lines.append(f"  - task: {item.get('task', None)}")
            lines.append(f"  - status: {item.get('status', 'open')}")

    lines.append("\n# 5. SPEAKER INSIGHTS\n")
    insights = final_data.get("speaker_insights", [])
    if not insights:
        lines.append("Không đủ bằng chứng để trích xuất speaker insights.")
    else:
        for item in insights:
            lines.append(f"- **{item.get('speaker', 'unknown')}**: {item.get('insight', '')}")
            spans = item.get("evidence_spans", [])
            if spans:
                lines.append(f"  - evidence_spans: {', '.join(fmt_span(x) for x in spans)}")

    lines.append("\n# 6. CLEANED TRANSCRIPT\n")
    for row in final_data.get("cleaned_transcript", []):
        lines.append(f"[{row['start']:.1f}s - {row['end']:.1f}s] {row['speaker']}: {row['text']}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")

# ---- config helpers --------------------------------------------------------
def _cfg_for_mode(base_cfg: PipelineConfig, qwen_mode: str) -> PipelineConfig:
    cfg = replace(base_cfg)
    cfg.do_sample = False
    if qwen_mode == "fast":
        cfg.max_input_tokens = min(cfg.max_input_tokens, 1792)
        cfg.normalize_max_new_tokens = min(cfg.normalize_max_new_tokens, 96)
        cfg.local_summary_max_new_tokens = min(cfg.local_summary_max_new_tokens, 128)
        cfg.reducer_max_new_tokens = 0
        cfg.smart_skip_normalize = True
        cfg.smart_skip_summary = True
        cfg.prefer_rule_normalize = True
        cfg.use_rule_reducer = True
        cfg.block_max_chars = min(cfg.block_max_chars, 2200)
        cfg.block_max_duration_sec = min(cfg.block_max_duration_sec, 50.0)
        cfg.block_max_utterances = min(cfg.block_max_utterances, 7)
        cfg.repetition_penalty = max(cfg.repetition_penalty, 1.08)
    else:
        cfg.max_input_tokens = min(cfg.max_input_tokens, 2304)
        cfg.normalize_max_new_tokens = min(cfg.normalize_max_new_tokens, 128)
        cfg.local_summary_max_new_tokens = min(cfg.local_summary_max_new_tokens, 144)
        cfg.reducer_max_new_tokens = 0 if cfg.use_rule_reducer else cfg.reducer_max_new_tokens
        cfg.block_max_chars = min(cfg.block_max_chars, 2600)
        cfg.block_max_duration_sec = min(cfg.block_max_duration_sec, 60.0)
        cfg.block_max_utterances = min(cfg.block_max_utterances, 8)
    return cfg


def _cfg_for_model(base_cfg: PipelineConfig, model_name: str) -> PipelineConfig:
    cfg = replace(base_cfg)
    cfg.model_name = model_name
    return cfg

# ---- block selection -------------------------------------------------------
def choose_llm_block_ids(blocks: List[Block], normalized_blocks: List[Dict[str, Any]], cfg: PipelineConfig, qwen_mode: str) -> List[int]:
    scored: List[Tuple[int, int]] = []
    for block, nb in zip(blocks, normalized_blocks):
        lines = nb.get("normalized_lines", [])
        score = compute_block_value_score(block, lines)
        scored.append((block.block_id, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    threshold = cfg.min_llm_score_fast if qwen_mode == "fast" else cfg.min_llm_score_stable
    limit = cfg.max_llm_blocks_fast if qwen_mode == "fast" else cfg.max_llm_blocks_stable

    chosen = [bid for bid, score in scored if score >= threshold][:limit]
    return chosen

# ---- pipeline --------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    normalize_model_name = args.normalize_model_name or args.model_name
    summary_model_name = args.summary_model_name or normalize_model_name

    cfg = PipelineConfig(
        model_name=normalize_model_name,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
        enable_thinking=args.enable_thinking,
    )
    cfg.max_input_tokens = args.max_input_tokens
    cfg.normalize_max_new_tokens = args.normalize_max_new_tokens
    cfg.local_summary_max_new_tokens = args.local_summary_max_new_tokens
    cfg.reducer_max_new_tokens = args.reducer_max_new_tokens
    cfg.temperature = args.temperature
    cfg.top_p = args.top_p
    cfg.repetition_penalty = args.repetition_penalty
    cfg.no_repeat_ngram_size = args.no_repeat_ngram_size
    cfg.do_sample = args.do_sample
    cfg.merge_gap_sec = args.merge_gap_sec
    cfg.block_max_chars = args.block_max_chars
    cfg.block_max_duration_sec = args.block_max_duration_sec
    cfg.block_max_utterances = args.block_max_utterances
    cfg.min_text_chars = args.min_text_chars
    cfg.keep_single_word_fillers = args.keep_fillers
    cfg.smart_skip_normalize = not args.disable_smart_skip_normalize
    cfg.smart_skip_summary = not args.disable_smart_skip_summary
    cfg.prefer_rule_normalize = not args.force_llm_normalize
    cfg.use_rule_reducer = not args.use_llm_reducer
    cfg = _cfg_for_mode(cfg, args.qwen_mode)

    perf: Dict[str, Any] = {}
    total_t0 = time.perf_counter()

    print(f"[1/7] Loading transcript: {args.input}")
    t0 = time.perf_counter()
    segments = load_transcript(args.input)
    perf["load_transcript_sec"] = round(time.perf_counter() - t0, 4)
    print(f"  -> raw segments: {len(segments)}")

    print("[2/7] Deterministic preprocessing")
    t0 = time.perf_counter()
    segments = filter_segments(segments, cfg)
    segments = merge_adjacent_segments(segments, cfg)
    blocks = split_into_blocks(segments, cfg)
    perf["preprocess_sec"] = round(time.perf_counter() - t0, 4)
    print(f"  -> filtered+merged segments: {len(segments)}")
    print(f"  -> blocks: {len(blocks)}")

    need_llm = not args.rule_only
    runner_cache: Dict[str, QwenRunner] = {}
    runner_load_times: Dict[str, float] = {}

    def get_runner(model_name: str) -> QwenRunner:
        if model_name not in runner_cache:
            print(f"[3/7] Loading model lazily: {model_name}", flush=True)
            t_load = time.perf_counter()
            runner_cache[model_name] = QwenRunner(_cfg_for_model(cfg, model_name))
            runner_load_times[model_name] = round(time.perf_counter() - t_load, 4)
        return runner_cache[model_name]

    if need_llm:
        print(
            f"[3/7] Qwen mode={args.qwen_mode} | normalize_model={normalize_model_name} | "
            f"summary_model={summary_model_name} | lazy loading enabled",
            flush=True,
        )
    else:
        print("[3/7] rule_only=True -> skip model loading", flush=True)

    normalized_blocks: List[Dict[str, Any]] = []
    print("[4/7] Stage A - normalize each block")
    t0 = time.perf_counter()
    for i, block in enumerate(blocks):
        use_llm = need_llm and not should_skip_llm_normalize(block, cfg)
        runner = get_runner(normalize_model_name) if use_llm else None
        mode = "llm" if use_llm else "rule"
        print(f"  -> normalize block {block.block_id} [{block.start:.1f}s - {block.end:.1f}s] mode={mode}", flush=True)
        normalized_blocks.append(normalize_block(runner, block, cfg))
        print(f"PROGRESS:{i+1}/{max(1, len(blocks)*2)}", flush=True)
    perf["normalize_stage_sec"] = round(time.perf_counter() - t0, 4)

    llm_block_ids = choose_llm_block_ids(blocks, normalized_blocks, cfg, args.qwen_mode) if need_llm else []
    perf["selected_llm_blocks"] = llm_block_ids[:]
    print(f"[5/7] Stage B - summarize each normalized block | llm_blocks={len(llm_block_ids)}/{len(normalized_blocks)}", flush=True)

    block_summaries: List[Dict[str, Any]] = []
    perf["summary_repair_calls"] = 0
    t0 = time.perf_counter()
    for i, nb in enumerate(normalized_blocks):
        block_id = int(nb["block_id"])
        use_llm = need_llm and (block_id in llm_block_ids) and (not should_skip_llm_summary(nb, cfg))
        runner = get_runner(summary_model_name) if use_llm else None
        mode = "llm" if use_llm else "rule"
        print(f"  -> summarize block {block_id} mode={mode}", flush=True)
        if use_llm:
            block_summaries.append(summarize_block_llm(runner, nb, cfg, args.qwen_mode, perf))
        else:
            block_summaries.append(summarize_block_rule(nb))
        print(f"PROGRESS:{len(blocks) + i + 1}/{max(1, len(blocks)*2)}", flush=True)
    perf["summary_stage_sec"] = round(time.perf_counter() - t0, 4)

    print("[6/7] Stage C - reducer")
    t0 = time.perf_counter()
    reducer_runner = None
    if need_llm and (not cfg.use_rule_reducer) and cfg.reducer_max_new_tokens > 0:
        reducer_runner = get_runner(summary_model_name)
    reduced = reduce_summaries(reducer_runner, block_summaries, normalized_blocks, cfg)
    perf["reducer_stage_sec"] = round(time.perf_counter() - t0, 4)

    print("[7/7] Building final output")
    t0 = time.perf_counter()
    perf["rule_only"] = args.rule_only
    perf["qwen_mode"] = args.qwen_mode
    perf["do_sample"] = cfg.do_sample
    perf["normalize_model_name"] = normalize_model_name
    perf["summary_model_name"] = summary_model_name
    perf["loaded_models"] = list(runner_cache.keys())
    perf["model_load_sec_total"] = round(sum(runner_load_times.values()), 4)
    perf["model_load_breakdown"] = runner_load_times
    perf["normalize_rule_blocks"] = sum(1 for nb in normalized_blocks if "rule_normalized" in nb.get("notes", []))
    perf["normalize_llm_blocks"] = sum(1 for nb in normalized_blocks if "llm_normalized" in nb.get("notes", []))
    perf["summary_rule_blocks"] = sum(1 for bs in block_summaries if "rule_summary" in bs.get("notes", []))
    perf["summary_llm_blocks"] = sum(1 for bs in block_summaries if any(n.startswith("llm_extract") for n in bs.get("notes", [])))

    final_data = build_final_output(args.input, blocks, normalized_blocks, block_summaries, reduced, perf)
    perf["build_output_sec"] = round(time.perf_counter() - t0, 4)
    perf["total_wall_sec"] = round(time.perf_counter() - total_t0, 4)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(final_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] JSON written to: {output_json}", flush=True)

    if args.output_md:
        export_markdown(final_data, args.output_md)
        print(f"[DONE] Markdown written to: {args.output_md}", flush=True)

    if args.dump_blocks_dir:
        dump_dir = Path(args.dump_blocks_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        for block in blocks:
            (dump_dir / f"block_{block.block_id:03d}_raw.txt").write_text(block.text, encoding="utf-8")
        for nb in normalized_blocks:
            (dump_dir / f"block_{nb['block_id']:03d}_normalized.json").write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        for bs in block_summaries:
            (dump_dir / f"block_{bs['block_id']:03d}_summary.json").write_text(json.dumps(bs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] Intermediate blocks dumped to: {dump_dir}", flush=True)

    return final_data


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Optimized Qwen pipeline for DER + Whisper summarization")
    ap.add_argument("--input", type=str, required=True, help="Path to transcript .jsonl or bracket-format .txt")
    ap.add_argument("--output_json", type=str, required=True, help="Path to final structured JSON output")
    ap.add_argument("--output_md", type=str, default=None, help="Optional path to final Markdown output")
    ap.add_argument("--dump_blocks_dir", type=str, default=None, help="Optional directory to dump intermediate blocks")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Backward compatible single-model option")
    ap.add_argument("--normalize_model_name", type=str, default=None, help="Model used for normalization stage")
    ap.add_argument("--summary_model_name", type=str, default=None, help="Model used for summary/reducer stage")
    ap.add_argument("--qwen_mode", type=str, choices=["stable", "fast"], default="stable", help="Runtime mode preset")
    ap.add_argument("--adapter_path", type=str, default=None, help="Optional PEFT adapter path")
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    ap.add_argument("--enable_thinking", action="store_true", help="Backward compatibility only")
    ap.add_argument("--rule_only", action="store_true", help="Skip model loading and use only rule-based pipeline")
    ap.add_argument("--use_llm_reducer", action="store_true", help="Use LLM reducer instead of rule-based reducer")
    ap.add_argument("--force_llm_normalize", action="store_true", help="Allow LLM normalize for very noisy blocks instead of preferring rule normalize")

    ap.add_argument("--max_input_tokens", type=int, default=2304)
    ap.add_argument("--normalize_max_new_tokens", type=int, default=128)
    ap.add_argument("--local_summary_max_new_tokens", type=int, default=128)
    ap.add_argument("--reducer_max_new_tokens", type=int, default=448)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling; default is deterministic greedy")

    ap.add_argument("--merge_gap_sec", type=float, default=1.0)
    ap.add_argument("--block_max_chars", type=int, default=2600)
    ap.add_argument("--block_max_duration_sec", type=float, default=60.0)
    ap.add_argument("--block_max_utterances", type=int, default=8)
    ap.add_argument("--min_text_chars", type=int, default=2)
    ap.add_argument("--keep_fillers", action="store_true", help="Keep one-word filler rows like 'ờ', 'à'")
    ap.add_argument("--disable_smart_skip_normalize", action="store_true", help="Disable skip heuristics for normalization")
    ap.add_argument("--disable_smart_skip_summary", action="store_true", help="Disable skip heuristics for summary")
    return ap


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
