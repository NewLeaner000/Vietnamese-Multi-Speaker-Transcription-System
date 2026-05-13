#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-step Qwen pipeline for noisy DER + Whisper transcripts.

Pipeline:
1) Load transcript from JSONL or bracket-format text.
2) Deterministic preprocessing:
   - clean text
   - remove very noisy / empty rows
   - merge adjacent same-speaker rows
   - split into topical blocks on transition cues and size limits
3) Stage A: normalize each block with Qwen (strict no-hallucination prompt).
4) Stage B: summarize each normalized block into structured JSON.
5) Stage C: reduce all block summaries into one final structured JSON.
6) Optional Markdown export.

Designed to work with:
- Qwen/Qwen2.5-7B-Instruct base model
- optional LoRA/PEFT adapter
- 4-bit quantization on RTX 4060 / 3090Ti

Author: OpenAI assistant
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")

# Compatibility patches for environments similar to the user's current project.
import accelerate.utils.memory  # type: ignore
if not hasattr(accelerate.utils.memory, "clear_device_cache"):
    accelerate.utils.memory.clear_device_cache = lambda **kwargs: None

import transformers  # type: ignore
if not hasattr(transformers, "EncoderDecoderCache"):
    transformers.EncoderDecoderCache = type("EncoderDecoderCache", (object,), {})

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False


# =========================
# Configuration
# =========================

@dataclass
class PipelineConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: Optional[str] = None
    load_in_4bit: bool = True
    max_input_tokens: int = 8192
    normalize_max_new_tokens: int = 1200
    local_summary_max_new_tokens: int = 900
    reducer_max_new_tokens: int = 1400
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False

    # deterministic preprocessing
    merge_gap_sec: float = 1.2
    block_max_chars: int = 5000
    block_max_duration_sec: float = 240.0
    min_text_chars: int = 2
    keep_single_word_fillers: bool = False
    normalize_batch_max_lines: int = 12
    normalize_batch_max_chars: int = 2200
    normalize_retry_split_threshold: int = 4


TRANSITION_PATTERNS = [
    r"quý vị thân mến.*quay trở lại",
    r"chúng tôi sẽ quay trở lại",
    r"nghỉ giải lao",
    r"mời bạn vào",
    r"xin mời khách mời",
    r"bây giờ vấn đề.*mang tới",
    r"chúng ta cùng đến với",
    r"phần thưởng",
    r"chiếc hộp",
    r"game",
    r"trò chơi",
    r"review",
]

FILLERS = {
    "ờ", "ờm", "ừm", "à", "ạ", "dạ", "ha", "hả", "ơ", "ừ", "ừ ha", "ừm ha"
}


# =========================
# Data structures
# =========================

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
        parts = []
        for s in self.segments:
            parts.append(f"[{s.start:.1f}s - {s.end:.1f}s] {s.speaker}: {s.text}")
        return "\n".join(parts)


# =========================
# Utility
# =========================

BRACKET_LINE_RE = re.compile(
    r"^\[(?P<start>[\d\.]+)s\s*-\s*(?P<end>[\d\.]+)s\]\s*(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)

RAW_LINE_RE = re.compile(
    r"^(?:\S+)\s+\d+\s+(?P<speaker>speaker[_\- ]?\d+|\S+)\s+(?P<start>[\d\.]+)\s+(?P<end>[\d\.]+)\s+(?P<text>.+)$",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def clean_segment_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace(" ,", ",").replace(" .", ".")
    text = re.sub(r"([,\.\?!])\1{1,}", r"\1", text)
    return text.strip()


def maybe_is_filler(text: str) -> bool:
    txt = normalize_whitespace(text).lower().strip(" .,!?:;-")
    return txt in FILLERS


def has_terminal_punctuation(text: str) -> bool:
    return bool(text) and text[-1] in ".!?…"


def looks_like_transition(text: str) -> bool:
    t = normalize_whitespace(text).lower()
    return any(re.search(p, t) for p in TRANSITION_PATTERNS)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text
    in_string = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def escape_raw_control_chars_in_strings(text: str) -> str:
    out = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue
        out.append(ch)
        if ch == '"':
            in_string = True
    return ''.join(out)


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def safe_json_loads(text: str) -> Any:
    text = strip_code_fences(text)
    candidates = [text, extract_first_json_object(text)]
    last_error = None
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        variants = [
            cand,
            remove_trailing_commas(cand),
            escape_raw_control_chars_in_strings(cand),
            remove_trailing_commas(escape_raw_control_chars_in_strings(cand)),
        ]
        for variant in variants:
            try:
                return json.loads(variant)
            except Exception as e:
                last_error = e
    if last_error:
        raise last_error
    raise ValueError("Không tìm thấy JSON hợp lệ trong output model")



# =========================
# Load transcript
# =========================


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
            text = item.get("text", item.get("content", ""))
            text = clean_segment_text(text)
            if not text:
                continue
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
            if m:
                segments.append(
                    Segment(
                        start=float(m.group("start")),
                        end=float(m.group("end")),
                        speaker=clean_segment_text(m.group("speaker")),
                        text=clean_segment_text(m.group("text")),
                        source_idx=idx,
                    )
                )
                continue

            m2 = RAW_LINE_RE.match(line)
            if m2:
                segments.append(
                    Segment(
                        start=float(m2.group("start")),
                        end=float(m2.group("end")),
                        speaker=clean_segment_text(m2.group("speaker")),
                        text=clean_segment_text(m2.group("text")),
                        source_idx=idx,
                    )
                )
    segments = [s for s in segments if s.text]
    segments.sort(key=lambda s: (s.start, s.end, s.source_idx))
    return segments


def load_transcript(path: str) -> List[Segment]:
    if path.lower().endswith(".jsonl"):
        return load_transcript_jsonl(path)
    return load_transcript_text(path)


# =========================
# Deterministic preprocessing
# =========================


def filter_segments(segments: List[Segment], cfg: PipelineConfig) -> List[Segment]:
    kept: List[Segment] = []
    for s in segments:
        text = clean_segment_text(s.text)
        if len(text) < cfg.min_text_chars:
            continue
        if (not cfg.keep_single_word_fillers) and maybe_is_filler(text):
            continue
        kept.append(Segment(start=s.start, end=s.end, speaker=s.speaker, text=text, source_idx=s.source_idx))
    return kept


def merge_adjacent_segments(segments: List[Segment], cfg: PipelineConfig) -> List[Segment]:
    if not segments:
        return []

    merged: List[Segment] = [segments[0]]
    for s in segments[1:]:
        prev = merged[-1]
        gap = max(0.0, s.start - prev.end)
        same_speaker = s.speaker == prev.speaker
        should_merge = False

        if same_speaker and gap <= cfg.merge_gap_sec:
            should_merge = True
        elif same_speaker and gap <= (cfg.merge_gap_sec * 2) and not has_terminal_punctuation(prev.text):
            should_merge = True

        if should_merge:
            joiner = " "
            merged[-1] = Segment(
                start=prev.start,
                end=max(prev.end, s.end),
                speaker=prev.speaker,
                text=clean_segment_text(prev.text + joiner + s.text),
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

    def flush(block_segments: List[Segment], is_transition_heavy: bool) -> None:
        if not block_segments:
            return
        blocks.append(
            Block(
                block_id=len(blocks),
                start=block_segments[0].start,
                end=block_segments[-1].end,
                segments=list(block_segments),
                is_transition_heavy=is_transition_heavy,
            )
        )

    for s in segments:
        seg_text = f"[{s.start:.1f}s - {s.end:.1f}s] {s.speaker}: {s.text}\n"
        seg_chars = len(seg_text)
        current_duration = 0.0 if not cur else (s.end - cur_start)
        is_transition = looks_like_transition(s.text)

        force_new_block = False
        if cur and is_transition and current_duration >= 30:
            force_new_block = True
        if cur and (cur_chars + seg_chars > cfg.block_max_chars):
            force_new_block = True
        if cur and (current_duration > cfg.block_max_duration_sec):
            force_new_block = True

        if force_new_block:
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


# =========================
# Model wrapper
# =========================

class QwenRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        tok = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        return tok

    def _load_model(self):
        quantization_config = None
        if self.cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        if self.cfg.adapter_path:
            if not _HAS_PEFT:
                raise RuntimeError("peft chưa được cài nhưng bạn đã truyền --adapter_path")
            model = PeftModel.from_pretrained(model, self.cfg.adapter_path)

        return model

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=self.cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        try:
            return safe_json_loads(text)
        except Exception as e:
            raise RuntimeError(f"Model không trả về JSON hợp lệ. Raw output:\n{text}\n\nParse error: {e}")


# =========================
# Prompts
# =========================

NORMALIZE_SYSTEM_PROMPT = """Bạn là bộ chuẩn hóa transcript hội thoại tiếng Việt.
Nhiệm vụ duy nhất của bạn là làm sạch transcript ASR.

Quy tắc bắt buộc:
- Chỉ dùng thông tin có trong input.
- Không được bịa chi tiết, không suy diễn, không rút gọn thành summary.
- Không được đổi speaker nếu input chưa chắc tên thật.
- Nếu input là speaker_03, speaker_08... thì giữ nguyên.
- Nếu thấy các đoạn khác chủ đề thì KHÔNG gộp nội dung giữa các đoạn đó.
- Loại bỏ từ lặp/ngập ngừng ở mức tối thiểu nhưng không làm đổi nghĩa.
- Sửa câu cho dễ đọc hơn, nhưng vẫn trung thành với ý gốc.
- Mỗi clean_text phải nằm trên MỘT DÒNG DUY NHẤT, không chèn xuống dòng trong chuỗi.
- Giữ timestamp block-level như input từng dòng.

Trả về JSON đúng schema:
{
  "normalized_lines": [
    {"start": 0.0, "end": 1.0, "speaker": "speaker_01", "clean_text": "..."}
  ],
  "notes": ["ghi chú ngắn, nếu có"]
}
"""

LOCAL_SUMMARY_SYSTEM_PROMPT = """Bạn là bộ tóm tắt block hội thoại tiếng Việt cho transcript DER + Whisper.
Bạn chỉ được dùng thông tin có trong input block.

Quy tắc bắt buộc:
- Không bịa.
- Không tự tạo action item nếu block không có giao việc rõ ràng.
- Nếu không chắc, ghi 'không đủ bằng chứng'.
- Nếu block là chuyển cảnh / nghỉ giải lao / giới thiệu khách mời thì ghi rõ đó là segment chuyển tiếp.
- Mọi action item phải có evidence_span.
- Speaker insight phải giữ đúng speaker label hiện có.

Trả về JSON đúng schema:
{
  "block_id": 0,
  "segment_type": "discussion|transition|game|guest_intro|other",
  "overview": "...",
  "key_points": ["..."],
  "action_items": [
    {
      "owner": "speaker_03|null",
      "task": "...",
      "deadline": "...|null",
      "evidence_span": [12.3, 15.8],
      "confidence": "high|medium|low"
    }
  ],
  "speaker_insights": [
    {
      "speaker": "speaker_03",
      "insight": "...",
      "evidence_spans": [[12.3, 20.4]]
    }
  ],
  "facts": [
    {
      "fact": "...",
      "evidence_spans": [[12.3, 20.4]]
    }
  ]
}
"""

REDUCER_SYSTEM_PROMPT = """Bạn là bộ tổng hợp summary toàn cuộc họp từ các block summaries.
Input đã là các block summaries có cấu trúc. Bạn KHÔNG được thêm fact mới ngoài input này.

Quy tắc bắt buộc:
- Nếu các block thuộc các câu chuyện khác nhau, phải ghi rõ là nhiều segment/chủ đề.
- Không tạo action item nếu không đủ bằng chứng trong block summaries.
- Nếu tên thật người nói chưa có, giữ speaker_xx.
- Mọi action item phải có evidence_span.
- Không dùng văn phong dài dòng.

Trả về JSON đúng schema:
{
  "meeting_overview": "...",
  "meeting_type": "meeting|talkshow|interview|discussion|other",
  "segments": [
    {"title": "...", "block_ids": [0,1], "summary": "..."}
  ],
  "action_items": [
    {
      "owner": "speaker_03|null",
      "task": "...",
      "deadline": "...|null",
      "evidence_span": [12.3, 15.8],
      "confidence": "high|medium|low"
    }
  ],
  "speaker_insights": [
    {"speaker": "speaker_03", "insight": "...", "evidence_spans": [[12.3, 20.4]]}
  ],
  "risk_flags": ["..."],
  "quality_notes": ["..."]
}
"""


# =========================
# Pipeline stages
# =========================


def split_segments_for_normalize(segments: List[Segment], cfg: PipelineConfig) -> List[List[Segment]]:
    batches: List[List[Segment]] = []
    cur: List[Segment] = []
    cur_chars = 0
    for seg in segments:
        seg_text = f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}: {seg.text}\n"
        seg_chars = len(seg_text)
        if cur and (len(cur) >= cfg.normalize_batch_max_lines or (cur_chars + seg_chars) > cfg.normalize_batch_max_chars):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(seg)
        cur_chars += seg_chars
    if cur:
        batches.append(cur)
    return batches


def normalize_batch(runner: QwenRunner, batch: List[Segment], block: Block, cfg: PipelineConfig) -> Dict[str, Any]:
    batch_text = "\n".join(f"[{s.start:.1f}s - {s.end:.1f}s] {s.speaker}: {s.text}" for s in batch)
    user_prompt = (
        f"Block ID: {block.block_id}\n"
        f"Batch time: {batch[0].start:.1f}s - {batch[-1].end:.1f}s\n"
        f"Transition heavy: {block.is_transition_heavy}\n\n"
        f"Transcript batch:\n{batch_text}\n"
    )
    return runner.generate_json(NORMALIZE_SYSTEM_PROMPT, user_prompt, cfg.normalize_max_new_tokens)


def fallback_passthrough_batch(batch: List[Segment]) -> List[Dict[str, Any]]:
    return [
        {
            "start": float(s.start),
            "end": float(s.end),
            "speaker": str(s.speaker),
            "clean_text": clean_segment_text(str(s.text)),
        }
        for s in batch
    ]


def normalize_block(runner: QwenRunner, block: Block, cfg: PipelineConfig) -> Dict[str, Any]:
    batches = split_segments_for_normalize(block.segments, cfg)
    clean_lines: List[Dict[str, Any]] = []
    notes: List[str] = []

    def run_recursive(batch: List[Segment]) -> None:
        nonlocal clean_lines, notes
        try:
            result = normalize_batch(runner, batch, block, cfg)
            normalized_lines = result.get("normalized_lines", [])
            accepted = []
            for line in normalized_lines:
                try:
                    accepted.append({
                        "start": float(line["start"]),
                        "end": float(line["end"]),
                        "speaker": str(line["speaker"]),
                        "clean_text": clean_segment_text(str(line["clean_text"])),
                    })
                except Exception:
                    continue
            if accepted:
                clean_lines.extend(accepted)
            else:
                clean_lines.extend(fallback_passthrough_batch(batch))
                notes.append(f"fallback_passthrough_empty_batch_{batch[0].source_idx}_{batch[-1].source_idx}")
            if isinstance(result.get("notes", []), list):
                notes.extend(result.get("notes", []))
        except Exception as e:
            if len(batch) > cfg.normalize_retry_split_threshold:
                mid = len(batch) // 2
                run_recursive(batch[:mid])
                run_recursive(batch[mid:])
            else:
                clean_lines.extend(fallback_passthrough_batch(batch))
                notes.append(f"fallback_passthrough_exception_{batch[0].source_idx}_{batch[-1].source_idx}: {e}")

    for batch in batches:
        run_recursive(batch)

    clean_lines.sort(key=lambda x: (x["start"], x["end"]))
    return {
        "block_id": block.block_id,
        "start": block.start,
        "end": block.end,
        "is_transition_heavy": block.is_transition_heavy,
        "normalized_lines": clean_lines,
        "notes": notes,
    }


def summarize_block(runner: QwenRunner, normalized_block: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, Any]:
    lines = normalized_block.get("normalized_lines", [])
    text = "\n".join(
        f"[{x['start']:.1f}s - {x['end']:.1f}s] {x['speaker']}: {x['clean_text']}"
        for x in lines
    )
    user_prompt = (
        f"Block ID: {normalized_block['block_id']}\n"
        f"Time: {normalized_block['start']:.1f}s - {normalized_block['end']:.1f}s\n"
        f"Transcript normalized block:\n{text}\n"
    )
    result = runner.generate_json(LOCAL_SUMMARY_SYSTEM_PROMPT, user_prompt, cfg.local_summary_max_new_tokens)
    result.setdefault("block_id", normalized_block["block_id"])
    return result


def reduce_summaries(runner: QwenRunner, block_summaries: List[Dict[str, Any]], cfg: PipelineConfig) -> Dict[str, Any]:
    packed = json.dumps(block_summaries, ensure_ascii=False, indent=2)
    user_prompt = f"Dưới đây là block summaries:\n{packed}"
    return runner.generate_json(REDUCER_SYSTEM_PROMPT, user_prompt, cfg.reducer_max_new_tokens)


def build_cleaned_transcript_output(normalized_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for block in normalized_blocks:
        for line in block.get("normalized_lines", []):
            rows.append({
                "block_id": block["block_id"],
                "start": line["start"],
                "end": line["end"],
                "speaker": line["speaker"],
                "text": line["clean_text"],
            })
    rows.sort(key=lambda x: (x["start"], x["end"], x["block_id"]))
    return rows


def build_final_output(
    source_path: str,
    blocks: List[Block],
    normalized_blocks: List[Dict[str, Any]],
    block_summaries: List[Dict[str, Any]],
    reduced: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "source": source_path,
        "stats": {
            "num_blocks": len(blocks),
            "num_normalized_blocks": len(normalized_blocks),
            "num_block_summaries": len(block_summaries),
        },
        "meeting_overview": reduced.get("meeting_overview", ""),
        "meeting_type": reduced.get("meeting_type", "other"),
        "segments": reduced.get("segments", []),
        "action_items": reduced.get("action_items", []),
        "speaker_insights": reduced.get("speaker_insights", []),
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
        },
    }


# =========================
# Markdown export
# =========================


def export_markdown(final_data: Dict[str, Any], out_path: str) -> None:
    def fmt_span(span: Any) -> str:
        if isinstance(span, list) and len(span) == 2:
            try:
                return f"[{float(span[0]):.1f}s - {float(span[1]):.1f}s]"
            except Exception:
                return str(span)
        return str(span)

    lines: List[str] = []
    lines.append("# 1. TỔNG QUAN CUỘC HỌP\n")
    lines.append(final_data.get("meeting_overview", "") or "Không đủ bằng chứng để kết luận tổng quan rõ ràng.")
    lines.append("\n# 2. HÀNH ĐỘNG TRIỂN KHAI (ACTION ITEMS)\n")

    action_items = final_data.get("action_items", [])
    if not action_items:
        lines.append("Không có action item rõ ràng trong transcript này.")
    else:
        for item in action_items:
            owner = item.get("owner", None)
            task = item.get("task", None)
            deadline = item.get("deadline", None)
            ev = fmt_span(item.get("evidence_span", None))
            conf = item.get("confidence", "unknown")
            lines.append(f"- owner: {owner}")
            lines.append(f"  - task: {task}")
            lines.append(f"  - deadline: {deadline}")
            lines.append(f"  - evidence_span: {ev}")
            lines.append(f"  - confidence: {conf}")

    lines.append("\n# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)\n")
    insights = final_data.get("speaker_insights", [])
    if not insights:
        lines.append("Không đủ bằng chứng để trích xuất speaker insights rõ ràng.")
    else:
        for item in insights:
            lines.append(f"- **{item.get('speaker', 'unknown')}**: {item.get('insight', '')}")
            spans = item.get("evidence_spans", [])
            if spans:
                spans_str = ", ".join(fmt_span(x) for x in spans)
                lines.append(f"  - evidence_spans: {spans_str}")

    lines.append("\n# 4. KỊCH BẢN CHUẨN HOÁ (CLEANED TRANSCRIPT)\n")
    for row in final_data.get("cleaned_transcript", []):
        lines.append(f"[{row['start']:.1f}s - {row['end']:.1f}s] {row['speaker']}: {row['text']}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# =========================
# Main
# =========================


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = PipelineConfig(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
        max_input_tokens=args.max_input_tokens,
        normalize_max_new_tokens=args.normalize_max_new_tokens,
        local_summary_max_new_tokens=args.local_summary_max_new_tokens,
        reducer_max_new_tokens=args.reducer_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
        merge_gap_sec=args.merge_gap_sec,
        block_max_chars=args.block_max_chars,
        block_max_duration_sec=args.block_max_duration_sec,
        min_text_chars=args.min_text_chars,
        keep_single_word_fillers=args.keep_fillers,
        normalize_batch_max_lines=args.normalize_batch_max_lines,
        normalize_batch_max_chars=args.normalize_batch_max_chars,
        normalize_retry_split_threshold=args.normalize_retry_split_threshold,
    )

    print(f"[1/7] Loading transcript: {args.input}")
    segments = load_transcript(args.input)
    print(f"  -> raw segments: {len(segments)}")

    print("[2/7] Deterministic preprocessing")
    segments = filter_segments(segments, cfg)
    segments = merge_adjacent_segments(segments, cfg)
    blocks = split_into_blocks(segments, cfg)
    print(f"  -> filtered+merged segments: {len(segments)}")
    print(f"  -> blocks: {len(blocks)}")

    print("[3/7] Loading model(s)")
    normalize_model_name = args.normalize_model_name or args.model_name
    summary_model_name = args.summary_model_name or args.model_name
    normalize_adapter_path = args.normalize_adapter_path or args.adapter_path
    summary_adapter_path = args.summary_adapter_path or args.adapter_path

    normalize_cfg = replace(
        cfg,
        model_name=normalize_model_name,
        adapter_path=normalize_adapter_path,
    )
    summary_cfg = replace(
        cfg,
        model_name=summary_model_name,
        adapter_path=summary_adapter_path,
    )

    normalize_runner = QwenRunner(normalize_cfg)
    summary_runner = normalize_runner
    if (
        summary_cfg.model_name != normalize_cfg.model_name
        or summary_cfg.adapter_path != normalize_cfg.adapter_path
    ):
        summary_runner = QwenRunner(summary_cfg)

    normalized_blocks: List[Dict[str, Any]] = []
    print("[4/7] Stage A - normalize each block")
    for block in blocks:
        print(f"  -> normalize block {block.block_id} [{block.start:.1f}s - {block.end:.1f}s]")
        normalized_blocks.append(normalize_block(normalize_runner, block, normalize_cfg))

    block_summaries: List[Dict[str, Any]] = []
    print("[5/7] Stage B - summarize each normalized block")
    for nb in normalized_blocks:
        print(f"  -> summarize block {nb['block_id']}")
        block_summaries.append(summarize_block(summary_runner, nb, summary_cfg))

    print("[6/7] Stage C - reducer")
    reduced = reduce_summaries(summary_runner, block_summaries, summary_cfg)

    print("[7/7] Building final output")
    final_data = build_final_output(args.input, blocks, normalized_blocks, block_summaries, reduced)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(final_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] JSON written to: {output_json}")

    if args.output_md:
        export_markdown(final_data, args.output_md)
        print(f"[DONE] Markdown written to: {args.output_md}")

    if args.dump_blocks_dir:
        dump_dir = Path(args.dump_blocks_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        for block in blocks:
            (dump_dir / f"block_{block.block_id:03d}_raw.txt").write_text(block.text, encoding="utf-8")
        for nb in normalized_blocks:
            (dump_dir / f"block_{nb['block_id']:03d}_normalized.json").write_text(
                json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        for bs in block_summaries:
            (dump_dir / f"block_{bs['block_id']:03d}_summary.json").write_text(
                json.dumps(bs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"[DONE] Intermediate blocks dumped to: {dump_dir}")

    return final_data


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Multi-step Qwen pipeline for DER + Whisper meeting summarization")
    ap.add_argument("--input", type=str, required=True, help="Path to transcript .jsonl or bracket-format .txt")
    ap.add_argument("--output_json", type=str, required=True, help="Path to final structured JSON output")
    ap.add_argument("--output_md", type=str, default=None, help="Optional path to final Markdown output")
    ap.add_argument("--dump_blocks_dir", type=str, default=None, help="Optional directory to dump intermediate blocks")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--normalize_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name for normalization stage")
    ap.add_argument("--summary_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name for summary stage")
    ap.add_argument("--adapter_path", type=str, default=None, help="Optional PEFT adapter path")
    ap.add_argument("--normalize_adapter_path", type=str, default=None, help="Optional adapter path for normalize model")
    ap.add_argument("--summary_adapter_path", type=str, default=None, help="Optional adapter path for summary model")
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")

    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.add_argument("--normalize_max_new_tokens", type=int, default=1200)
    ap.add_argument("--local_summary_max_new_tokens", type=int, default=900)
    ap.add_argument("--reducer_max_new_tokens", type=int, default=1400)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--greedy", action="store_true", help="Use greedy decoding")

    ap.add_argument("--merge_gap_sec", type=float, default=1.2)
    ap.add_argument("--block_max_chars", type=int, default=5000)
    ap.add_argument("--block_max_duration_sec", type=float, default=240.0)
    ap.add_argument("--min_text_chars", type=int, default=2)
    ap.add_argument("--keep_fillers", action="store_true", help="Keep one-word filler rows like 'ờ', 'à'")
    ap.add_argument("--normalize_batch_max_lines", type=int, default=12, help="Max lines per normalization sub-batch")
    ap.add_argument("--normalize_batch_max_chars", type=int, default=2200, help="Max chars per normalization sub-batch")
    ap.add_argument("--normalize_retry_split_threshold", type=int, default=4, help="If a normalize batch fails and has more than this many lines, split recursively")
    return ap


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
