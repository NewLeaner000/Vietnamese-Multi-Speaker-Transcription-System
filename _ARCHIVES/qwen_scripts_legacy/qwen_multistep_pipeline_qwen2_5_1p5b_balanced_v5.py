#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast multi-step Qwen pipeline for noisy DER + Whisper transcripts.
Optimized for RTX 4060 Laptop 8GB with Qwen2.5-1.5B-Instruct.

Main optimizations:
- Default model: Qwen/Qwen2.5-1.5B-Instruct
- Qwen2.5 lightweight instruct model for lower latency / lower VRAM
- 4-bit quantization by default
- Smart skip of LLM normalization for clean / transition blocks
- Smart skip of LLM summary for very short transition blocks
- Rule-based reducer by default (saves one full LLM call)
- Smaller generation budgets and input truncation

Input:
- transcript.jsonl with keys like start/end/speaker/text
- or text file with lines: [13.8s - 25.0s] speaker_08: text

Output:
- structured JSON
- optional Markdown
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

# Compatibility patches similar to the user's current environment.
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


@dataclass
class PipelineConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: Optional[str] = None
    load_in_4bit: bool = True
    enable_thinking: bool = False  # unused for Qwen2.5; kept for backward compatibility

    # lower budgets for 4060 8GB
    max_input_tokens: int = 3072
    normalize_max_new_tokens: int = 384
    local_summary_max_new_tokens: int = 256
    reducer_max_new_tokens: int = 0  # default: disable LLM reducer

    # Conservative decoding for structured JSON tasks on 8GB VRAM.
    temperature: float = 0.3
    top_p: float = 0.8
    do_sample: bool = False

    # preprocessing
    merge_gap_sec: float = 1.2
    block_max_chars: int = 3200
    block_max_duration_sec: float = 180.0
    min_text_chars: int = 2
    keep_single_word_fillers: bool = False

    # fast-path toggles
    smart_skip_normalize: bool = True
    smart_skip_summary: bool = True
    always_rule_for_transition_summary: bool = True
    use_rule_reducer: bool = True


TRANSITION_PATTERNS = [
    r"quý vị thân mến.*quay trở lại",
    r"chúng tôi sẽ quay trở lại",
    r"nghỉ giải lao",
    r"mời bạn vào",
    r"xin mời.*khách mời",
    r"chúng ta cùng đến với",
    r"chiếc hộp",
    r"trò chơi",
    r"game",
    r"review",
    r"phần thưởng",
    r"quảng cáo",
]

FILLERS = {
    "ờ", "ờm", "ừm", "à", "ạ", "dạ", "ha", "hả", "ơ", "ừ", "ừ ha", "ừm ha"
}

BRACKET_LINE_RE = re.compile(
    r"^\[(?P<start>[\d\.]+)s\s*-\s*(?P<end>[\d\.]+)s\]\s*(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)
RAW_LINE_RE = re.compile(
    r"^(?:\S+)\s+\d+\s+(?P<speaker>speaker[_\- ]?\d+|\S+)\s+(?P<start>[\d\.]+)\s+(?P<end>[\d\.]+)\s+(?P<text>.+)$",
    re.IGNORECASE,
)


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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def count_repeated_tokens(text: str) -> int:
    toks = normalize_whitespace(text).lower().split()
    if len(toks) < 2:
        return 0
    repeats = 0
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            repeats += 1
    return repeats


def estimate_noise_score(text: str) -> int:
    score = 0
    if count_repeated_tokens(text) >= 2:
        score += 1
    if re.search(r"\b(ờ|ờm|à|ừm|dạ)\b", text.lower()):
        score += 1
    if re.search(r"([a-zA-ZÀ-ỹ]+)(\s+\1){2,}", text, flags=re.IGNORECASE):
        score += 1
    if len(text) > 0 and len(re.findall(r"[!?\.]{2,}", text)) > 0:
        score += 1
    return score


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
        should_merge = False
        if same_speaker and gap <= cfg.merge_gap_sec:
            should_merge = True
        elif same_speaker and gap <= (cfg.merge_gap_sec * 2) and not has_terminal_punctuation(prev.text):
            should_merge = True
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

    def flush(cur_segments: List[Segment], is_transition_heavy: bool) -> None:
        if not cur_segments:
            return
        blocks.append(
            Block(
                block_id=len(blocks),
                start=cur_segments[0].start,
                end=cur_segments[-1].end,
                segments=list(cur_segments),
                is_transition_heavy=is_transition_heavy,
            )
        )

    for s in segments:
        seg_text = f"[{s.start:.1f}s - {s.end:.1f}s] {s.speaker}: {s.text}\n"
        seg_chars = len(seg_text)
        current_duration = 0.0 if not cur else (s.end - cur_start)
        is_transition = looks_like_transition(s.text)
        force_new_block = False
        if cur and is_transition and current_duration >= 25:
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


class QwenRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.device = self._resolve_device()

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

        kwargs = dict(
            trust_remote_code=True,
            device_map="auto",
        )
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = "sdpa"

        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **kwargs)

        if self.cfg.adapter_path:
            if not _HAS_PEFT:
                raise RuntimeError("peft chưa được cài nhưng bạn đã truyền --adapter_path")
            model = PeftModel.from_pretrained(model, self.cfg.adapter_path)

        model.eval()
        return model

    def _resolve_device(self):
        try:
            return self.model.device
        except Exception:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        apply_kwargs = dict(tokenize=False, add_generation_prompt=True)
        prompt = self.tokenizer.apply_chat_template(messages, **apply_kwargs)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_input_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.cfg.do_sample,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = self.cfg.temperature
            gen_kwargs["top_p"] = self.cfg.top_p

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        try:
            return safe_json_loads(text)
        except Exception as e:
            raise RuntimeError(f"Model không trả về JSON hợp lệ. Raw output:\n{text}\n\nParse error: {e}")


NORMALIZE_SYSTEM_PROMPT = (
    "Bạn là bộ chuẩn hóa transcript tiếng Việt. "
    "Chỉ làm sạch ASR, không tóm tắt, không suy diễn, không đổi speaker, giữ đúng nội dung. "
    "Nếu speaker là speaker_xx thì giữ nguyên. "
    "BẮT BUỘC trả về JSON hợp lệ, không dùng markdown fence, không giải thích thêm. "
    "Chỉ trả về đúng 1 khóa duy nhất là normalized_lines. "
    'Định dạng: {"normalized_lines":[{"start":0.0,"end":1.0,"speaker":"speaker_01","clean_text":"..."}]}'
)

LOCAL_SUMMARY_SYSTEM_PROMPT = (
    "Bạn là bộ tóm tắt block hội thoại tiếng Việt. "
    "BẮT BUỘC chỉ trả lời bằng tiếng Việt. Không dùng tiếng Anh. Không dùng câu mẫu như 'The conversation revolves around'. "
    "Chỉ dùng thông tin trong block. Không bịa. Không tự tạo action item nếu không có giao việc rõ ràng. "
    "Nếu không chắc thì ghi 'không đủ bằng chứng'. "
    "Trả về JSON với các khóa: block_id, segment_type, overview, key_points, action_items, speaker_insights, facts."
)

REDUCER_SYSTEM_PROMPT = (
    "Bạn là bộ tổng hợp block summaries. Chỉ dùng thông tin trong input, không thêm fact mới. "
    "Nếu có nhiều câu chuyện khác nhau thì phải tách segment. Trả về JSON với các khóa: "
    "meeting_overview, meeting_type, segments, action_items, speaker_insights, risk_flags, quality_notes."
)


def block_to_lines(block: Block) -> List[Dict[str, Any]]:
    return [
        {"start": s.start, "end": s.end, "speaker": s.speaker, "clean_text": s.text}
        for s in block.segments
    ]


def should_skip_llm_normalize(block: Block, cfg: PipelineConfig) -> bool:
    if not cfg.smart_skip_normalize:
        return False
    text = block.text
    noise = estimate_noise_score(text)
    if block.is_transition_heavy and (block.duration <= 35 or len(block.segments) <= 4):
        return True
    if noise == 0 and len(text) <= 1800:
        return True
    return False


def _normalized_text_blob(normalized_block: Dict[str, Any]) -> str:
    return " ".join(x.get("clean_text", "") for x in normalized_block.get("normalized_lines", [])).lower()


def should_skip_llm_summary(normalized_block: Dict[str, Any], cfg: PipelineConfig) -> bool:
    if not cfg.smart_skip_summary:
        return False
    lines = normalized_block.get("normalized_lines", [])
    is_transition = normalized_block.get("is_transition_heavy", False)
    total_chars = sum(len(x.get("clean_text", "")) for x in lines)
    text_blob = _normalized_text_blob(normalized_block)
    if cfg.always_rule_for_transition_summary and is_transition:
        return True
    if any(k in text_blob for k in ["nghỉ giải lao", "quay trở lại", "chiếc hộp", "khách mời", "trò chơi", "phần thưởng"]):
        return True
    if len(lines) <= 3 or total_chars <= 420:
        return True
    return False


def looks_generic_or_wrong_summary(summary: Dict[str, Any]) -> bool:
    overview = normalize_whitespace(str(summary.get("overview", "")))
    if not overview:
        return True
    low = overview.lower()
    generic_markers = [
        "the conversation revolves around",
        "the conversation is about",
        "speaker 0",
        "speaker 1",
        "mental health treatment",
        "document and the need for it to be updated",
        "product and expresses dissatisfaction",
    ]
    if any(m in low for m in generic_markers):
        return True
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in overview)
    vietnamese_hint = bool(re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]|\b(và|của|đoạn|khách|chương trình|người nói|cuộc trò chuyện)\b", overview.lower()))
    if ascii_letters >= 25 and not vietnamese_hint:
        return True
    return False



CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def has_structured_artifact(text: str) -> bool:
    t = text or ""
    return "{'insight':" in t or '{"insight":' in t or t.strip().startswith('{') or t.strip().startswith('[')


def cleanup_repeated_sentences(text: str) -> str:
    parts = [normalize_whitespace(x) for x in re.split(r"(?<=[\.!?])\s+", text or "") if normalize_whitespace(x)]
    out = []
    seen = set()
    for p in parts:
        low = p.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(p)
    return " ".join(out).strip()


def sanitize_summary_text(text: str, fallback: str = "") -> str:
    text = normalize_whitespace(str(text or ""))
    if not text:
        return fallback
    if has_cjk(text) or has_structured_artifact(text):
        return fallback
    if looks_generic_or_wrong_summary({"overview": text}):
        return fallback
    return cleanup_repeated_sentences(text) or fallback


def sanitize_insight_text(text: str) -> str:
    text = normalize_whitespace(str(text or ""))
    if not text:
        return ""
    if has_cjk(text) or has_structured_artifact(text):
        return ""
    generic_bad = [
        "the conversation revolves around",
        "the conversation is about",
        "speaker 0",
        "speaker 1",
    ]
    low = text.lower()
    if any(x in low for x in generic_bad):
        return ""
    return cleanup_repeated_sentences(text)


def is_generic_speaker_insight(text: str) -> bool:
    text = normalize_whitespace(str(text or ""))
    if not text:
        return True
    low = text.lower()
    generic_markers = [
        "tham gia trao đổi trong đoạn này",
        "có nhiều lượt phát biểu trong đoạn này",
        "đoạn trao đổi chính của cuộc trò chuyện",
        "không đủ bằng chứng để tóm tắt sâu hơn",
        "đoạn trao đổi khác",
    ]
    return any(x in low for x in generic_markers)


def _segment_phrase_for_speaker(seg_type: str) -> str:
    mapping = {
        "transition": "phần dẫn dắt chương trình hoặc chuyển cảnh",
        "game": "phần trò chơi hoặc tương tác nhẹ",
        "guest_intro": "phần giới thiệu khách mời hoặc mở chủ đề",
        "family_story": "phần trao đổi về gia đình, vợ chồng, phân chia trách nhiệm hoặc dạy con",
        "health_story": "phần trao đổi về sức khỏe, áp lực, ăn uống hoặc giảm cân",
        "work_story": "phần trao đổi về công việc, khách hàng hoặc tình huống nghề nghiệp",
        "discussion": "phần trao đổi chính của cuộc trò chuyện",
        "other": "một phần trao đổi khác trong cuộc trò chuyện",
    }
    return mapping.get(seg_type, "một phần trao đổi khác trong cuộc trò chuyện")


def build_rule_speaker_insight(seg_type: str, many_turns: bool) -> str:
    phrase = _segment_phrase_for_speaker(seg_type)
    if many_turns:
        return f"Có nhiều lượt phát biểu, chủ yếu ở {phrase}."
    return f"Chủ yếu tham gia {phrase}."


def build_rule_speaker_main_summary(seg_types: List[str]) -> str:
    seg_types = [str(x or 'discussion') for x in seg_types]
    if not seg_types:
        return "Người nói này có tham gia cuộc hội thoại, nhưng chưa đủ bằng chứng để rút ra nội dung chính rõ hơn."
    uniq = []
    seen = set()
    for t in seg_types:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    main = uniq[0]
    extra = uniq[1] if len(uniq) > 1 else None
    main_phrase_map = {
        'transition': 'dẫn dắt chương trình, chuyển cảnh hoặc giới thiệu phần tiếp theo',
        'game': 'trò chơi hoặc tương tác nhẹ giữa các người nói',
        'guest_intro': 'giới thiệu khách mời hoặc mở ra chủ đề mới',
        'family_story': 'câu chuyện gia đình, vợ chồng, phân chia trách nhiệm hoặc việc dạy con',
        'health_story': 'sức khỏe, áp lực, ăn uống hoặc giảm cân',
        'work_story': 'công việc, khách hàng hoặc tình huống nghề nghiệp',
        'discussion': 'phần trao đổi chính của cuộc trò chuyện',
        'other': 'một phần trao đổi khác trong cuộc trò chuyện',
    }
    main_phrase = main_phrase_map.get(main, 'phần trao đổi chính của cuộc trò chuyện')
    if extra and extra != 'transition':
        extra_phrase = main_phrase_map.get(extra, 'một phần trao đổi khác')
        return f"Nội dung chính của người nói này tập trung vào {main_phrase}; ngoài ra còn xuất hiện ở {extra_phrase}."
    if main == 'transition':
        return "Người nói này chủ yếu xuất hiện ở phần dẫn dắt chương trình, chuyển cảnh hoặc giới thiệu phần tiếp theo."
    return f"Nội dung chính của người nói này tập trung vào {main_phrase}."


def is_usable_speaker_summary_candidate(text: str) -> bool:
    text = sanitize_insight_text(text)
    if not text:
        return False
    low = text.lower()
    bad = [
        'đoạn trao đổi',
        'không đủ bằng chứng',
        'chuyển cảnh',
        'dẫn chương trình',
        'giới thiệu phần tiếp theo',
        'tham gia trao đổi trong đoạn này',
        'có nhiều lượt phát biểu trong đoạn này',
    ]
    if any(x in low for x in bad):
        return False
    return len(text.split()) >= 6


def pick_best_block_for_speaker(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    def score(item: Dict[str, Any]) -> float:
        seg_type = str(item.get("_seg_type", "discussion"))
        spans = item.get("evidence_spans", []) or []
        total_span = 0.0
        for span in spans:
            if isinstance(span, list) and len(span) == 2:
                try:
                    total_span += max(0.0, float(span[1]) - float(span[0]))
                except Exception:
                    pass
        seg_bonus = {
            "family_story": 7.0,
            "health_story": 7.0,
            "work_story": 7.0,
            "guest_intro": 5.0,
            "game": 4.0,
            "discussion": 3.0,
            "transition": 1.0,
            "other": 2.0,
        }.get(seg_type, 2.0)

        insight = sanitize_insight_text(str(item.get("insight", "")))
        detail_bonus = 0.0
        if insight and not is_generic_speaker_insight(insight):
            detail_bonus += 6.0

        for kp in item.get("_key_points", []) or []:
            kp_clean = sanitize_summary_text(str(kp), "")
            if kp_clean and not is_generic_speaker_insight(kp_clean):
                detail_bonus += 3.0
                break

        overview = sanitize_summary_text(str(item.get("_overview", "")), "")
        if overview and not is_generic_speaker_insight(overview):
            detail_bonus += 2.0

        return seg_bonus + min(total_span / 20.0, 4.0) + detail_bonus

    valid = [x for x in items if isinstance(x, dict)]
    if not valid:
        return None
    return max(valid, key=score)


def build_best_speaker_insight_text(best_item: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
    raw = sanitize_insight_text(str(best_item.get("insight", "")))
    if raw and not is_generic_speaker_insight(raw):
        return raw

    for kp in best_item.get("_key_points", []) or []:
        kp_clean = sanitize_summary_text(str(kp), "")
        if kp_clean and not is_generic_speaker_insight(kp_clean):
            return kp_clean

    overview = sanitize_summary_text(str(best_item.get("_overview", "")), "")
    if overview and not is_generic_speaker_insight(overview):
        return overview

    seg_type = str(best_item.get("_seg_type", "discussion"))
    many_turns = len(items) >= 2
    return build_rule_speaker_insight(seg_type, many_turns)


def build_best_speaker_main_summary(best_item: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
    candidates = []
    raw = sanitize_insight_text(str(best_item.get('insight', '')))
    if raw:
        candidates.append(raw)
    for kp in best_item.get('_key_points', []) or []:
        kp_clean = sanitize_summary_text(str(kp), '')
        if kp_clean:
            candidates.append(kp_clean)
    overview = sanitize_summary_text(str(best_item.get('_overview', '')), '')
    if overview:
        candidates.append(overview)
    for cand in candidates:
        if is_usable_speaker_summary_candidate(cand):
            return cleanup_repeated_sentences(cand)

    # Fallback: infer from dominant segment types across this speaker's strongest blocks.
    def seg_score(item: Dict[str, Any]) -> float:
        score = 1.0
        for span in item.get('evidence_spans', []) or []:
            if isinstance(span, list) and len(span) == 2:
                try:
                    score += max(0.0, float(span[1]) - float(span[0]))
                except Exception:
                    pass
        return score

    sorted_items = sorted(items, key=seg_score, reverse=True)
    seg_types = [str(x.get('_seg_type', 'discussion')) for x in sorted_items[:2]]
    return build_rule_speaker_main_summary(seg_types)


def fallback_overview_for_type(seg_type: str) -> str:
    mapping = {
        "transition": "Đoạn chuyển cảnh, dẫn chương trình hoặc giới thiệu phần tiếp theo.",
        "game": "Đoạn trò chơi hoặc tương tác nhẹ giữa các người nói.",
        "guest_intro": "Đoạn giới thiệu khách mời hoặc hé lộ chủ đề mới.",
        "family_story": "Đoạn trao đổi về đời sống gia đình, vợ chồng hoặc cách phân chia trách nhiệm.",
        "health_story": "Đoạn chia sẻ về sức khỏe, áp lực, ăn uống hoặc giảm cân.",
        "work_story": "Đoạn trao đổi về công việc, khách hàng hoặc tình huống nghề nghiệp.",
        "discussion": "Đoạn trao đổi chính của cuộc trò chuyện; không đủ bằng chứng để tóm tắt sâu hơn.",
    }
    return mapping.get(seg_type, "Đoạn trao đổi khác.")


def build_meeting_overview_from_groups(groups: List[Dict[str, Any]], meeting_type: str) -> str:
    seg_types = [g.get("_seg_type", "discussion") for g in groups]
    titles = [g.get("title", "") for g in groups]
    if not groups:
        return "Không đủ bằng chứng để kết luận tổng quan rõ ràng."

    if meeting_type == "talkshow":
        parts = []
        if "guest_intro" in seg_types:
            parts.append("Đây là một đoạn talkshow có phần mở đầu và giới thiệu khách mời.")
        else:
            parts.append("Đây là một đoạn talkshow với nhiều phần dẫn dắt và trao đổi.")
        if "family_story" in seg_types:
            parts.append("Nội dung chính xoay quanh câu chuyện gia đình, cách phân chia trách nhiệm giữa vợ chồng hoặc việc dạy con.")
        elif "health_story" in seg_types:
            parts.append("Nội dung chính xoay quanh sức khỏe, áp lực hoặc thói quen sinh hoạt.")
        elif "work_story" in seg_types:
            parts.append("Nội dung chính xoay quanh công việc hoặc tình huống nghề nghiệp.")
        if "transition" in seg_types:
            parts.append("Transcript có nhiều đoạn chuyển cảnh và dẫn chương trình nên cần tránh gộp nhầm các phần khác nhau.")
        return " ".join(parts[:3]).strip()

    good = []
    for g in groups:
        if g.get("_seg_type") == "transition":
            continue
        s = sanitize_summary_text(g.get("summary", ""), "")
        if s:
            good.append(s)
    if good:
        return " ".join(good[:2]).strip()
    if titles:
        uniq = []
        seen = set()
        for t in titles:
            t = normalize_whitespace(t)
            if not t:
                continue
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            uniq.append(t)
        if uniq:
            return "Các phần chính gồm: " + ", ".join(uniq[:3]) + "."
    return "Không đủ bằng chứng để kết luận tổng quan rõ ràng."


def build_conversation_main_summary(groups: List[Dict[str, Any]], meeting_type: str) -> str:
    if not groups:
        return "Không đủ bằng chứng để tóm tắt nội dung chính của cuộc hội thoại."
    non_transition = [g for g in groups if g.get('_seg_type') != 'transition']
    useful = []
    for g in non_transition:
        s = sanitize_summary_text(g.get('summary', ''), '')
        if s and 'không đủ bằng chứng' not in s.lower() and s.lower() != 'đoạn trao đổi khác.':
            useful.append(s)
    if useful:
        uniq = []
        seen = set()
        for s in useful:
            low = s.lower()
            if low in seen:
                continue
            seen.add(low)
            uniq.append(s)
        intro = {
            'talkshow': 'Nội dung chính của cuộc hội thoại',
            'discussion': 'Nội dung chính của cuộc trao đổi',
            'other': 'Nội dung chính của đoạn hội thoại',
        }.get(meeting_type, 'Nội dung chính của đoạn hội thoại')
        return cleanup_repeated_sentences(intro + ' xoay quanh ' + '; '.join(uniq[:2]).rstrip('. ') + '.')
    return build_meeting_overview_from_groups(groups, meeting_type)


def infer_segment_type_from_text(text_blob: str, is_transition_heavy: bool) -> str:
    if is_transition_heavy or any(k in text_blob for k in ["nghỉ giải lao", "quay trở lại", "chiếc hộp"]):
        return "transition"
    if any(k in text_blob for k in ["trò chơi", "game", "đoán", "calo", "thử thách"]):
        return "game"
    if any(k in text_blob for k in ["khách mời", "xin mời", "giới thiệu", "mời bạn vào"]):
        return "guest_intro"
    if any(k in text_blob for k in ["vợ chồng", "gia đình", "đưa đón con", "thời khóa biểu", "khiêu vũ"]):
        return "family_story"
    if any(k in text_blob for k in ["stress", "ăn nhiều", "giảm cân", "sức khỏe", "nhan sắc"]):
        return "health_story"
    if any(k in text_blob for k in ["file", "khách hàng", "công việc", "nghề nghiệp"]):
        return "work_story"
    return "discussion"


def segment_title_from_type(seg_type: str, summary: str) -> str:
    mapping = {
        "transition": "Chuyển cảnh / dẫn chương trình",
        "game": "Trò chơi / tương tác",
        "guest_intro": "Giới thiệu khách mời",
        "family_story": "Chia sẻ về gia đình",
        "health_story": "Chia sẻ về sức khỏe / áp lực",
        "work_story": "Chia sẻ về công việc",
        "discussion": "Trao đổi chính",
        "other": "Trao đổi khác",
    }
    return mapping.get(seg_type, "Trao đổi khác")


def normalize_block_rule(block: Block) -> Dict[str, Any]:
    clean_lines = []
    for s in block.segments:
        text = clean_segment_text(s.text)
        text = re.sub(r"\b(ờm|ừm|ờ)\b", "", text, flags=re.IGNORECASE)
        text = normalize_whitespace(text)
        clean_lines.append({
            "start": s.start,
            "end": s.end,
            "speaker": s.speaker,
            "clean_text": text,
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
    compact_lines = [
        {
            "start": round(s.start, 1),
            "end": round(s.end, 1),
            "speaker": s.speaker,
            "text": s.text,
        }
        for s in block.segments
    ]
    user_prompt = (
        f"Block ID: {block.block_id}\n"
        f"Time: {block.start:.1f}s - {block.end:.1f}s\n"
        "Input lines JSON:\n"
        f"{json.dumps(compact_lines, ensure_ascii=False)}\n"
    )
    try:
        result = runner.generate_json(NORMALIZE_SYSTEM_PROMPT, user_prompt, cfg.normalize_max_new_tokens)
    except Exception:
        return normalize_block_rule(block)

    clean_lines = []
    for line in result.get("normalized_lines", []):
        try:
            clean_lines.append({
                "start": float(line["start"]),
                "end": float(line["end"]),
                "speaker": str(line["speaker"]),
                "clean_text": clean_segment_text(str(line["clean_text"])),
            })
        except Exception:
            continue
    if not clean_lines:
        return normalize_block_rule(block)
    return {
        "block_id": block.block_id,
        "start": block.start,
        "end": block.end,
        "is_transition_heavy": block.is_transition_heavy,
        "normalized_lines": clean_lines,
        "notes": ["llm_normalized"],
    }


def summarize_block_rule(normalized_block: Dict[str, Any]) -> Dict[str, Any]:
    lines = normalized_block.get("normalized_lines", [])
    text_blob = _normalized_text_blob(normalized_block)
    segment_type = infer_segment_type_from_text(text_blob, normalized_block.get("is_transition_heavy", False))

    overview_map = {
        "transition": "Đoạn chuyển cảnh, dẫn chương trình hoặc giới thiệu phần tiếp theo.",
        "game": "Đoạn trò chơi hoặc tương tác nhẹ giữa các người nói.",
        "guest_intro": "Đoạn giới thiệu khách mời hoặc hé lộ chủ đề mới.",
        "family_story": "Đoạn trao đổi về đời sống gia đình, vợ chồng hoặc cách phân chia trách nhiệm.",
        "health_story": "Đoạn chia sẻ về sức khỏe, áp lực, ăn uống hoặc giảm cân.",
        "work_story": "Đoạn trao đổi về công việc, khách hàng hoặc tình huống nghề nghiệp.",
        "discussion": "Đoạn trao đổi chính của cuộc trò chuyện; không đủ bằng chứng để tóm tắt sâu hơn theo rule-based.",
    }
    overview = overview_map.get(segment_type, "Đoạn trao đổi khác.")

    speakers = []
    seen = set()
    for x in lines:
        spk = x.get("speaker", "unknown")
        if spk not in seen:
            seen.add(spk)
            speakers.append(spk)

    insights = []
    for spk in speakers[:5]:
        spk_lines = [x for x in lines if x.get("speaker") == spk]
        if not spk_lines:
            continue
        first = spk_lines[0]
        last = spk_lines[-1]
        role_hint = "Tham gia trao đổi trong đoạn này."
        if len(spk_lines) >= 2:
            role_hint = "Có nhiều lượt phát biểu trong đoạn này."
        insights.append({
            "speaker": spk,
            "insight": role_hint,
            "evidence_spans": [[first["start"], last["end"]]],
        })

    facts = []
    if lines:
        facts.append({
            "fact": overview,
            "evidence_spans": [[lines[0]["start"], lines[-1]["end"]]],
        })

    key_points = [overview]
    if lines:
        content = next((normalize_whitespace(x.get("clean_text", "")) for x in lines if len(normalize_whitespace(x.get("clean_text", ""))) >= 25), "")
        if content:
            key_points.append(content[:180])

    return {
        "block_id": normalized_block["block_id"],
        "segment_type": segment_type,
        "overview": overview,
        "key_points": key_points[:3],
        "action_items": [],
        "speaker_insights": insights,
        "facts": facts,
        "notes": ["rule_summary"],
    }


def summarize_block_llm(runner: QwenRunner, normalized_block: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, Any]:
    lines = normalized_block.get("normalized_lines", [])
    text = "\n".join(
        f"[{x['start']:.1f}s - {x['end']:.1f}s] {x['speaker']}: {x['clean_text']}" for x in lines
    )
    user_prompt = (
        f"Block ID: {normalized_block['block_id']}\n"
        f"Time: {normalized_block['start']:.1f}s - {normalized_block['end']:.1f}s\n"
        f"Transcript normalized block:\n{text}\n"
    )
    try:
        result = runner.generate_json(LOCAL_SUMMARY_SYSTEM_PROMPT, user_prompt, cfg.local_summary_max_new_tokens)
    except Exception:
        return summarize_block_rule(normalized_block)
    result.setdefault("block_id", normalized_block["block_id"])
    result.setdefault("segment_type", infer_segment_type_from_text(_normalized_text_blob(normalized_block), normalized_block.get("is_transition_heavy", False)))
    result.setdefault("overview", "")
    result.setdefault("key_points", [])
    result.setdefault("action_items", [])
    result.setdefault("speaker_insights", [])
    result.setdefault("facts", [])
    if looks_generic_or_wrong_summary(result):
        return summarize_block_rule(normalized_block)
    return result


def normalize_block(runner: Optional[QwenRunner], block: Block, cfg: PipelineConfig) -> Dict[str, Any]:
    if should_skip_llm_normalize(block, cfg) or runner is None:
        return normalize_block_rule(block)
    return normalize_block_llm(runner, block, cfg)


def summarize_block(runner: Optional[QwenRunner], normalized_block: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, Any]:
    if should_skip_llm_summary(normalized_block, cfg) or runner is None:
        return summarize_block_rule(normalized_block)
    return summarize_block_llm(runner, normalized_block, cfg)


def dedup_action_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = (
            str(item.get("owner", None)).strip().lower(),
            str(item.get("task", None)).strip().lower(),
            tuple(item.get("evidence_span", [])) if isinstance(item.get("evidence_span"), list) else (),
        )
        if key in seen or not item.get("task"):
            continue
        seen.add(key)
        out.append(item)
    return out


def reduce_summaries_rule(block_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped_segments: List[Dict[str, Any]] = []
    action_items: List[Dict[str, Any]] = []
    per_speaker: Dict[str, List[Dict[str, Any]]] = {}
    risk_flags: List[str] = []

    def push_group(seg_type: str, block_ids: List[int], summaries: List[str]) -> None:
        if not block_ids:
            return
        cleaned = [sanitize_summary_text(s, "") for s in summaries if s]
        cleaned = [s for s in cleaned if s]
        merged_summary = cleanup_repeated_sentences(" ".join(cleaned))
        if not merged_summary:
            merged_summary = fallback_overview_for_type(seg_type)
        grouped_segments.append({
            "title": segment_title_from_type(seg_type, merged_summary),
            "block_ids": block_ids[:],
            "summary": merged_summary,
            "_seg_type": seg_type,
        })

    current_type = None
    current_block_ids: List[int] = []
    current_summaries: List[str] = []
    non_transition_types = set()

    for bs in block_summaries:
        bid = bs.get("block_id")
        seg_type = bs.get("segment_type") or infer_segment_type_from_text(str(bs.get("overview", "")).lower(), False)
        overview = sanitize_summary_text(str(bs.get("overview", "")), fallback_overview_for_type(seg_type))
        if seg_type != "transition":
            non_transition_types.add(seg_type)

        if current_type is None:
            current_type = seg_type
        if seg_type != current_type or (current_block_ids and bid != current_block_ids[-1] + 1):
            push_group(current_type, current_block_ids, current_summaries)
            current_type = seg_type
            current_block_ids = []
            current_summaries = []
        current_block_ids.append(bid)
        current_summaries.append(overview)

        for item in bs.get("action_items", []):
            if item and isinstance(item, dict):
                action_items.append(item)

        raw_insights = bs.get("speaker_insights", [])
        if isinstance(raw_insights, dict):
            raw_insights = [{"speaker": k, "insight": v, "evidence_spans": []} for k, v in raw_insights.items()]
        for item in raw_insights:
            if not isinstance(item, dict):
                continue
            spk = item.get("speaker", "unknown")
            insight = sanitize_insight_text(item.get("insight", ""))
            cleaned_item = {
                "speaker": spk,
                "insight": insight,
                "evidence_spans": item.get("evidence_spans", []),
                "_block_id": bid,
                "_seg_type": seg_type,
                "_overview": overview,
                "_key_points": bs.get("key_points", []),
            }
            per_speaker.setdefault(spk, []).append(cleaned_item)

    push_group(current_type, current_block_ids, current_summaries)

    final_insights = []
    final_speaker_main_summaries = []
    for speaker, items in sorted(per_speaker.items()):
        best_item = pick_best_block_for_speaker(items)
        if best_item is None:
            continue

        insight_text = build_best_speaker_insight_text(best_item, items)
        insight_text = sanitize_insight_text(insight_text) or build_rule_speaker_insight(str(best_item.get("_seg_type", "discussion")), len(items) >= 2)
        main_summary = build_best_speaker_main_summary(best_item, items)
        main_summary = sanitize_summary_text(main_summary, "") or build_rule_speaker_main_summary([str(best_item.get("_seg_type", "discussion"))])

        best_spans = []
        for span in best_item.get("evidence_spans", []):
            if isinstance(span, list) and len(span) == 2:
                best_spans.append(span)
        if not best_spans:
            for item in items:
                for span in item.get("evidence_spans", []):
                    if isinstance(span, list) and len(span) == 2:
                        best_spans.append(span)
                if best_spans:
                    break

        final_insights.append({
            "speaker": speaker,
            "insight": cleanup_repeated_sentences(insight_text),
            "evidence_spans": best_spans[:3],
        })
        final_speaker_main_summaries.append({
            "speaker": speaker,
            "main_summary": cleanup_repeated_sentences(main_summary),
            "evidence_spans": best_spans[:3],
        })

    if len(non_transition_types) > 1:
        risk_flags.append("Transcript có nhiều segment/chủ đề khác nhau; cần chú ý tránh gộp nhầm.")

    if "family_story" in non_transition_types or "health_story" in non_transition_types:
        meeting_type = "talkshow"
    elif "work_story" in non_transition_types or "game" in non_transition_types:
        meeting_type = "discussion"
    else:
        meeting_type = "other"

    meeting_overview = build_meeting_overview_from_groups(grouped_segments, meeting_type)
    conversation_main_summary = build_conversation_main_summary(grouped_segments, meeting_type)

    clean_segments = []
    for g in grouped_segments:
        clean_segments.append({
            "title": g["title"],
            "block_ids": g["block_ids"],
            "summary": g["summary"],
        })

    return {
        "meeting_overview": meeting_overview,
        "conversation_main_summary": conversation_main_summary,
        "meeting_type": meeting_type,
        "segments": clean_segments,
        "action_items": dedup_action_items(action_items),
        "speaker_insights": final_insights,
        "speaker_main_summaries": final_speaker_main_summaries,
        "risk_flags": risk_flags,
        "quality_notes": ["rule_reducer_v5_main_conversation_and_per_speaker_summary"],
    }


def reduce_summaries_llm(runner: QwenRunner, block_summaries: List[Dict[str, Any]], cfg: PipelineConfig) -> Dict[str, Any]:
    packed = json.dumps(block_summaries, ensure_ascii=False, indent=2)
    user_prompt = f"Dưới đây là block summaries:\n{packed}"
    return runner.generate_json(REDUCER_SYSTEM_PROMPT, user_prompt, cfg.reducer_max_new_tokens)


def reduce_summaries(runner: Optional[QwenRunner], block_summaries: List[Dict[str, Any]], cfg: PipelineConfig) -> Dict[str, Any]:
    if cfg.use_rule_reducer or runner is None or cfg.reducer_max_new_tokens <= 0:
        return reduce_summaries_rule(block_summaries)
    return reduce_summaries_llm(runner, block_summaries, cfg)


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


def build_final_output(source_path: str, blocks: List[Block], normalized_blocks: List[Dict[str, Any]], block_summaries: List[Dict[str, Any]], reduced: Dict[str, Any]) -> Dict[str, Any]:
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
        "speaker_main_summaries": reduced.get("speaker_main_summaries", reduced.get("speaker_insights", [])),
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
            lines.append(f"- owner: {item.get('owner', None)}")
            lines.append(f"  - task: {item.get('task', None)}")
            lines.append(f"  - deadline: {item.get('deadline', None)}")
            lines.append(f"  - evidence_span: {fmt_span(item.get('evidence_span', None))}")
            lines.append(f"  - confidence: {item.get('confidence', 'unknown')}")

    lines.append("\n# 3. CHI TIẾT THEO NGƯỜI NÓI (SPEAKER INSIGHTS)\n")
    insights = final_data.get("speaker_insights", [])
    if not insights:
        lines.append("Không đủ bằng chứng để trích xuất speaker insights rõ ràng.")
    else:
        for item in insights:
            lines.append(f"- **{item.get('speaker', 'unknown')}**: {item.get('insight', '')}")
            spans = item.get("evidence_spans", [])
            if spans:
                lines.append(f"  - evidence_spans: {', '.join(fmt_span(x) for x in spans)}")

    lines.append("\n# 4. KỊCH BẢN CHUẨN HOÁ (CLEANED TRANSCRIPT)\n")
    for row in final_data.get("cleaned_transcript", []):
        lines.append(f"[{row['start']:.1f}s - {row['end']:.1f}s] {row['speaker']}: {row['text']}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = PipelineConfig(
        model_name=args.model_name,
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
    cfg.do_sample = args.do_sample
    cfg.merge_gap_sec = args.merge_gap_sec
    cfg.block_max_chars = args.block_max_chars
    cfg.block_max_duration_sec = args.block_max_duration_sec
    cfg.min_text_chars = args.min_text_chars
    cfg.keep_single_word_fillers = args.keep_fillers
    cfg.smart_skip_normalize = not args.disable_smart_skip_normalize
    cfg.smart_skip_summary = not args.disable_smart_skip_summary
    cfg.use_rule_reducer = not args.use_llm_reducer

    print(f"[1/7] Loading transcript: {args.input}")
    segments = load_transcript(args.input)
    print(f"  -> raw segments: {len(segments)}")

    print("[2/7] Deterministic preprocessing")
    segments = filter_segments(segments, cfg)
    segments = merge_adjacent_segments(segments, cfg)
    blocks = split_into_blocks(segments, cfg)
    print(f"  -> filtered+merged segments: {len(segments)}")
    print(f"  -> blocks: {len(blocks)}")

    need_llm = True
    if args.rule_only:
        need_llm = False

    runner: Optional[QwenRunner] = None
    if need_llm:
        print("[3/7] Loading model")
        runner = QwenRunner(cfg)
    else:
        print("[3/7] rule_only=True -> skip model loading")

    normalized_blocks: List[Dict[str, Any]] = []
    print("[4/7] Stage A - normalize each block")
    for block in blocks:
        mode = "rule" if should_skip_llm_normalize(block, cfg) or runner is None else "llm"
        print(f"  -> normalize block {block.block_id} [{block.start:.1f}s - {block.end:.1f}s] mode={mode}")
        normalized_blocks.append(normalize_block(runner, block, cfg))

    block_summaries: List[Dict[str, Any]] = []
    print("[5/7] Stage B - summarize each normalized block")
    for nb in normalized_blocks:
        mode = "rule" if should_skip_llm_summary(nb, cfg) or runner is None else "llm"
        print(f"  -> summarize block {nb['block_id']} mode={mode}")
        block_summaries.append(summarize_block(runner, nb, cfg))

    print("[6/7] Stage C - reducer")
    reduced = reduce_summaries(runner, block_summaries, cfg)

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
    ap = argparse.ArgumentParser(description="Fast multi-step Qwen2.5 pipeline for DER + Whisper summarization")
    ap.add_argument("--input", type=str, required=True, help="Path to transcript .jsonl or bracket-format .txt")
    ap.add_argument("--output_json", type=str, required=True, help="Path to final structured JSON output")
    ap.add_argument("--output_md", type=str, default=None, help="Optional path to final Markdown output")
    ap.add_argument("--dump_blocks_dir", type=str, default=None, help="Optional directory to dump intermediate blocks")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_path", type=str, default=None, help="Optional PEFT adapter path")
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    ap.add_argument("--enable_thinking", action="store_true", help="Ignored for Qwen2.5 (kept only for backward compatibility)")
    ap.add_argument("--rule_only", action="store_true", help="Skip model loading and use only rule-based pipeline")
    ap.add_argument("--use_llm_reducer", action="store_true", help="Use LLM reducer instead of rule-based reducer")

    ap.add_argument("--max_input_tokens", type=int, default=3072)
    ap.add_argument("--normalize_max_new_tokens", type=int, default=384)
    ap.add_argument("--local_summary_max_new_tokens", type=int, default=256)
    ap.add_argument("--reducer_max_new_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling; default is greedy for speed/stability")

    ap.add_argument("--merge_gap_sec", type=float, default=1.2)
    ap.add_argument("--block_max_chars", type=int, default=3200)
    ap.add_argument("--block_max_duration_sec", type=float, default=180.0)
    ap.add_argument("--min_text_chars", type=int, default=2)
    ap.add_argument("--keep_fillers", action="store_true", help="Keep one-word filler rows like 'ờ', 'à'")
    ap.add_argument("--disable_smart_skip_normalize", action="store_true", help="Force LLM normalize for all blocks")
    ap.add_argument("--disable_smart_skip_summary", action="store_true", help="Force LLM summary for all blocks")
    return ap


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
