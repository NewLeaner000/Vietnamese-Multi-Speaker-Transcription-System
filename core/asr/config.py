from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

DEFAULT_MODEL_NAME = "vinai/PhoWhisper-large"
DEFAULT_LANGUAGE = "vi"
DEFAULT_TASK = "transcribe"
DEFAULT_SAMPLE_RATE = 16000

MIN_KEEP_SEC = 2.5
PREFERRED_MAX_SEC = 18.0
HARD_MAX_SEC = 30.0
MERGE_GAP_SEC = 0.50          # gap tối đa để merge 2 segment khác speaker
MERGE_GAP_SAME_SPK_SEC = 1.50 # gap tối đa để merge 2 segment CÙNG speaker (có điều kiện)
PAD_LEFT_SEC = 0.10
PAD_RIGHT_SEC = 0.20

MAX_OVERLAP_RATIO_STAGE1 = 0.18
MAX_OVERLAP_RATIO_STAGE2 = 0.30

CLASS_TO_BUCKET: Dict[str, str] = {
    "vif": "gold_real",
    "conan": "silver_real",
    "dustin_on_go": "silver_real",
    "dustin": "silver_real",
    "chuyen_ho_chuyen_minh": "silver_real",
    "chuyen_ho": "silver_real",
    "coi_mo": "hard_real",
    "coi_moi": "hard_real",
}

KEY_CANDIDATES = {
    "start": ["start", "start_time", "segment_start", "begin", "st", "s"],
    "end": ["end", "end_time", "segment_end", "finish", "et", "e"],
    "text": ["text", "transcript", "sentence", "utterance", "content", "normalized_text"],
    "speaker": ["speaker", "speaker_id", "spk", "spkid", "label"],
}

STAGE1_SOURCE_RATIOS: Dict[str, float] = {
    "vivos": 0.10,
}

STAGE2_SOURCE_RATIOS: Dict[str, float] = {
    "vivos": 0.05,
}

@dataclass(frozen=True)
class StageSpec:
    name: str
    epochs: int
    allowed_overlap_ratio: float
    source_ratios: Dict[str, float]

STAGE1_SPEC = StageSpec(
    name="stage1",
    epochs=2,
    allowed_overlap_ratio=MAX_OVERLAP_RATIO_STAGE1,
    source_ratios=STAGE1_SOURCE_RATIOS,
)

STAGE2_SPEC = StageSpec(
    name="stage2",
    epochs=1,
    allowed_overlap_ratio=MAX_OVERLAP_RATIO_STAGE2,
    source_ratios=STAGE2_SOURCE_RATIOS,
)