
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from pipeline_config import QWEN_NORMALIZE_MODEL_DEFAULT, QWEN_SCRIPT_PATH, QWEN_SUMMARY_MODEL_DEFAULT


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _parse_speaker(data_name: str) -> str:
    if not data_name:
        return "speaker_unknown"
    parts = data_name.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else data_name


def _is_hallucination(text: str) -> bool:
    hallucinations = [
        "tuy nhiên không phải ai cũng có thể thực hiện điều này",
        "một ngày sinh hoạt và thời",
        "một ngày qua tôi vẫn mong đợi",
    ]
    text_l = text.lower().strip()
    for h in hallucinations:
        if h in text_l:
            return True
    if len(text_l.split()) <= 1 and text_l in ["unk", "là", "ờ", "ừ", "hả"]:
        return True
    return False


def _calculate_similarity(s1: str, s2: str) -> float:
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    return len(intersection) / max(len(set1), len(set2))


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


def convert_asr_csv_to_jsonl(csv_path: Path, jsonl_path: Path) -> dict:
    t0 = time.perf_counter()
    raw_rows = []
    kept_rows = 0
    skipped_rows = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)
            text = (row.get("predicted_text") or "").strip()
            if not text or _is_hallucination(text):
                skipped_rows += 1
                continue
            try:
                start = float(row.get("start", 0.0))
                end = float(row.get("end", start))
            except ValueError:
                skipped_rows += 1
                continue

            data_name = row.get("data_name", "")
            speaker = (row.get("speaker") or "").strip() or _parse_speaker(data_name)
            kept_rows += 1
            row["_normalized"] = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }

    filtered_rows = []
    duplicates_removed = 0
    for row in raw_rows:
        current = row.get("_normalized")
        if not current:
            continue
        is_duplicate = False
        for existing in filtered_rows:
            time_diff = abs(current["start"] - existing["start"]) + abs(current["end"] - existing["end"])
            if time_diff < 1.0:
                sim = _calculate_similarity(current["text"], existing["text"])
                if sim > 0.8:
                    is_duplicate = True
                    duplicates_removed += 1
                    break
        if not is_duplicate:
            filtered_rows.append(current)

    if not filtered_rows:
        raise RuntimeError(f"No valid ASR rows found in {csv_path} after filtering.")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in filtered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    return {
        "raw_rows": len(raw_rows),
        "kept_rows": kept_rows,
        "skipped_rows": skipped_rows,
        "duplicates_removed": duplicates_removed,
        "final_rows": len(filtered_rows),
        "elapsed_sec": round(elapsed, 4),
    }


def build_qwen_command(
    transcript_jsonl: Path,
    output_json: Path,
    output_md: Path,
    normalize_model_name: str,
    summary_model_name: str,
    no_4bit: bool,
    qwen_mode: str = "stable",
) -> list[str]:
    _ensure_exists(QWEN_SCRIPT_PATH, "Qwen pipeline script")
    command = [sys.executable, str(QWEN_SCRIPT_PATH)]
    command += ["--input", str(transcript_jsonl)]
    command += ["--output_json", str(output_json)]
    command += ["--output_md", str(output_md)]
    command += ["--normalize_model_name", normalize_model_name.replace("\\", "/")]
    command += ["--summary_model_name", summary_model_name.replace("\\", "/")]
    command += ["--qwen_mode", qwen_mode]
    if no_4bit:
        command += ["--no_4bit"]
    return command


def run_qwen_pipeline(
    asr_csv_path: Path,
    output_dir: Path,
    normalize_model_name: Optional[str] = None,
    summary_model_name: Optional[str] = None,
    no_4bit: bool = False,
    qwen_mode: str = "stable",
):
    _ensure_exists(asr_csv_path, "ASR result CSV")
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_jsonl = output_dir / "qwen_transcript.jsonl"
    qwen_output_json = output_dir / "qwen_summary.json"
    qwen_output_md = output_dir / "qwen_summary.md"

    normalize_model_name = normalize_model_name or QWEN_NORMALIZE_MODEL_DEFAULT
    summary_model_name = summary_model_name or QWEN_SUMMARY_MODEL_DEFAULT

    yield f"[QWEN-BRIDGE] mode={qwen_mode} | normalize_model={normalize_model_name} | summary_model={summary_model_name} | 4bit={'off' if no_4bit else 'on'}"

    stats = convert_asr_csv_to_jsonl(asr_csv_path, transcript_jsonl)
    yield (
        "[QWEN-BRIDGE] transcript_jsonl_ready | "
        f"raw_rows={stats['raw_rows']} | kept_rows={stats['kept_rows']} | skipped_rows={stats['skipped_rows']} | "
        f"duplicates_removed={stats['duplicates_removed']} | final_rows={stats['final_rows']} | "
        f"elapsed={_format_seconds(stats['elapsed_sec'])}"
    )

    command = build_qwen_command(
        transcript_jsonl=transcript_jsonl,
        output_json=qwen_output_json,
        output_md=qwen_output_md,
        normalize_model_name=normalize_model_name,
        summary_model_name=summary_model_name,
        no_4bit=no_4bit,
        qwen_mode=qwen_mode,
    )

    yield f"[QWEN-BRIDGE] command_ready | script={QWEN_SCRIPT_PATH.name} | output_dir={output_dir}"

    process_start = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        yield line.strip()

    process.wait()
    total_elapsed = time.perf_counter() - process_start
    yield f"[QWEN-BRIDGE] process_finished | return_code={process.returncode} | elapsed={_format_seconds(total_elapsed)}"

    if process.returncode != 0:
        raise RuntimeError(f"Qwen pipeline failed with return code {process.returncode}")
