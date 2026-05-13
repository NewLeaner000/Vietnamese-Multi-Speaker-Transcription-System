import sys
from pathlib import Path
sys.path.append(r"C:\ai_diarizen5090\website")
from qwen_infer_bridge import convert_asr_csv_to_jsonl

csv_path = Path(r"C:\ai_diarizen5090\website\output\asr\asr_results.csv")
jsonl_path = Path(r"C:\ai_diarizen5090\website\output\qwen\qwen_transcript_v2.jsonl")

print("Converting CSV to JSONL with deduplication...")
convert_asr_csv_to_jsonl(csv_path, jsonl_path)
print(f"Done. Check {jsonl_path}")
