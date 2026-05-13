import sys
import os
from pathlib import Path
import json

# Add website dir to path to import bridges
sys.path.append(r"C:\ai_diarizen5090\website")

from der_infer_bridge import run_der_pipeline
from asr_infer_bridge import run_asr_pipeline
from qwen_infer_bridge import run_qwen_pipeline
from pipeline_config import DEFAULT_OUTPUT_DIR, DER_CHECKPOINT_DEFAULT

def auto_run():
    # 1. Setup Paths
    base_dis = Path(r"C:\ai_diarizen5090\website\Dis")
    audio_path = base_dis / "raw.wav"
    enroll_dir = base_dis / "enrollment"
    output_dir = Path(r"C:\ai_diarizen5090\website\output_auto") # Use a clean output dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    der_out = output_dir / "der"
    asr_out = output_dir / "asr"
    qwen_out = output_dir / "qwen"
    
    print(">>> PHASE 1: DIARIZATION (DER)")
    print(f"    Audio: {audio_path}")
    print(f"    Enrollment: {enroll_dir}")
    
    for line in run_der_pipeline(
        audio_path=audio_path,
        enrollment_dir=enroll_dir,
        n_speakers=4,
        checkpoint_path=Path(DER_CHECKPOINT_DEFAULT) if DER_CHECKPOINT_DEFAULT else None,
        output_dir=der_out
    ):
        print(f"    {line}")

    # 2. Find RTTM
    rttm_files = list(der_out.rglob("*.rttm"))
    if not rttm_files:
        print("ERROR: No RTTM found!")
        return
    rttm_path = rttm_files[0]
    print(f">>> Found RTTM: {rttm_path}")

    # 3. ASR
    from pipeline_config import ASR_CHECKPOINT_DEFAULT
    print(">>> PHASE 2: ASR TRANSCRIPTION")
    for line in run_asr_pipeline(
        audio_path=audio_path,
        rttm_path=rttm_path,
        asr_mode="whisper_only",
        output_dir=asr_out,
        checkpoint_path=ASR_CHECKPOINT_DEFAULT
    ):
        if line.startswith("PROGRESS:"):
            print(f"    {line}", end="\r")
    print("\n    ASR Completed.")

    # 4. QWEN
    print(">>> PHASE 3: QWEN SUMMARIZATION (CLEANED)")
    qwen_msg = run_qwen_pipeline(
        asr_csv_path=asr_out / "asr_results.csv",
        output_dir=qwen_out
    )
    print("    Qwen Completed.")
    print("-" * 50)
    print("AUTO RUN FINISHED.")

if __name__ == "__main__":
    auto_run()
