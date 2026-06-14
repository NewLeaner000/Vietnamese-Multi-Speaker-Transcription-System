import os
from pathlib import Path
from app.ai_core.audio_preprocess_input import normalize_audio_to_mono16k

enrollment_dir = Path("uploads/enrollment_9")
job_id = 9

print(f"Starting loop on {enrollment_dir}")
for e_file in enrollment_dir.rglob("*"):
    if e_file.is_file():
        with open(f"d:/website/test_debug_job_{job_id}.log", "a") as f:
            f.write(f"Processing: {e_file}\n")
        print(f"File: {e_file}, suffix: {e_file.suffix}")
        if e_file.suffix.lower() != ".wav":
            # Convert sang wav 16kHz Mono
            wav_path = e_file.with_suffix(".wav")
            try:
                print(f"Calling normalize on {e_file}")
                normalize_audio_to_mono16k(str(e_file), str(wav_path))
                print("Returned from normalize")
                e_file.unlink()
                with open(f"d:/website/test_debug_job_{job_id}.log", "a") as f:
                    f.write(f"Converted {e_file.name} -> {wav_path.name} OK\n")
            except Exception as e:
                print(f"Caught exception: {e}")
                with open(f"d:/website/test_debug_job_{job_id}.log", "a") as f:
                    f.write(f"FAILED to convert {e_file.name}: {e}\n")
        else:
            print("Is wav")
print("Done loop")
