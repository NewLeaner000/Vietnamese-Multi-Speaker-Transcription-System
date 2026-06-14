import os
import sys

# append backend path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ai_core.der.engine import _discover_enrollment_groups
import torchaudio

enrollment_dir = os.path.abspath("uploads/enrollment_2")
print("Discovering groups in:", enrollment_dir)
groups = _discover_enrollment_groups(enrollment_dir)
print("Groups found:", groups)

from app.ai_core.der.engine import _load_audio_mono16k
for spk_name, files in groups:
    for fpath in files:
        print(f"Testing load for {fpath}...")
        try:
            wav = _load_audio_mono16k(fpath)
            print("Loaded shape:", wav.shape)
        except Exception as e:
            print("Failed to load:", e)
