import os
import time

files = [
    r"C:\ai_diarizen5090\website\output\input_audio\raw.wav",
    r"C:\ai_diarizen5090\website\output\der\der_report.json",
    r"C:\ai_diarizen5090\website\output\asr\asr_results.csv",
    r"C:\ai_diarizen5090\website\output\qwen\qwen_summary.md"
]

print(f"{'File':<60} | {'Modified Time'}")
print("-" * 85)

results = []
for f in files:
    if os.path.exists(f):
        mtime = os.path.getmtime(f)
        t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        print(f"{os.path.basename(f):<60} | {t_str}")
        results.append(mtime)
    else:
        print(f"{os.path.basename(f):<60} | NOT FOUND")

if len(results) >= 2:
    print("\n[Duration Estimates]")
    for i in range(1, len(results)):
        dur = results[i] - results[i-1]
        phase = ["DER", "ASR", "Qwen"][i-1]
        print(f"Phase {i} ({phase}): {dur:.2f} seconds ({dur/60:.2f} minutes)")
