import csv
import json
import re

csv_path = r'C:\ai_diarizen5090\website\output_auto\asr\asr_results.csv'
jsonl_path = r'C:\ai_diarizen5090\website\output_auto\asr\qwen_input.jsonl'

def extract_speaker(data_name):
    # Match names like Khang, Sang, Phuong, Duy at the end of the data_name
    match = re.search(r'_([^_]+)$', data_name)
    if match:
        return match.group(1)
    return "Unknown"

with open(csv_path, 'r', encoding='utf-8') as csvfile, \
     open(jsonl_path, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['predicted_text'].strip()
        if not text:
            continue
            
        item = {
            "start": float(row['start']),
            "end": float(row['end']),
            "speaker": extract_speaker(row['data_name']),
            "text": text
        }
        jsonlfile.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Successfully converted {csv_path} to {jsonl_path}")
