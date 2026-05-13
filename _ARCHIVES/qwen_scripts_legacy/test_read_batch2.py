import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
for d in ['data7', 'data8', 'data9', 'data10', 'data11']:
    path = f'c:/ai_diarizen5090/code_qwen_25_7b/data_labeled/Tong_hop_data_labelled/{d}/labeled/transcript.jsonl'
    print(f'\n--- BEGIN {d} ---')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f.readlines()][:15]
            for l in lines:
                print(f"{l.get('speaker', 'Unknown')}: {l.get('text', '')}")
    except Exception as e:
        print(f"Error: {e}")
