import json
import csv
import re
from pathlib import Path

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return get_levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_wer(reference, hypothesis):
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = get_levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)

def evaluate():
    gt_path = Path(r"c:\ai_diarizen5090\code_pho_whisper\results\inference_rttm\results_rttm.jsonl")
    my_path = Path(r"c:\ai_diarizen5090\website\output_auto\asr\asr_results.csv")
    
    if not gt_path.exists() or not my_path.exists():
        print(f"Error: Missing files. GT: {gt_path.exists()}, MY: {my_path.exists()}")
        return

    # Load Ground Truth
    gt_segments = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            gt_segments.append(json.loads(line))
    
    # Load My ASR
    my_segments = []
    with my_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            my_segments.append(row)

    print(f"Comparing {len(my_segments)} hypothesis segments vs {len(gt_segments)} reference segments.")

    matches = []
    speaker_map_counts = {} # (my_speaker, gt_speaker) -> count

    for my_seg in my_segments:
        m_start = float(my_seg['start'])
        m_end = float(my_seg['end'])
        m_text = my_seg['predicted_text']
        m_label = my_seg['data_name'].split('_')[-2] + "_" + my_seg['data_name'].split('_')[-1] # unknown_X
        
        # Simple time-based overlap matching
        best_gt = None
        max_overlap = 0
        for gt_seg in gt_segments:
            overlap = min(m_end, gt_seg['end']) - max(m_start, gt_seg['start'])
            if overlap > max_overlap:
                max_overlap = overlap
                best_gt = gt_seg
        
        if best_gt and max_overlap > 0.5:
            wer = calculate_wer(best_gt['transcription'], m_text)
            matches.append({
                'my_label': m_label,
                'gt_speaker': best_gt['speaker'],
                'wer': wer,
                'text': m_text,
                'ref': best_gt['transcription']
            })
            
            # Count for speaker mapping
            key = (m_label, best_gt['speaker'])
            speaker_map_counts[key] = speaker_map_counts.get(key, 0) + 1

    # Determine best speaker mapping
    final_mapping = {}
    seen_gt = set()
    # Sort by frequency to get best matches first
    sorted_counts = sorted(speaker_map_counts.items(), key=lambda x: x[1], reverse=True)
    for (m_label, gt_speaker), count in sorted_counts:
        if m_label not in final_mapping:
            final_mapping[m_label] = gt_speaker
            seen_gt.add(gt_speaker)

    report_path = Path(r"c:\ai_diarizen5090\scratch\asr_evaluation_report.txt")
    with report_path.open("w", encoding="utf-8") as rf:
        rf.write("[SPEAKER MAPPING RESULTS]\n")
        for m, g in final_mapping.items():
            rf.write(f"  {m} -> {g}\n")

        rf.write("\n[WER RESULTS]\n")
        total_wer = sum(m['wer'] for m in matches)
        avg_wer = total_wer / len(matches) if matches else 0
        rf.write(f"  Average WER: {avg_wer:.2%}\n")
        rf.write(f"  Matched Segments: {len(matches)}/{len(my_segments)}\n")
        
        rf.write("\n[SAMPLE COMPARISON (SIDE-BY-SIDE)]\n")
        for i, m in enumerate(matches):
            rf.write(f"--- Segment {i} ---\n")
            rf.write(f"Speaker: {m['my_label']} (Mapped: {m['gt_speaker']})\n")
            rf.write(f"REF: {m['ref']}\n")
            rf.write(f"HYP: {m['text']}\n")
            rf.write(f"WER: {m['wer']:.2%}\n\n")

    # Output simplified summary to console
    print(f"Evaluation complete. Average WER: {avg_wer:.2%}")
    print(f"Speaker mapping determined. Results saved to {report_path}")

evaluate()
