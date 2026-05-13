import re
import sys

# Force UTF-8 for printing
sys.stdout.reconfigure(encoding='utf-8')

def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

def cleanup_repeated_sentences(text: str) -> str:
    if not text:
        return ""
    # 1. Deduplicate sequential identical phrases (e.g., "word word word")
    # This regex catches repeating chunks of 1-5 words
    # We use (?:...)+ to catch any number of repeats
    for _ in range(2): # Run twice to catch nested repeats
        text = re.sub(r'\b(.+?)(?:\s+\1\b){1,}', r'\1', text, flags=re.IGNORECASE)
    
    # 2. Deduplicate whole sentences
    parts = [normalize_whitespace(x) for x in re.split(r"(?<=[\.!?])\s+", text) if normalize_whitespace(x)]
    out = []
    seen = set()
    for p in parts:
        low = p.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(p)
    return " ".join(out).strip()

sample = "Tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là tất nhiên là t."

print(f"Original length: {len(sample)}")
cleaned = cleanup_repeated_sentences(sample)
print(f"Cleaned: {cleaned}")
print(f"Match: {cleaned == 'Tất nhiên là t.'}")
