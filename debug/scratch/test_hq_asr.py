import torch
import soundfile as sf
import json
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def test_hq():
    model_path = r"C:\ai_diarizen5090\code_pho_whisper\checkpoints\training\stage2\best_adapter"
    audio_path = r"C:\ai_diarizen5090\website\Dis\raw.wav"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path}...")
    
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # Segment 1: [0.581s - 3.364s] from GT
    start_time = 0.581
    end_time = 3.364
    
    data, sr = sf.read(audio_path, start=int(start_time * 48000), stop=int(end_time * 48000))
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Resample to 16k if needed (sf.read sr is likely 48k)
    import librosa
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    
    inputs = processor(data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    if device == "cuda":
        inputs = inputs.to(torch.float16)
        
    print("Generating transcription...")
    with torch.no_grad():
        generated_ids = model.generate(
            inputs,
            max_new_tokens=256,
            language="vi",
            task="transcribe",
            num_beams=5
        )
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    with open(r"c:\ai_diarizen5090\scratch\test_hq_result.txt", "w", encoding="utf-8") as f:
        f.write(f"GT: Vậy thôi có gì đâu.\n")
        f.write(f"HQ ASR: {transcription}\n")
    
    print("Test complete. Results saved to c:\\ai_diarizen5090\\scratch\\test_hq_result.txt")

if __name__ == "__main__":
    test_hq()
