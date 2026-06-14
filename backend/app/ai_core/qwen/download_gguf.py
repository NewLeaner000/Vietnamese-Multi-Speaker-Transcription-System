import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Fix stdout encoding for Windows
sys.stdout.reconfigure(encoding="utf-8")

def main():
    repo_id = "bartowski/Qwen2.5-7B-Instruct-GGUF"
    filename = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    
    # Define local path
    base_dir = Path(__file__).resolve().parent
    local_dir = base_dir / "checkpoints"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {filename} from {repo_id}...")
    print(f"Destination: {local_dir}")
    print("This is a ~4.5GB file, it may take a few minutes depending on your connection.")
    
    try:
        # Download the file
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Better for Windows compatibility
        )
        print(f"\n[DONE] Download complete!")
        print(f"File saved to: {local_path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")

if __name__ == "__main__":
    main()
