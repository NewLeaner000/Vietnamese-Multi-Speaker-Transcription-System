import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from app.ai_core.pipeline_config import ASR_SCRIPT_PATH


def _find_python_executable() -> str:
    return sys.executable


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def build_asr_command(
    audio_path: Path,
    rttm_path: Path,
    asr_mode: str,
    output_dir: Path,
    checkpoint_path: Optional[Path],
    use_fast_mode: bool = False,
) -> List[str]:
    python_exec = _find_python_executable()
    _ensure_exists(ASR_SCRIPT_PATH, "ASR script")
    command = [python_exec, str(ASR_SCRIPT_PATH)]
    command += [
        "--audio", str(audio_path), 
        "--rttm", str(rttm_path), 
        "--mode", asr_mode, 
        "--output_dir", str(output_dir),
    ]
    if use_fast_mode:
        command += ["--fast"]
    if checkpoint_path:
        # HuggingFace Hub requires forward slashes even on Windows
        command += ["--checkpoint", str(checkpoint_path).replace("\\", "/")]
    return command


def run_asr_pipeline(
    audio_path: Path,
    rttm_path: Path,
    asr_mode: str,
    output_dir: Path,
    checkpoint_path: Optional[Path] = None,
    use_fast_mode: bool = False,
):
    command = build_asr_command(
        audio_path, rttm_path, asr_mode, output_dir, checkpoint_path, 
        use_fast_mode=use_fast_mode
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout for easier parsing
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    
    for line in process.stdout:
        yield line.strip()
        
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ASR pipeline failed with return code {process.returncode}")

