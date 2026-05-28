import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from pipeline_config import DER_SCRIPT_PATH, PYANNOTE_SCRIPT_PATH


def _find_python_executable() -> str:
    return sys.executable


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def build_der_command(
    audio_path: Path,
    enrollment_dir: Optional[Path],
    n_speakers: int,
    checkpoint_path: Optional[Path],
    output_dir: Path,
    script_path: Path,
    segmentation_step: float = 0.1,
) -> List[str]:
    python_exec = _find_python_executable()
    _ensure_exists(script_path, "DER script")
    command = [python_exec, str(script_path)]
    command += ["--wav", str(audio_path), "--n_speakers", str(n_speakers), "--out_json", str(output_dir / "der_results.json"), "--skip_der"]
    command += ["--step", str(segmentation_step)]
    if enrollment_dir:
        command += ["--enrollment_dir", str(enrollment_dir)]
    if checkpoint_path:
        command += ["--ckpt", str(checkpoint_path)]
    return command


def run_der_pipeline(
    audio_path: Path,
    enrollment_dir: Optional[Path],
    n_speakers: int,
    checkpoint_path: Optional[Path],
    output_dir: Path,
    script_path: Path,
    segmentation_step: float = 0.1,
):
    command = build_der_command(
        audio_path, enrollment_dir, n_speakers, checkpoint_path, output_dir, script_path,
        segmentation_step=segmentation_step
    )

    yield f"  [DEBUG] Command: {' '.join(command)}"
    if enrollment_dir:
        enr_p = Path(enrollment_dir)
        if enr_p.exists():
            contents = list(enr_p.glob("**/*"))
            yield f"  [DEBUG] Enrollment dir exists: {enr_p}"
            yield f"  [DEBUG] Contents: {[str(c.relative_to(enr_p)) for c in contents if c.is_file()]}"
        else:
            yield f"  [DEBUG] Enrollment dir NOT found: {enr_p}"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    
    for line in process.stdout:
        yield line.strip()
        
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"DER pipeline failed with return code {process.returncode}")
