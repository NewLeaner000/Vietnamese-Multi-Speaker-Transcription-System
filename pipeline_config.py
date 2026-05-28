from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent

# Default paths to the live DER and ASR pipeline entrypoints.
# The interface is designed to run inside one environment, using the current
# Python interpreter for both DER and ASR subprocess execution.
DER_SCRIPT_PATH = Path(os.environ.get(
    "DER_SCRIPT_PATH",
    ROOT_DIR / "core" / "der" / "engine.py",
)).resolve()
PYANNOTE_SCRIPT_PATH = Path(os.environ.get(
    "PYANNOTE_SCRIPT_PATH",
    ROOT_DIR / "core" / "der" / "pyannote_engine.py",
)).resolve()
ASR_SCRIPT_PATH = Path(os.environ.get(
    "ASR_SCRIPT_PATH",
    ROOT_DIR / "asr_runner.py",
)).resolve()

QWEN_SCRIPT_PATH = Path(os.environ.get(
    "QWEN_SCRIPT_PATH",
    ROOT_DIR / "core" / "qwen" / "engine.py",
)).resolve()
QWEN_NORMALIZE_MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_SUMMARY_MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"

# Default checkpoint paths for DER and ASR (Local Core)
DER_CHECKPOINT_DEFAULT = str(ROOT_DIR / "core" / "der" / "checkpoints" / "best_model.pth")
PYANNOTE_CHECKPOINT_DEFAULT = str(ROOT_DIR / "core" / "der" / "checkpoints" / "pyannote_best.pth")
ASR_CHECKPOINT_DEFAULT = str(ROOT_DIR / "core" / "asr" / "checkpoints" / "best_adapter")

# Default output folder for interface-run results. The interface writes DER and ASR
# artifacts under this folder to keep results isolated from training and source code.
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
