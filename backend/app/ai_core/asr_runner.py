import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _copy_or_link(src: Path, dst: Path) -> None:
    """Create a stable file path for core/asr/engine.py without changing ASR logic."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_engine_input(audio_path: Path, rttm_path: Path, output_dir: Path) -> Path:
    """
    Adapt website single-file DER output to the directory layout expected by
    core/asr/engine.py:

        <root>/data0001/mixture.wav
        <root>/data0001/labeled/hyp_low025.rttm

    This only prepares paths/files for the engine. It does not change decoding,
    filtering, retry, trim, threshold, or any ASR parameter.
    """
    engine_root = output_dir / "_engine_input"
    data_dir = engine_root / "data0001"
    labeled_dir = data_dir / "labeled"

    if engine_root.exists():
        shutil.rmtree(engine_root)
    labeled_dir.mkdir(parents=True, exist_ok=True)

    _copy_or_link(audio_path, data_dir / "mixture.wav")
    _copy_or_link(rttm_path, labeled_dir / "hyp_low025.rttm")

    metadata_path = engine_root / "metadata.txt"
    metadata_path.write_text(
        "data0001 | website_pipeline | | 1\n",
        encoding="utf-8",
    )
    return engine_root


def run_core_asr_engine(
    engine_input_dir: Path,
    model_path: Path,
    output_path: Path,
    use_fast_mode: bool = False,
):
    script_path = Path(__file__).resolve().parent / "asr" / "engine.py"
    if not script_path.exists():
        raise FileNotFoundError(f"ASR helper script not found: {script_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(script_path),
        "--dir",
        str(engine_input_dir),
        "--model",
        str(model_path),
        "--out_csv",
        str(output_path),
    ]

    if use_fast_mode:
        command += ["--compute_type", "int8_float16"]
        print(
            "[INFO] ASR fast mode: using int8_float16 compute type for extra speed.",
            flush=True,
        )

    print("[INFO] Running core ASR engine with its default parameters.", flush=True)
    print(f"[INFO] ASR engine input dir: {engine_input_dir}", flush=True)
    print(f"[INFO] ASR output CSV: {output_path}", flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    if process.stdout is not None:
        for line in process.stdout:
            print(line.strip(), flush=True)

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ASR process failed with return code {process.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR bridge wrapper for website full pipeline.")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--rttm", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["whisper_only", "wer_v2", "dicow_only"], default="whisper_only")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=os.environ.get("ASR_MODEL_PATH"))
    parser.add_argument("--fast", action="store_true")
    # Backward-compatible only. The current core ASR engine has no batch-size argument.
    parser.add_argument("--batch_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--enrollment",
        type=str,
        default=None,
        help=(
            "Kept for CLI compatibility. Enrollment metadata is not consumed by "
            "the current core ASR engine."
        ),
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    rttm_path = Path(args.rttm)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not rttm_path.exists():
        raise FileNotFoundError(f"RTTM file not found: {rttm_path}")

    asr_model_path = args.checkpoint
    if not asr_model_path:
        raise ValueError("ASR model path is required.")

    if args.batch_size is not None:
        print(
            "[INFO] --batch_size was provided to the wrapper, but it is not "
            "forwarded because the current core ASR engine has no batch-size parameter.",
            flush=True,
        )

    print(f"[INFO] Running ASR mode {args.mode} with model {asr_model_path}", flush=True)
    engine_input_dir = prepare_engine_input(audio_path, rttm_path, output_dir)
    output_csv = output_dir / "asr_results.csv"
    run_core_asr_engine(
        engine_input_dir=engine_input_dir,
        model_path=Path(asr_model_path),
        output_path=output_csv,
        use_fast_mode=args.fast,
    )
    print(f"[INFO] ASR results written to: {output_csv}", flush=True)


if __name__ == "__main__":
    main()

