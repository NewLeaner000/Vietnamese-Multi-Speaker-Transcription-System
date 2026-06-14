"""
convert_to_ct2.py
─────────────────
One-time script to convert a HuggingFace Whisper checkpoint
to CTranslate2 format for use with faster-whisper.

Usage:
    python convert_to_ct2.py
    python convert_to_ct2.py --quantization int8_float16
    python convert_to_ct2.py --model path/to/custom/checkpoint --output_dir path/to/output
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def main():
    default_model = str(Path(__file__).resolve().parent / "checkpoints" / "best_adapter")
    default_output = str(Path(__file__).resolve().parent / "checkpoints" / "best_adapter_ct2")

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Whisper checkpoint to CTranslate2 format."
    )
    parser.add_argument(
        "--model", type=str, default=default_model,
        help=f"Path to HuggingFace checkpoint (default: {default_model})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=default_output,
        help=f"Output directory for CTranslate2 model (default: {default_output})"
    )
    parser.add_argument(
        "--quantization", type=str, default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
        help="Quantization type (default: float16)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite output directory if it already exists"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output_dir)

    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    if output_path.exists() and not args.force:
        print(f"ERROR: Output directory already exists: {output_path}")
        print("Use --force to overwrite.")
        sys.exit(1)

    print("=" * 60)
    print("  CTranslate2 Whisper Model Conversion")
    print("=" * 60)
    print(f"  Source model:    {model_path}")
    print(f"  Output dir:      {output_path}")
    print(f"  Quantization:    {args.quantization}")
    print("=" * 60)

    try:
        import ctranslate2
    except ImportError:
        print("\n❌ ctranslate2 is not installed.")
        print("   Install it with: pip install ctranslate2")
        sys.exit(1)

    print(f"\n  ctranslate2 version: {ctranslate2.__version__}")

    # In ctranslate2 v4.x, use TransformersConverter (handles Whisper automatically)
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        print("\n❌ TransformersConverter not found in ctranslate2.")
        print("   Please upgrade: pip install --upgrade ctranslate2")
        sys.exit(1)

    if args.force and output_path.exists():
        shutil.rmtree(output_path)

    print("\nConverting... (this may take 2-5 minutes)\n")

    t0 = time.perf_counter()
    try:
        converter = TransformersConverter(
            str(model_path),
            copy_files=["preprocessor_config.json"],
        )
        converter.convert(
            str(output_path),
            quantization=args.quantization,
        )
    except Exception as e:
        print(f"\n❌ Conversion FAILED: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0

    print(f"\n✅ Conversion completed in {elapsed:.1f} seconds.")
    print(f"   Output: {output_path}")
    print(f"\n   To use this model, set --model {output_path} when running ASR.")


if __name__ == "__main__":
    main()
