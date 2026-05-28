import os
import sys
from dotenv import load_dotenv
load_dotenv()  # Load biến từ .env tự động

# Windows-friendly base directory: use the current script folder instead of a hardcoded macOS path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import json
import warnings
import shutil
import subprocess
import tempfile
from collections import defaultdict
import re
import argparse
import traceback
import math
import types
import sys


def _install_pyannote_receptive_field_compat():
    module_name = "pyannote.audio.utils.receptive_field"
    if module_name in sys.modules:
        return

    def conv1d_num_frames(num_samples, kernel_size=5, stride=1, padding=0, dilation=1):
        return 1 + (num_samples + 2 * padding - dilation * (kernel_size - 1) - 1) // stride

    def multi_conv_num_frames(num_samples, kernel_size=None, stride=None, padding=None, dilation=None):
        num_frames = num_samples
        for k, s, p, d in zip(kernel_size, stride, padding, dilation):
            num_frames = conv1d_num_frames(
                num_frames, kernel_size=k, stride=s, padding=p, dilation=d
            )
        return num_frames

    def conv1d_receptive_field_size(num_frames=1, kernel_size=5, stride=1, dilation=1):
        effective_kernel_size = 1 + (kernel_size - 1) * dilation
        return effective_kernel_size + (num_frames - 1) * stride

    def multi_conv_receptive_field_size(num_frames, kernel_size=None, stride=None, padding=None, dilation=None):
        receptive_field_size = num_frames
        for k, s, d in reversed(list(zip(kernel_size, stride, dilation))):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=k,
                stride=s,
                dilation=d,
            )
        return receptive_field_size

    def conv1d_receptive_field_center(frame=0, kernel_size=5, stride=1, padding=0, dilation=1):
        effective_kernel_size = 1 + (kernel_size - 1) * dilation
        return frame * stride + (effective_kernel_size - 1) // 2 - padding

    def multi_conv_receptive_field_center(frame, kernel_size=None, stride=None, padding=None, dilation=None):
        receptive_field_center = frame
        for k, s, p, d in reversed(list(zip(kernel_size, stride, padding, dilation))):
            receptive_field_center = conv1d_receptive_field_center(
                frame=receptive_field_center,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
            )
        return receptive_field_center

    compat_module = types.ModuleType(module_name)
    compat_module.conv1d_num_frames = conv1d_num_frames
    compat_module.multi_conv_num_frames = multi_conv_num_frames
    compat_module.conv1d_receptive_field_size = conv1d_receptive_field_size
    compat_module.multi_conv_receptive_field_size = multi_conv_receptive_field_size
    compat_module.conv1d_receptive_field_center = conv1d_receptive_field_center
    compat_module.multi_conv_receptive_field_center = multi_conv_receptive_field_center

    sys.modules[module_name] = compat_module


_install_pyannote_receptive_field_compat()

from pyannote.audio.core.model import Model as _PyannoteBaseModel
from pyannote.audio.core.io import Audio
import importlib


def _pyannote_model_init_compat(self, sample_rate=16000, num_channels=1, task=None, **kwargs):
    self._diarizen_chunk_size = kwargs.pop("duration", None)
    self._diarizen_max_speakers_per_chunk = kwargs.pop("max_speakers_per_chunk", None)
    self._diarizen_max_speakers_per_frame = kwargs.pop("max_speakers_per_frame", None)
    if num_channels != 1:
        num_channels = 1

    super(_PyannoteBaseModel, self).__init__()
    assert (
        num_channels == 1
    ), "Only mono audio is supported for now (num_channels = 1)"
    self.save_hyperparameters("sample_rate", "num_channels")
    self.task = task
    self.audio = Audio(sample_rate=sample_rate, mono="downmix")

    if task is None and self._diarizen_max_speakers_per_frame is not None:
        num_speakers = int(self._diarizen_max_speakers_per_frame)
        duration = float(self._diarizen_chunk_size or 0.01)
        self._specifications = _Specifications_cls(
            problem=_Problem_cls.MONO_LABEL_CLASSIFICATION,
            resolution=_Resolution_cls.FRAME,
            duration=duration,
            classes=[f"spk{i}" for i in range(num_speakers)],
            powerset_max_classes=num_speakers,
        )

_PyannoteBaseModel.__init__ = _pyannote_model_init_compat


def _install_diariZen_model_compat():
    try:
        from diarizen.models.eend import model_wavlm_conformer
    except ModuleNotFoundError:
        return

    ModelClass = model_wavlm_conformer.Model
    original_dimension = ModelClass.dimension.fget

    def _compat_dimension(self):
        try:
            return original_dimension(self)
        except Exception:
            if hasattr(self, "_diarizen_max_speakers_per_frame") and self._diarizen_max_speakers_per_frame:
                return self._diarizen_max_speakers_per_frame
            return 4

    def _compat_default_activation(self):
        import torch.nn as nn
        return nn.Sigmoid()

    ModelClass.dimension = property(_compat_dimension)
    ModelClass.default_activation = _compat_default_activation


def _install_diariZen_pipeline_compat():
    """Patch DiariZen pipeline to work with pyannote 3.x clustering API."""
    try:
        from diarizen.pipelines import inference as diariZen_inference
    except (ModuleNotFoundError, ImportError):
        return

    original_init = diariZen_inference.DiariZenPipeline.__init__

    def patched_init(self, diarizen_hub, embedding_model, config_parse=None, rttm_out_dir=None):
        # Load config internally like the original does
        import toml
        from pathlib import Path
        
        config_path = Path(diarizen_hub / "config.toml")
        config = toml.load(config_path.as_posix())

        if config_parse is not None:
            print('Overriding with parsed config.')
            config["inference"]["args"] = config_parse["inference"]["args"]
            config["clustering"]["args"] = config_parse["clustering"]["args"]

        # FIX: Convert VBxClustering to AgglomerativeClustering for pyannote 3.x compatibility
        if config.get("clustering", {}).get("args", {}).get("method") == "VBxClustering":
            print("[COMPAT] Converting VBxClustering to AgglomerativeClustering for pyannote 3.x")
            config["clustering"]["args"]["method"] = "AgglomerativeClustering"
            # Ensure required fields for AgglomerativeClustering exist
            if "min_cluster_size" not in config["clustering"]["args"]:
                config["clustering"]["args"]["min_cluster_size"] = 2

        # Call original with modified config
        from pathlib import Path
        import importlib
        import torch
        
        inference_config = config["inference"]["args"]
        clustering_config = config["clustering"]["args"]

        print(f'Loaded configuration: {config}')

        model_cls_path = config["model"]["path"]
        model_module_name, model_class_name = model_cls_path.rsplit(".", 1)
        model_module = importlib.import_module(model_module_name)
        model_cls = getattr(model_module, model_class_name)
        segmentation_model = model_cls(**config["model"]["args"])

        checkpoint_path = Path(diarizen_hub / "pytorch_model.bin")
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            segmentation_model.load_state_dict(state_dict, strict=False)

        # Call parent __init__ with converted clustering method
        from pyannote.audio.pipelines import SpeakerDiarization
        SpeakerDiarization.__init__(
            self,
            segmentation=segmentation_model,
            segmentation_step=inference_config["segmentation_step"],
            embedding=embedding_model,
            embedding_exclude_overlap=True,
            clustering=clustering_config["method"],
            embedding_batch_size=inference_config["batch_size"],
            segmentation_batch_size=inference_config["batch_size"],
        )

        self.apply_median_filtering = inference_config["apply_median_filtering"]
        self.min_speakers = clustering_config["min_speakers"]
        self.max_speakers = clustering_config["max_speakers"]

        if clustering_config["method"] == "AgglomerativeClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": clustering_config.get("min_cluster_size", clustering_config.get("ahc_threshold", 10)),
                    "threshold": clustering_config.get("ahc_threshold", 0.6),
                }
            }
        elif clustering_config["method"] == "OracleClustering":
            self.PIPELINE_PARAMS = {}
        else:
            # Original VBxClustering params (for reference)
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "ahc_criterion": clustering_config.get("ahc_criterion", "distance"),
                    "ahc_threshold": clustering_config.get("ahc_threshold", 0.6),
                    "Fa": clustering_config.get("Fa", 0.07),
                    "Fb": clustering_config.get("Fb", 0.8),
                }
            }

        self.instantiate(self.PIPELINE_PARAMS)

        if rttm_out_dir is not None:
            import os
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir

        assert self._segmentation.model.specifications.powerset is True

    diariZen_inference.DiariZenPipeline.__init__ = patched_init

_install_diariZen_model_compat()
_install_diariZen_pipeline_compat()

import numpy as np
import torch
import torchaudio

# FIX 3: Set up torch.load monkeypatch BEFORE importing pyannote to ensure weights_only=False
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    # Force weights_only=False for compatibility with all pickle types
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)

torch.load = _safe_load

# Also patch lightning_fabric's torch.load if it exists
try:
    import lightning_fabric.utilities.cloud_io as cloud_io
    if hasattr(cloud_io, '_load'):
        _orig_fabric_load = cloud_io._load
        def _safe_fabric_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _orig_fabric_load(*args, **kwargs)
        cloud_io._load = _safe_fabric_load
except Exception:
    pass

# FIX 5: Robust audio loading for WAV on Windows when torchcodec/FFmpeg is unavailable.
_orig_torchaudio_load = torchaudio.load

def _normalize_waveform_np(waveform_np):
    """Normalize various integer and float audio dtypes to float32 in [-1, 1]."""
    orig_dtype = waveform_np.dtype
    waveform_np = np.asarray(waveform_np)
    if np.issubdtype(orig_dtype, np.floating):
        # Already float-like – cast to float32 if necessary
        waveform_np = waveform_np.astype(np.float32, copy=False)
    elif orig_dtype == np.int16:
        # 16‑bit PCM
        waveform_np = waveform_np.astype(np.float32) / 32768.0
    elif orig_dtype == np.int32:
        # 32‑bit PCM
        waveform_np = waveform_np.astype(np.float32) / 2147483648.0
    elif orig_dtype == np.uint8:
        # 8‑bit unsigned PCM
        waveform_np = (waveform_np.astype(np.float32) - 128.0) / 128.0
    else:
        # Fallback for uncommon integer types
        info = np.iinfo(orig_dtype) if np.issubdtype(orig_dtype, np.integer) else None
        if info is not None:
            scale = float(max(abs(info.min), abs(info.max)))
            waveform_np = waveform_np.astype(np.float32) / max(scale, 1.0)
        else:
            waveform_np = waveform_np.astype(np.float32)
    return waveform_np


def _load_wav_via_scipy(filepath, frame_offset=0, num_frames=-1, normalize=False, channels_first=True):
    """Load a WAV file via scipy.io.wavfile with optional slicing and normalization."""
    from scipy.io import wavfile
    sr, waveform_np = wavfile.read(filepath)
    waveform_np = _normalize_waveform_np(waveform_np)
    # Convert to torch tensor, adding channel dimension for mono
    if waveform_np.ndim == 1:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    else:
        if channels_first:
            waveform = torch.from_numpy(np.ascontiguousarray(waveform_np.T))
        else:
            waveform = torch.from_numpy(np.ascontiguousarray(waveform_np))
    total_frames = waveform.shape[1] if channels_first else waveform.shape[0]
    # Slice according to frame_offset/num_frames
    start = max(0, int(frame_offset))
    if num_frames is None or int(num_frames) < 0:
        end = total_frames
    else:
        end = min(total_frames, start + int(num_frames))
    if channels_first:
        waveform = waveform[:, start:end]
    else:
        waveform = waveform[start:end, :]
    if normalize:
        peak = float(waveform.abs().max()) if waveform.numel() > 0 else 0.0
        if peak > 1e-7:
            waveform = waveform / peak
    return waveform.contiguous(), int(sr)


def _safe_audio_load(filepath, frame_offset=0, num_frames=-1, normalize=False, channels_first=True, format=None):
    """Attempt to load audio via torchaudio; fall back to scipy for WAV if necessary."""
    ext = os.path.splitext(str(filepath))[1].lower()
    # For WAV, bypass torchcodec entirely whenever possible.
    if ext == ".wav":
        try:
            return _load_wav_via_scipy(
                filepath,
                frame_offset=frame_offset,
                num_frames=num_frames,
                normalize=normalize,
                channels_first=channels_first,
            )
        except Exception:
            pass
    # Otherwise, try torchaudio first
    try:
        return _orig_torchaudio_load(
            filepath,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=normalize,
            channels_first=channels_first,
            format=format,
        )
    except RuntimeError as e:
        if ext == ".wav" and ("torchcodec" in str(e) or "FFmpeg" in str(e) or "libtorchcodec" in str(e)):
            # Fallback to scipy for WAV
            try:
                return _load_wav_via_scipy(
                    filepath,
                    frame_offset=frame_offset,
                    num_frames=num_frames,
                    normalize=normalize,
                    channels_first=channels_first,
                )
            except Exception as fallback_err:
                raise RuntimeError(f"Original torchaudio.load failed: {e}. Fallback also failed: {fallback_err}")
        raise


def _patched_torchaudio_load(filepath, frame_offset=0, num_frames=-1, normalize=False, channels_first=True, format=None):
    return _safe_audio_load(
        filepath,
        frame_offset=frame_offset,
        num_frames=num_frames,
        normalize=normalize,
        channels_first=channels_first,
        format=format,
    )


torchaudio.load = _patched_torchaudio_load

# FIX 6: Patch SpeakerDiarization to remove unsupported 'plda' argument in newer pyannote versions
def _patch_speaker_diarization():
    try:
        from pyannote.audio.pipelines import SpeakerDiarization
        _original_init = SpeakerDiarization.__init__
        
        def _patched_init(self, *args, **kwargs):
            # Remove 'plda' if present (unsupported in pyannote >= 3.0.1)
            kwargs.pop('plda', None)
            return _original_init(self, *args, **kwargs)
        
        SpeakerDiarization.__init__ = _patched_init
    except Exception as e:
        print(f"[DEBUG] Failed to patch SpeakerDiarization: {e}")

_patch_speaker_diarization()

# Import pyannote classes now, BEFORE setting safe_globals
from pyannote.audio.core.task import Problem as _Problem_cls, Resolution as _Resolution_cls, Specifications as _Specifications_cls

# FIX 4: Add safe globals for torch.load to accept Specifications and related classes
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from omegaconf import DictConfig, ListConfig
        torch.serialization.add_safe_globals([_Specifications_cls, _Problem_cls, _Resolution_cls, DictConfig, ListConfig])
    except Exception:
        try:
            torch.serialization.add_safe_globals([_Specifications_cls, _Problem_cls, _Resolution_cls])
        except Exception:
            pass

from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import AgglomerativeClustering

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

warnings.filterwarnings("ignore")
TARGET_SR = 16000

# =========================
# DEVICE / MODEL
# Windows + RTX 4060: prioritize CUDA first.
# =========================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[CONFIG] Using device: {DEVICE}")

# =========================
# GPU PERFORMANCE FLAGS (RTX 4060 8 GB)
# =========================
if DEVICE.type == "cuda":
    # Allow TF32 on Ampere+ Tensor Cores → ~3× faster matmuls with negligible quality loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cuDNN auto-tune: picks fastest conv algorithm for fixed input shapes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Limit memory-fragmentation overhead for the 8 GB budget
    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True")
    torch.cuda.set_device(0)
    print(f"[CONFIG] CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"[CONFIG] VRAM total : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"[CONFIG] TF32={torch.backends.cuda.matmul.allow_tf32} | "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}")

# =========================
# CONFIG
# =========================
FIXED_N_CLUSTERS = 3

# ---------------------------------------------------------------------------
# GLOBAL FLAGS
#
# SKIP_DER: When True, the pipeline skips computing diarization error rate (DER)
# and related metrics. DER is an evaluation metric that compares system
# hypotheses against a ground‑truth reference. In production scenarios
# where reference RTTM files are unavailable or when reporting error
# statistics is undesirable, this flag can be enabled to bypass DER
# computation. The core diarization logic remains unchanged. When
# SKIP_DER is True the "der", "miss", "fa" and "conf" keys will be set to
# None in the JSON output and corresponding console prints will show
# "N/A" instead of numeric values.
SKIP_DER = False

# FIX 1: Add missing constant used in compute_overlap_identity_metrics / build_stage_report.
EVAL_FRAME_SEC = 0.010   # 10 ms evaluation frame for overlap & identity metrics

# Optional enrollment bootstrap
ENROLLMENT_DIR = None
ENROLLMENT_MATCH_SIM = 0.42
ENROLLMENT_MATCH_MARGIN = 0.05
ENROLLMENT_MIN_AUDIO_SEC = 0.50

# EEND inference
CHUNK_SEC = 16.0

# Phase 1 — prototype/anchor policy
HIGH_THR_FIXED = 0.70
HIGH_THR_START = HIGH_THR_FIXED
HIGH_THR_MIN = HIGH_THR_FIXED
HIGH_THR_STEP = 0.0

PRIMARY_MIN_DUR = 0.4
LONG_SEG_SEC = 3.0
EMB_WINDOW_SEC = 1.5
EMB_HOP_SEC = 0.75
SHRINK_SEC = 0.10
MIN_ANCHOR_PER_CLUSTER = 1

# Phase 2 — attachment / rescue
LOW_THRESHOLD = 0.04
AUX_MIN_DUR = 0.30
MIN_DUR_OUT = 0.10
MIN_DUR_FORCE = 0.03
MIN_CLEAN_DUR_FORCE = 0.05
FORCE_FRAGMENT_MIN_TOTAL_ACTIVE_SEC = 0.06
FORCE_FRAGMENT_DENSE_WIN_SEC = 0.40
FORCE_FRAGMENT_PAD_SEC = 0.08
FORCE_FRAGMENT_MIN_EMBED_SEC = 0.24
FORCE_FRAGMENT_MAX_WIN_SEC = 0.80
PRIMARY_ASSIGN_SIM = 0.20
AUX_ASSIGN_SIM = 0.22
FALLBACK_ASSIGN_SIM = 0.26
MAX_PROPAGATE_CHUNK_GAP = 4

# Smoothing / post-process
MED_WIN = 11
MERGE_GAP_SEC = 0.30

# Stable single-speaker zones
STABLE_ZONE_ENABLED = True
STABLE_ZONE_MIN_DUR_SEC = 0.90
STABLE_ZONE_MIN_MARGIN = 0.14
STABLE_ZONE_MAX_TOP1_MINUS_CUR = 0.06
STABLE_ZONE_VETO_MIN_COVER_SEC = 0.20
STABLE_ZONE_MAX_WEAK_RATIO = 0.58
STABLE_ZONE_MAX_WEAK_NEIGHBOR_SEC = 0.12
STABLE_ZONE_CONTEXT_SEC = 0.80
STABLE_ZONE_WEAK_MAX_CONTEXT_SEC = 0.10
STABLE_ZONE_STRONG_CONTEXT_MIN_SEC = 0.30

# Centroid purification
CENTROID_PURIFY_ENABLED = True
CENTROID_PURIFY_ITERS = 1
CENTROID_PURIFY_MIN_SIM = 0.35
CENTROID_PURIFY_MIN_MARGIN = 0.12
CENTROID_PURIFY_MIN_USED_DUR = 1.20
CENTROID_PURIFY_MIN_PER_CLUSTER = 3
CENTROID_PURIFY_TOTAL_MIN = 18
CENTROID_PURIFY_MAX_CLUSTER_IMBALANCE = 6.0
CENTROID_PURIFY_BLEND = 0.15
CENTROID_PURIFY_ACCEPT_TIERS = {"primary", "aux"}
CENTROID_PURIFY_MIN_STABLE_SAMPLES = 18

# Fallback governance
FALLBACK_ROW_WEIGHT = 0.20
AUX_ROW_WEIGHT = 0.70
PRIMARY_ROW_WEIGHT = 1.00
RESEG_ROW_WEIGHT = 0.90
FALLBACK_EXCLUDE_FROM_STABLE = True
FALLBACK_EXCLUDE_FROM_TRUSTED = True
FALLBACK_OVERRIDE_MAX_RATIO = 0.55
FALLBACK_VBX_REMAP_BLOCK_RATIO = 0.45
FALLBACK_PAIR_SUPPORT_BLOCK_RATIO = 0.35
FALLBACK_SCORE_BLOCK_RATIO = 0.45
FALLBACK_NEIGHBOR_BLOCK_RATIO = 0.45
FALLBACK_DISABLE_STABLE_FOR_SPK_RATIO = 0.55

# Speaker-hub control
HUB_PENALTY_ENABLED = True
HUB_SHARE_FREE = 0.34
HUB_SHARE_HARD = 0.44
HUB_REMAP_EXTRA_GAIN = 0.11
HUB_GATE_EXTRA_SUPPORT_SEC = 0.18
HUB_GAIN_EXP = 1.55
HUB_FALLBACK_BOOST = 0.30
HUB_COUNT_WEIGHT = 0.25
HUB_DURATION_WEIGHT = 0.55
HUB_FALLBACK_WEIGHT = 0.20

VBX_NOISE_COVERAGE_THR = 0.50

# Post-silence filter
POST_SILENCE_FILTER_ENABLED = True
POST_SILENCE_ENERGY_THR = 0.0025
POST_SILENCE_FRAME_SEC = 0.020
POST_SILENCE_HOP_SEC = 0.010
POST_SILENCE_MIN_SPEECH_RATIO = 0.15
POST_SILENCE_FALLBACK_RATIO = 0.08
POST_SILENCE_SMOOTHING_FRAMES = 15
POST_SILENCE_PROTECT_HIGH_EEND = 0.45

# Temporal single-speaker resegmentation
TEMPORAL_RESEG_ENABLED = False
TEMPORAL_RESEG_MAX_BRIDGE_SEC = 1.20
TEMPORAL_RESEG_MIN_NEIGHBOR_SEC = 0.80
TEMPORAL_RESEG_MAX_CUR_MARGIN = 0.18
TEMPORAL_RESEG_MIN_ALT_GAIN = 0.08
TEMPORAL_RESEG_NEIGHBOR_SEC = 1.20
TEMPORAL_RESEG_MIN_SUPPORT_SEC = 0.35
TEMPORAL_RESEG_SCORE_ONLY_MAX_DUR = 1.60

# Real VBx single-speaker backbone refinement
VBX_REAL_ENABLED = False   # [SCHEDULE T0] Tắt để đo pipeline lõi sạch; bật lại sau khi centroid/assign ổn
VBX_REPO_DIR = os.path.join(BASE_DIR, "third_party/VBx")
VBX_PYTHON = sys.executable
VBX_BACKEND = "onnx"
VBX_MODEL_DIR = os.path.join(VBX_REPO_DIR, "VBx/models/ResNet101_16kHz")
VBX_WEIGHTS = os.path.join(VBX_MODEL_DIR, "nnet/final.onnx")
VBX_TRANSFORM = os.path.join(VBX_MODEL_DIR, "transform.h5")
VBX_PLDA = os.path.join(VBX_MODEL_DIR, "plda")
VBX_INIT = "AHC+VB"
VBX_AHC_THRESHOLD = -0.015
VBX_LDA_DIM = 128
VBX_FA = 0.3
VBX_FB = 17.0
VBX_LOOPP = 0.99
VBX_MIN_SPEECH_SEG_SEC = 0.20
VBX_MAP_USE_STABLE_ZONES = True
VBX_MAP_MIN_OVERLAP_SEC = 0.25
VBX_KEEP_LOCAL_IF_UNMAPPED = True
VBX_STRICT_REQUIREMENTS = True
VBX_MAP_OVERLAP_WEIGHT = 1.00
VBX_MAP_ACOUSTIC_WEIGHT = 0.80
VBX_MAP_NEIGHBOR_WEIGHT = 0.55
VBX_MAP_MIN_GLOBAL_SCORE = 0.38
VBX_MAP_MIN_MAPPED_SPEAKERS = 2
VBX_MAP_MIN_UNIQUE_LOCAL_SPEAKERS = 2
VBX_MAP_FALLBACK_TO_TRUSTED_SINGLE = True
VBX_REMAP_MIN_INTERVAL_SEC = 0.25
VBX_REMAP_MIN_GAIN = 0.06
VBX_REMAP_MIN_NEIGHBOR_GAIN = 0.08
VBX_REMAP_MAX_CURRENT_MARGIN = 0.16
VBX_REMAP_NEIGHBOR_SEC = 0.90
VBX_REMAP_REQUIRE_TARGET_TOP1 = False

# VBx speaker pre-merge
VBX_PREMERGE_ENABLED = True
VBX_PREMERGE_COSINE_THR = 0.75
VBX_PREMERGE_MAX_CLUSTERS = FIXED_N_CLUSTERS + 1

# FIX 5: Cannot-link penalty — use value just above max cosine distance (2.0).
# The old value of 1000.0 caused severe distortion of average-linkage distances.
# Any value > 2.0 is sufficient to guarantee the pair never merges under average
# linkage when the actual distances are in [0, 2.0].
CANNOT_LINK_DIST = 3.0

# Score dump for analyze_v3
FORENSIC_DIR = os.path.join(BASE_DIR, "finetune_9_3/report/eend_eda/v72_claude_silence_forensics")

# Paths
WAV_PATH = os.path.join(BASE_DIR, "data", "vivo", "meeting_20260327_112255_174badfa.wav")
CLEAN_RTTM_PATH = os.path.join(BASE_DIR, "data", "vivo", "meeting_20260327_112255_174badfa.rttm")
UTT_ID = "overlap_10_3spk_dual_thr_v72_claude_silence"

OUT_JSON = os.path.join(
    BASE_DIR,
    "out",
    "result.json"
)

# Optional finetuned checkpoint (.pth)
CKPT_PATH = None
HF_MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
PYANNOTE_HF_MODEL_ID = "pyannote/speaker-diarization-community-1"

def parse_args():
    global WAV_PATH, CLEAN_RTTM_PATH, UTT_ID, OUT_JSON, FORENSIC_DIR
    global ENROLLMENT_DIR, FIXED_N_CLUSTERS, VBX_PREMERGE_MAX_CLUSTERS, CKPT_PATH, HF_MODEL_ID
    parser = argparse.ArgumentParser(description="EEND-EDA Inference v7.2 Fix 3")
    parser.add_argument("--wav", type=str, help="Path to input WAV file")
    parser.add_argument("--rttm", type=str, help="Path to reference RTTM file")
    parser.add_argument("--utt_id", type=str, help="Utterance ID")
    parser.add_argument("--out_json", type=str, help="Path to output JSON file")
    parser.add_argument("--enrollment_dir", type=str)
    parser.add_argument("--n_speakers", type=int)
    parser.add_argument("--ckpt", type=str, help="Path to finetuned checkpoint (.pth)")
    parser.add_argument("--hf_model", type=str, default=HF_MODEL_ID, help="Hugging Face model id to use as the pretrained base")

    # When specified, skip computing diarization error rate (DER) and related
    # metrics. This is useful for production workloads where reference RTTM
    # annotations are not available or when DER should not be included in the
    # output. The flag merely controls reporting; all core diarization logic
    # remains intact.
    parser.add_argument("--skip_der", action="store_true", help="Skip DER computation and remove DER metrics from output")

    args, unknown = parser.parse_known_args()

    if args.wav:
        WAV_PATH = args.wav
    if args.rttm:
        CLEAN_RTTM_PATH = args.rttm
    if args.utt_id:
        UTT_ID = args.utt_id
    if args.out_json:
        OUT_JSON = args.out_json
    if args.ckpt:
        CKPT_PATH = args.ckpt
    if args.hf_model:
        HF_MODEL_ID = args.hf_model
    base_report_dir = os.path.dirname(OUT_JSON)
    FORENSIC_DIR = os.path.join(base_report_dir, f"forensics_{UTT_ID}")
    if args.enrollment_dir:
        ENROLLMENT_DIR = args.enrollment_dir
    if args.n_speakers is not None and args.n_speakers > 0:
        FIXED_N_CLUSTERS = int(args.n_speakers)
        VBX_PREMERGE_MAX_CLUSTERS = FIXED_N_CLUSTERS + 1

    # Apply SKIP_DER flag
    if getattr(args, "skip_der", False):
        global SKIP_DER
        SKIP_DER = True

parse_args()

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_rttm(path: str, uri: str) -> Annotation:
    ann = Annotation(uri=uri)
    if not os.path.exists(path):
        return ann
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            p = line.strip().split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                s = float(p[3])
                dur = float(p[4])
                lbl = p[7]
                ann[Segment(s, s + dur), f"ref_{idx}"] = lbl
    return ann

def save_rttm(annotation: Annotation, path: str):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seg, _, label in annotation.itertracks(yield_label=True):
            dur = seg.end - seg.start
            f.write(
                f"SPEAKER {annotation.uri} 1 {seg.start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>\n"
            )

def build_cluster_name_map(init_meta: dict) -> dict:
    """Build a mapping from cluster index (int) to speaker name (str).

    Uses ``init_meta["speaker_seed_info"]`` which is populated by
    ``initialize_phase1_centroids``.  When enrollment is active, the
    ``seed_name`` equals the enrollment folder name, so the RTTM labels will
    match the names provided in ``--enrollment_dir``.

    For any cluster whose seed_name is not available (e.g. unknown speakers
    bootstrapped from Phase 1), the label falls back to ``SPK_{index}`` so
    the existing behaviour is preserved.
    """
    name_map: dict = {}
    seed_info = (init_meta or {}).get("speaker_seed_info") or []
    for item in seed_info:
        idx = item.get("index")
        name = item.get("seed_name") or f"SPK_{idx}"
        if idx is not None:
            name_map[int(idx)] = str(name)
    return name_map


def remap_rttm_labels(annotation: "Annotation", cluster_name_map: dict) -> "Annotation":
    """Return a new Annotation with ``SPK_{i}`` labels replaced by the
    enrollment speaker name stored in *cluster_name_map*.

    If *cluster_name_map* is empty (no enrollment / mapping unavailable)
    the original annotation is returned unchanged.
    """
    if not cluster_name_map:
        return annotation

    remapped = Annotation(uri=annotation.uri)
    for seg, track, label in annotation.itertracks(yield_label=True):
        # label is e.g. "SPK_0" — extract the numeric cluster id
        new_label = label
        if label.startswith("SPK_"):
            try:
                cluster_id = int(label.split("_", 1)[1])
                new_label = cluster_name_map.get(cluster_id, label)
            except (ValueError, IndexError):
                pass
        remapped[seg, track] = new_label
    return remapped


def extract_active_segments(mask_1d: np.ndarray, offset_sec: float, dt_sec: float, min_dur_sec: float):
    segs = []
    active = False
    s_f = 0
    for f in range(mask_1d.shape[0]):
        if mask_1d[f] and not active:
            active = True
            s_f = f
        elif not mask_1d[f] and active:
            active = False
            e_f = f
            dur = (e_f - s_f) * dt_sec
            if dur >= min_dur_sec:
                segs.append({
                    "s_f": s_f, "e_f": e_f,
                    "s_sec": offset_sec + s_f * dt_sec,
                    "e_sec": offset_sec + e_f * dt_sec,
                    "dur": dur,
                })
    if active:
        e_f = mask_1d.shape[0]
        dur = (e_f - s_f) * dt_sec
        if dur >= min_dur_sec:
            segs.append({
                "s_f": s_f, "e_f": e_f,
                "s_sec": offset_sec + s_f * dt_sec,
                "e_sec": offset_sec + e_f * dt_sec,
                "dur": dur,
            })
    return segs

def _make_ahc_fixed_k(n_clusters: int):
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")

def infer_all_chunk_probs(model, wav_tensor: torch.Tensor, chunk_sec: float):
    outputs = []
    total_samples = wav_tensor.shape[1]
    chunk_samples = int(chunk_sec * TARGET_SR)
    total_chunks = len(range(0, total_samples, chunk_samples))
    for c_idx, start_samp in enumerate(range(0, total_samples, chunk_samples)):
        end_samp = min(start_samp + chunk_samples, total_samples)
        chunk = wav_tensor[:, start_samp:end_samp]
        if chunk.shape[1] < 160:
            continue
        pad_len = max(0, int(18 * TARGET_SR) - chunk.shape[1])
        # Ensure chunk is on DEVICE before padding so the whole tensor stays on GPU
        chunk = chunk.to(DEVICE)
        padding = torch.zeros((1, pad_len), device=DEVICE)
        padded_chunk = torch.cat([chunk, padding], dim=1)

        # NOTE: pyannote internals call .cpu().numpy() on the output tensor;
        # autocast (fp16/bf16) would break that conversion → use plain no_grad only.
        # GPU speedup already comes from model.to(DEVICE) + TF32 flags above.
        with torch.no_grad():
            actual_sec = chunk.shape[1] / TARGET_SR
            if getattr(model, "_direct_use_raw_forward", False):
                probs = _get_direct_chunk_probs(model, padded_chunk)
            else:
                seg_soft = model.get_segmentations({"waveform": padded_chunk, "sample_rate": TARGET_SR})
                if not hasattr(seg_soft, 'data') or len(seg_soft.data) == 0:
                    print(f"  [DEBUG] Chunk {c_idx} | Model returned empty output, skipping")
                    continue
                probs = seg_soft.data[0]
            total_window_sec = padded_chunk.shape[1] / TARGET_SR
            frame_rate = probs.shape[0] / total_window_sec
            actual_frames = min(probs.shape[0], int(actual_sec * frame_rate))
            probs = probs[:actual_frames]
            probs = probs.astype(np.float32)
        print(f"  [DEBUG] Chunk {c_idx} | Shape: {probs.shape} | Max Prob: {probs.max():.4f}")
        probs = median_filter(probs, size=(MED_WIN, 1)).astype(np.float32)
        dt_sec = (chunk.shape[1] / TARGET_SR) / max(1, probs.shape[0])
        outputs.append({
            "chunk_idx": c_idx,
            "offset_sec": start_samp / TARGET_SR,
            "probs": probs,
            "dt_sec": dt_sec,
        })
        print(f"PROGRESS:{c_idx+1}/{total_chunks}", flush=True)
        # Release GPU memory every 10 chunks to stay within 8 GB budget
        if DEVICE.type == "cuda" and (c_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    return outputs

def compute_der(ref: Annotation, hyp: Annotation):
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    detail = metric(ref, hyp, detailed=True)
    total = float(detail["total"]) if detail["total"] > 0 else 1e-8
    der = abs(metric) * 100.0
    miss = detail["missed detection"] / total * 100.0
    fa = detail["false alarm"] / total * 100.0
    conf = detail["confusion"] / total * 100.0
    return der, miss, fa, conf

def get_resampler():
    if TARGET_SR == 16000:
        return None
    return torchaudio.transforms.Resample(orig_freq=TARGET_SR, new_freq=16000).to(DEVICE)

def get_embedding_interval(ecapa, wav_tensor: torch.Tensor, resampler, s_sec: float, e_sec: float):
    ns = s_sec + SHRINK_SEC
    ne = e_sec - SHRINK_SEC
    if ne - ns < 0.2:
        return None, 0.0

    si = int(ns * TARGET_SR)
    ei = int(ne * TARGET_SR)
    max_s = wav_tensor.shape[1]
    si = max(0, min(si, max_s))
    ei = max(0, min(ei, max_s))
    if (ei - si) < int(0.2 * TARGET_SR):
        return None, 0.0

    slc = wav_tensor[:, si:ei]
    if slc.shape[1] < int(0.2 * TARGET_SR):
        return None, 0.0

    try:
        slc = slc.to(DEVICE)
        slc_16k = resampler(slc) if resampler is not None else slc
        # NOTE: embedding model also calls .cpu().numpy() on output;
        # keep plain no_grad — TF32 + GPU already provide the speedup.
        with torch.no_grad():
            emb_obj = None
            for attr in ("_embedding", "embedding", "model"):
                candidate = getattr(ecapa, attr, None)
                if callable(candidate):
                    emb_obj = candidate
                    break
            if emb_obj is None:
                raise AttributeError(
                    f"Cannot find a callable embedding attribute on {type(ecapa).__name__}. "
                    "Checked: _embedding, embedding, model."
                )
            emb = emb_obj(slc_16k.unsqueeze(0))
            if hasattr(emb, "detach"):
                emb = emb.squeeze().detach().cpu().numpy()
            else:
                emb = np.squeeze(emb)
            emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm <= 1e-8:
            return None, 0.0
        return emb / norm, float(ne - ns)
    except Exception as e:
        print(f"  [DEBUG] Embedding error on {s_sec:.2f}-{e_sec:.2f}: {e}")
        return None, 0.0

def get_segment_embedding(ecapa, wav_tensor: torch.Tensor, resampler, s_sec: float, e_sec: float):
    dur = e_sec - s_sec
    if dur < AUX_MIN_DUR:
        return None, 0.0, 0

    if dur <= LONG_SEG_SEC:
        emb, used_dur = get_embedding_interval(ecapa, wav_tensor, resampler, s_sec, e_sec)
        return emb, used_dur, 1 if emb is not None else 0

    sub_embs = []
    sub_durs = []
    cur = s_sec
    while cur + EMB_WINDOW_SEC <= e_sec + 1e-8:
        emb, used_dur = get_embedding_interval(ecapa, wav_tensor, resampler, cur, cur + EMB_WINDOW_SEC)
        if emb is not None:
            sub_embs.append(emb)
            sub_durs.append(used_dur)
        cur += EMB_HOP_SEC

    if not sub_embs:
        emb, used_dur = get_embedding_interval(ecapa, wav_tensor, resampler, s_sec, e_sec)
        return emb, used_dur, 1 if emb is not None else 0

    mat = np.stack(sub_embs).astype(np.float32)
    weights = np.array(sub_durs, dtype=np.float32)
    merged = np.average(mat, axis=0, weights=weights)
    norm = np.linalg.norm(merged)
    if norm <= 1e-8:
        return None, 0.0, len(sub_embs)
    return merged / norm, float(weights.sum()), len(sub_embs)

def _longest_clean_span(clean_mask: np.ndarray, frame_offset: int, offset_sec: float, dt_sec: float):
    best_s, best_e, cur_s = 0, 0, 0
    in_clean = False
    for f, v in enumerate(clean_mask):
        if v and not in_clean:
            in_clean = True
            cur_s = f
        elif not v and in_clean:
            in_clean = False
            if f - cur_s > best_e - best_s:
                best_s, best_e = cur_s, f
    if in_clean and len(clean_mask) - cur_s > best_e - best_s:
        best_s, best_e = cur_s, len(clean_mask)

    abs_s = offset_sec + (frame_offset + best_s) * dt_sec
    abs_e = offset_sec + (frame_offset + best_e) * dt_sec
    return abs_s, abs_e

def collect_anchor_embeddings(chunk_outputs, wav_tensor, ecapa, resampler, threshold: float):
    embeddings, durations, seg_info = [], [], []
    stats = {
        "primary_segments": 0,
        "subseg_used_segments": 0,
        "discarded_too_short": 0,
        "discarded_overlap": 0,
        "discarded_clean_ratio": 0,  # [T1] segment bị loại vì clean span < 80% duration
    }

    for item in chunk_outputs:
        c_idx = item["chunk_idx"]
        offset = item["offset_sec"]
        probs = item["probs"]
        dt_sec = item["dt_sec"]
        act_map = (probs > threshold)
        _, n_tracks = act_map.shape

        for t_idx in range(n_tracks):
            segs = extract_active_segments(act_map[:, t_idx], offset, dt_sec, PRIMARY_MIN_DUR)
            for seg in segs:
                seg_frames = act_map[seg["s_f"]:seg["e_f"], :]
                overlap_mask = np.sum(seg_frames, axis=1) > 1
                has_overlap = np.any(overlap_mask)

                if not has_overlap:
                    # Segment hoàn toàn clean → dùng toàn bộ như cũ
                    used_s, used_e = seg["s_sec"], seg["e_sec"]
                else:
                    # [T1] Thay vì discard toàn bộ, tìm longest clean span
                    clean_mask = ~overlap_mask
                    if not np.any(clean_mask):
                        stats["discarded_overlap"] += 1
                        continue

                    clean_s, clean_e = _longest_clean_span(clean_mask, seg["s_f"], offset, dt_sec)
                    clean_dur = float(clean_e - clean_s)
                    seg_dur = float(seg["e_sec"] - seg["s_sec"])

                    # Yêu cầu: clean span phải đủ dài VÀ chiếm >= 80% duration gốc
                    if clean_dur < PRIMARY_MIN_DUR or (clean_dur / max(seg_dur, 1e-8)) < 0.80:
                        stats["discarded_clean_ratio"] += 1
                        continue

                    used_s, used_e = clean_s, clean_e

                emb, used_dur, n_sub = get_segment_embedding(
                    ecapa, wav_tensor, resampler, used_s, used_e
                )
                if emb is None or used_dur < 0.2:
                    stats["discarded_too_short"] += 1
                    continue

                embeddings.append(emb)
                durations.append(used_dur)
                seg_info.append({
                    "chunk": c_idx,
                    "track": t_idx,
                    "s_sec": used_s,
                    "e_sec": used_e,
                })
                stats["primary_segments"] += 1
                if n_sub > 1:
                    stats["subseg_used_segments"] += 1

    return embeddings, durations, seg_info, stats

def cluster_anchors(embeddings, durations, seg_info, n_clusters: int):
    if len(embeddings) < n_clusters:
        return None

    mat = np.stack(embeddings).astype(np.float32)
    dist = squareform(pdist(mat, metric="cosine")).astype(np.float32)
    # FIX 5: Clip to [0, 2] first (valid cosine range), then apply cannot-link.
    # CANNOT_LINK_DIST = 3.0 is just above the max valid cosine distance of 2.0,
    # so it blocks merging without extreme numerical distortion in average linkage.
    dist = np.clip(dist, 0.0, 2.0)

    n = len(seg_info)
    for i in range(n):
        for j in range(i + 1, n):
            if seg_info[i]["chunk"] == seg_info[j]["chunk"] and seg_info[i]["track"] != seg_info[j]["track"]:
                dist[i, j] = CANNOT_LINK_DIST
                dist[j, i] = CANNOT_LINK_DIST

    try:
        labels = _make_ahc_fixed_k(n_clusters).fit_predict(dist)
        return labels
    except Exception as _ahc_exc:
        print(f"  [DEBUG] cluster_anchors AHC exception ({type(_ahc_exc).__name__}): {_ahc_exc}")
        # Check for NaN/inf in distance matrix to aid diagnosis
        _nan_count = int(np.sum(~np.isfinite(dist)))
        if _nan_count > 0:
            print(f"  [DEBUG] Distance matrix has {_nan_count} non-finite entries — likely fp16 NaN from autocast")
        return None

def build_weighted_centroids(embeddings, durations, labels, n_clusters: int):
    dim = embeddings[0].shape[0]
    centroids = np.zeros((n_clusters, dim), dtype=np.float32)
    weight_sums = np.zeros(n_clusters, dtype=np.float32)

    for emb, dur, lbl in zip(embeddings, durations, labels):
        centroids[lbl] += emb * dur
        weight_sums[lbl] += dur

    if np.any(weight_sums < 1e-8):
        return None

    for k in range(n_clusters):
        centroids[k] /= weight_sums[k]
        norm = np.linalg.norm(centroids[k])
        if norm > 1e-8:
            centroids[k] /= norm
    return centroids

def build_phase1_anchors(chunk_outputs, wav_tensor, ecapa, resampler, n_clusters: int):
    thr = float(HIGH_THR_FIXED)
    print(f"  [DEBUG] build_phase1_anchors called. fixed_high_thr={thr:.2f}")
    embs, durs, sinfo, stats = collect_anchor_embeddings(
        chunk_outputs, wav_tensor, ecapa, resampler, threshold=thr
    )
    if len(embs) < n_clusters:
        print(f"  [DEBUG] fixed high_thr={thr:.2f} | embs={len(embs)} | stats={stats}")
        return None, None, 0, stats

    labels = cluster_anchors(embs, durs, sinfo, n_clusters)
    if labels is None:
        print(f"  [DEBUG] fixed high_thr={thr:.2f} | clustering failed | stats={stats}")
        return None, None, 0, stats

    counts = np.bincount(labels, minlength=n_clusters)
    if np.any(counts < MIN_ANCHOR_PER_CLUSTER):
        print(f"  [DEBUG] fixed high_thr={thr:.2f} | insufficient per-cluster anchors={counts.tolist()}")
        return None, None, 0, stats

    centroids = build_weighted_centroids(embs, durs, labels, n_clusters)
    if centroids is None:
        print(f"  [DEBUG] fixed high_thr={thr:.2f} | centroid build failed")
        return None, None, 0, stats

    print(
        f"  [Phase 1] fixed_high_thr={thr:.2f} | anchors={len(embs)} | "
        f"per_cluster={counts.tolist()} | subseg_used={stats['subseg_used_segments']}"
    )
    return centroids, thr, len(embs), stats

def _best_sim_and_margin(vec, centroids):
    if centroids is None or len(centroids) == 0:
        return -1.0, 0.0, None
    sims = np.dot(centroids, vec.astype(np.float32))
    order = np.argsort(-sims)
    best_idx = int(order[0])
    best = float(sims[best_idx])
    second = float(sims[int(order[1])]) if len(order) > 1 else -1.0
    return best, float(best - second), best_idx

def _is_audio_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    return ext in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma"}

def _load_audio_mono16k(path: str):
    import soundfile as sf
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data).float().unsqueeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav

def _discover_enrollment_groups(enrollment_dir: str):
    if not enrollment_dir:
        return []
    enrollment_dir = os.path.abspath(enrollment_dir)
    if not os.path.isdir(enrollment_dir):
        return []

    subdirs = []
    print(f"  [Discovery] Scanning: {enrollment_dir}")
    for name in sorted(os.listdir(enrollment_dir)):
        full = os.path.join(enrollment_dir, name)
        if os.path.isdir(full):
            print(f"  [Discovery] Found possible speaker dir: {name}")
            files = []
            for root, _, fnames in os.walk(full):
                for fname in sorted(fnames):
                    fpath = os.path.join(root, fname)
                    if _is_audio_file(fpath):
                        print(f"    - Found audio: {fname}")
                        files.append(fpath)
            if files:
                subdirs.append((name, files))
    if subdirs:
        return subdirs

    root_audio = []
    for name in sorted(os.listdir(enrollment_dir)):
        full = os.path.join(enrollment_dir, name)
        if os.path.isfile(full) and _is_audio_file(full):
            root_audio.append((os.path.splitext(name)[0], [full]))
    return root_audio

def build_enrollment_centroids(enrollment_dir, ecapa, resampler, max_speakers=None):
    stats = {
        "enabled": bool(enrollment_dir),
        "dir": enrollment_dir,
        "groups_found": 0,
        "groups_used": 0,
        "files_used": 0,
        "files_skipped": 0,
        "speaker_names": [],
    }
    print(f"  [Enrollment] Discovering groups in: {enrollment_dir}")
    groups = _discover_enrollment_groups(enrollment_dir)
    stats["groups_found"] = len(groups)
    if not groups:
        return None, [], stats

    if max_speakers is not None and max_speakers > 0:
        groups = groups[:max_speakers]

    centroids = []
    names = []
    for spk_name, files in groups:
        print(f"  [Enrollment] Processing speaker '{spk_name}' with {len(files)} files")
        speaker_items = []
        for fpath in files:
            try:
                wav = _load_audio_mono16k(fpath)
                dur_sec = float(wav.shape[1] / TARGET_SR)
                print(f"    - File: {os.path.basename(fpath)} | Duration: {dur_sec:.2f}s")
            except Exception as e:
                print(f"    - Error loading {fpath}: {e}")
                stats["files_skipped"] += 1
                continue
            if dur_sec < ENROLLMENT_MIN_AUDIO_SEC:
                stats["files_skipped"] += 1
                continue
            emb, used_dur, _ = get_segment_embedding(ecapa, wav, resampler, 0.0, dur_sec)
            if emb is None or used_dur < 0.2:
                print(f"    - Embedding failed or too short: used_dur={used_dur if used_dur else 0:.2f}s")
                stats["files_skipped"] += 1
                continue
            print(f"    - Embedding OK: used_dur={used_dur:.2f}s")
            speaker_items.append((emb.astype(np.float32), float(used_dur)))
            stats["files_used"] += 1

        if not speaker_items:
            continue

        dim = speaker_items[0][0].shape[0]
        centroid = _weighted_centroid(speaker_items, dim)
        if centroid is None:
            continue
        centroids.append(centroid.astype(np.float32))
        names.append(spk_name)

    if not centroids:
        return None, [], stats

    stats["groups_used"] = len(centroids)
    stats["speaker_names"] = list(names)
    return np.stack(centroids).astype(np.float32), names, stats

def _combine_enrollment_with_phase1(enrollment_centroids, enrollment_names, phase1_centroids, n_clusters):
    k = int(enrollment_centroids.shape[0])
    n = int(n_clusters)
    if k <= 0:
        return phase1_centroids, []
    if k >= n:
        meta = []
        for i in range(n):
            meta.append({
                "index": int(i),
                "seed_type": "enrollment",
                "seed_name": enrollment_names[i] if i < len(enrollment_names) else f"enrollment_{i}",
                "matched_phase1_cluster": None,
            })
        return enrollment_centroids[:n].copy(), meta

    sims = 1.0 - cdist(enrollment_centroids, phase1_centroids, metric="cosine")
    ridx, cidx = linear_sum_assignment(-sims)
    matched = {int(r): int(c) for r, c in zip(ridx.tolist(), cidx.tolist())}
    unmatched_phase1 = [j for j in range(phase1_centroids.shape[0]) if j not in set(matched.values())]
    need = max(0, n - k)
    extra_idx = unmatched_phase1[:need]
    combined = []
    meta = []
    for i in range(k):
        combined.append(enrollment_centroids[i].astype(np.float32))
        meta.append({
            "index": int(i),
            "seed_type": "enrollment",
            "seed_name": enrollment_names[i] if i < len(enrollment_names) else f"enrollment_{i}",
            "matched_phase1_cluster": matched.get(i),
        })
    for offset, src_idx in enumerate(extra_idx):
        combined.append(phase1_centroids[src_idx].astype(np.float32))
        meta.append({
            "index": int(k + offset),
            "seed_type": "phase1_bootstrap",
            "seed_name": f"unknown_{offset + 1}",
            "matched_phase1_cluster": int(src_idx),
        })

    if len(combined) != n:
        return None, []
    return np.stack(combined).astype(np.float32), meta

def _bootstrap_missing_centroids_from_clean(chunk_outputs, wav_tensor, ecapa, resampler, known_centroids, n_missing):
    if n_missing <= 0:
        dim = int(known_centroids.shape[1]) if known_centroids is not None and known_centroids.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32), None, 0, {
            "primary_segments": 0,
            "subseg_used_segments": 0,
            "discarded_too_short": 0,
            "discarded_overlap": 0,
            "discarded_known_like": 0,
        }

    thr = float(HIGH_THR_FIXED)
    embs, durs, sinfo, stats = collect_anchor_embeddings(
        chunk_outputs, wav_tensor, ecapa, resampler, threshold=thr
    )
    f_embs, f_durs, f_sinfo = [], [], []
    stats = dict(stats)
    stats["discarded_known_like"] = 0
    for emb, dur, info in zip(embs, durs, sinfo):
        best, margin, _ = _best_sim_and_margin(emb, known_centroids)
        if best >= ENROLLMENT_MATCH_SIM and margin >= ENROLLMENT_MATCH_MARGIN:
            stats["discarded_known_like"] += 1
            continue
        f_embs.append(emb)
        f_durs.append(dur)
        f_sinfo.append(info)

    if len(f_embs) < n_missing:
        return None, None, 0, stats

    labels = cluster_anchors(f_embs, f_durs, f_sinfo, n_missing)
    if labels is None:
        return None, None, 0, stats
    counts = np.bincount(labels, minlength=n_missing)
    if np.any(counts < MIN_ANCHOR_PER_CLUSTER):
        return None, None, 0, stats

    centroids = build_weighted_centroids(f_embs, f_durs, labels, n_missing)
    if centroids is None:
        return None, None, 0, stats
    return centroids, thr, len(f_embs), stats

def initialize_phase1_centroids(chunk_outputs, wav_tensor, ecapa, resampler, n_clusters):
    init_meta = {
        "mode": "phase1_bootstrap",
        "speaker_seed_info": [],
        "enrollment": None,
        "phase1": None,
        "fallback": None,
    }

    if not ENROLLMENT_DIR:
        centroids, high_thr_used, n_anchors, p1_stats = build_phase1_anchors(
            chunk_outputs, wav_tensor, ecapa, resampler, n_clusters
        )
        init_meta["phase1"] = p1_stats
        if centroids is not None:
            init_meta["speaker_seed_info"] = [
                {"index": int(i), "seed_type": "phase1_bootstrap", "seed_name": f"unknown_{i+1}", "matched_phase1_cluster": int(i)}
                for i in range(int(n_clusters))
            ]
        return centroids, high_thr_used, n_anchors, p1_stats, init_meta

    enrollment_centroids, enrollment_names, enr_stats = build_enrollment_centroids(
        ENROLLMENT_DIR, ecapa, resampler, max_speakers=n_clusters
    )
    init_meta["enrollment"] = enr_stats
    if enrollment_centroids is None or enrollment_centroids.shape[0] == 0:
        print("  [Phase 1] Enrollment provided but no valid enrollment embeddings. Falling back to Phase 1.")
        centroids, high_thr_used, n_anchors, p1_stats = build_phase1_anchors(
            chunk_outputs, wav_tensor, ecapa, resampler, n_clusters
        )
        init_meta["mode"] = "phase1_bootstrap"
        init_meta["phase1"] = p1_stats
        if centroids is not None:
            init_meta["speaker_seed_info"] = [
                {"index": int(i), "seed_type": "phase1_bootstrap", "seed_name": f"unknown_{i+1}", "matched_phase1_cluster": int(i)}
                for i in range(int(n_clusters))
            ]
        return centroids, high_thr_used, n_anchors, p1_stats, init_meta

    k = int(enrollment_centroids.shape[0])
    if k >= n_clusters:
        init_meta["mode"] = "enrollment_only"
        init_meta["speaker_seed_info"] = [
            {"index": int(i), "seed_type": "enrollment", "seed_name": enrollment_names[i], "matched_phase1_cluster": None}
            for i in range(int(n_clusters))
        ]
        pseudo_stats = {
            "primary_segments": 0,
            "subseg_used_segments": 0,
            "discarded_too_short": 0,
            "discarded_overlap": 0,
            "enrollment_only": True,
        }
        return enrollment_centroids[:n_clusters].copy(), None, int(k), pseudo_stats, init_meta

    phase1_centroids, high_thr_used, n_anchors, p1_stats = build_phase1_anchors(
        chunk_outputs, wav_tensor, ecapa, resampler, n_clusters
    )
    init_meta["phase1"] = p1_stats
    if phase1_centroids is not None:
        combined, seed_info = _combine_enrollment_with_phase1(
            enrollment_centroids, enrollment_names, phase1_centroids, n_clusters
        )
        if combined is not None:
            init_meta["mode"] = "partial_enrollment_plus_phase1"
            init_meta["speaker_seed_info"] = seed_info
            return combined, high_thr_used, n_anchors, p1_stats, init_meta

    n_missing = int(n_clusters - k)
    missing_centroids, missing_thr, missing_anchors, missing_stats = _bootstrap_missing_centroids_from_clean(
        chunk_outputs, wav_tensor, ecapa, resampler, enrollment_centroids, n_missing
    )
    init_meta["fallback"] = missing_stats
    if missing_centroids is None:
        return None, None, 0, None, init_meta

    combined = np.concatenate([enrollment_centroids, missing_centroids], axis=0).astype(np.float32)
    init_meta["mode"] = "partial_enrollment_plus_clean_unknown"
    init_meta["speaker_seed_info"] = [
        {"index": int(i), "seed_type": "enrollment", "seed_name": enrollment_names[i], "matched_phase1_cluster": None}
        for i in range(k)
    ] + [
        {"index": int(k + i), "seed_type": "clean_bootstrap", "seed_name": f"unknown_{i+1}", "matched_phase1_cluster": None}
        for i in range(n_missing)
    ]
    return combined, missing_thr, int(k + missing_anchors), missing_stats, init_meta

def tier_threshold(tier: str) -> float:
    if tier == "primary":
        return PRIMARY_ASSIGN_SIM
    if tier == "aux":
        return AUX_ASSIGN_SIM
    return FALLBACK_ASSIGN_SIM

def tier_weight(tier: str) -> float:
    if tier == "primary":
        return 3.0
    if tier == "aux":
        return 1.5
    return 0.75

def collect_assign_embeddings(chunk_outputs, wav_tensor, ecapa, resampler, threshold: float):
    embeddings, seg_info = [], []
    stats = {
        "primary": 0,
        "aux": 0,
        "fallback": 0,
        "discarded_lt_0_5": 0,
    }

    for item in chunk_outputs:
        c_idx = item["chunk_idx"]
        offset = item["offset_sec"]
        probs = item["probs"]
        dt_sec = item["dt_sec"]
        act_map = (probs > threshold)
        _, n_tracks = act_map.shape

        for t_idx in range(n_tracks):
            segs = extract_active_segments(act_map[:, t_idx], offset, dt_sec, AUX_MIN_DUR)
            for seg in segs:
                clean_mask = (np.sum(act_map[seg["s_f"]:seg["e_f"], :], axis=1) == 1)

                tier = None
                used_s = seg["s_sec"]
                used_e = seg["e_sec"]
                clean_dur = 0.0

                if np.any(clean_mask):
                    clean_s, clean_e = _longest_clean_span(clean_mask, seg["s_f"], offset, dt_sec)
                    clean_dur = clean_e - clean_s
                    if clean_dur >= PRIMARY_MIN_DUR:
                        tier = "primary"
                        used_s, used_e = clean_s, clean_e
                    elif clean_dur >= AUX_MIN_DUR:
                        tier = "aux"
                        used_s, used_e = clean_s, clean_e
                    else:
                        tier = "fallback"
                        used_s, used_e = seg["s_sec"], seg["e_sec"]
                else:
                    tier = "fallback"
                    used_s, used_e = seg["s_sec"], seg["e_sec"]

                if (used_e - used_s) < AUX_MIN_DUR:
                    stats["discarded_lt_0_5"] += 1
                    continue

                emb, used_dur, n_sub = get_segment_embedding(
                    ecapa, wav_tensor, resampler, used_s, used_e
                )
                if emb is None:
                    stats["discarded_lt_0_5"] += 1
                    continue

                embeddings.append(emb)
                seg_info.append({
                    "chunk": c_idx,
                    "track": t_idx,
                    "s_sec": seg["s_sec"],
                    "e_sec": seg["e_sec"],
                    "tier": tier,
                    "clean_dur": clean_dur,
                    "used_s_sec": used_s,
                    "used_e_sec": used_e,
                    "used_dur": used_dur,
                    "n_sub": n_sub,
                })
                stats[tier] += 1

    return embeddings, seg_info, stats

def assign_to_centroids(embeddings, seg_info, centroids):
    if len(embeddings) == 0:
        return {}, {}, {
            "assigned_embeddings": 0,
            "assigned_primary": 0,
            "assigned_aux": 0,
            "assigned_fallback": 0,
        }, []

    mat = np.stack(embeddings).astype(np.float32)
    cos_dist = cdist(mat, centroids, metric="cosine")
    cos_sim = 1.0 - cos_dist

    init_labels = np.argmax(cos_sim, axis=1)
    init_best = np.max(cos_sim, axis=1)
    valid = np.array([init_best[i] >= tier_threshold(seg_info[i]["tier"]) for i in range(len(seg_info))], dtype=bool)
    raw_labels = init_labels.copy()
    raw_best = init_best.copy()

    def _any_time_overlap(idxs_a, idxs_b, seg_info):
        """True nếu bất kỳ cặp emb nào giữa 2 track_id chồng nhau về thời gian.
        Dùng để phân biệt collision thực sự (cùng lúc) vs same speaker ở different times."""
        for xa in idxs_a:
            s_a = float(seg_info[xa]["s_sec"])
            e_a = float(seg_info[xa]["e_sec"])
            for xb in idxs_b:
                s_b = float(seg_info[xb]["s_sec"])
                e_b = float(seg_info[xb]["e_sec"])
                if s_a < e_b - 1e-6 and s_b < e_a - 1e-6:
                    return True
        return False

    chunk_groups = defaultdict(list)
    for i, info in enumerate(seg_info):
        if valid[i]:
            chunk_groups[info["chunk"]].append(i)

    for c_idx, idxs in chunk_groups.items():
        track_to_idxs = defaultdict(list)
        for i in idxs:
            track_to_idxs[seg_info[i]["track"]].append(i)

        unique_tracks = list(track_to_idxs.keys())
        for ti in range(len(unique_tracks)):
            for tj in range(ti + 1, len(unique_tracks)):
                ta, tb = unique_tracks[ti], unique_tracks[tj]
                cand_a = [x for x in track_to_idxs[ta] if valid[x]]
                cand_b = [x for x in track_to_idxs[tb] if valid[x]]
                if not cand_a or not cand_b:
                    continue

                ia = max(cand_a, key=lambda x: raw_best[x])
                ib = max(cand_b, key=lambda x: raw_best[x])

                if raw_labels[ia] != raw_labels[ib]:
                    continue

                # [FixUnmap] Chỉ fire collision nếu 2 tracks thực sự chồng nhau thời gian.
                has_time_overlap = _any_time_overlap(
                    track_to_idxs[ta], track_to_idxs[tb], seg_info
                )

                if not has_time_overlap:
                    # Không chồng nhau về thời gian → cả hai giữ label của mình.
                    # Không cần collision resolution: same centroid ở different times = OK.
                    continue

                # Có chồng nhau thời gian → collision thực sự, xử lý như cũ
                keep, move = (ia, ib) if raw_best[ia] >= raw_best[ib] else (ib, ia)
                move_tier = seg_info[move]["tier"]

                alternatives = np.argsort(-cos_sim[move])
                reassigned = False
                for alt in alternatives:
                    if alt == raw_labels[keep]:
                        continue
                    if cos_sim[move, alt] >= tier_threshold(move_tier):
                        track_id = seg_info[move]["track"]
                        for idx in track_to_idxs[track_id]:
                            raw_labels[idx] = int(alt)
                            raw_best[idx] = float(cos_sim[idx, alt])
                            valid[idx] = raw_best[idx] >= tier_threshold(seg_info[idx]["tier"])
                        reassigned = True
                        break

                if not reassigned:
                    # [T2] Thử gán vào second-best với ngưỡng mềm hơn trước khi drop.
                    track_id = seg_info[move]["track"]
                    move_tier = seg_info[move]["tier"]
                    soft_thr = tier_threshold(move_tier) * 0.8
                    soft_reassigned = False
                    for alt in alternatives:
                        if alt == raw_labels[keep]:
                            continue
                        if cos_sim[move, alt] >= soft_thr:
                            for idx in track_to_idxs[track_id]:
                                raw_labels[idx] = int(alt)
                                raw_best[idx] = float(cos_sim[idx, alt])
                                valid[idx] = raw_best[idx] >= soft_thr
                            soft_reassigned = True
                            break
                    if not soft_reassigned:
                        for idx in track_to_idxs[track_id]:
                            valid[idx] = False

    embed_details = []
    for i, info in enumerate(seg_info):
        assigned_label = int(raw_labels[i])
        assigned_score = float(cos_sim[i, assigned_label])
        order = np.argsort(-cos_sim[i])
        second_label = None
        second_score = -1.0
        for alt in order:
            alt = int(alt)
            if alt != assigned_label:
                second_label = alt
                second_score = float(cos_sim[i, alt])
                break
        margin = float(assigned_score - second_score) if second_label is not None else float(assigned_score)
        embed_details.append({
            "valid": bool(valid[i]),
            "assigned_label": assigned_label,
            "assigned_score": assigned_score,
            "second_label": second_label,
            "second_score": second_score,
            "margin": margin,
            "tier": info["tier"],
        })

    votes = defaultdict(lambda: defaultdict(float))
    track_meta = {}
    stats = {
        "assigned_embeddings": 0,
        "assigned_primary": 0,
        "assigned_aux": 0,
        "assigned_fallback": 0,
    }

    for i, info in enumerate(seg_info):
        det = embed_details[i]
        if not det["valid"]:
            continue
        key = (info["chunk"], info["track"])
        lbl = int(det["assigned_label"])
        weight = tier_weight(info["tier"]) * float(max(det["assigned_score"], 1e-4))
        votes[key][lbl] += weight

        meta = track_meta.setdefault(key, {
            "best_sim_values": [],
            "tier_votes": defaultdict(float),
            "label_votes": defaultdict(float),
            "all_sims": [],
            "used_dur": [],
            "top1_scores": [],
            "top2_scores": [],
            "margins": [],
        })
        meta["best_sim_values"].append(float(det["assigned_score"]))
        meta["tier_votes"][info["tier"]] += weight
        meta["label_votes"][lbl] += weight
        meta["all_sims"].append(cos_sim[i].astype(np.float32))
        meta["used_dur"].append(float(info["used_dur"]))
        meta["top1_scores"].append(float(det["assigned_score"]))
        meta["top2_scores"].append(float(det["second_score"]) if det["second_label"] is not None else 0.0)
        meta["margins"].append(float(det["margin"]))

        stats["assigned_embeddings"] += 1
        stats[f"assigned_{info['tier']}"] += 1

    mapping = {key: max(counter, key=counter.get) for key, counter in votes.items()}

    final_track_meta = {}
    for key, assigned_lbl in mapping.items():
        meta = track_meta[key]
        avg_sims = np.mean(np.stack(meta["all_sims"], axis=0), axis=0)
        order = np.argsort(-avg_sims)
        top1_label = int(order[0])
        top2_label = int(order[1]) if len(order) > 1 else None
        top1_score = float(avg_sims[top1_label]) if top1_label is not None else 0.0
        top2_score = float(avg_sims[top2_label]) if top2_label is not None else 0.0
        total_tier_vote = float(sum(meta["tier_votes"].values())) if meta["tier_votes"] else 0.0
        primary_ratio = float(meta["tier_votes"].get("primary", 0.0) / max(total_tier_vote, 1e-8))
        aux_ratio = float(meta["tier_votes"].get("aux", 0.0) / max(total_tier_vote, 1e-8))
        fallback_ratio = float(meta["tier_votes"].get("fallback", 0.0) / max(total_tier_vote, 1e-8))
        final_track_meta[key] = {
            "assigned_label": int(assigned_lbl),
            "best_sim": float(np.mean(meta["best_sim_values"])) if meta["best_sim_values"] else 0.0,
            "avg_sim_vector": avg_sims.astype(np.float32),
            "dominant_tier": max(meta["tier_votes"], key=meta["tier_votes"].get),
            "avg_used_dur": float(np.mean(meta["used_dur"])) if meta["used_dur"] else 0.0,
            "top1_label": top1_label,
            "top2_label": top2_label,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "top1_minus_top2": float(top1_score - top2_score),
            "avg_margin": float(np.mean(meta["margins"])) if meta["margins"] else 0.0,
            "primary_vote_ratio": primary_ratio,
            "aux_vote_ratio": aux_ratio,
            "fallback_vote_ratio": fallback_ratio,
        }

    return mapping, final_track_meta, stats, embed_details

def _weighted_centroid(items, dim: int):
    acc = np.zeros((dim,), dtype=np.float32)
    wsum = 0.0
    for emb, weight in items:
        acc += emb * float(weight)
        wsum += float(weight)
    if wsum <= 1e-8:
        return None
    acc /= wsum
    norm = np.linalg.norm(acc)
    if norm <= 1e-8:
        return None
    return acc / norm

def purify_centroids(embeddings, seg_info, embed_details, centroids, n_clusters: int):
    stats = {
        "enabled": bool(CENTROID_PURIFY_ENABLED),
        "iterations": int(CENTROID_PURIFY_ITERS),
        "health_ok": False,
        "skip_reason": None,
        "selected_total": 0,
        "selected_per_cluster": [0] * n_clusters,
        "candidate_total": 0,
        "candidate_per_cluster": [0] * n_clusters,
        "rejected_invalid": 0,
        "rejected_tier": 0,
        "rejected_low_sim": 0,
        "rejected_low_margin": 0,
        "rejected_short": 0,
        "min_cluster_count": 0,
        "max_cluster_count": 0,
        "cluster_imbalance_ratio": None,
        "centroid_shift_cosine": [],
    }
    if not CENTROID_PURIFY_ENABLED or len(embeddings) == 0:
        stats["skip_reason"] = "disabled_or_no_embeddings"
        return centroids, stats

    current = centroids.copy()
    dim = current.shape[1]
    candidate_cache = []
    for emb, info, det in zip(embeddings, seg_info, embed_details):
        if not det.get("valid", False):
            stats["rejected_invalid"] += 1
            continue
        if info.get("tier") not in CENTROID_PURIFY_ACCEPT_TIERS:
            stats["rejected_tier"] += 1
            continue
        if float(info.get("used_dur", 0.0)) < CENTROID_PURIFY_MIN_USED_DUR:
            stats["rejected_short"] += 1
            continue
        if float(det.get("assigned_score", 0.0)) < CENTROID_PURIFY_MIN_SIM:
            stats["rejected_low_sim"] += 1
            continue
        if float(det.get("margin", 0.0)) < CENTROID_PURIFY_MIN_MARGIN:
            stats["rejected_low_margin"] += 1
            continue
        lbl = int(det["assigned_label"])
        weight = float(info.get("used_dur", 0.0)) * max(float(det.get("assigned_score", 0.0)), 1e-4)
        candidate_cache.append((lbl, emb.astype(np.float32), weight))
        stats["candidate_total"] += 1
        stats["candidate_per_cluster"][lbl] += 1

    counts = list(stats["candidate_per_cluster"])
    stats["min_cluster_count"] = int(min(counts)) if counts else 0
    stats["max_cluster_count"] = int(max(counts)) if counts else 0
    denom = max(float(stats["min_cluster_count"]), 1.0)
    stats["cluster_imbalance_ratio"] = float(stats["max_cluster_count"] / denom) if counts else None

    if stats["candidate_total"] < CENTROID_PURIFY_TOTAL_MIN:
        stats["skip_reason"] = "insufficient_total_candidates"
        return centroids, stats
    if stats["min_cluster_count"] < CENTROID_PURIFY_MIN_PER_CLUSTER:
        stats["skip_reason"] = "insufficient_per_cluster_candidates"
        return centroids, stats
    if stats["cluster_imbalance_ratio"] is not None and stats["cluster_imbalance_ratio"] > CENTROID_PURIFY_MAX_CLUSTER_IMBALANCE:
        stats["skip_reason"] = "cluster_imbalance_too_high"
        return centroids, stats

    stats["health_ok"] = True
    for _ in range(max(1, int(CENTROID_PURIFY_ITERS))):
        buckets = defaultdict(list)
        stats["selected_total"] = 0
        stats["selected_per_cluster"] = [0] * n_clusters
        for lbl, emb, weight in candidate_cache:
            buckets[lbl].append((emb, weight))
            stats["selected_total"] += 1
            stats["selected_per_cluster"][lbl] += 1

        # [T4] Hub undersample: tính duration share của mỗi cluster.
        # Nếu cluster k chiếm > HUB_SHARE_HARD tổng duration →
        # chỉ dùng top-N samples có margin (weight) cao nhất để tránh centroid drift về hub.
        total_weight = sum(w for items in buckets.values() for _, w in items)
        hub_undersampled = []
        if total_weight > 1e-8:
            for k in range(n_clusters):
                k_weight = sum(w for _, w in buckets[k])
                share = k_weight / total_weight
                if share > float(HUB_SHARE_HARD):
                    # Giữ top-N samples (sort by weight desc), N = int(HUB_SHARE_HARD * len(buckets[k]))
                    n_keep = max(1, int(float(HUB_SHARE_HARD) * len(buckets[k])))
                    buckets[k] = sorted(buckets[k], key=lambda x: -x[1])[:n_keep]
                    hub_undersampled.append(k)
            if hub_undersampled:
                stats["hub_undersampled_clusters"] = hub_undersampled

        new_centroids = current.copy()
        shifts = []
        skipped_clusters = []
        for k in range(n_clusters):
            n_samples_k = stats["selected_per_cluster"][k]
            if n_samples_k < CENTROID_PURIFY_MIN_STABLE_SAMPLES:
                shifts.append(1.0)
                skipped_clusters.append(k)
                continue
            refined = _weighted_centroid(buckets[k], dim)
            if refined is None:
                shifts.append(1.0)
                continue
            blended = (1.0 - CENTROID_PURIFY_BLEND) * refined + CENTROID_PURIFY_BLEND * current[k]
            norm = np.linalg.norm(blended)
            if norm > 1e-8:
                blended = blended / norm
            shift = float(np.dot(current[k], blended))
            shifts.append(shift)
            new_centroids[k] = blended.astype(np.float32)
        stats["centroid_shift_cosine"].append(shifts)
        stats["skipped_clusters_unstable"] = skipped_clusters
        current = new_centroids

    return current, stats

def annotation_to_mask_dict(annotation: Annotation, total_dur: float, frame_sec: float):
    labels = sorted({str(label) for _, _, label in annotation.itertracks(yield_label=True)})
    n_frames = int(np.ceil(total_dur / frame_sec))
    masks = {lbl: np.zeros((n_frames,), dtype=bool) for lbl in labels}
    for seg, _, lbl in annotation.itertracks(yield_label=True):
        lbl = str(lbl)
        s = max(0, int(np.floor(seg.start / frame_sec)))
        e = min(n_frames, int(np.ceil(seg.end / frame_sec)))
        if e > s:
            masks[lbl][s:e] = True
    return masks

def derive_hyp_to_ref_label_mapping(ref: Annotation, hyp: Annotation, total_dur: float, frame_sec: float):
    ref_masks = annotation_to_mask_dict(ref, total_dur, frame_sec)
    hyp_masks = annotation_to_mask_dict(hyp, total_dur, frame_sec)
    mapping = {}
    for h_lbl, h_mask in hyp_masks.items():
        best_ref = None
        best_overlap = -1.0
        for r_lbl, r_mask in ref_masks.items():
            ov = float(np.sum(h_mask & r_mask)) * frame_sec
            if ov > best_overlap:
                best_overlap = ov
                best_ref = r_lbl
        mapping[h_lbl] = best_ref if best_ref is not None else h_lbl
    return mapping, ref_masks, hyp_masks

def compute_overlap_identity_metrics(ref: Annotation, hyp: Annotation, total_dur: float, frame_sec: float = EVAL_FRAME_SEC):
    mapping, ref_masks, hyp_masks = derive_hyp_to_ref_label_mapping(ref, hyp, total_dur, frame_sec)
    n_frames = int(np.ceil(total_dur / frame_sec))
    ref_labels = sorted(ref_masks.keys())
    mapped_hyp_masks = defaultdict(lambda: np.zeros((n_frames,), dtype=bool))
    for h_lbl, h_mask in hyp_masks.items():
        mapped_hyp_masks[mapping.get(h_lbl, h_lbl)] |= h_mask

    ref_count = np.zeros((n_frames,), dtype=np.int16)
    hyp_count = np.zeros((n_frames,), dtype=np.int16)
    for lbl in ref_labels:
        ref_count += ref_masks[lbl].astype(np.int16)
    for lbl, mask in mapped_hyp_masks.items():
        hyp_count += mask.astype(np.int16)

    ref_overlap = ref_count >= 2
    hyp_overlap = hyp_count >= 2
    ref_single = ref_count == 1

    tp = float(np.sum(ref_overlap & hyp_overlap)) * frame_sec
    fp = float(np.sum((~ref_overlap) & hyp_overlap)) * frame_sec
    fn = float(np.sum(ref_overlap & (~hyp_overlap))) * frame_sec
    hyp_overlap_sec = float(np.sum(hyp_overlap)) * frame_sec
    ref_overlap_sec = float(np.sum(ref_overlap)) * frame_sec
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    exact2 = ref_count == 2
    both_correct_any = both_correct_exact = one_correct = zero_correct = 0.0
    detected_as_overlap = collapsed_to_nonoverlap = 0.0
    false_positive_overlap_on_single = kept_true_plus_extra = both_wrong_overlap_on_single = 0.0

    exact_idx = np.where(exact2)[0]
    for i in exact_idx.tolist():
        ref_set = {lbl for lbl in ref_labels if ref_masks[lbl][i]}
        hyp_set = {lbl for lbl, mask in mapped_hyp_masks.items() if mask[i]}
        inter = len(ref_set & hyp_set)
        if inter >= 2:
            both_correct_any += frame_sec
            if hyp_set == ref_set:
                both_correct_exact += frame_sec
        elif inter == 1:
            one_correct += frame_sec
        else:
            zero_correct += frame_sec
        if len(hyp_set) >= 2:
            detected_as_overlap += frame_sec
        else:
            collapsed_to_nonoverlap += frame_sec

    single_idx = np.where(ref_single & hyp_overlap)[0]
    for i in single_idx.tolist():
        ref_set = {lbl for lbl in ref_labels if ref_masks[lbl][i]}
        hyp_set = {lbl for lbl, mask in mapped_hyp_masks.items() if mask[i]}
        false_positive_overlap_on_single += frame_sec
        if ref_set and len(ref_set & hyp_set) >= 1:
            kept_true_plus_extra += frame_sec
        else:
            both_wrong_overlap_on_single += frame_sec

    return {
        "label_mapping_hyp_to_ref": mapping,
        "overlap": {
            "reference_overlap_seconds": ref_overlap_sec,
            "hypothesis_overlap_seconds": hyp_overlap_sec,
            "true_positive_seconds": tp,
            "false_positive_seconds": fp,
            "false_negative_seconds": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "exact2_identity": {
            "reference_exactly_2_speaker_overlap_seconds": float(np.sum(exact2)) * frame_sec,
            "both_correct_any_seconds": both_correct_any,
            "both_correct_exact_seconds": both_correct_exact,
            "one_correct_seconds": one_correct,
            "zero_correct_seconds": zero_correct,
            "detected_as_overlap_seconds": detected_as_overlap,
            "collapsed_to_nonoverlap_seconds": collapsed_to_nonoverlap,
        },
        "false_positive_overlap_on_single": {
            "seconds": false_positive_overlap_on_single,
            "kept_true_plus_extra_seconds": kept_true_plus_extra,
            "both_wrong_seconds": both_wrong_overlap_on_single,
        },
    }

def build_stage_report(stage_name: str, ref: Annotation, hyp: Annotation, total_dur: float):
    """
    Build a report summarizing diarization quality for a particular stage.

    Parameters
    ----------
    stage_name : str
        Descriptive name of the current processing stage.
    ref : Annotation
        Reference annotation (ground truth) used for computing DER. May be empty
        if no reference is provided.
    hyp : Annotation
        System hypothesis annotation to evaluate.
    total_dur : float
        Total duration of the audio in seconds, used when computing overlap metrics.

    Returns
    -------
    dict
        A dictionary containing the stage name, DER statistics and overlap
        metrics. When the global flag SKIP_DER is True the DER values are
        reported as None.
    """
    # Compute DER only when enabled. We still compute overlap metrics regardless
    # of SKIP_DER because overlap statistics may be useful for downstream
    # processing.
    if not SKIP_DER:
        der, miss, fa, conf = compute_der(ref, hyp)
        der_f = float(der)
        miss_f = float(miss)
        fa_f = float(fa)
        conf_f = float(conf)
    else:
        der_f = None
        miss_f = None
        fa_f = None
        conf_f = None
    ov = compute_overlap_identity_metrics(ref, hyp, total_dur, frame_sec=EVAL_FRAME_SEC)
    return {
        "stage": stage_name,
        "der": der_f,
        "miss": miss_f,
        "fa": fa_f,
        "conf": conf_f,
        "overlap_metrics": ov,
    }

def _pack_force_assign_meta_from_emb(key, emb, tier, used_dur, centroids):
    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm <= 1e-8:
        return None
    emb = (emb / norm).astype(np.float32)
    sims = 1.0 - cdist(emb[None, :], centroids, metric="cosine")[0]
    order = np.argsort(-sims)
    top1_label = int(order[0])
    top2_label = int(order[1]) if len(order) > 1 else None
    top1_score = float(sims[top1_label])
    top2_score = float(sims[top2_label]) if top2_label is not None else 0.0
    primary_ratio = 1.0 if tier == "primary" else 0.0
    aux_ratio = 1.0 if tier == "aux" else 0.0
    fallback_ratio = 1.0 if tier == "fallback" else 0.0
    return {
        "assigned_label": top1_label,
        "best_sim": top1_score,
        "avg_sim_vector": sims.astype(np.float32),
        "dominant_tier": tier,
        "avg_used_dur": float(used_dur),
        "top1_label": top1_label,
        "top2_label": top2_label,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "top1_minus_top2": float(top1_score - top2_score),
        "avg_margin": float(top1_score - top2_score),
        "primary_vote_ratio": primary_ratio,
        "aux_vote_ratio": aux_ratio,
        "fallback_vote_ratio": fallback_ratio,
    }


def _get_force_assign_embedding(ecapa, wav_tensor: torch.Tensor, resampler, s_sec: float, e_sec: float):
    emb, used_dur = get_embedding_interval(ecapa, wav_tensor, resampler, s_sec, e_sec)
    if emb is not None:
        return emb, used_dur

    raw_dur = float(e_sec - s_sec)
    if raw_dur < 0.20:
        return None, 0.0

    si = int(s_sec * TARGET_SR)
    ei = int(e_sec * TARGET_SR)
    max_s = wav_tensor.shape[1]
    si = max(0, min(si, max_s))
    ei = max(0, min(ei, max_s))
    if (ei - si) < int(0.20 * TARGET_SR):
        return None, 0.0

    slc = wav_tensor[:, si:ei]
    if slc.shape[1] < int(0.20 * TARGET_SR):
        return None, 0.0

    try:
        slc = slc.to(DEVICE)
        slc_16k = resampler(slc) if resampler is not None else slc
        with torch.no_grad():
            emb_obj = None
            for attr in ("_embedding", "embedding", "model"):
                candidate = getattr(ecapa, attr, None)
                if callable(candidate):
                    emb_obj = candidate
                    break
            if emb_obj is None:
                raise AttributeError(
                    f"Cannot find a callable embedding attribute on {type(ecapa).__name__}. "
                    "Checked: _embedding, embedding, model."
                )
            emb = emb_obj(slc_16k.unsqueeze(0))
            if hasattr(emb, "detach"):
                emb = emb.squeeze().detach().cpu().numpy()
            else:
                emb = np.squeeze(emb)
            emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm <= 1e-8:
            return None, 0.0
        return emb / norm, raw_dur
    except Exception:
        return None, 0.0


def _find_densest_active_window(mask_1d: np.ndarray, window_frames: int):
    n = int(mask_1d.shape[0])
    if n <= 0:
        return 0, 0, 0
    window_frames = max(1, min(int(window_frames), n))
    active = mask_1d.astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(active)])
    best_s, best_e, best_count = 0, window_frames, -1
    for s in range(0, n - window_frames + 1):
        e = s + window_frames
        count = int(csum[e] - csum[s])
        if count > best_count:
            best_s, best_e, best_count = s, e, count
    return best_s, best_e, best_count


def _expand_span_to_min_frames(s_f: int, e_f: int, n_frames: int, min_frames: int):
    cur = int(e_f - s_f)
    if cur >= int(min_frames):
        return int(s_f), int(e_f)
    extra = int(min_frames - cur)
    left = extra // 2
    right = extra - left
    s_f = max(0, int(s_f - left))
    e_f = min(int(n_frames), int(e_f + right))
    cur = int(e_f - s_f)
    if cur >= int(min_frames):
        return int(s_f), int(e_f)
    if s_f == 0:
        e_f = min(int(n_frames), int(min_frames))
    elif e_f == int(n_frames):
        s_f = max(0, int(n_frames - min_frames))
    return int(s_f), int(e_f)


def _build_fragmented_force_segment(mask_1d: np.ndarray, offset_sec: float, dt_sec: float):
    active_idx = np.flatnonzero(mask_1d)
    if active_idx.size == 0:
        return None, 'no_active'

    total_active_sec = float(active_idx.size * dt_sec)
    if total_active_sec < FORCE_FRAGMENT_MIN_TOTAL_ACTIVE_SEC:
        return None, 'too_sparse'

    dense_win_frames = max(1, int(round(FORCE_FRAGMENT_DENSE_WIN_SEC / max(dt_sec, 1e-8))))
    s_f, e_f, active_count = _find_densest_active_window(mask_1d, dense_win_frames)
    if active_count <= 0:
        return None, 'no_dense_window'

    pad_frames = max(1, int(round(FORCE_FRAGMENT_PAD_SEC / max(dt_sec, 1e-8))))
    s_f = max(0, int(s_f - pad_frames))
    e_f = min(int(mask_1d.shape[0]), int(e_f + pad_frames))

    min_embed_frames = max(1, int(round(FORCE_FRAGMENT_MIN_EMBED_SEC / max(dt_sec, 1e-8))))
    s_f, e_f = _expand_span_to_min_frames(s_f, e_f, int(mask_1d.shape[0]), min_embed_frames)

    max_win_frames = max(min_embed_frames, int(round(FORCE_FRAGMENT_MAX_WIN_SEC / max(dt_sec, 1e-8))))
    if (e_f - s_f) > max_win_frames:
        mid = int(round((s_f + e_f) / 2.0))
        half = max_win_frames // 2
        s_f = max(0, mid - half)
        e_f = min(int(mask_1d.shape[0]), s_f + max_win_frames)
        s_f = max(0, int(e_f - max_win_frames))

    seg = {
        's_f': int(s_f),
        'e_f': int(e_f),
        's_sec': float(offset_sec + s_f * dt_sec),
        'e_sec': float(offset_sec + e_f * dt_sec),
        'dur': float((e_f - s_f) * dt_sec),
        'fragment_total_active_sec': total_active_sec,
        'fragment_dense_active_frames': int(active_count),
        'source': 'fragment_dense_window',
    }
    return seg, 'ok'


def build_force_assign_track_meta(embeddings, seg_info, centroids, chunk_outputs=None, wav_tensor=None, ecapa=None, resampler=None, threshold=None, mapped_keys=None):
    global LAST_FORCE_ASSIGN_META_DEBUG
    debug_stats = {
        "force_meta_bucket_keys": 0,
        "force_meta_bucket_entries_created": 0,
        "force_meta_tracks_seen": 0,
        "force_meta_tracks_existing_skipped": 0,
        "force_meta_tracks_already_mapped_skipped": 0,
        "force_meta_tracks_with_segments": 0,
        "force_meta_tracks_no_segments": 0,
        "force_meta_tracks_no_active": 0,
        "force_meta_tracks_fragment_window": 0,
        "force_meta_tracks_fragment_too_sparse": 0,
        "force_meta_tracks_fragment_no_dense_window": 0,
        "force_meta_tracks_clean_primary": 0,
        "force_meta_tracks_clean_aux": 0,
        "force_meta_tracks_clean_too_short": 0,
        "force_meta_tracks_embedding_none": 0,
        "force_meta_entries_created_from_force_only": 0,
        "force_meta_total_entries": 0,
        "force_meta_force_only_sample_keys": [],
    }
    if len(embeddings) == 0 and chunk_outputs is None:
        LAST_FORCE_ASSIGN_META_DEBUG = dict(debug_stats)
        return {}

    buckets = defaultdict(lambda: {
        "weighted_emb_sum": None,
        "weight_sum": 0.0,
        "tier_weights": defaultdict(float),
        "used_dur": [],
    })

    for emb, info in zip(embeddings, seg_info):
        key = (info["chunk"], info["track"])
        weight = max(float(info.get("used_dur", 0.0)), 1e-4)
        bucket = buckets[key]
        if bucket["weighted_emb_sum"] is None:
            bucket["weighted_emb_sum"] = np.zeros_like(emb, dtype=np.float32)
        bucket["weighted_emb_sum"] += emb.astype(np.float32) * weight
        bucket["weight_sum"] += weight
        bucket["tier_weights"][info.get("tier", "fallback")] += weight
        bucket["used_dur"].append(float(info.get("used_dur", 0.0)))

    debug_stats["force_meta_bucket_keys"] = int(len(buckets))
    out = {}
    for key, bucket in buckets.items():
        if bucket["weighted_emb_sum"] is None or bucket["weight_sum"] <= 1e-8:
            continue
        avg_emb = bucket["weighted_emb_sum"] / float(bucket["weight_sum"])
        norm = np.linalg.norm(avg_emb)
        if norm <= 1e-8:
            continue
        avg_emb = (avg_emb / norm).astype(np.float32)
        sims = 1.0 - cdist(avg_emb[None, :], centroids, metric="cosine")[0]
        order = np.argsort(-sims)
        top1_label = int(order[0])
        top2_label = int(order[1]) if len(order) > 1 else None
        top1_score = float(sims[top1_label])
        top2_score = float(sims[top2_label]) if top2_label is not None else 0.0
        total_tier_weight = float(sum(bucket["tier_weights"].values()))
        dominant_tier = max(bucket["tier_weights"], key=bucket["tier_weights"].get) if bucket["tier_weights"] else "fallback"
        out[key] = {
            "assigned_label": top1_label,
            "best_sim": top1_score,
            "avg_sim_vector": sims.astype(np.float32),
            "dominant_tier": dominant_tier,
            "avg_used_dur": float(np.mean(bucket["used_dur"])) if bucket["used_dur"] else 0.0,
            "top1_label": top1_label,
            "top2_label": top2_label,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "top1_minus_top2": float(top1_score - top2_score),
            "avg_margin": float(top1_score - top2_score),
            "primary_vote_ratio": float(bucket["tier_weights"].get("primary", 0.0) / max(total_tier_weight, 1e-8)),
            "aux_vote_ratio": float(bucket["tier_weights"].get("aux", 0.0) / max(total_tier_weight, 1e-8)),
            "fallback_vote_ratio": float(bucket["tier_weights"].get("fallback", 0.0) / max(total_tier_weight, 1e-8)),
        }
        debug_stats["force_meta_bucket_entries_created"] += 1

    if chunk_outputs is None or wav_tensor is None or ecapa is None or threshold is None:
        debug_stats["force_meta_total_entries"] = int(len(out))
        LAST_FORCE_ASSIGN_META_DEBUG = dict(debug_stats)
        return out

    existing_keys = set(out.keys())
    mapped_keys = set(mapped_keys or [])
    for item in chunk_outputs:
        c_idx = item["chunk_idx"]
        offset = item["offset_sec"]
        probs = item["probs"]
        dt_sec = item["dt_sec"]
        act_map = (probs > float(threshold))
        _, n_tracks = act_map.shape

        for t_idx in range(n_tracks):
            key = (c_idx, t_idx)
            debug_stats["force_meta_tracks_seen"] += 1
            if key in existing_keys:
                debug_stats["force_meta_tracks_existing_skipped"] += 1
                continue
            if key in mapped_keys:
                debug_stats["force_meta_tracks_already_mapped_skipped"] += 1
                continue

            segs = extract_active_segments(act_map[:, t_idx], offset, dt_sec, MIN_DUR_FORCE)
            seg_source = "continuous_segment"
            if not segs:
                debug_stats["force_meta_tracks_no_segments"] += 1
                frag_seg, frag_reason = _build_fragmented_force_segment(act_map[:, t_idx], offset, dt_sec)
                if frag_seg is None:
                    if frag_reason == "no_active":
                        debug_stats["force_meta_tracks_no_active"] += 1
                    elif frag_reason == "too_sparse":
                        debug_stats["force_meta_tracks_fragment_too_sparse"] += 1
                    elif frag_reason == "no_dense_window":
                        debug_stats["force_meta_tracks_fragment_no_dense_window"] += 1
                    continue
                seg = frag_seg
                seg_source = "fragment_dense_window"
                debug_stats["force_meta_tracks_fragment_window"] += 1
            else:
                debug_stats["force_meta_tracks_with_segments"] += 1
                seg = max(segs, key=lambda s: (float(s["dur"]), -float(s["s_sec"])))
            clean_mask = (np.sum(act_map[seg["s_f"]:seg["e_f"], :], axis=1) == 1)
            tier = "fallback"
            used_s = seg["s_sec"]
            used_e = seg["e_sec"]
            clean_dur = 0.0

            if np.any(clean_mask):
                clean_s, clean_e = _longest_clean_span(clean_mask, seg["s_f"], offset, dt_sec)
                clean_dur = float(clean_e - clean_s)
                if clean_dur >= PRIMARY_MIN_DUR:
                    tier = "primary"
                    used_s, used_e = clean_s, clean_e
                    debug_stats["force_meta_tracks_clean_primary"] += 1
                elif clean_dur >= MIN_CLEAN_DUR_FORCE:
                    tier = "aux"
                    used_s, used_e = clean_s, clean_e
                    debug_stats["force_meta_tracks_clean_aux"] += 1
                else:
                    debug_stats["force_meta_tracks_clean_too_short"] += 1

            emb, used_dur = _get_force_assign_embedding(ecapa, wav_tensor, resampler, used_s, used_e)
            if emb is None:
                debug_stats["force_meta_tracks_embedding_none"] += 1
                continue

            meta = _pack_force_assign_meta_from_emb(key, emb, tier, used_dur, centroids)
            if meta is None:
                debug_stats["force_meta_tracks_embedding_none"] += 1
                continue
            meta["source"] = "force_only_embedding"
            meta["segment_source"] = seg_source
            meta["segment_start"] = float(seg["s_sec"])
            meta["segment_end"] = float(seg["e_sec"])
            meta["clean_dur"] = clean_dur
            if "fragment_total_active_sec" in seg:
                meta["fragment_total_active_sec"] = float(seg["fragment_total_active_sec"])
            if "fragment_dense_active_frames" in seg:
                meta["fragment_dense_active_frames"] = int(seg["fragment_dense_active_frames"])
            out[key] = meta
            debug_stats["force_meta_entries_created_from_force_only"] += 1
            if len(debug_stats["force_meta_force_only_sample_keys"]) < 8:
                debug_stats["force_meta_force_only_sample_keys"].append(f"{key}|{meta.get('segment_source','?')}|tier={meta.get('dominant_tier', tier)}")

    debug_stats["force_meta_total_entries"] = int(len(out))
    LAST_FORCE_ASSIGN_META_DEBUG = dict(debug_stats)
    print(
        "  [ForceMeta] "
        f"seen={debug_stats['force_meta_tracks_seen']} | "
        f"existing_skip={debug_stats['force_meta_tracks_existing_skipped']} | "
        f"already_mapped_skip={debug_stats['force_meta_tracks_already_mapped_skipped']} | "
        f"with_segments={debug_stats['force_meta_tracks_with_segments']} | "
        f"no_segments={debug_stats['force_meta_tracks_no_segments']} | "
        f"no_active={debug_stats['force_meta_tracks_no_active']} | "
        f"fragment_window={debug_stats['force_meta_tracks_fragment_window']} | "
        f"fragment_too_sparse={debug_stats['force_meta_tracks_fragment_too_sparse']} | "
        f"fragment_no_dense={debug_stats['force_meta_tracks_fragment_no_dense_window']} | "
        f"clean_primary={debug_stats['force_meta_tracks_clean_primary']} | "
        f"clean_aux={debug_stats['force_meta_tracks_clean_aux']} | "
        f"clean_too_short={debug_stats['force_meta_tracks_clean_too_short']} | "
        f"emb_none={debug_stats['force_meta_tracks_embedding_none']} | "
        f"bucket_entries={debug_stats['force_meta_bucket_entries_created']} | "
        f"force_only_entries={debug_stats['force_meta_entries_created_from_force_only']} | "
        f"total_entries={debug_stats['force_meta_total_entries']}"
    )
    if debug_stats["force_meta_force_only_sample_keys"]:
        print("  [ForceMeta sample] " + " ; ".join(debug_stats["force_meta_force_only_sample_keys"]))
    return out


GLOBAL_PROPAGATE_MIN_CONFIDENCE = 0.50  # [FixUnmap] was 0.65

ORPHAN_FORCE_PRIMARY_MIN_SIM = 0.34
ORPHAN_FORCE_PRIMARY_MIN_MARGIN = 0.06
ORPHAN_FORCE_FALLBACK_MIN_SIM = 0.40
ORPHAN_FORCE_FALLBACK_MIN_MARGIN = 0.08

LAST_FORCE_ASSIGN_META_DEBUG = {}


def _allow_orphan_force_assign(meta):
    if meta is None:
        return False, "missing_meta"
    tier = str(meta.get("dominant_tier", "fallback"))
    top1 = float(meta.get("top1_score", 0.0))
    margin = float(meta.get("top1_minus_top2", 0.0))
    if tier in {"primary", "aux"}:
        min_sim = ORPHAN_FORCE_PRIMARY_MIN_SIM
        min_margin = ORPHAN_FORCE_PRIMARY_MIN_MARGIN
    else:
        min_sim = ORPHAN_FORCE_FALLBACK_MIN_SIM
        min_margin = ORPHAN_FORCE_FALLBACK_MIN_MARGIN
    if top1 < float(min_sim):
        return False, "low_score"
    if margin < float(min_margin):
        return False, "low_margin"
    return True, "accepted"

def propagate_missing_tracks(chunk_outputs, mapping, track_meta, n_clusters: int, force_assign_track_meta=None, threshold=None):
    final_map = dict(mapping)
    final_meta = dict(track_meta)
    stats = {
        "propagated_tracks": 0,
        "force_assigned_tracks": 0,
        "force_assign_candidates_total": 0,
        "force_assign_lookup_hits_pass1": 0,
        "force_assign_lookup_misses_pass1": 0,
        "force_assign_lookup_misses_pass1_active": 0,
        "force_assign_lookup_misses_pass1_inactive": 0,
        "force_assign_same_track_other_chunk_pass1": 0,
        "force_assign_lookup_hits_pass3": 0,
        "force_assign_lookup_misses_pass3": 0,
        "force_assign_lookup_misses_pass3_active": 0,
        "force_assign_lookup_misses_pass3_inactive": 0,
        "force_assign_same_track_other_chunk_pass3": 0,
        "force_assign_orphan_accepted": 0,
        "force_assign_orphan_rejected_low_score": 0,
        "force_assign_orphan_rejected_low_margin": 0,
        "force_assign_orphan_rejected_other": 0,
        "global_propagated_tracks": 0,           # [T3]
        "global_propagate_rejected_low_conf": 0, # [T3]
        "unmapped_tracks": 0,
        "unmapped_tracks_active": 0,
        "unmapped_tracks_inactive": 0,
        "force_assign_sample_pass1_miss_active": [],
        "force_assign_sample_pass3_miss_active": [],
    }

    same_track_index = defaultdict(list)
    for (chunk_id, track_id), g in mapping.items():
        same_track_index[track_id].append((chunk_id, g))
    for track_id in same_track_index:
        same_track_index[track_id].sort(key=lambda x: x[0])

    same_track_meta = defaultdict(list)
    for key, meta in track_meta.items():
        c_idx, t_idx = key
        same_track_meta[t_idx].append((c_idx, meta))
    for t_idx in same_track_meta:
        same_track_meta[t_idx].sort(key=lambda x: x[0])

    force_assign_track_meta = force_assign_track_meta or {}
    threshold = float(LOW_THRESHOLD if threshold is None else threshold)
    force_keys_by_track = defaultdict(int)
    for fk in force_assign_track_meta.keys():
        force_keys_by_track[int(fk[1])] += 1

    # Pass 1: neighbor propagation + force assign (không đổi)
    for item in chunk_outputs:
        c_idx = item["chunk_idx"]
        n_tracks = item["probs"].shape[1]
        for t_idx in range(n_tracks):
            key = (c_idx, t_idx)
            if key in final_map:
                continue

            if t_idx in same_track_index and same_track_index[t_idx]:
                nearest = min(same_track_index[t_idx], key=lambda x: abs(x[0] - c_idx))
                if abs(nearest[0] - c_idx) <= MAX_PROPAGATE_CHUNK_GAP:
                    final_map[key] = nearest[1]
                    stats["propagated_tracks"] += 1
                    if t_idx in same_track_meta and same_track_meta[t_idx]:
                        nearest_meta = min(same_track_meta[t_idx], key=lambda x: abs(x[0] - c_idx))[1]
                        final_meta[key] = dict(nearest_meta)
                    continue

            forced_meta = force_assign_track_meta.get(key)
            if forced_meta is not None:
                stats["force_assign_lookup_hits_pass1"] += 1
                stats["force_assign_candidates_total"] += 1
                final_map[key] = int(forced_meta["assigned_label"])
                final_meta[key] = dict(forced_meta)
                stats["force_assigned_tracks"] += 1
                continue
            stats["force_assign_lookup_misses_pass1"] += 1
            active_here = bool(np.any(item["probs"][:, t_idx] > threshold))
            if active_here:
                stats["force_assign_lookup_misses_pass1_active"] += 1
                if len(stats["force_assign_sample_pass1_miss_active"]) < 8:
                    stats["force_assign_sample_pass1_miss_active"].append(str(key))
            else:
                stats["force_assign_lookup_misses_pass1_inactive"] += 1
            if int(t_idx) in force_keys_by_track:
                stats["force_assign_same_track_other_chunk_pass1"] += 1

    # [T3] Pass 2: Global track voting cho tracks vẫn còn unmapped
    # Build global label vote: với mỗi track_id, tổng hợp label được assign nhiều nhất
    # (weighted by best_sim) từ toàn bộ mapping đã có.
    # Chỉ accept nếu confidence >= GLOBAL_PROPAGATE_MIN_CONFIDENCE để tránh permutation lỗi.
    track_label_votes = defaultdict(lambda: defaultdict(float))
    for (chunk_id, track_id), label in final_map.items():
        meta = final_meta.get((chunk_id, track_id), {})
        weight = float(meta.get("best_sim", 0.1)) if meta else 0.1
        weight = max(weight, 0.01)
        track_label_votes[track_id][label] += weight

    global_track_label = {}
    global_track_confidence = {}
    for track_id, vote_dict in track_label_votes.items():
        total_vote = sum(vote_dict.values())
        best_label = max(vote_dict, key=vote_dict.get)
        confidence = vote_dict[best_label] / max(total_vote, 1e-8)
        global_track_label[track_id] = best_label
        global_track_confidence[track_id] = confidence

    remaining_unmapped = 0
    for item in chunk_outputs:
        c_idx = item["chunk_idx"]
        n_tracks = item["probs"].shape[1]
        for t_idx in range(n_tracks):
            key = (c_idx, t_idx)
            if key in final_map:
                continue

            if t_idx in global_track_label:
                conf = global_track_confidence.get(t_idx, 0.0)
                if conf >= GLOBAL_PROPAGATE_MIN_CONFIDENCE:
                    final_map[key] = global_track_label[t_idx]
                    stats["global_propagated_tracks"] += 1
                    if t_idx in same_track_meta and same_track_meta[t_idx]:
                        nm = min(same_track_meta[t_idx], key=lambda x: abs(x[0] - c_idx))[1]
                        final_meta[key] = dict(nm)
                else:
                    stats["global_propagate_rejected_low_conf"] += 1
                    remaining_unmapped += 1
            else:
                remaining_unmapped += 1

    # [FixUnmap Pass-3] Hard fallback: slot vẫn unmapped sau global propagation
    # → lấy nearest assigned neighbor theo t_idx, không giới hạn gap, không conf gate.
    # Lý do: EEND detect activity thực → không assign = Miss; worst case là Conf.
    stats["hard_fallback_tracks"] = 0
    stats["hard_fallback_no_history"] = 0
    for item in chunk_outputs:
        c_idx = item["chunk_idx"]
        n_tracks = item["probs"].shape[1]
        for t_idx in range(n_tracks):
            key = (c_idx, t_idx)
            if key in final_map:
                continue
            if t_idx in same_track_index and same_track_index[t_idx]:
                nearest = min(same_track_index[t_idx], key=lambda x: abs(x[0] - c_idx))
                final_map[key] = nearest[1]
                stats["hard_fallback_tracks"] += 1
                if t_idx in same_track_meta and same_track_meta[t_idx]:
                    nm = min(same_track_meta[t_idx], key=lambda x: abs(x[0] - c_idx))[1]
                    final_meta[key] = dict(nm)
            elif t_idx in global_track_label:
                final_map[key] = global_track_label[t_idx]
                stats["hard_fallback_tracks"] += 1
            else:
                forced_meta = force_assign_track_meta.get(key)
                if forced_meta is not None:
                    stats["force_assign_lookup_hits_pass3"] += 1
                    stats["force_assign_candidates_total"] += 1
                    allow_force, reject_reason = _allow_orphan_force_assign(forced_meta)
                    if allow_force:
                        final_map[key] = int(forced_meta["assigned_label"])
                        final_meta[key] = dict(forced_meta)
                        stats["force_assigned_tracks"] += 1
                        stats["force_assign_orphan_accepted"] += 1
                    else:
                        if reject_reason == "low_score":
                            stats["force_assign_orphan_rejected_low_score"] += 1
                        elif reject_reason == "low_margin":
                            stats["force_assign_orphan_rejected_low_margin"] += 1
                        else:
                            stats["force_assign_orphan_rejected_other"] += 1
                        stats["hard_fallback_no_history"] += 1
                else:
                    stats["force_assign_lookup_misses_pass3"] += 1
                    active_here = bool(np.any(item["probs"][:, t_idx] > threshold))
                    if active_here:
                        stats["force_assign_lookup_misses_pass3_active"] += 1
                        if len(stats["force_assign_sample_pass3_miss_active"]) < 8:
                            stats["force_assign_sample_pass3_miss_active"].append(str(key))
                    else:
                        stats["force_assign_lookup_misses_pass3_inactive"] += 1
                    if int(t_idx) in force_keys_by_track:
                        stats["force_assign_same_track_other_chunk_pass3"] += 1
                    stats["hard_fallback_no_history"] += 1

    remaining_after_fallback = sum(
        1 for item in chunk_outputs
        for t_idx in range(item["probs"].shape[1])
        if (item["chunk_idx"], t_idx) not in final_map
    )
    stats["unmapped_tracks"] = remaining_after_fallback
    active_unmapped = 0
    inactive_unmapped = 0
    for item in chunk_outputs:
        c_idx = item["chunk_idx"]
        n_tracks = item["probs"].shape[1]
        for t_idx in range(n_tracks):
            key = (c_idx, t_idx)
            if key in final_map:
                continue
            if bool(np.any(item["probs"][:, t_idx] > threshold)):
                active_unmapped += 1
            else:
                inactive_unmapped += 1
    stats["unmapped_tracks_active"] = active_unmapped
    stats["unmapped_tracks_inactive"] = inactive_unmapped

    print(
        f"  [Propagate] neighbor={stats['propagated_tracks']} | "
        f"force={stats['force_assigned_tracks']} | "
        f"force_candidates={stats['force_assign_candidates_total']} | "
        f"force_hit_p1={stats['force_assign_lookup_hits_pass1']} | "
        f"force_miss_p1={stats['force_assign_lookup_misses_pass1']} | "
        f"force_miss_p1_active={stats['force_assign_lookup_misses_pass1_active']} | "
        f"force_same_track_p1={stats['force_assign_same_track_other_chunk_pass1']} | "
        f"force_hit_p3={stats['force_assign_lookup_hits_pass3']} | "
        f"force_miss_p3={stats['force_assign_lookup_misses_pass3']} | "
        f"force_miss_p3_active={stats['force_assign_lookup_misses_pass3_active']} | "
        f"force_same_track_p3={stats['force_assign_same_track_other_chunk_pass3']} | "
        f"force_orphan_ok={stats['force_assign_orphan_accepted']} | "
        f"force_orphan_low_score={stats['force_assign_orphan_rejected_low_score']} | "
        f"force_orphan_low_margin={stats['force_assign_orphan_rejected_low_margin']} | "
        f"global={stats['global_propagated_tracks']} | "
        f"global_rej={stats['global_propagate_rejected_low_conf']} | "
        f"hard_fallback={stats['hard_fallback_tracks']} | "
        f"no_history={stats['hard_fallback_no_history']} | "
        f"unmapped={stats['unmapped_tracks']} | "
        f"unmapped_active={stats['unmapped_tracks_active']} | "
        f"unmapped_inactive={stats['unmapped_tracks_inactive']}"
    )
    return final_map, final_meta, stats

def merge_same_speaker_gaps(hyp: Annotation, max_gap: float):
    per_label = defaultdict(list)
    for segment, _, label in hyp.itertracks(yield_label=True):
        per_label[label].append((segment.start, segment.end))

    merged = Annotation(uri=hyp.uri)
    for label, spans in per_label.items():
        if not spans:
            continue
        spans.sort()
        cur_s, cur_e = spans[0]
        seg_idx = 0
        for s, e in spans[1:]:
            if s - cur_e <= max_gap:
                cur_e = max(cur_e, e)
            else:
                merged[Segment(cur_s, cur_e), f"{label}_{seg_idx}"] = label
                seg_idx += 1
                cur_s, cur_e = s, e
        merged[Segment(cur_s, cur_e), f"{label}_{seg_idx}"] = label
    return merged

def apply_silence_filter(rows, wav_tensor: torch.Tensor, uri: str):
    stats = {
        "enabled": bool(POST_SILENCE_FILTER_ENABLED),
        "removed": 0,
        "kept": 0,
        "removed_dur_sec": 0.0,
        "removed_by_tier": {},
    }

    def _build_hyp(r_list):
        h = Annotation(uri=uri)
        for idx, r in enumerate(r_list):
            h[Segment(r["start"], r["end"]), f"sf_{idx}"] = r["speaker"]
        return h

    if not POST_SILENCE_FILTER_ENABLED or not rows:
        stats["enabled"] = False
        return _build_hyp(rows), rows, stats

    wav_np = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
    total_samples = len(wav_np)
    frame_samples = max(1, int(POST_SILENCE_FRAME_SEC * TARGET_SR))
    hop_samples   = max(1, int(POST_SILENCE_HOP_SEC   * TARGET_SR))

    rms_frames = []
    pos = 0
    while pos + frame_samples <= total_samples:
        frame = wav_np[pos : pos + frame_samples]
        rms_frames.append(float(np.sqrt(np.mean(frame ** 2))))
        pos += hop_samples
    if pos < total_samples:
        frame = wav_np[pos:]
        rms_frames.append(float(np.sqrt(np.mean(frame ** 2))))

    if not rms_frames:
        stats["enabled"] = False
        stats["reason"] = "no_rms_frames"
        return _build_hyp(rows), rows, stats

    rms_arr    = np.array(rms_frames, dtype=np.float32)
    global_rms = float(np.sqrt(np.mean(rms_arr ** 2))) + 1e-8
    norm_rms   = rms_arr / global_rms

    activity_mask = (norm_rms >= POST_SILENCE_ENERGY_THR).astype(np.float32)
    if POST_SILENCE_SMOOTHING_FRAMES > 1:
        activity_mask = median_filter(activity_mask, size=POST_SILENCE_SMOOTHING_FRAMES)
    n_frames = len(activity_mask)

    kept_rows = []
    for r in rows:
        if float(r.get("top1_score", 0.0)) >= POST_SILENCE_PROTECT_HIGH_EEND:
            kept_rows.append(r)
            stats["kept"] += 1
            continue

        s_f = int(r["start"] / POST_SILENCE_HOP_SEC)
        e_f = int(r["end"]   / POST_SILENCE_HOP_SEC)
        s_f = max(0, min(s_f, n_frames - 1))
        e_f = max(s_f + 1,   min(e_f, n_frames))
        seg_mask     = activity_mask[s_f:e_f]
        speech_ratio = float(np.mean(seg_mask)) if len(seg_mask) > 0 else 0.0

        tier = r.get("tier", "fallback")
        thr  = (POST_SILENCE_FALLBACK_RATIO
                if tier == "fallback"
                else POST_SILENCE_MIN_SPEECH_RATIO)

        if speech_ratio < thr:
            stats["removed"]          += 1
            stats["removed_dur_sec"]  += float(r["end"] - r["start"])
            stats["removed_by_tier"][tier] = stats["removed_by_tier"].get(tier, 0) + 1
        else:
            kept_rows.append(r)
            stats["kept"] += 1

    return _build_hyp(kept_rows), kept_rows, stats

def build_raw_hypothesis(chunk_outputs, final_map, track_meta, threshold: float, uri: str):
    hyp = Annotation(uri=uri)
    rows = []
    for item in chunk_outputs:
        c_idx = item["chunk_idx"]
        offset = item["offset_sec"]
        probs = item["probs"]
        dt_sec = item["dt_sec"]
        act_map = (probs > threshold)
        _, n_tracks = act_map.shape

        for t_idx in range(n_tracks):
            key = (c_idx, t_idx)
            g_id = final_map.get(key)
            if g_id is None:
                continue
            meta = track_meta.get(key, None)
            best_sim = float(meta["best_sim"]) if meta is not None else 0.0
            dominant_tier = meta["dominant_tier"] if meta is not None else "propagated"
            avg_sim_vector = meta["avg_sim_vector"] if meta is not None else np.zeros((FIXED_N_CLUSTERS,), dtype=np.float32)
            top1_label = meta.get("top1_label") if meta is not None else None
            top2_label = meta.get("top2_label") if meta is not None else None
            top1_score = float(meta.get("top1_score", 0.0)) if meta is not None else 0.0
            top2_score = float(meta.get("top2_score", 0.0)) if meta is not None else 0.0
            top1_minus_top2 = float(meta.get("top1_minus_top2", top1_score - top2_score)) if meta is not None else float(top1_score - top2_score)
            segs = extract_active_segments(act_map[:, t_idx], offset, dt_sec, MIN_DUR_OUT)
            for s_idx, seg in enumerate(segs):
                hyp[Segment(seg["s_sec"], seg["e_sec"]), f"trk_{c_idx}_{t_idx}_{s_idx}"] = f"SPK_{g_id}"
                rows.append({
                    "chunk": c_idx,
                    "track": t_idx,
                    "speaker": f"SPK_{g_id}",
                    "start": float(seg["s_sec"]),
                    "end": float(seg["e_sec"]),
                    "best_sim": best_sim,
                    "tier": dominant_tier,
                    "avg_sim_vector": avg_sim_vector.copy(),
                    "top1_label": top1_label,
                    "top2_label": top2_label,
                    "top1_score": top1_score,
                    "top2_score": top2_score,
                    "top1_minus_top2": top1_minus_top2,
                    "primary_vote_ratio": float(meta.get("primary_vote_ratio", 0.0)) if meta is not None else 0.0,
                    "aux_vote_ratio": float(meta.get("aux_vote_ratio", 0.0)) if meta is not None else 0.0,
                    "fallback_vote_ratio": float(meta.get("fallback_vote_ratio", 1.0 if dominant_tier == "fallback" else 0.0)) if meta is not None else 0.0,
                })
    return hyp, rows

def _collect_boundaries(rows):
    xs = set()
    for r in rows:
        if r["end"] > r["start"]:
            xs.add(round(r["start"], 6))
            xs.add(round(r["end"], 6))
    return sorted(xs)

def _active_rows_at(rows, mid):
    return [r for r in rows if r["start"] <= mid < r["end"]]

def _merge_intervals(intervals):
    if not intervals:
        return []
    is_dict = isinstance(intervals[0], dict)
    intervals = sorted(intervals, key=lambda x: x["start"] if is_dict else x[0])
    first_s = intervals[0]["start"] if is_dict else intervals[0][0]
    first_e = intervals[0]["end"] if is_dict else intervals[0][1]
    merged = [[first_s, first_e]]
    for item in intervals[1:]:
        s = item["start"] if is_dict else item[0]
        e = item["end"] if is_dict else item[1]
        if s <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged]

def _find_exactly_two_speaker_overlap_intervals(rows):
    intervals = []
    bounds = _collect_boundaries(rows)
    if len(bounds) < 2:
        return intervals
    prev = None
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s <= 1e-9:
            continue
        mid = 0.5 * (s + e)
        active = _active_rows_at(rows, mid)
        speakers = tuple(sorted({r["speaker"] for r in active}))
        if len(speakers) != 2:
            prev = None
            continue
        if prev is not None and prev["speakers"] == speakers and abs(prev["end"] - s) <= 1e-9:
            prev["end"] = e
        else:
            prev = {"start": s, "end": e, "speakers": speakers}
            intervals.append(prev)
    return intervals

def _overlap_dur(a_s, a_e, b_s, b_e):
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))

def _speaker_overlap_rows(rows, speaker, s, e):
    out = []
    for r in rows:
        if r["speaker"] != speaker:
            continue
        ov = _overlap_dur(r["start"], r["end"], s, e)
        if ov > 1e-9:
            out.append((r, ov))
    return out

def _speaker_outside_neighbor_support(rows, speaker, s, e, window):
    total = 0.0
    left_s, left_e = max(0.0, s - window), s
    right_s, right_e = e, e + window
    for r in rows:
        if r["speaker"] != speaker:
            continue
        w = _row_tier_weight(r)
        if _row_is_fallback_heavy(r, FALLBACK_NEIGHBOR_BLOCK_RATIO):
            w *= 0.10
        total += w * _overlap_dur(r["start"], r["end"], left_s, left_e)
        total += w * _overlap_dur(r["start"], r["end"], right_s, right_e)
    return total

def _apply_removals_to_rows(rows, removals):
    if not removals:
        return rows
    merged_removals = {spk: _merge_intervals(spans) for spk, spans in removals.items() if spans}
    new_rows = []
    for r in rows:
        speaker = r["speaker"]
        spans = merged_removals.get(speaker, [])
        fragments = [(r["start"], r["end"])]
        for rs, re in spans:
            next_frags = []
            for fs, fe in fragments:
                ov = _overlap_dur(fs, fe, rs, re)
                if ov <= 1e-9:
                    next_frags.append((fs, fe))
                    continue
                if fs < rs:
                    next_frags.append((fs, rs))
                if re < fe:
                    next_frags.append((re, fe))
            fragments = next_frags
            if not fragments:
                break
        for fs, fe in fragments:
            if fe - fs > 1e-9:
                nr = dict(r)
                nr["start"] = float(fs)
                nr["end"] = float(fe)
                # FIX 8: Recompute duration-derived field so downstream scoring
                # reflects the actual fragment duration, not the original segment.
                nr["_fragment_of_original"] = True
                new_rows.append(nr)
    return new_rows

def _speaker_idx_from_label(spk_label: str):
    try:
        return int(str(spk_label).split("_")[1])
    except Exception:
        return None

def _interval_score_vector(rows, s, e, n_spk: int):
    acc = np.zeros((n_spk,), dtype=np.float32)
    total = 0.0
    for r in rows:
        ov = _overlap_dur(r["start"], r["end"], s, e)
        if ov <= 1e-9:
            continue
        vec = np.asarray(r.get("avg_sim_vector", np.zeros((n_spk,), dtype=np.float32)), dtype=np.float32)
        if vec.shape[0] != n_spk:
            padded = np.zeros((n_spk,), dtype=np.float32)
            padded[: min(n_spk, vec.shape[0])] = vec[: min(n_spk, vec.shape[0])]
            vec = padded
        w = ov * _row_tier_weight(r)
        if _row_fallback_ratio(r) >= FALLBACK_SCORE_BLOCK_RATIO:
            w *= 0.05
        acc += w * vec
        total += w
    if total <= 1e-8:
        return acc
    return acc / float(total)

def _dominant_tier_from_rows(rows):
    if not rows:
        return "reseg"
    votes = defaultdict(float)
    for r in rows:
        votes[str(r.get("tier", "reseg"))] += float(max(r.get("end", 0.0) - r.get("start", 0.0), 1e-4))
    return max(votes, key=votes.get) if votes else "reseg"

def _row_tier_weight(row_or_tier):
    tier = row_or_tier if isinstance(row_or_tier, str) else str(row_or_tier.get("tier", "reseg"))
    if tier == "primary":
        return float(PRIMARY_ROW_WEIGHT)
    if tier == "aux":
        return float(AUX_ROW_WEIGHT)
    if tier == "fallback":
        return float(FALLBACK_ROW_WEIGHT)
    return float(RESEG_ROW_WEIGHT)

def _row_fallback_ratio(row):
    return float(row.get("fallback_vote_ratio", 1.0 if str(row.get("tier", "")) == "fallback" else 0.0))

def _row_is_fallback_heavy(row, thr=FALLBACK_OVERRIDE_MAX_RATIO):
    return _row_fallback_ratio(row) >= float(thr)

def _speaker_duration_map(rows):
    out = defaultdict(float)
    for r in rows:
        out[str(r["speaker"])] += max(0.0, float(r["end"] - r["start"]))
    return dict(out)

def _speaker_hub_penalty(rows, speaker):
    if not HUB_PENALTY_ENABLED:
        return 0.0
    speaker = str(speaker)
    if not rows:
        return 0.0
    dur_map = _speaker_duration_map(rows)
    total_dur = sum(dur_map.values())
    if total_dur <= 1e-8:
        return 0.0
    row_count_total = max(len(rows), 1)
    row_count_spk = sum(1 for r in rows if str(r.get("speaker")) == speaker)
    dur_share = float(dur_map.get(speaker, 0.0) / total_dur)
    count_share = float(row_count_spk / row_count_total)
    spk_rows = [r for r in rows if str(r.get("speaker")) == speaker]
    if spk_rows:
        spk_dur = sum(max(0.0, float(r.get("end", 0.0) - r.get("start", 0.0))) for r in spk_rows)
        fallback_dur = sum(max(0.0, float(r.get("end", 0.0) - r.get("start", 0.0))) for r in spk_rows if _row_is_fallback_heavy(r, FALLBACK_DISABLE_STABLE_FOR_SPK_RATIO))
        fallback_share = float(fallback_dur / max(spk_dur, 1e-8))
    else:
        fallback_share = 0.0
    free = float(HUB_SHARE_FREE)
    hard = float(HUB_SHARE_HARD)
    dur_term = max(0.0, (dur_share - free) / max(hard - free, 1e-6))
    cnt_term = max(0.0, (count_share - free) / max(hard - free, 1e-6))
    raw = HUB_DURATION_WEIGHT * dur_term + HUB_COUNT_WEIGHT * cnt_term + HUB_FALLBACK_WEIGHT * fallback_share * HUB_FALLBACK_BOOST
    if raw <= 0.0:
        return 0.0
    return float(raw ** HUB_GAIN_EXP)

def _row_template_from_active_rows(active_rows, speaker, s, e, n_spk: int):
    vec = _interval_score_vector(active_rows, s, e, n_spk)
    spk_idx = _speaker_idx_from_label(speaker)
    if spk_idx is None or spk_idx >= n_spk:
        spk_idx = int(np.argmax(vec)) if vec.size else 0
    order = np.argsort(-vec) if vec.size else np.asarray([spk_idx], dtype=np.int64)
    top1 = int(order[0]) if len(order) >= 1 else spk_idx
    top2 = int(order[1]) if len(order) >= 2 else None
    top1_score = float(vec[top1]) if vec.size else 0.0
    top2_score = float(vec[top2]) if (top2 is not None and vec.size) else 0.0
    best_sim = float(vec[spk_idx]) if vec.size else 0.0
    return {
        "chunk": -1,
        "track": -1,
        "speaker": speaker,
        "start": float(s),
        "end": float(e),
        "best_sim": best_sim,
        "tier": _dominant_tier_from_rows(active_rows),
        "avg_sim_vector": np.asarray(vec, dtype=np.float32),
        "top1_label": top1,
        "top2_label": top2,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "top1_minus_top2": float(top1_score - top2_score),
    }

def _merge_rows_same_speaker(rows, max_gap=0.0):
    if not rows:
        return []
    grouped = defaultdict(list)
    for r in rows:
        grouped[str(r["speaker"])].append(dict(r))

    def _merge_two(a, b):
        da = max(0.0, float(a["end"] - a["start"]))
        db = max(0.0, float(b["end"] - b["start"]))
        tot = max(da + db, 1e-8)
        va = np.asarray(a.get("avg_sim_vector", np.zeros((FIXED_N_CLUSTERS,), dtype=np.float32)), dtype=np.float32)
        vb = np.asarray(b.get("avg_sim_vector", np.zeros((FIXED_N_CLUSTERS,), dtype=np.float32)), dtype=np.float32)
        if va.shape[0] != FIXED_N_CLUSTERS:
            tmp = np.zeros((FIXED_N_CLUSTERS,), dtype=np.float32)
            tmp[: min(FIXED_N_CLUSTERS, va.shape[0])] = va[: min(FIXED_N_CLUSTERS, va.shape[0])]
            va = tmp
        if vb.shape[0] != FIXED_N_CLUSTERS:
            tmp = np.zeros((FIXED_N_CLUSTERS,), dtype=np.float32)
            tmp[: min(FIXED_N_CLUSTERS, vb.shape[0])] = vb[: min(FIXED_N_CLUSTERS, vb.shape[0])]
            vb = tmp
        vec = ((da * va) + (db * vb)) / float(tot)
        order = np.argsort(-vec)
        top1 = int(order[0]) if len(order) >= 1 else None
        top2 = int(order[1]) if len(order) >= 2 else None
        tier = a.get("tier", "reseg") if da >= db else b.get("tier", "reseg")
        return {
            "chunk": -1,
            "track": -1,
            "speaker": a["speaker"],
            "start": float(min(a["start"], b["start"])),
            "end": float(max(a["end"], b["end"])),
            "best_sim": float(vec[_speaker_idx_from_label(a["speaker"])] if _speaker_idx_from_label(a["speaker"]) is not None else np.max(vec)),
            "tier": tier,
            "avg_sim_vector": vec.astype(np.float32),
            "top1_label": top1,
            "top2_label": top2,
            "top1_score": float(vec[top1]) if top1 is not None else 0.0,
            "top2_score": float(vec[top2]) if top2 is not None else 0.0,
            "top1_minus_top2": float((vec[top1] - vec[top2]) if (top1 is not None and top2 is not None) else 0.0),
        }

    out = []
    for speaker, items in grouped.items():
        items = sorted(items, key=lambda x: (x["start"], x["end"]))
        cur = items[0]
        for nxt in items[1:]:
            if float(nxt["start"]) <= float(cur["end"]) + float(max_gap) + 1e-9:
                cur = _merge_two(cur, nxt)
            else:
                out.append(cur)
                cur = nxt
        out.append(cur)
    return sorted(out, key=lambda x: (x["start"], x["end"], x["speaker"]))

def _iter_atomic_intervals(rows):
    bounds = _collect_boundaries(rows)
    if len(bounds) < 2:
        return []
    items = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s <= 1e-9:
            continue
        mid = 0.5 * (s + e)
        active = _active_rows_at(rows, mid)
        if not active:
            continue
        items.append((float(s), float(e), active))
    return items

def find_stable_single_speaker_zones(rows):
    zones = []
    if not STABLE_ZONE_ENABLED:
        return zones
    cur = None
    for s, e, active in _iter_atomic_intervals(rows):
        speakers = sorted({r["speaker"] for r in active})
        if len(speakers) != 1:
            cur = None
            continue
        speaker = speakers[0]
        tpl = _row_template_from_active_rows(active, speaker, s, e, FIXED_N_CLUSTERS)
        spk_idx = _speaker_idx_from_label(speaker)
        top1_minus_cur = float(tpl["top1_score"] - (tpl["avg_sim_vector"][spk_idx] if spk_idx is not None else tpl["top1_score"]))
        fallback_heavy = any(_row_is_fallback_heavy(r, FALLBACK_DISABLE_STABLE_FOR_SPK_RATIO) for r in active) if FALLBACK_EXCLUDE_FROM_STABLE else False
        is_stable = (
            (e - s) > 1e-9 and
            float(tpl["top1_minus_top2"]) >= STABLE_ZONE_MIN_MARGIN and
            top1_minus_cur <= STABLE_ZONE_MAX_TOP1_MINUS_CUR and
            (tpl["top1_label"] == spk_idx if spk_idx is not None else True) and
            (not fallback_heavy)
        )
        if not is_stable:
            cur = None
            continue
        if cur is not None and cur["speaker"] == speaker and abs(cur["end"] - s) <= 1e-9:
            cur["end"] = e
        else:
            cur = {"speaker": speaker, "start": s, "end": e}
            zones.append(cur)
    return [z for z in zones if (z["end"] - z["start"]) >= STABLE_ZONE_MIN_DUR_SEC]

def _stable_zone_coverage(zones, speaker, s, e):
    total = 0.0
    for z in zones:
        if z["speaker"] != speaker:
            continue
        total += _overlap_dur(float(z["start"]), float(z["end"]), s, e)
    return float(total)

def _stable_zone_context_coverage(zones, speaker, s, e, context_sec=STABLE_ZONE_CONTEXT_SEC):
    total = 0.0
    left_s, left_e = float(s - context_sec), float(s)
    right_s, right_e = float(e), float(e + context_sec)
    for z in zones:
        if z["speaker"] != speaker:
            continue
        total += _overlap_dur(float(z["start"]), float(z["end"]), left_s, left_e)
        total += _overlap_dur(float(z["start"]), float(z["end"]), right_s, right_e)
    return float(total)

def _collect_trusted_single_speaker_rows(rows, min_margin=0.06, min_dur=0.25):
    trusted = []
    for s, e, active in _iter_atomic_intervals(rows):
        speakers = sorted({r["speaker"] for r in active})
        if len(speakers) != 1:
            continue
        speaker = speakers[0]
        tpl = _row_template_from_active_rows(active, speaker, s, e, FIXED_N_CLUSTERS)
        spk_idx = _speaker_idx_from_label(speaker)
        if spk_idx is None:
            continue
        cur_score = float(tpl["avg_sim_vector"][spk_idx])
        top1_ok = (tpl["top1_label"] == spk_idx)
        fallback_heavy = any(_row_is_fallback_heavy(r, FALLBACK_DISABLE_STABLE_FOR_SPK_RATIO) for r in active) if FALLBACK_EXCLUDE_FROM_TRUSTED else False
        if top1_ok and float(tpl["top1_minus_top2"]) >= float(min_margin) and (e - s) >= float(min_dur) and (not fallback_heavy):
            trusted.append({
                "speaker": speaker,
                "start": float(s),
                "end": float(e),
                "score": cur_score,
            })
    merged = []
    trusted = sorted(trusted, key=lambda x: (x["speaker"], x["start"], x["end"]))
    for rec in trusted:
        if merged and merged[-1]["speaker"] == rec["speaker"] and rec["start"] <= merged[-1]["end"] + 1e-9:
            merged[-1]["end"] = max(merged[-1]["end"], rec["end"])
            merged[-1]["score"] = max(float(merged[-1].get("score", 0.0)), float(rec.get("score", 0.0)))
        else:
            merged.append(dict(rec))
    return merged

def _speaker_neighbor_context_from_rows(rows, speaker, s, e, context_sec):
    left = _speaker_outside_neighbor_support(rows, speaker, s, s, context_sec)
    right = _speaker_outside_neighbor_support(rows, speaker, e, e, context_sec)
    return float(left + right)

def _normalize_matrix(mat):
    arr = np.asarray(mat, dtype=np.float32)
    if arr.size == 0:
        return arr
    m = float(np.max(arr))
    return arr / m if m > 1e-8 else np.zeros_like(arr)

def _mapping_health_from_debug(debug):
    mapping = debug.get("mapping", {})
    mapped_pairs = [(k, v) for k, v in mapping.items() if v is not None]
    mapped_vbx = len(mapped_pairs)
    unique_local = len({v for _, v in mapped_pairs})
    total_vbx = len(debug.get("vbx_speakers", []))
    original_vbx = len(debug.get("original_vbx_speakers", debug.get("vbx_speakers", [])))
    coverage = float(mapped_vbx / max(total_vbx, 1))
    coverage_original = float(mapped_vbx / max(original_vbx, 1))
    selected_scores = [float(x) for x in debug.get("selected_global_scores", [])]
    avg_score = float(np.mean(selected_scores)) if selected_scores else 0.0
    is_noise_env = (coverage_original < VBX_NOISE_COVERAGE_THR)
    return {
        "mapped_vbx_speakers": mapped_vbx,
        "total_vbx_speakers": total_vbx,
        "original_vbx_speakers": original_vbx,
        "coverage": coverage,
        "coverage_original": coverage_original,
        "is_noise_environment": is_noise_env,
        "unique_local_speakers": unique_local,
        "avg_selected_global_score": avg_score,
        "healthy": bool(
            mapped_vbx >= min(VBX_MAP_MIN_MAPPED_SPEAKERS, max(total_vbx, 1)) and
            unique_local >= min(VBX_MAP_MIN_UNIQUE_LOCAL_SPEAKERS, max(total_vbx, 1)) and
            avg_score >= VBX_MAP_MIN_GLOBAL_SCORE
        ),
    }

def _rows_union_speech_intervals(rows, min_dur=0.0):
    intervals = _merge_intervals([(float(r["start"]), float(r["end"])) for r in rows if float(r["end"]) > float(r["start"])])
    if min_dur <= 0.0:
        return intervals
    return [(s, e) for s, e in intervals if (e - s) >= float(min_dur)]

def _write_vbx_lab(path: str, intervals):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s, e in intervals:
            if e - s > 1e-9:
                f.write(f"{s:.3f} {e:.3f}\n")

def _prepare_vbx_workdir(wav_path: str, rows, work_tag: str):
    workdir = tempfile.mkdtemp(prefix=f"vbx_{work_tag}_")
    wav_dir = os.path.join(workdir, "wav")
    vad_dir = os.path.join(workdir, "vad")
    exp_dir = os.path.join(workdir, "exp")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(vad_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(wav_path))[0]
    wav_copy = os.path.join(wav_dir, f"{stem}.wav")
    shutil.copy2(wav_path, wav_copy)

    list_path = os.path.join(exp_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write(stem + "\n")

    speech_intervals = _rows_union_speech_intervals(rows, min_dur=VBX_MIN_SPEECH_SEG_SEC)
    lab_path = os.path.join(vad_dir, f"{stem}.lab")
    _write_vbx_lab(lab_path, speech_intervals)
    return {
        "workdir": workdir,
        "wav_dir": wav_dir,
        "vad_dir": vad_dir,
        "exp_dir": exp_dir,
        "stem": stem,
        "wav_copy": wav_copy,
        "lab_path": lab_path,
        "list_path": list_path,
        "ark_path": os.path.join(exp_dir, f"{stem}.ark"),
        "seg_path": os.path.join(exp_dir, f"{stem}.seg"),
        "rttm_path": os.path.join(exp_dir, f"{stem}.rttm"),
        "speech_intervals": speech_intervals,
    }

def _validate_vbx_requirements():
    missing = []
    required_paths = [
        VBX_REPO_DIR,
        os.path.join(VBX_REPO_DIR, "VBx", "predict.py"),
        os.path.join(VBX_REPO_DIR, "VBx", "vbhmm.py"),
        VBX_WEIGHTS,
        VBX_TRANSFORM,
        VBX_PLDA,
    ]
    for p in required_paths:
        if not os.path.exists(p):
            missing.append(p)
    return missing

def _run_subprocess_checked(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed\nCMD: {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(" ".join(cmd), proc.stdout[-4000:], proc.stderr[-4000:])
        )
    return {"stdout": proc.stdout, "stderr": proc.stderr}

def _run_real_vbx_on_rows(rows, wav_path: str, work_tag: str):
    missing = _validate_vbx_requirements()
    if missing:
        raise RuntimeError("VBx real backend is enabled but required paths are missing: " + "; ".join(missing))

    prep = _prepare_vbx_workdir(wav_path, rows, work_tag)
    predict_py = os.path.join(VBX_REPO_DIR, "VBx", "predict.py")
    vbhmm_py = os.path.join(VBX_REPO_DIR, "VBx", "vbhmm.py")

    predict_cmd = [
        VBX_PYTHON, predict_py,
        "--in-file-list", prep["list_path"],
        "--in-lab-dir", prep["vad_dir"],
        "--in-wav-dir", prep["wav_dir"],
        "--out-ark-fn", prep["ark_path"],
        "--out-seg-fn", prep["seg_path"],
        "--weights", VBX_WEIGHTS,
        "--backend", VBX_BACKEND,
    ]
    pred_logs = _run_subprocess_checked(predict_cmd, cwd=VBX_REPO_DIR)

    vbhmm_cmd = [
        VBX_PYTHON, vbhmm_py,
        "--init", VBX_INIT,
        "--out-rttm-dir", prep["exp_dir"],
        "--xvec-ark-file", prep["ark_path"],
        "--segments-file", prep["seg_path"],
        "--xvec-transform", VBX_TRANSFORM,
        "--plda-file", VBX_PLDA,
        "--threshold", str(VBX_AHC_THRESHOLD),
        "--lda-dim", str(VBX_LDA_DIM),
        "--Fa", str(VBX_FA),
        "--Fb", str(VBX_FB),
        "--loopP", str(VBX_LOOPP),
    ]
    vb_logs = _run_subprocess_checked(vbhmm_cmd, cwd=VBX_REPO_DIR)

    ann = load_rttm(prep["rttm_path"], prep["stem"])
    vbx_rows = []
    for seg, _, lbl in ann.itertracks(yield_label=True):
        vbx_rows.append({
            "speaker": str(lbl),
            "start": float(seg.start),
            "end": float(seg.end),
        })
    return prep, vbx_rows, {
        "predict_cmd": predict_cmd,
        "vbhmm_cmd": vbhmm_cmd,
        "predict_stdout_tail": pred_logs["stdout"][-2000:],
        "predict_stderr_tail": pred_logs["stderr"][-2000:],
        "vbhmm_stdout_tail": vb_logs["stdout"][-2000:],
        "vbhmm_stderr_tail": vb_logs["stderr"][-2000:],
    }

def _active_speakers_simple(rows, mid):
    return [r["speaker"] for r in rows if float(r["start"]) <= mid < float(r["end"])]

def _premerge_vbx_speakers(vbx_rows, ecapa, wav_tensor, resampler, n_local: int):
    """
    FIX 6: Corrected pre-merge using distance-threshold AHC instead of a fixed-k
    loop with a dominance check that could fall through to k = n_vbx (no merging).

    Algorithm:
    1. Extract representative embedding per VBx speaker
    2. Run AHC with distance_threshold = 1 - VBX_PREMERGE_COSINE_THR
    3. If result > VBX_PREMERGE_MAX_CLUSTERS, re-run with fixed k as a safety cap
    4. Relabel vbx_rows according to merged cluster map
    """
    vbx_spks = sorted({str(r["speaker"]) for r in vbx_rows})
    n_vbx = len(vbx_spks)

    stats = {
        "enabled": True,
        "n_vbx_before": n_vbx,
        "n_vbx_after": n_vbx,
        "merge_map": {},
        "skipped_reason": None,
    }

    if not VBX_PREMERGE_ENABLED:
        stats["enabled"] = False
        stats["skipped_reason"] = "disabled"
        return vbx_rows, {s: s for s in vbx_spks}, stats

    if n_vbx <= n_local:
        stats["skipped_reason"] = "vbx_count_le_local"
        return vbx_rows, {s: s for s in vbx_spks}, stats

    # Extract representative embedding per VBx speaker
    spk_embs = {}
    for vspk in vbx_spks:
        segs = [r for r in vbx_rows if str(r["speaker"]) == vspk]
        sub_embs, sub_durs = [], []
        for seg in segs:
            dur = float(seg["end"]) - float(seg["start"])
            if dur < 0.25:
                continue
            emb, used, _ = get_segment_embedding(
                ecapa, wav_tensor, resampler,
                float(seg["start"]), float(seg["end"])
            )
            if emb is not None:
                sub_embs.append(emb)
                sub_durs.append(used)
        if not sub_embs:
            continue
        mat = np.stack(sub_embs).astype(np.float32)
        weights = np.array(sub_durs, dtype=np.float32)
        merged = np.average(mat, axis=0, weights=weights)
        norm = np.linalg.norm(merged)
        if norm > 1e-8:
            spk_embs[vspk] = merged / norm

    if len(spk_embs) < 2:
        stats["skipped_reason"] = "insufficient_embeddings"
        return vbx_rows, {s: s for s in vbx_spks}, stats

    valid_spks = sorted(spk_embs.keys())
    emb_mat = np.stack([spk_embs[s] for s in valid_spks]).astype(np.float32)
    cos_dist = squareform(pdist(emb_mat, metric="cosine")).astype(np.float32)
    cos_dist = np.clip(cos_dist, 0.0, 2.0)

    # FIX 6: Use distance_threshold AHC so merging is driven by actual similarity,
    # not by a fixed-k dominance heuristic that degenerates to no-merge.
    dist_thr = float(1.0 - VBX_PREMERGE_COSINE_THR)
    try:
        try:
            agg = AgglomerativeClustering(
                n_clusters=None, metric="precomputed", linkage="average",
                distance_threshold=dist_thr
            )
        except TypeError:
            agg = AgglomerativeClustering(
                n_clusters=None, affinity="precomputed", linkage="average",
                distance_threshold=dist_thr
            )
        labels = agg.fit_predict(cos_dist)
        n_merged = int(len(set(labels)))
        # Safety cap: if merge produced more clusters than allowed, re-run with fixed k
        if n_merged > VBX_PREMERGE_MAX_CLUSTERS:
            labels = _make_ahc_fixed_k(VBX_PREMERGE_MAX_CLUSTERS).fit_predict(cos_dist)
    except Exception as ex:
        stats["skipped_reason"] = f"ahc_failed:{ex}"
        return vbx_rows, {s: s for s in vbx_spks}, stats

    # Build merge map: old VBx speaker → merged label string
    merge_map = {}
    for i, vspk in enumerate(valid_spks):
        merge_map[vspk] = f"vbx_merged_{int(labels[i])}"

    # Speakers without embeddings: assign to temporally nearest known speaker's cluster
    for vspk in vbx_spks:
        if vspk not in merge_map:
            earliest_cur = min(
                (float(r["start"]) for r in vbx_rows if str(r["speaker"]) == vspk),
                default=0.0,
            )
            best_label = "vbx_merged_0"
            best_dist = float("inf")
            for vs, ml in merge_map.items():
                earliest_vs = min(
                    (float(r["start"]) for r in vbx_rows if str(r["speaker"]) == vs),
                    default=0.0,
                )
                d = abs(earliest_vs - earliest_cur)
                if d < best_dist:
                    best_dist = d
                    best_label = ml
            merge_map[vspk] = best_label

    n_after = len(set(merge_map.values()))
    stats["n_vbx_after"] = n_after
    stats["merge_map"] = dict(merge_map)

    merged_rows = []
    for r in vbx_rows:
        nr = dict(r)
        nr["speaker"] = merge_map.get(str(r["speaker"]), str(r["speaker"]))
        merged_rows.append(nr)

    return merged_rows, merge_map, stats

def _build_vbx_to_local_map(rows, vbx_rows, ecapa=None, wav_tensor=None, resampler=None, precomputed_stable_zones=None):
    local_spks = sorted({str(r["speaker"]) for r in rows})
    original_vbx_spks = sorted({str(r["speaker"]) for r in vbx_rows})

    premerge_stats = {"enabled": False, "skipped_reason": "no_ecapa"}
    if VBX_PREMERGE_ENABLED and ecapa is not None and wav_tensor is not None:
        vbx_rows, _, premerge_stats = _premerge_vbx_speakers(
            vbx_rows, ecapa, wav_tensor, resampler, len(local_spks)
        )

    vbx_spks = sorted({str(r["speaker"]) for r in vbx_rows})

    stable_anchor_rows = []
    if VBX_MAP_USE_STABLE_ZONES:
        _zones_for_map = precomputed_stable_zones if precomputed_stable_zones is not None else find_stable_single_speaker_zones(rows)
        for z in _zones_for_map:
            stable_anchor_rows.append({
                "speaker": z["speaker"],
                "start": float(z["start"]),
                "end": float(z["end"]),
            })
    trusted_anchor_rows = _collect_trusted_single_speaker_rows(rows)
    anchor_rows = stable_anchor_rows if stable_anchor_rows else trusted_anchor_rows
    if VBX_MAP_FALLBACK_TO_TRUSTED_SINGLE and anchor_rows:
        anchor_cov = sum(float(r["end"] - r["start"]) for r in anchor_rows)
        trusted_cov = sum(float(r["end"] - r["start"]) for r in trusted_anchor_rows)
        if trusted_cov > anchor_cov + 1.0:
            anchor_rows = trusted_anchor_rows
    elif trusted_anchor_rows:
        anchor_rows = trusted_anchor_rows

    overlap = np.zeros((len(vbx_spks), len(local_spks)), dtype=np.float32)
    acoustic = np.zeros_like(overlap)
    neighbor = np.zeros_like(overlap)

    for i, vspk in enumerate(vbx_spks):
        vseg_rows = [vr for vr in vbx_rows if str(vr["speaker"]) == vspk]
        total_vdur = sum(max(0.0, float(vr["end"] - vr["start"])) for vr in vseg_rows)
        for j, lspk in enumerate(local_spks):
            ov = 0.0
            for vr in vseg_rows:
                for lr in anchor_rows:
                    if str(lr["speaker"]) != lspk:
                        continue
                    ov += _overlap_dur(float(vr["start"]), float(vr["end"]), float(lr["start"]), float(lr["end"]))
            overlap[i, j] = ov

            ac_num = 0.0
            nei_num = 0.0
            for vr in vseg_rows:
                s = float(vr["start"])
                e = float(vr["end"])
                dur = max(0.0, e - s)
                if dur <= 1e-9:
                    continue
                vec = _interval_score_vector(rows, s, e, FIXED_N_CLUSTERS)
                idx = _speaker_idx_from_label(lspk)
                if idx is not None and idx < len(vec):
                    ac_num += dur * float(vec[idx])
                nei_num += _speaker_neighbor_context_from_rows(rows, lspk, s, e, VBX_REMAP_NEIGHBOR_SEC)
            acoustic[i, j] = float(ac_num / max(total_vdur, 1e-6))
            neighbor[i, j] = float(nei_num)

    overlap_n = _normalize_matrix(overlap)
    acoustic_n = _normalize_matrix(acoustic)
    neighbor_n = _normalize_matrix(neighbor)
    global_score = (
        VBX_MAP_OVERLAP_WEIGHT * overlap_n +
        VBX_MAP_ACOUSTIC_WEIGHT * acoustic_n +
        VBX_MAP_NEIGHBOR_WEIGHT * neighbor_n
    )

    mapping = {vspk: None for vspk in vbx_spks}
    selected_scores = []
    if global_score.size and np.max(global_score) > 0.0:
        cost = -global_score
        ridx, cidx = linear_sum_assignment(cost)
        for r, c in zip(ridx, cidx):
            raw_overlap = float(overlap[r, c])
            raw_score = float(global_score[r, c])
            if raw_score >= VBX_MAP_MIN_GLOBAL_SCORE and raw_overlap >= VBX_MAP_MIN_OVERLAP_SEC:
                mapping[vbx_spks[r]] = local_spks[c]
                selected_scores.append(raw_score)

    debug = {
        "vbx_speakers": vbx_spks,
        "original_vbx_speakers": original_vbx_spks,
        "local_speakers": local_spks,
        "premerge_stats": premerge_stats,
        "anchor_rows_used": anchor_rows,
        "stable_anchor_rows": stable_anchor_rows,
        "trusted_anchor_rows": trusted_anchor_rows,
        "overlap_matrix": overlap.tolist(),
        "acoustic_matrix": acoustic.tolist(),
        "neighbor_matrix": neighbor.tolist(),
        "global_score_matrix": global_score.tolist(),
        "selected_global_scores": selected_scores,
        "mapping": dict(mapping),
    }
    debug["health"] = _mapping_health_from_debug(debug)
    return mapping, debug

def apply_real_vbx_single_speaker_backbone(rows, uri: str, wav_path: str, work_tag: str,
                                            ecapa=None, wav_tensor=None, resampler=None):
    stats = {
        "enabled": bool(VBX_REAL_ENABLED),
        "used": False,
        "speech_intervals": 0,
        "vbx_rows": 0,
        "single_intervals": 0,
        "relabelled_intervals": 0,
        "relabelled_seconds": 0.0,
        "relabel_candidates": 0,
        "relabel_skipped_unmapped": 0,
        "relabel_skipped_overlap_context": 0,
        "relabel_rejected_low_gain": 0,
        "relabel_rejected_high_conf_current": 0,
        "relabel_rejected_neighbor": 0,
        "mapping": {},
        "mapping_debug": {},
        "mapping_health": {},
        "skipped_due_to_mapping_health": False,
        "workdir": None,
        "predict_cmd": None,
        "vbhmm_cmd": None,
    }
    if not VBX_REAL_ENABLED:
        hyp = Annotation(uri=uri)
        for idx, r in enumerate(rows):
            hyp[Segment(r["start"], r["end"]), f"vbx_{idx}"] = r["speaker"]
        return merge_same_speaker_gaps(hyp, max_gap=MERGE_GAP_SEC), rows, stats

    _vbx_workdir = None

    # FIX 4: Wrap entire function body in try/finally so the VBx temp directory is
    # always cleaned up regardless of which return path is taken (mapping health
    # failure, exception, or successful completion).
    try:
        try:
            prep, vbx_rows, cmd_logs = _run_real_vbx_on_rows(rows, wav_path, work_tag)
            _vbx_workdir = prep.get("workdir")
        except Exception:
            raise

        stats["used"] = True
        stats["speech_intervals"] = len(prep["speech_intervals"])
        stats["vbx_rows"] = len(vbx_rows)
        stats["workdir"] = prep["workdir"]
        stats["predict_cmd"] = cmd_logs["predict_cmd"]
        stats["vbhmm_cmd"] = cmd_logs["vbhmm_cmd"]
        stats["predict_stdout_tail"] = cmd_logs["predict_stdout_tail"]
        stats["predict_stderr_tail"] = cmd_logs["predict_stderr_tail"]
        stats["vbhmm_stdout_tail"] = cmd_logs["vbhmm_stdout_tail"]
        stats["vbhmm_stderr_tail"] = cmd_logs["vbhmm_stderr_tail"]

        # FIX 2: Compute stable_zones BEFORE passing it to _build_vbx_to_local_map.
        # Previously stable_zones was assigned after the map call, causing a NameError
        # (or silently using a stale global from a prior run).
        stable_zones = find_stable_single_speaker_zones(rows)
        candidate_overlap_mask = _merge_intervals(_find_exactly_two_speaker_overlap_intervals(rows))

        mapping, map_debug = _build_vbx_to_local_map(
            rows, vbx_rows,
            ecapa=ecapa, wav_tensor=wav_tensor, resampler=resampler,
            precomputed_stable_zones=stable_zones,
        )
        stats["mapping"] = dict(mapping)
        stats["mapping_debug"] = map_debug
        stats["mapping_health"] = map_debug.get("health", {})
        stats["is_noise_environment"] = map_debug.get("health", {}).get("is_noise_environment", False)

        if VBX_STRICT_REQUIREMENTS and not stats["mapping_health"].get("healthy", False):
            stats["skipped_due_to_mapping_health"] = True
            hyp = Annotation(uri=uri)
            for idx, r in enumerate(rows):
                hyp[Segment(r["start"], r["end"]), f"vbx_{idx}"] = r["speaker"]
            # FIX 4 cont.: return here — finally block will still run cleanup.
            return merge_same_speaker_gaps(hyp, max_gap=MERGE_GAP_SEC), rows, stats

        rebuilt = []
        for s, e, active in _iter_atomic_intervals(rows):
            speakers = sorted({r["speaker"] for r in active})
            if len(speakers) == 1:
                stats["single_intervals"] += 1
                current = speakers[0]
                mid = 0.5 * (s + e)
                vactive = sorted(set(_active_speakers_simple(vbx_rows, mid)))
                chosen = current
                stats["relabel_candidates"] += int(len(vactive) == 1)
                if len(vactive) == 1:
                    mapped = mapping.get(vactive[0])
                    if mapped is None:
                        stats["relabel_skipped_unmapped"] += 1
                    elif mapped != current:
                        tpl_cur = _row_template_from_active_rows(active, current, s, e, FIXED_N_CLUSTERS)
                        score_vec = np.asarray(tpl_cur.get("avg_sim_vector", np.zeros((FIXED_N_CLUSTERS,), dtype=np.float32)), dtype=np.float32)
                        cur_idx = _speaker_idx_from_label(current)
                        tgt_idx = _speaker_idx_from_label(mapped)
                        cur_score = float(score_vec[cur_idx]) if cur_idx is not None and cur_idx < len(score_vec) else 0.0
                        tgt_score = float(score_vec[tgt_idx]) if tgt_idx is not None and tgt_idx < len(score_vec) else 0.0
                        gain = float(tgt_score - cur_score)
                        cur_neighbor = _speaker_neighbor_context_from_rows(rows, current, s, e, VBX_REMAP_NEIGHBOR_SEC)
                        tgt_neighbor = _speaker_neighbor_context_from_rows(rows, mapped, s, e, VBX_REMAP_NEIGHBOR_SEC)
                        neighbor_gain = float(tgt_neighbor - cur_neighbor)
                        near_overlap = any(_overlap_dur(s, e, os, oe) > 1e-9 for os, oe in candidate_overlap_mask)
                        current_stable_ctx = _stable_zone_context_coverage(stable_zones, current, s, e, context_sec=VBX_REMAP_NEIGHBOR_SEC)
                        target_stable_ctx = _stable_zone_context_coverage(stable_zones, mapped, s, e, context_sec=VBX_REMAP_NEIGHBOR_SEC)
                        current_margin = float(tpl_cur.get("top1_minus_top2", 0.0))
                        current_fallback_ratio = float(tpl_cur.get("fallback_vote_ratio", 0.0))
                        target_hub_penalty = _speaker_hub_penalty(rows, mapped)
                        effective_min_gain = VBX_REMAP_MIN_GAIN + HUB_REMAP_EXTRA_GAIN * target_hub_penalty
                        effective_min_neighbor_gain = VBX_REMAP_MIN_NEIGHBOR_GAIN + 0.5 * HUB_GATE_EXTRA_SUPPORT_SEC * target_hub_penalty
                        accept = True
                        if (e - s) < VBX_REMAP_MIN_INTERVAL_SEC:
                            accept = False
                            stats["relabel_rejected_low_gain"] += 1
                        elif near_overlap:
                            accept = False
                            stats["relabel_skipped_overlap_context"] += 1
                        elif current_margin >= VBX_REMAP_MAX_CURRENT_MARGIN and current_stable_ctx >= STABLE_ZONE_STRONG_CONTEXT_MIN_SEC:
                            accept = False
                            stats["relabel_rejected_high_conf_current"] += 1
                        elif current_fallback_ratio >= FALLBACK_VBX_REMAP_BLOCK_RATIO:
                            accept = False
                            stats["relabel_rejected_high_conf_current"] += 1
                        elif gain < effective_min_gain:
                            accept = False
                            stats["relabel_rejected_low_gain"] += 1
                        elif neighbor_gain < effective_min_neighbor_gain and target_stable_ctx < current_stable_ctx:
                            accept = False
                            stats["relabel_rejected_neighbor"] += 1
                        elif VBX_REMAP_REQUIRE_TARGET_TOP1 and tpl_cur.get("top1_label") != tgt_idx:
                            accept = False
                            stats["relabel_rejected_low_gain"] += 1
                        if accept:
                            chosen = mapped
                tpl = _row_template_from_active_rows(active, chosen, s, e, FIXED_N_CLUSTERS)
                rebuilt.append(tpl)
                if chosen != current:
                    stats["relabelled_intervals"] += 1
                    stats["relabelled_seconds"] += float(e - s)
            else:
                for spk in speakers:
                    rebuilt.append(_row_template_from_active_rows(active, spk, s, e, FIXED_N_CLUSTERS))

        rebuilt = _merge_rows_same_speaker(rebuilt, max_gap=MERGE_GAP_SEC)
        hyp = Annotation(uri=uri)
        for idx, r in enumerate(rebuilt):
            hyp[Segment(r["start"], r["end"]), f"vbx_{idx}"] = r["speaker"]
        return merge_same_speaker_gaps(hyp, max_gap=MERGE_GAP_SEC), rebuilt, stats

    finally:
        # FIX 4: Guaranteed cleanup of VBx temp directory on ALL exit paths.
        if _vbx_workdir and os.path.isdir(_vbx_workdir):
            shutil.rmtree(_vbx_workdir, ignore_errors=True)

def apply_temporal_single_speaker_resegmentation(rows, uri: str):
    stats = {
        "enabled": bool(TEMPORAL_RESEG_ENABLED),
        "single_intervals": 0,
        "relabelled_intervals": 0,
        "relabelled_seconds": 0.0,
        "bridge_relabels": 0,
        "score_relabels": 0,
        "skipped_high_confidence": 0,
        "stable_zones": 0,
    }
    if not TEMPORAL_RESEG_ENABLED:
        hyp = Annotation(uri=uri)
        for idx, r in enumerate(rows):
            hyp[Segment(r["start"], r["end"]), f"tmp_{idx}"] = r["speaker"]
        return merge_same_speaker_gaps(hyp, max_gap=MERGE_GAP_SEC), rows, stats

    stable_zones = find_stable_single_speaker_zones(rows)
    stats["stable_zones"] = len(stable_zones)
    atomic = _iter_atomic_intervals(rows)
    single_records = []
    for idx, (s, e, active) in enumerate(atomic):
        speakers = sorted({r["speaker"] for r in active})
        if len(speakers) != 1:
            continue
        speaker = speakers[0]
        tpl = _row_template_from_active_rows(active, speaker, s, e, FIXED_N_CLUSTERS)
        single_records.append({
            "idx": idx,
            "start": s,
            "end": e,
            "duration": float(e - s),
            "speaker": speaker,
            "template": tpl,
            "active": active,
        })
    stats["single_intervals"] = len(single_records)
    record_by_idx = {r["idx"]: r for r in single_records}

    relabels = []
    for pos, rec in enumerate(single_records):
        tpl = rec["template"]
        current = rec["speaker"]
        cur_idx = _speaker_idx_from_label(current)
        vec = np.asarray(tpl["avg_sim_vector"], dtype=np.float32)
        if cur_idx is None or cur_idx >= vec.shape[0]:
            continue
        cur_score = float(vec[cur_idx])
        current_margin = float(tpl["top1_minus_top2"])
        if current_margin >= (TEMPORAL_RESEG_MAX_CUR_MARGIN + 0.06) and tpl["top1_label"] == cur_idx and rec["duration"] >= TEMPORAL_RESEG_MAX_BRIDGE_SEC:
            stats["skipped_high_confidence"] += 1
            continue

        candidate = None
        reason = None

        prev_rec = record_by_idx.get(rec["idx"] - 1)
        next_rec = record_by_idx.get(rec["idx"] + 1)
        if (
            prev_rec is not None and next_rec is not None and
            prev_rec["speaker"] == next_rec["speaker"] != current and
            prev_rec["duration"] >= TEMPORAL_RESEG_MIN_NEIGHBOR_SEC and
            next_rec["duration"] >= TEMPORAL_RESEG_MIN_NEIGHBOR_SEC and
            rec["duration"] <= TEMPORAL_RESEG_MAX_BRIDGE_SEC and
            current_margin <= TEMPORAL_RESEG_MAX_CUR_MARGIN
        ):
            bridge_spk = prev_rec["speaker"]
            bridge_support = _stable_zone_coverage(stable_zones, bridge_spk, rec["start"] - TEMPORAL_RESEG_NEIGHBOR_SEC, rec["end"] + TEMPORAL_RESEG_NEIGHBOR_SEC)
            if bridge_support >= TEMPORAL_RESEG_MIN_SUPPORT_SEC:
                candidate = bridge_spk
                reason = "bridge"

        if candidate is None:
            best_idx = int(np.argmax(vec)) if vec.size else cur_idx
            best_spk = f"SPK_{best_idx}"
            alt_gain = float(vec[best_idx] - cur_score)
            alt_support = _speaker_outside_neighbor_support(rows, best_spk, rec["start"], rec["end"], TEMPORAL_RESEG_NEIGHBOR_SEC)
            if (
                best_spk != current and
                alt_gain >= TEMPORAL_RESEG_MIN_ALT_GAIN and
                current_margin <= TEMPORAL_RESEG_MAX_CUR_MARGIN and
                rec["duration"] <= TEMPORAL_RESEG_SCORE_ONLY_MAX_DUR and
                alt_support >= TEMPORAL_RESEG_MIN_SUPPORT_SEC
            ):
                candidate = best_spk
                reason = "score"

        if candidate is None:
            continue
        new_tpl = dict(_row_template_from_active_rows(rec["active"], candidate, rec["start"], rec["end"], FIXED_N_CLUSTERS))
        relabels.append({
            "old": current,
            "new": candidate,
            "start": rec["start"],
            "end": rec["end"],
            "reason": reason,
            "template": new_tpl,
        })
        stats["relabelled_intervals"] += 1
        stats["relabelled_seconds"] += rec["duration"]
        if reason == "bridge":
            stats["bridge_relabels"] += 1
        else:
            stats["score_relabels"] += 1

    if not relabels:
        hyp = Annotation(uri=uri)
        for idx, r in enumerate(rows):
            hyp[Segment(r["start"], r["end"]), f"tmp_{idx}"] = r["speaker"]
        return merge_same_speaker_gaps(hyp, max_gap=MERGE_GAP_SEC), rows, stats

    removals = defaultdict(list)
    additions = []
    for rel in relabels:
        removals[rel["old"]].append((float(rel["start"]), float(rel["end"])))
        additions.append(dict(rel["template"]))

    new_rows = _apply_removals_to_rows(rows, removals)
    new_rows.extend(additions)
    new_rows = _merge_rows_same_speaker(new_rows, max_gap=MERGE_GAP_SEC)
    hyp = Annotation(uri=uri)
    for idx, r in enumerate(new_rows):
        hyp[Segment(r["start"], r["end"]), f"tmp_{idx}"] = r["speaker"]
    return merge_same_speaker_gaps(hyp, max_gap=MERGE_GAP_SEC), new_rows, stats


def _iter_named_module_candidates(root_obj, max_depth: int = 3):
    import torch.nn as nn

    seen_obj_ids = set()
    seen_mod_ids = set()
    queue = [("model", root_obj, 0)]

    while queue:
        path, obj, depth = queue.pop(0)
        oid = id(obj)
        if oid in seen_obj_ids:
            continue
        seen_obj_ids.add(oid)

        if isinstance(obj, nn.Module):
            mid = id(obj)
            if mid not in seen_mod_ids:
                seen_mod_ids.add(mid)
                yield path, obj

        if depth >= max_depth:
            continue

        names = set()
        try:
            names.update(getattr(obj, "__dict__", {}).keys())
        except Exception:
            pass
        try:
            names.update(dir(obj))
        except Exception:
            pass

        for name in sorted(names):
            if not name or name.startswith("__"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if child is None or callable(child):
                continue
            if isinstance(child, (str, bytes, int, float, bool)):
                continue
            queue.append((f"{path}.{name}", child, depth + 1))


def _state_match_score(module_state_keys, candidate_state_keys):
    module_key_set = set(module_state_keys)
    overlap = len(module_key_set & set(candidate_state_keys))
    coverage = overlap / max(1, len(module_key_set))
    return overlap, coverage


def _extract_logits_tensor(raw_out):
    if isinstance(raw_out, tuple):
        out = raw_out[0]
    elif isinstance(raw_out, dict):
        out = raw_out.get("logits", next(iter(raw_out.values())))
    else:
        out = raw_out

    if isinstance(out, np.ndarray):
        out = torch.from_numpy(out)
    if not torch.is_tensor(out):
        raise TypeError(f"Unsupported segmentation output type: {type(out).__name__}")

    if out.ndim == 3:
        return out[0]
    if out.ndim == 2:
        return out
    if out.ndim == 1:
        return out.unsqueeze(-1)
    raise ValueError(f"Unsupported segmentation output shape: {tuple(out.shape)}")


def _infer_state_num_speakers(state: dict):
    candidate_keys = [
        "classifier.weight",
        "model.classifier.weight",
        "_seg_model.classifier.weight",
        "classifier.bias",
        "model.classifier.bias",
        "_seg_model.classifier.bias",
    ]
    for key in candidate_keys:
        value = state.get(key)
        if torch.is_tensor(value):
            return int(value.shape[0])
    return None


def _extract_checkpoint_state(ckpt):
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format is invalid. Expected a dict-like checkpoint.")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid model_state/state_dict.")
    meta = {
        "epoch": ckpt.get("epoch"),
        "phase": ckpt.get("phase"),
        "val_decode_mode": ckpt.get("val_decode_mode"),
        "max_speakers_per_chunk": ckpt.get("max_speakers_per_chunk"),
        "max_speakers_per_frame": ckpt.get("max_speakers_per_frame"),
    }
    return state, meta


def _infer_powerset_params_from_num_classes(num_classes: int, prefer_chunk: int = 3):
    if not isinstance(num_classes, int) or num_classes <= 0:
        return None, None

    candidates = []
    max_chunk = max(8, int(prefer_chunk) + 3)
    for max_chunk_spk in range(1, max_chunk + 1):
        for max_frame_spk in range(1, min(4, max_chunk_spk) + 1):
            total = 0
            for k in range(0, max_frame_spk + 1):
                total += math.comb(max_chunk_spk, k)
            if total == num_classes:
                candidates.append((max_chunk_spk, max_frame_spk))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (abs(x[0] - prefer_chunk), x[1], x[0]))
    return candidates[0]


def _state_looks_like_pyannote_powerset(state: dict, meta: dict | None = None):
    if not isinstance(state, dict) or not state:
        return False

    meta = meta or {}
    key_set = set(state.keys())

    has_sincnet = any(k.startswith("sincnet.") for k in key_set)
    has_lstm = any(k.startswith("lstm.") for k in key_set)
    has_pyannote_linear_stack = any(k.startswith("linear.") for k in key_set)
    classifier_out = _infer_state_num_speakers(state)

    meta_chunk = meta.get("max_speakers_per_chunk")
    meta_frame = meta.get("max_speakers_per_frame")
    if isinstance(meta_chunk, int) and isinstance(meta_frame, int):
        expected = 0
        for k in range(0, int(meta_frame) + 1):
            expected += math.comb(int(meta_chunk), k)
        if classifier_out == expected:
            return True

    inferred_chunk, inferred_frame = _infer_powerset_params_from_num_classes(
        int(classifier_out) if classifier_out is not None else -1,
        prefer_chunk=FIXED_N_CLUSTERS,
    )
    if inferred_chunk is not None and classifier_out is not None and classifier_out > FIXED_N_CLUSTERS:
        return True

    if (has_sincnet or has_lstm) and has_pyannote_linear_stack and classifier_out is not None:
        return True

    decode_mode = str(meta.get("val_decode_mode") or "").lower()
    if "powerset" in decode_mode:
        return True

    return False


def _patch_hf_hub_auth_compat():
    """Compatibility shim for older pyannote/audio calling newer huggingface_hub.

    Some pyannote/audio versions still pass `use_auth_token=...` into
    `hf_hub_download` / `snapshot_download`, while newer huggingface_hub
    versions only accept `token=...`. Patch both public and internal entry
    points before importing pyannote.audio so legacy calls keep working.
    """
    try:
        import huggingface_hub
    except Exception:
        return

    def _wrap_auth_kw(func):
        if func is None or getattr(func, "_auth_kw_compat_patched", False):
            return func

        def _wrapped(*args, **kwargs):
            if "use_auth_token" in kwargs:
                if "token" not in kwargs:
                    kwargs["token"] = kwargs["use_auth_token"]
                kwargs.pop("use_auth_token", None)
            return func(*args, **kwargs)

        _wrapped._auth_kw_compat_patched = True
        _wrapped.__name__ = getattr(func, "__name__", "wrapped_hf_hub_func")
        _wrapped.__doc__ = getattr(func, "__doc__", None)
        return _wrapped

    try:
        if hasattr(huggingface_hub, "hf_hub_download"):
            huggingface_hub.hf_hub_download = _wrap_auth_kw(huggingface_hub.hf_hub_download)
        if hasattr(huggingface_hub, "snapshot_download"):
            huggingface_hub.snapshot_download = _wrap_auth_kw(huggingface_hub.snapshot_download)
    except Exception:
        pass

    try:
        import huggingface_hub.file_download as _hf_file_download
        if hasattr(_hf_file_download, "hf_hub_download"):
            _hf_file_download.hf_hub_download = _wrap_auth_kw(_hf_file_download.hf_hub_download)
    except Exception:
        pass

    try:
        import huggingface_hub._snapshot_download as _hf_snapshot_download
        if hasattr(_hf_snapshot_download, "snapshot_download"):
            _hf_snapshot_download.snapshot_download = _wrap_auth_kw(_hf_snapshot_download.snapshot_download)
    except Exception:
        pass


def _load_pyannote_pipeline_with_fallback(pretrained_id: str):
    """Load pyannote pipeline with fallback for config placeholder issues."""
    _patch_hf_hub_auth_compat()
    from pyannote.audio.pipelines import SpeakerDiarization
    
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_HUB_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass
    
    # Try standard loading first
    try:
        from pyannote.audio import Pipeline
        for kwargs in (
            {"token": hf_token} if hf_token else {},
            {"use_auth_token": hf_token} if hf_token else {},
            {},
        ):
            try:
                model = Pipeline.from_pretrained(pretrained_id, **kwargs)
                if model is not None:
                    return model
            except Exception:
                continue
    except Exception:
        pass
    
    # Fallback: create default SpeakerDiarization pipeline
    print(f"  [DEBUG] Standard loading failed, creating default SpeakerDiarization pipeline")
    try:
        return SpeakerDiarization(clustering="AgglomerativeClustering")
    except Exception as e:
        raise RuntimeError(f"Failed to load/create pyannote pipeline: {e}")

def _load_pyannote_pipeline(pretrained_id: str):
    return _load_pyannote_pipeline_with_fallback(pretrained_id)


def _build_base_model(ckpt_path: str | None, hf_model_id: str):
    ckpt = None
    state = None
    meta = {}
    ckpt_family = "none"

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state, meta = _extract_checkpoint_state(ckpt)
        ckpt_family = "pyannote_powerset" if _state_looks_like_pyannote_powerset(state, meta) else "diarizen_or_multilabel"

    if ckpt_family == "pyannote_powerset":
        chosen_hf = hf_model_id
        if not chosen_hf or chosen_hf == "BUT-FIT/diarizen-wavlm-large-s80-md-v2":
            chosen_hf = PYANNOTE_HF_MODEL_ID
        print("  [DEBUG] Detected checkpoint family: pyannote_powerset")
        print(f"  [DEBUG] Loading pyannote base pipeline: {chosen_hf}")
        model = _load_pyannote_pipeline(chosen_hf)
    else:
        print("  [DEBUG] Detected checkpoint family: diarizen_or_multilabel")
        print("  [DEBUG] Importing DiariZenPipeline...")
        from diarizen.pipelines.inference import DiariZenPipeline
        print(f"  [DEBUG] Loading pre-trained DiariZen model: {hf_model_id}")
        model = DiariZenPipeline.from_pretrained(hf_model_id)
        chosen_hf = hf_model_id

    if model is None:
        raise RuntimeError(
            f"Base model construction failed for checkpoint_family={ckpt_family} and hf_model={chosen_hf}."
        )

    model._checkpoint_family = ckpt_family
    model._checkpoint_meta = meta
    model._checkpoint_hf_model = chosen_hf
    return model, ckpt, state, meta


def _find_classifier_owner(module):
    import torch.nn as nn
    if hasattr(module, "classifier") and isinstance(getattr(module, "classifier"), nn.Linear):
        return module, "classifier", getattr(module, "classifier")
    for path, submodule in module.named_modules():
        if hasattr(submodule, "classifier") and isinstance(getattr(submodule, "classifier"), nn.Linear):
            return submodule, "classifier", getattr(submodule, "classifier")
    return None, None, None


def _maybe_adapt_module_for_checkpoint(module, load_state: dict):
    import torch.nn as nn

    target_out = _infer_state_num_speakers(load_state)
    owner, attr, classifier = _find_classifier_owner(module)

    if target_out is not None and classifier is not None and classifier.out_features != target_out:
        new_classifier = nn.Linear(
            classifier.in_features,
            target_out,
            bias=(classifier.bias is not None),
        )
        new_classifier = new_classifier.to(device=classifier.weight.device, dtype=classifier.weight.dtype)
        nn.init.xavier_uniform_(new_classifier.weight)
        if new_classifier.bias is not None:
            nn.init.zeros_(new_classifier.bias)
        setattr(owner, attr, new_classifier)
        print(
            f"  [DEBUG] Replaced classifier at {type(owner).__name__}.{attr}: "
            f"{classifier.out_features} -> {target_out} outputs"
        )

    if hasattr(module, "activation"):
        module.activation = nn.Identity()
        print("  [DEBUG] Forced segmentation activation = Identity for raw-logit inference.")

    return int(target_out) if target_out is not None else None


def _choose_state_variant_for_module(module, state: dict):
    raw_keys = list(state.keys())
    stripped_seg_state = {
        k[len("_seg_model."):]: v for k, v in state.items() if k.startswith("_seg_model.")
    }
    stripped_model_state = {
        k[len("model."):]: v for k, v in state.items() if k.startswith("model.")
    }

    variants = [
        ("raw", state),
        ("_seg_model", stripped_seg_state),
        ("model", stripped_model_state),
    ]

    module_keys = list(module.state_dict().keys())
    scored = []
    for variant_name, variant_state in variants:
        if not variant_state:
            continue
        overlap, coverage = _state_match_score(module_keys, list(variant_state.keys()))
        scored.append((variant_name, overlap, coverage, variant_state))

    if not scored:
        return "raw", state, 0, 0.0

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored[0]


def _load_state_into_best_submodule(model, state):
    """Load checkpoint into the actual DiariZen segmentation nn.Module.

    For md-v2 fine-tuned checkpoints, the classifier was replaced during training
    from the original powerset head to a fixed multi-label speaker head. This
    loader first finds the best internal segmentation module, adapts the
    classifier shape to the checkpoint, forces raw-logit inference, and only
    then loads the state dict.
    """

    candidate_modules = []

    seg_wrapper = getattr(model, "_segmentation", None)
    if seg_wrapper is not None:
        inner_model = getattr(seg_wrapper, "model", None)
        if inner_model is not None:
            candidate_modules.append(("_segmentation.model", inner_model))
        if hasattr(seg_wrapper, "load_state_dict"):
            candidate_modules.append(("_segmentation", seg_wrapper))

    for attr in ("model", "segmentation", "_model"):
        obj = getattr(model, attr, None)
        if obj is not None and hasattr(obj, "load_state_dict"):
            candidate_modules.append((attr, obj))

    for path, module in _iter_named_module_candidates(model, max_depth=8):
        candidate_modules.append((path, module))

    # deduplicate while preserving order
    dedup = []
    seen = set()
    for path, module in candidate_modules:
        key = id(module)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((path, module))
    candidate_modules = dedup

    if not candidate_modules:
        raise RuntimeError(
            "Could not find any internal torch.nn.Module to receive the checkpoint. "
            "Your diarizen/pyannote install is likely incompatible with this script."
        )

    ranked = []
    for path, module in candidate_modules:
        try:
            variant_name, overlap, coverage, variant_state = _choose_state_variant_for_module(module, state)
            if overlap <= 0:
                continue
            ranked.append((path, module, variant_name, overlap, coverage, variant_state))
        except Exception:
            continue

    if not ranked:
        raise RuntimeError(
            "Could not match checkpoint keys against any internal submodule. "
            "Most likely causes: wrong base model, wrong diarizen version, or malformed checkpoint."
        )

    ranked.sort(key=lambda x: (x[3], x[4]), reverse=True)
    best_path, best_module, best_variant, best_overlap, best_cov, load_state = ranked[0]

    print(f"  [DEBUG] Best internal module candidate: {best_path}")
    print(f"  [DEBUG] Match mode={best_variant} | overlap={best_overlap} | coverage={best_cov:.4f}")

    inferred_num_speakers = _maybe_adapt_module_for_checkpoint(best_module, load_state)

    incompatible = best_module.load_state_dict(load_state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(f"  [DEBUG] Checkpoint loaded into {best_path}.")
    print(f"  [DEBUG] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    model._direct_segmentation_module = best_module
    model._direct_use_raw_forward = True
    model._direct_num_speakers = int(inferred_num_speakers) if inferred_num_speakers is not None else FIXED_N_CLUSTERS
    return model



def _load_finetuned_checkpoint_into_model(model, ckpt_path: str):
    if not ckpt_path:
        model._direct_use_raw_forward = False
        return model

    print(f"  [DEBUG] Loading finetuned checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state, meta = _extract_checkpoint_state(ckpt)

    print(f"  [DEBUG] Checkpoint keys: {len(state)} parameter tensors")
    is_pyannote = _state_looks_like_pyannote_powerset(state, meta)
    model._checkpoint_family = "pyannote_powerset" if is_pyannote else "diarizen_or_multilabel"
    model._checkpoint_meta = meta

    loaded_model = _load_state_into_best_submodule(model, state)

    if is_pyannote:
        from pyannote.audio.utils.powerset import Powerset

        n_classes = _infer_state_num_speakers(state)
        max_chunk = meta.get("max_speakers_per_chunk")
        max_frame = meta.get("max_speakers_per_frame")

        if not isinstance(max_chunk, int) or not isinstance(max_frame, int):
            inferred_chunk, inferred_frame = _infer_powerset_params_from_num_classes(
                int(n_classes) if n_classes is not None else -1,
                prefer_chunk=FIXED_N_CLUSTERS,
            )
            max_chunk = inferred_chunk if inferred_chunk is not None else FIXED_N_CLUSTERS
            max_frame = inferred_frame if inferred_frame is not None else min(2, FIXED_N_CLUSTERS)

        loaded_model._direct_decode_mode = "pyannote_powerset_soft"
        loaded_model._direct_powerset = Powerset(int(max_chunk), int(max_frame))
        loaded_model._direct_num_speakers = int(max_chunk)

        print(
            f"  [DEBUG] Using pyannote powerset decode | "
            f"max_speakers_per_chunk={int(max_chunk)} | max_speakers_per_frame={int(max_frame)}"
        )
    else:
        loaded_model._direct_decode_mode = "sigmoid_multilabel"

    return loaded_model


def _get_direct_chunk_probs(model, padded_chunk: torch.Tensor):
    module = getattr(model, "_direct_segmentation_module", None)
    if module is None:
        raise RuntimeError("Direct segmentation module was not attached to the pipeline.")

    raw_out = module(padded_chunk.view(1, 1, -1))
    logits = _extract_logits_tensor(raw_out)

    decode_mode = getattr(model, "_direct_decode_mode", "sigmoid_multilabel")
    if decode_mode == "pyannote_powerset_soft":
        powerset = getattr(model, "_direct_powerset", None)
        if powerset is None:
            raise RuntimeError("Pyannote powerset decoder was not attached to the model.")
        probs = torch.softmax(logits, dim=-1)
        if hasattr(powerset, "mapping") and probs.device != powerset.mapping.device:
            probs = probs.to(powerset.mapping.device)
        probs = powerset.to_multilabel(probs)
        if hasattr(powerset, "mapping") and probs.device != logits.device:
            probs = probs.to(logits.device)
    else:
        probs = torch.sigmoid(logits)

    return probs.detach().cpu().numpy().astype(np.float32)

def main():
    print(f"DEVICE: {DEVICE}")

    model, _, _, _ = _build_base_model(CKPT_PATH, HF_MODEL_ID)
    print(f"  [DEBUG] Base model loaded: {getattr(model, '_checkpoint_hf_model', HF_MODEL_ID)}")
    model = _load_finetuned_checkpoint_into_model(model, CKPT_PATH)
    # ── GPU: move model to DEVICE and switch to inference mode ──────────────
    if hasattr(model, "to"):
        model = model.to(DEVICE)
    if hasattr(model, "eval"):
        model.eval()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        print(f"  [DEBUG] VRAM after model load: "
              f"{torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated / "
              f"{torch.cuda.memory_reserved()  / 1024**2:.0f} MB reserved")
    print("  [DEBUG] Model loaded successfully.")

    ecapa = model

    ref = load_rttm(CLEAN_RTTM_PATH, UTT_ID)

    wav, sr = torchaudio.load(WAV_PATH)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    # ── GPU: keep wav on GPU so every chunk slice is already on DEVICE ───────
    wav = wav.to(DEVICE)

    resampler = get_resampler()

    total_dur = wav.shape[1] / TARGET_SR
    print(f"File Duration: {total_dur:.2f}s")

    print(f"\nRunning EEND-EDA chunk inference... (chunk={CHUNK_SEC}s)")
    chunk_outputs = infer_all_chunk_probs(model, wav, chunk_sec=CHUNK_SEC)
    if not chunk_outputs:
        print("No chunk outputs. Abort.")
        return
    print(f"Chunks inferred: {len(chunk_outputs)}")
    # Free activation memory before the embedding/clustering phase
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[Phase 1] Initializing speaker centroids...")
    centroids, high_thr_used, n_anchors, p1_stats, init_meta = initialize_phase1_centroids(
        chunk_outputs, wav, ecapa, resampler, FIXED_N_CLUSTERS
    )
    if centroids is None:
        print("Phase 1 failed: not enough valid centroids to initialize. Abort.")
        return

    p1_primary = int((p1_stats or {}).get("primary_segments", 0)) if isinstance(p1_stats, dict) else 0
    p1_subseg = int((p1_stats or {}).get("subseg_used_segments", 0)) if isinstance(p1_stats, dict) else 0
    high_thr_display = f"{float(high_thr_used):.2f}" if high_thr_used is not None else "N/A"
    print(
        f"  Init mode={init_meta.get('mode')} | shape={centroids.shape} | high_thr={high_thr_display} | "
        f"primary={p1_primary} | subseg_used={p1_subseg}"
    )
    if init_meta.get("speaker_seed_info"):
        for item in init_meta["speaker_seed_info"]:
            print(
                f"    - idx={item['index']} | seed={item['seed_type']} | name={item['seed_name']} | "
                f"phase1_match={item.get('matched_phase1_cluster')}"
            )

    # Free GPU memory after Phase 1 embedding extraction before Phase 2
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    low_thr = float(LOW_THRESHOLD)
    print(f"\n[Phase 2] Running main flow with fixed low threshold...")
    print(f"  n_clusters={FIXED_N_CLUSTERS} | high_thr={high_thr_display} | low_threshold={low_thr:.2f}")
    print("-" * 170)
    print(
        f"{'LOW_THR':<8} | {'Emb':<5} | {'P/A/F':<12} | {'Assigned':<8} | "
        f"{'Prop':<5} | {'GlobProp':<9} | {'Unmap':<5} | {'Purify':<6} | {'VBxRel':<6} | "
        f"{'DER%':<8} | {'Miss%':<8} | {'FA%':<8} | {'Conf%':<8} | {'OvF1':<8}"
    )
    print("-" * 170)

    result = None
    os.makedirs(FORENSIC_DIR, exist_ok=True)

    embs, sinfo, collect_stats = collect_assign_embeddings(
        chunk_outputs, wav, ecapa, resampler, threshold=low_thr
    )

    n_emb = len(embs)
    tiers_str = f"{collect_stats['primary']}/{collect_stats['aux']}/{collect_stats['fallback']}"
    if n_emb == 0:
        print(
            f"{low_thr:<8.2f} | {0:<5d} | {tiers_str:<12} | {'SKIP':<8} | "
            f"{'-':<5} | {'-':<5} | {'-':<6} | {'-':<6} | {'NA':<7} | {'NA':<8} | {'NA':<8} | {'NA':<8} | {'NA':<8} | {'NA':<8}"
        )
        # Prepare result dictionary. DER and related metrics are only inserted
        # when SKIP_DER is False. Otherwise they are set to None to indicate
        # the computation was skipped. Keeping the keys present avoids front‑end
        # code breaking on missing keys.
        result = {
            "low_threshold": float(low_thr),
            "high_threshold_used": (float(high_thr_used) if high_thr_used is not None else None),
            "num_assign_embeddings": 0,
            "skipped": True,
            "skip_reason": "no_embeddings",
            "collect_stats": collect_stats,
            "centroid_init": init_meta,
        }
    else:
        _, _, _, embed_details0 = assign_to_centroids(embs, sinfo, centroids)

        purified_centroids, purify_stats = purify_centroids(embs, sinfo, embed_details0, centroids, FIXED_N_CLUSTERS)
        base_mapping, track_meta, assign_stats, embed_details = assign_to_centroids(embs, sinfo, purified_centroids)
        force_assign_track_meta = build_force_assign_track_meta(
            embs, sinfo, purified_centroids,
            chunk_outputs=chunk_outputs,
            wav_tensor=wav,
            ecapa=ecapa,
            resampler=resampler,
            threshold=low_thr,
            mapped_keys=set(base_mapping.keys()),
        )
        final_mapping, final_track_meta, prop_stats = propagate_missing_tracks(
            chunk_outputs, base_mapping, track_meta, FIXED_N_CLUSTERS,
            force_assign_track_meta=force_assign_track_meta,
            threshold=low_thr,
        )

        raw_hyp, raw_rows = build_raw_hypothesis(chunk_outputs, final_mapping, final_track_meta, low_thr, UTT_ID)

        sf_hyp, sf_rows, sf_stats = apply_silence_filter(raw_rows, wav, UTT_ID)
        raw_hyp, raw_rows = sf_hyp, sf_rows

        low_tag = f"{int(round(low_thr * 100)):03d}"

        vbx_hyp, vbx_rows, vbx_stats = apply_real_vbx_single_speaker_backbone(
            raw_rows, UTT_ID, WAV_PATH, f"low{low_tag}",
            ecapa=ecapa, wav_tensor=wav, resampler=resampler,
        )
        final_stage = build_stage_report("post_real_vbx", ref, vbx_hyp, total_dur)
        hyp, final_rows = vbx_hyp, vbx_rows

        der, miss, fa, conf = final_stage["der"], final_stage["miss"], final_stage["fa"], final_stage["conf"]
        ov_f1 = final_stage["overlap_metrics"]["overlap"]["f1"]

        hyp_rttm_path = os.path.join(FORENSIC_DIR, f"hyp_low{low_tag}.rttm")
        raw_rttm_path = os.path.join(FORENSIC_DIR, f"raw_low{low_tag}.rttm")
        # ── Remap SPK_{i} → enrollment speaker name ──────────────────────────
        # build_cluster_name_map trích {cluster_idx: seed_name} từ
        # speaker_seed_info do initialize_phase1_centroids điền vào.
        # Khi không dùng enrollment, map rỗng và nhãn giữ nguyên SPK_{i}.
        cluster_name_map = build_cluster_name_map(init_meta)
        if cluster_name_map:
            print(f"  [INFO] Remapping RTTM labels: {cluster_name_map}")
        hyp     = remap_rttm_labels(hyp,     cluster_name_map)
        raw_hyp = remap_rttm_labels(raw_hyp, cluster_name_map)
        # ─────────────────────────────────────────────────────────────────────
        save_rttm(hyp, hyp_rttm_path)
        save_rttm(raw_hyp, raw_rttm_path)

        # Format DER related metrics depending on SKIP_DER flag. When DER is skipped
        # the values are displayed as "N/A" to indicate they were not computed.
        if SKIP_DER:
            der_str = miss_str = fa_str = conf_str = "N/A"
        else:
            der_str = f"{der:<8.2f}"
            miss_str = f"{miss:<8.2f}"
            fa_str = f"{fa:<8.2f}"
            conf_str = f"{conf:<8.2f}"
        print(
            f"{low_thr:<8.2f} | {n_emb:<5d} | {tiers_str:<12} | "
            f"{len(base_mapping):<8d} | {prop_stats['propagated_tracks']:<5d} | "
            f"{prop_stats.get('global_propagated_tracks', 0):<9d} | "
            f"{prop_stats['unmapped_tracks']:<5d} | {purify_stats['selected_total']:<6d} | {vbx_stats['relabelled_intervals']:<6d} | "
            f"{der_str} | {miss_str} | {fa_str} | {conf_str} | {ov_f1:<8.3f}"
        )

        result = {
            "low_threshold": float(low_thr),
            "high_threshold_used": (float(high_thr_used) if high_thr_used is not None else None),
            "num_assign_embeddings": int(n_emb),
            "collect_stats": collect_stats,
            "assign_stats_final": assign_stats,
            "centroid_purify_stats": purify_stats,
            "force_assign_meta_debug": LAST_FORCE_ASSIGN_META_DEBUG,
            "silence_filter_stats": sf_stats,
            "propagate_stats": prop_stats,
            "real_vbx_stats": vbx_stats,
            "mapped_tracks_before_propagate": int(len(base_mapping)),
            "mapped_tracks_after_propagate": int(len(final_mapping)),
            "raw_segment_rows": int(len(raw_rows)),
            "vbx_segment_rows": int(len(vbx_rows)),
            "final_segment_rows": int(len(final_rows)),
            "artifacts": {
                "raw_rttm": raw_rttm_path,
                "hyp_rttm": hyp_rttm_path,
            },
            "final_overlap_metrics": final_stage["overlap_metrics"],
            "skipped": False,
            "der": (float(der) if not SKIP_DER else None),
            "miss": (float(miss) if not SKIP_DER else None),
            "fa": (float(fa) if not SKIP_DER else None),
            "conf": (float(conf) if not SKIP_DER else None),
        }

    print("-" * 160)
    if result and not result.get("skipped", False):
        # Print final DER statistics only when SKIP_DER is disabled. Otherwise, display
        # an informative message that DER metrics were omitted.
        if not SKIP_DER:
            print(
                f"\nFinal DER: {result['der']:.2f}% @ low_thr={result['low_threshold']:.2f} "
                f"(Miss={result['miss']:.2f}%, FA={result['fa']:.2f}%, Conf={result['conf']:.2f}%)"
            )
        else:
            print(
                f"\nFinal DER: N/A (DER computation skipped) @ low_thr={result['low_threshold']:.2f}"
            )
        ov = result["final_overlap_metrics"]["overlap"]
        print(
            f"Final overlap: Precision={ov['precision']:.3f} Recall={ov['recall']:.3f} F1={ov['f1']:.3f}"
        )

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUT_JSON}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()