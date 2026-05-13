# ViMeet — Vietnamese Multi-Speaker Transcription System

> **Deep Learning Framework for Automatic Meeting Transcription in Noisy Environments** 
> End-to-end pipeline: Speaker Diarization → ASR → Structured Transcript

---

## Overview

**ViMeet** là hệ thống ghi biên bản cuộc họp tự động tiếng Việt, được xây dựng như đồ án cuối kỳ môn AI/Deep Learning. Hệ thống giải quyết bài toán **multi-speaker transcription** trong điều kiện thực tế (nhiễu nền, overlap giọng nói, nhiều người nói đồng thời).

### Bài toán giải quyết
- **Input**: File audio cuộc họp (`.wav`) với nhiều người nói, nhiễu nền
- **Output**: Transcript có cấu trúc — *ai nói gì, lúc nào*

### Pipeline chính

```
Audio Input
 │
 ▼
[Preprocessing] ── Normalize to mono 16kHz
 │
 ▼
[Speaker Diarization] ── DiariZen (WavLM-Large) + Enrollment Embeddings
 │
 ▼ RTTM file (who spoke when)
 │
 ▼
[ASR per Segment] ── PhoWhisper-Large (Vietnamese)
 │
 ▼
[Post-processing] ── Merge, align, format
 │
 ▼
Structured Transcript (JSON / TXT / CSV)
```

---

## ️ Repository Structure

```
vimeet/
├── app_streamlit.py # Main Streamlit UI — entry point
├── pipeline_config.py # Global config: paths, model defaults
│
├── core/ # Core ML inference engines
│ ├── asr/
│ │ ├── engine.py # PhoWhisper ASR inference
│ │ ├── config.py # ASR hyperparameters & constants
│ │ ├── manifest_builder.py # RTTM → JSONL manifest conversion
│ │ └── checkpoints/ # Fine-tuned ASR model weights (not tracked)
│ │ └── best_adapter/
│ ├── der/
│ │ ├── engine.py # DiariZen diarization inference
│ │ ├── pyannote_engine.py # PyAnnote baseline engine
│ │ └── checkpoints/ # Fine-tuned DER model weights (not tracked)
│ └── qwen/
│ └── engine.py # Qwen 2.5 transcript normalization & summary
│
├── ui/ # Streamlit UI components
│ ├── theme.py # CSS theme & i18n translator
│ ├── components.py # Reusable UI widgets
│ ├── transcript_view.py # Speaker-tagged transcript display
│ ├── summary_view.py # ASR workspace & result viewer
│ └── run_history.py # Run history management
│
├── audio_preprocess_input.py # Audio normalization utilities
├── asr_infer_bridge.py # Bridge: UI → ASR subprocess
├── der_infer_bridge.py # Bridge: UI → DER subprocess
├── qwen_infer_bridge.py # Bridge: UI → Qwen subprocess
├── asr_runner.py # ASR orchestration runner
│
├── output/ # Generated results (gitignored)
│ ├── der/ # RTTM diarization outputs
│ ├── asr/ # Transcript CSV/text outputs
│ └── qwen/ # Qwen summary outputs
│
├── debug/ # Dev/evaluation scripts
│ ├── asr_test/ # ASR test manifests
│ └── scratch/ # Experimental evaluation scripts
│
├── _ARCHIVES/ # Legacy scripts (version history)
│ └── qwen_scripts_legacy/ # Qwen pipeline iteration history (v2–v8)
│
├── requirements_extra_web.txt # Additional UI dependencies
├── run_streamlit.ps1 # Windows launch script
└── install_extra_web.ps1 # Windows installation script
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Speaker Diarization** | DiariZen (WavLM-Large), fine-tuned + enrollment-based speaker ID |
| ️ **Vietnamese ASR** | PhoWhisper-Large, segment-level inference from RTTM |
| **Speaker Attribution** | Enrollment directory → identity-aware diarization |
| **Structured Output** | Timeline JSON, speaker-tagged transcript, RTTM artifacts |
| **Web Interface** | Streamlit UI with demo & debug mode |
| **Low-VRAM Support** | Qwen 4-bit quantization for RTX 4060 (8GB VRAM) |
| **Run History** | Lưu và xem lại các lần xử lý trước |

---

## Quick Start

### Requirements
- Python 3.10
- CUDA GPU (tested: RTX 3090 Ti for training, RTX 4060 8GB for inference)
- Windows 11 (PowerShell scripts included)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/vimeet.git
cd vimeet

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate # Windows
# source .venv/bin/activate # Linux/macOS

# 3. Install PyTorch (match your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements_extra_web.txt
```

### Download Model Checkpoints

Checkpoints không được lưu trong repo. Đặt vào đúng thư mục sau khi tải:

```
core/
├── der/checkpoints/best_model.pth ← DiariZen fine-tuned checkpoint
├── asr/checkpoints/best_adapter/ ← PhoWhisper adapter weights
└── qwen/ ← (optional) Qwen local weights
```

> ️ Xem phần [Model Checkpoints](#-model-checkpoints) để biết link tải.

### Run

```bash
streamlit run app_streamlit.py
```

Hoặc dùng script Windows:

```powershell
.\run_streamlit.ps1
```

---

## Models Used

### Speaker Diarization
| Model | Role |
|---|---|
| `BUT-FIT/diarizen-wavlm-large-s80-md` | Backbone diarization — WavLM-Large based |
| Fine-tuned checkpoint (`.pth`) | Custom-trained on Vietnamese meeting data |
| Enrollment embeddings | Speaker identity matching từ mẫu giọng |

### Automatic Speech Recognition
| Model | Role |
|---|---|
| `vinai/PhoWhisper-large` | Vietnamese ASR — primary model |
| DiCoW | Alternative ASR backbone (benchmark) |

### Post-processing / Summary
| Model | Role |
|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (4-bit) | Transcript normalization, fast summary |
| `Qwen/Qwen2.5-7B-Instruct` (4-bit) | Deep meeting summary |

---

## ️ Usage Guide

### 1. Upload Audio
Chọn file `.wav` (mono hoặc stereo, bất kỳ sample rate).

### 2. Configure Pipeline

**Diarization settings:**
- `DER Checkpoint` — đường dẫn đến `.pth` fine-tuned (hoặc dùng model gốc)
- `Enrollment directory` — thư mục chứa mẫu giọng của từng speaker đã biết
- `Number of speakers` — số lượng người nói ước tính
- `Low threshold` — ngưỡng sensitivity (default: `0.25`)

**ASR settings:**
- Mode: `whisper_only` / `wer_v2` / `dicow_only`
- `ASR Checkpoint` — đường dẫn đến adapter weights

### 3. Run & Monitor
Pipeline chạy theo thứ tự: Preprocess → DER → RTTM → ASR → Transcript

### 4. View Results
```
[00:00 - 00:05] Speaker_A: Chào mọi người, hôm nay chúng ta bàn về...
[00:06 - 00:12] Speaker_B: Đồng ý, trước tiên cần review kết quả...
```

---

## Output Files

Mỗi lần chạy tạo ra các file trong `output/`:

| File | Description |
|---|---|
| `hyp_low025.rttm` | Final diarization output (RTTM format) |
| `raw_low025.rttm` | Raw diarization (before post-processing) |
| `transcript.csv` | ASR output với timestamps + speaker labels |
| `transcript.json` | Structured transcript (JSON) |
| `summary.md` | Meeting summary (if Qwen enabled) |

---

## Configuration

Tùy chỉnh default paths trong `pipeline_config.py`:

```python
DER_SCRIPT_PATH = "core/der/engine.py"
ASR_SCRIPT_PATH = "asr_runner.py"
DER_CHECKPOINT_DEFAULT = "core/der/checkpoints/best_model.pth"
QWEN_NORMALIZE_MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
```

Hoặc dùng environment variables:
```bash
export DER_SCRIPT_PATH=/custom/path/der_engine.py
export DER_CHECKPOINT=/path/to/checkpoint.pth
```

---

## Experiments & Results

*(Điền kết quả thực nghiệm của bạn ở đây)*

| Metric | Baseline | Fine-tuned |
|---|---|---|
| DER (%) | — | — |
| WER (%) | — | — |
| cpWER (%) | — | — |

**Test set**: Vietnamese meeting data (noisy, multi-speaker) 
**Hardware**: NVIDIA RTX 3090 Ti (training), RTX 4060 8GB (inference)

---

## Model Checkpoints

> Checkpoints không được lưu trong repo do kích thước lớn.

*(Thêm link Google Drive / HuggingFace Hub của bạn ở đây)*

```
Checkpoints
├── DER best checkpoint → [link]
├── ASR adapter weights → [link]
└── Qwen fine-tuned (opt.) → [link]
```

---

## ️ .gitignore Recommended

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
*.egg-info/

# Model weights & large files
*.pth
*.bin
*.safetensors
core/*/checkpoints/

# Outputs
output/
*.rttm
*.jsonl

# Logs & debug
debug/scratch/*.log
debug/scratch/*.txt

# OS
.DS_Store
Thumbs.db
```

---

## ‍ Development Notes

- **Single venv**: Tất cả components (DER, ASR, Qwen, UI) dùng chung 1 virtualenv
- **Hardware target**: RTX 4060 8GB (inference) / RTX 3090 Ti (training/dev)
- **Bridge pattern**: UI → subprocess bridges (`*_infer_bridge.py`) → core engines
- **Debug mode**: Toggle trong UI để xem full subprocess logs

---

## References

- [DiariZen — BUT-FIT](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md)
- [PhoWhisper — VinAI](https://huggingface.co/vinai/PhoWhisper-large)
- [WavLM Large](https://arxiv.org/abs/2110.13900)
- [Qwen 2.5 — Alibaba](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio)

---

## Author

**[Tên của bạn]** 
Đồ án cuối kỳ — [Tên môn học / Khóa học] 
[Trường / Khoa] 
[Email] · [LinkedIn] · [GitHub]

---

*Đồ án được thực hiện như một hệ thống nghiên cứu và prototype — không phải sản phẩm thương mại.*