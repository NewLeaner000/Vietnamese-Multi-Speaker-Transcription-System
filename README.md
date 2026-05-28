# ViMeet — Vietnamese Multi-Speaker Transcription System

> **Deep Learning Framework for Automatic Meeting Transcription in Noisy Environments**
> End-to-end pipeline: Speaker Diarization → ASR → Structured Transcript → AI-Powered Q&A

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

**ViMeet** is an automatic Vietnamese meeting transcription system, developed as a final capstone project in AI/Deep Learning. The system addresses the **multi-speaker transcription** problem under real-world conditions — background noise, overlapping speech, and multiple simultaneous speakers.

- **Input**: Meeting audio file (`.wav`) with multiple speakers and background noise
- **Output**: Structured transcript (*who said what, and when*) + AI-generated summary + interactive Q&A chatbot

### Main Pipeline

```
Audio Input
 |
 v
[Preprocessing]        Normalize to mono 16kHz
 |
 v
[Speaker Diarization]  DiariZen (WavLM-Large) + Enrollment Embeddings
 |
 v  RTTM file (who spoke when)
 |
 v
[ASR per Segment]      PhoWhisper-Large (Vietnamese)
 |
 v
[Post-processing]      Merge, align, format
 |
 v
Structured Transcript (JSON / TXT / CSV)
 |
 v
[Qwen Summary]         Meeting overview, speaker focus, action items
 |
 v
[RAG Chatbot]          Ask questions about the meeting content
```

---

## Key Features

| Feature | Detail |
|---|---|
| Speaker Diarization | DiariZen (WavLM-Large), fine-tuned + enrollment-based speaker ID |
| Vietnamese ASR | PhoWhisper-Large, segment-level inference from RTTM |
| Speaker Attribution | Enrollment directory → identity-aware diarization |
| Structured Output | Timeline JSON, speaker-tagged transcript, RTTM artifacts |
| AI Meeting Summary | Qwen 2.5 generates overview, topics, speaker summaries, action items |
| RAG Chatbot | Ask natural-language questions about meeting content (Gemini API or local model) |
| Interactive Transcript | Click-to-play audio segments synchronized with transcript |
| Web Interface | Streamlit UI — bilingual (EN/VI), dark/light theme |
| Low-VRAM Support | Qwen 4-bit quantization for RTX 4060 (8GB VRAM) |
| Run History | Save and review previous processing runs |

---

## RAG Chatbot — Meeting Q&A

After transcription and summarization, users can query the meeting content through an embedded chatbot powered by **Retrieval-Augmented Generation (RAG)**.

### How it works

```
User Question
 |
 v
[Embedding]     Encode question using Gemini or sentence-transformers
 |
 v
[FAISS Retrieval]  Find top-k relevant transcript chunks
 |                 Speaker-aware: filters by speaker name if mentioned
 v
[LLM Answer]    Generate answer grounded in retrieved context
 |
 v
Response (Vietnamese)
```

### Two backend options

| Backend | Model | When to use |
|---|---|---|
| Gemini API | `gemini-2.5-flash` + `gemini-embedding-001` | Best answer quality, requires internet + API key |
| Local | `Qwen2.5-1.5B-Instruct` + `paraphrase-multilingual-MiniLM-L12-v2` | No API key, runs fully offline (CPU) |

### Features

- **Speaker-aware retrieval**: Detects speaker names in the question (e.g. "What did Sang say?") and prioritizes that speaker's utterances
- **Dual data source**: Query from raw ASR transcript or Qwen-generated summary JSON
- **Conversation history**: Maintains last 6 turns for multi-turn Q&A
- **Lazy loading**: Index is only built when the user clicks Start — no auto-run on tab open
- **Progress tracking**: Batch embedding with quota-aware retry (exponential backoff for Gemini free tier)
- **Reset / Stop**: Clear chat history or fully unload the index from memory

### Example queries

```
"Tóm tắt những điểm chính của cuộc họp"
"Sang nói gì về vấn đề thời gian?"
"Các việc cần làm sau cuộc họp là gì?"
"Khang đề xuất phương án nào?"
```

---

## Repository Structure

```
vimeet/
├── app_streamlit.py            Main Streamlit UI — entry point
├── pipeline_config.py          Global config: paths, model defaults
│
├── core/                       Core ML inference engines
│   ├── asr/
│   │   ├── engine.py           PhoWhisper ASR inference
│   │   ├── config.py           ASR hyperparameters & constants
│   │   ├── manifest_builder.py RTTM to JSONL manifest conversion
│   │   └── checkpoints/        Fine-tuned ASR model weights (not tracked)
│   ├── der/
│   │   ├── engine.py           DiariZen diarization inference
│   │   ├── pyannote_engine.py  PyAnnote baseline engine
│   │   └── checkpoints/        Fine-tuned DER model weights (not tracked)
│   ├── qwen/
│   │   └── engine.py           Qwen 2.5 transcript normalization & summary
│   └── rag/
│       └── engine.py           RAG pipeline — embedding, FAISS index, chatbot
│
├── ui/                         Streamlit UI components
│   ├── theme.py                CSS theme & i18n translator (EN/VI)
│   ├── components.py           Reusable UI widgets
│   ├── transcript_view.py      Interactive transcript with click-to-play audio
│   ├── summary_view.py         ASR workspace, summary tabs, inline chatbot
│   ├── chatbot_view.py         Standalone chatbot page
│   └── run_history.py          Run history management
│
├── audio_preprocess_input.py   Audio normalization utilities
├── asr_infer_bridge.py         Bridge: UI to ASR subprocess
├── der_infer_bridge.py         Bridge: UI to DER subprocess
├── qwen_infer_bridge.py        Bridge: UI to Qwen subprocess
├── asr_runner.py               ASR orchestration runner
│
├── output/                     Generated results (gitignored)
│   ├── der/                    RTTM diarization outputs
│   ├── asr/                    Transcript CSV/JSON outputs
│   └── qwen/                   Qwen summary outputs
│
├── requirements_extra_web.txt  Additional UI/RAG dependencies
├── run_streamlit.ps1           Windows launch script
└── install_extra_web.ps1       Windows installation script
```

---

## Quick Start

### Requirements

- Python 3.10
- CUDA GPU (tested: RTX 3090 Ti for training, RTX 4060 8GB for inference)
- Windows 11 (PowerShell scripts included)
- (Optional) Gemini API key for cloud-based chatbot backend

### Installation

```bash
# 1. Clone repository
git clone https://github.com/NewLeaner000/Vietnamese-Multi-Speaker-Transcription-System.git
cd Vietnamese-Multi-Speaker-Transcription-System

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install PyTorch (match your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt

# 5. (Optional) Install RAG/chatbot dependencies
pip install -r requirements_extra_web.txt
```

### Environment variables (optional)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=AIzaSy...   # Required for Gemini chatbot backend
```

### Run

```bash
streamlit run app_streamlit.py
```

Or use the Windows script:

```powershell
.\run_streamlit.ps1
```

---

## Interface

**Main Interface — Run & Configuration**

![Main Interface](assets/workspace_interface_02.png)

**Workspace — Transcript, Summary & Chatbot**

![Workspace Interface](assets/main_interface_01.png)

The Streamlit interface has four main sections:

| Tab | Description |
|---|---|
| **Run** | Upload audio, configure pipeline, monitor progress |
| **Workspace** | View transcript, summary, mind map, and chatbot |
| **Files** | Browse and download all output files per session |
| **History** | Review and reopen previous processing runs |

---

## Model Checkpoints

### Fine-tuned Checkpoints (download required)

These checkpoints are not stored in the repository due to file size. Download and place them at the correct paths before running.

| Component | Path | Download |
|---|---|---|
| DER checkpoint (DiariZen fine-tuned) | `core/der/checkpoints/best_model.pth` | [Google Drive](https://drive.google.com/drive/folders/1pjW941P41BSZlNZcP9fkC1-t1TEVFptG?usp=sharing) |
| ASR checkpoint (PhoWhisper adapter) | `core/asr/checkpoints/best_adapter/` | [Google Drive](https://drive.google.com/drive/folders/1pjW941P41BSZlNZcP9fkC1-t1TEVFptG?usp=sharing) |

### Qwen 2.5 (auto-downloaded from HuggingFace)

| Model | HuggingFace ID | Usage |
|---|---|---|
| Qwen 2.5 1.5B Instruct (4-bit) | `Qwen/Qwen2.5-1.5B-Instruct` | Transcript normalization, fast summary, local chatbot |
| Qwen 2.5 7B Instruct (4-bit) | `Qwen/Qwen2.5-7B-Instruct` | Deep meeting summary |

> 4-bit quantization is applied automatically at runtime to reduce VRAM usage.

---

## Models Used

### Speaker Diarization

| Model | Role |
|---|---|
| `BUT-FIT/diarizen-wavlm-large-s80-md` | Backbone diarization — WavLM-Large based |
| Fine-tuned checkpoint (`.pth`) | Custom-trained on Vietnamese meeting data |
| Enrollment embeddings | Speaker identity matching from voice samples |

### Automatic Speech Recognition

| Model | Role |
|---|---|
| `vinai/PhoWhisper-large` | Vietnamese ASR — primary model |
| DiCoW | Alternative ASR backbone (benchmark) |

### Post-processing / Summary

| Model | Role |
|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (4-bit) | Transcript normalization, fast summary, local chatbot LLM |
| `Qwen/Qwen2.5-7B-Instruct` (4-bit) | Deep meeting summary |

### RAG / Chatbot

| Component | Detail |
|---|---|
| Embedding (cloud) | `gemini-embedding-001` via Google Generative AI |
| Embedding (local) | `paraphrase-multilingual-MiniLM-L12-v2` via sentence-transformers |
| Vector store | FAISS (in-memory) |
| LLM (cloud) | `gemini-2.5-flash` via Gemini API |
| LLM (local) | `Qwen/Qwen2.5-1.5B-Instruct` via HuggingFace transformers |
| Framework | LangChain (retrieval, text splitting, message history) |

---

## Experiments and Results

### DiariZen on test_labeled Set (best fine-tuned model)

| Dataset | DER(%) | Miss(%) | FA(%) | Conf(%) | Ov.F1 | Pred.Seg | GT Seg | Baseline DER |
|---|---|---|---|---|---|---|---|---|
| chuyen_ho | 41.281 | 1.895 | 33.413 | 5.974 | 0.066 | 1042 | 632 | 59.164 |
| coi_moi | 60.186 | 6.084 | 41.661 | 12.441 | 0.128 | 1398 | 669 | 92.307 |
| dustin_1 | 26.412 | 3.958 | 16.018 | 6.436 | 0.106 | 1236 | 785 | 56.185 |
| vif_1 | 6.793 | 2.875 | 3.618 | 0.301 | 0.000 | 714 | 625 | 19.845 |
| vif_2 | 6.897 | 0.540 | 5.972 | 0.386 | 0.000 | 375 | 522 | 15.155 |

### Pyannote on test_labeled Set (best fine-tuned model)

| Dataset | DER(%) | Miss(%) | FA(%) | Conf(%) | Ov.F1 | Pred.Seg | GT Seg | Baseline DER |
|---|---|---|---|---|---|---|---|---|
| chuyen_ho | 32.742 | 3.891 | 26.218 | 2.633 | 0.090 | 713 | 632 | 50.801 |
| coi_moi | 51.042 | 6.231 | 40.364 | 4.448 | 0.095 | 1028 | 669 | 72.408 |
| dustin_1 | 24.516 | 3.404 | 17.706 | 3.405 | 0.070 | 879 | 785 | 44.078 |
| vif_1 | 6.617 | 0.468 | 5.795 | 0.355 | 0.000 | 423 | 625 | 18.248 |
| vif_2 | 7.521 | 0.132 | 7.072 | 0.317 | 0.000 | 318 | 522 | 13.404 |

### Comparison DiariZen vs Pyannote (Best Fine-tuned Model)

| Criterion | Advantage | DiariZen (Best FT) | Pyannote (Best FT) |
|---|---|---|---|
| DER: Synthetic Data (overall) | DiariZen | **13.793%** | 17.441% |
| Miss: Synthetic Data (overall) | DiariZen | **1.138%** | 4.279% |
| FA: Synthetic Data (overall) | Pyannote | 9.731% | **1.761%** |
| Confusion: Synthetic Data (overall) | DiariZen | **2.924%** | 11.401% |
| Overlap F1: Synthetic Data (overall) | Pyannote | 0.478 | **0.555** |
| DER: 2 speakers | DiariZen | **9.438%** | 13.672% |
| DER: 3 speakers | Pyannote | 17.858% | **17.008%** |
| DER: 4 speakers | DiariZen | **14.084%** | 21.643% |

**Test set**: Vietnamese meeting data (noisy, multi-speaker)
**Hardware**: NVIDIA RTX 3090 Ti (training), RTX 4060 8GB (inference)

---

## Development Notes

- **Single venv**: All components (DER, ASR, Qwen, RAG, UI) share one virtualenv
- **Hardware target**: RTX 4060 8GB (inference) / RTX 3090 Ti (training/dev)
- **Bridge pattern**: UI calls subprocess bridges (`*_infer_bridge.py`) → core engines
- **RAG architecture**: Utterance-level chunking + sliding window context, FAISS in-memory index, speaker-aware retrieval
- **Offline-capable**: Full pipeline including chatbot can run without internet using local models
- **Debug mode**: Toggle in UI to view full subprocess logs and error traces

---

## References

- [DiariZen — BUT-FIT](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md)
- [PhoWhisper — VinAI](https://huggingface.co/vinai/PhoWhisper-large)
- [WavLM Large](https://arxiv.org/abs/2110.13900)
- [Qwen 2.5 — Alibaba](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio)
- [LangChain](https://python.langchain.com/)
- [FAISS — Facebook AI](https://github.com/facebookresearch/faiss)
- [Gemini API — Google](https://ai.google.dev/)

---

## Authors

**Capstone Project — AIP491-SP26AI91**

Name | Student ID |
|---|---|
Tran Nguyen Quang Khang | SE183747 |
Nguyen Duy Phuong | SE183477 |
Truong Minh Sang | SE184204 |
Ho Khanh Duy | SE184539 |

**Supervisors:** Huynh Van Thong — Nguyen Hong Hai

---

*This project was developed as an academic research prototype and is not intended for commercial use.*
