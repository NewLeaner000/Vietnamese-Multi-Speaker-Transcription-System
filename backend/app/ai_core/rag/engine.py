"""
core/rag/engine.py
RAG pipeline cho Vietnamese Meeting Chatbot — optimized version.

Các cải tiến chính:
1. Chunk theo từng utterance (không gom thành 1 doc lớn rồi split)
2. Metadata đầy đủ: speaker, start, end, chunk_type
3. Speaker-aware retrieval: phát hiện tên speaker trong câu hỏi → filter metadata
4. Dynamic k: câu hỏi về 1 speaker lấy nhiều chunk hơn
5. Deduplicate chunks trùng nội dung
6. Batch embedding + exponential backoff (tránh 429 free-tier quota)
7. Dual backend: Gemini API hoặc Local HuggingFace (Qwen2.5-1.5B-Instruct)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Literal, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

# Backend literal type
LLMBackend = Literal["gemini", "local"]

# Default local model
try:
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(_root))
    from app.ai_core.pipeline_config import QWEN_GGUF_MODEL_DEFAULT
    DEFAULT_LOCAL_MODEL = QWEN_GGUF_MODEL_DEFAULT
except ImportError:
    DEFAULT_LOCAL_MODEL = "core/qwen/checkpoints/Qwen2.5-7B-Instruct-Q4_K_M.gguf"


# ── System prompt ─────────────────────────────────────────────────────────────
CHATBOT_SYSTEM_PROMPT = """Bạn là trợ lý phân tích nội dung cuộc họp tiếng Việt.
Chỉ trả lời dựa trên nội dung transcript/summary được cung cấp.
Nếu không tìm thấy thông tin, hãy nói rõ "Không có trong transcript".
Trả lời ngắn gọn, đúng trọng tâm, bằng tiếng Việt."""


# ── Load summary.json (từ Qwen) ───────────────────────────────────────────────
def load_documents_from_summary(summary_path: str) -> list[Document]:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {summary_path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    if overview := data.get("meeting_overview"):
        docs.append(Document(
            page_content=f"Tổng quan cuộc họp: {overview}",
            metadata={"chunk_type": "overview"},
        ))

    for seg in data.get("segments", []):
        title = seg.get("title", "")
        summary = seg.get("summary", "")
        if summary:
            docs.append(Document(
                page_content=f"[{title}] {summary}",
                metadata={"chunk_type": "segment", "title": title},
            ))

    action_items = data.get("action_items", [])
    if action_items:
        items_text = "\n".join(
            f"- {a.get('owner','?')}: {a.get('task','')} [{a.get('status','')}]"
            for a in action_items
        )
        docs.append(Document(
            page_content=f"Các việc cần làm:\n{items_text}",
            metadata={"chunk_type": "action_items"},
        ))

    for spk in data.get("speaker_main_summaries", []):
        speaker = spk.get("speaker", "")
        summary = spk.get("main_summary", "")
        if summary:
            docs.append(Document(
                page_content=f"Tóm tắt {speaker}: {summary}",
                metadata={"chunk_type": "speaker_summary", "speaker": speaker},
            ))

    return docs


# ── Load asr_results.json (từ ASR engine) ────────────────────────────────────
def load_documents_from_transcript(transcript_path: str) -> list[Document]:
    """
    Mỗi utterance → 1 Document riêng với metadata đầy đủ.
    Không gom thành 1 doc lớn → tránh mất nội dung khi chunk + retrieve.
    """
    path = Path(transcript_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    docs = []

    # ── 1. Mỗi utterance → 1 Document ────────────────────────────────────
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "Unknown")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        docs.append(Document(
            page_content=f"{speaker}: {text}",
            metadata={
                "chunk_type": "utterance",
                "speaker": speaker,
                "start": start,
                "end": end,
                "start_fmt": _fmt_time(start),
            },
        ))

    # ── 2. Sliding window chunks (ngữ cảnh) ──────────────────────────────
    window_size = 5
    for i in range(0, len(segments), window_size // 2):
        window = segments[i: i + window_size]
        if not window:
            continue
        lines = [
            f"[{s.get('speaker')}] ({_fmt_time(s['start'])}) {s.get('text','').strip()}"
            for s in window if s.get("text", "").strip()
        ]
        if not lines:
            continue
        docs.append(Document(
            page_content="\n".join(lines),
            metadata={
                "chunk_type": "window",
                "start": window[0].get("start", 0),
                "end": window[-1].get("end", 0),
            },
        ))

    return docs


def _fmt_time(seconds: float) -> str:
    """Chuyển giây → HH:MM:SS."""
    seconds = int(seconds or 0)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Retry helper (Gemini quota) ───────────────────────────────────────────────
def _with_retry(
    fn: Callable,
    max_retries: int = 5,
    base_delay: float = 60.0,
) -> any:
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if not is_quota or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Gemini quota hit (attempt %d/%d). Retry sau %.0fs...",
                attempt + 1, max_retries, delay,
            )
            time.sleep(delay)


# ── Embedding backends ────────────────────────────────────────────────────────
_EMBED_BATCH_SIZE = 80
_INTER_BATCH_DELAY = 62


def _build_gemini_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def _build_local_embeddings(model_name: str):
    """
    Dùng sentence-transformers local để embed — không cần API key, không quota.
    Mặc định dùng paraphrase-multilingual-MiniLM-L12-v2 (hỗ trợ tiếng Việt tốt).
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "Cần cài thêm: pip install sentence-transformers langchain-community"
        )
    # Dùng model multilingual nhỏ gọn, chạy được trên CPU
    embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    logger.info("Loading local embedding model: %s", embed_model)
    return HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_rag_index(
    docs: list[Document],
    api_key: str,
    backend: LLMBackend = "gemini",
    local_model_name: str = DEFAULT_LOCAL_MODEL,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> FAISS:
    """
    Build FAISS index với embedding.

    - backend="gemini": Dùng Gemini API (batch + retry cho free tier)
    - backend="local" : Dùng sentence-transformers local (không quota, chạy CPU)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    total = len(chunks)
    logger.info("Tổng chunks cần embed: %d (backend=%s)", total, backend)

    # ── Local backend: embed toàn bộ 1 lần, không cần batch/retry ────────
    if backend == "local":
        embeddings = _build_local_embeddings(local_model_name)
        if on_progress:
            on_progress(0, total)
        index = FAISS.from_documents(chunks, embeddings)
        if on_progress:
            on_progress(total, total)
        return index

    # ── Gemini backend: batch + retry ─────────────────────────────────────
    embeddings = _build_gemini_embeddings(api_key)
    index: Optional[FAISS] = None
    done = 0

    for batch_start in range(0, total, _EMBED_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + _EMBED_BATCH_SIZE]
        is_last = (batch_start + _EMBED_BATCH_SIZE) >= total

        logger.info("Embedding batch %d–%d / %d ...", batch_start + 1, batch_start + len(batch), total)

        batch_index: FAISS = _with_retry(
            lambda b=batch: FAISS.from_documents(b, embeddings)
        )

        if index is None:
            index = batch_index
        else:
            index.merge_from(batch_index)

        done += len(batch)
        if on_progress:
            on_progress(done, total)

        if not is_last:
            logger.info("Đợi %ds trước batch tiếp theo...", _INTER_BATCH_DELAY)
            time.sleep(_INTER_BATCH_DELAY)

    return index


# ── Local LLM (llama.cpp) ─────────────────────────────────────────────────────
class _LocalHFLLM:
    """
    Wrapper gọi Qwen2.5-7B-Instruct (GGUF) qua llama.cpp.
    Tương thích với interface .invoke(messages).
    """

    def __init__(self, model_path: str = DEFAULT_LOCAL_MODEL):
        try:
            import sys
            if sys.platform == "win32":
                import os
                from pathlib import Path
                torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
                if torch_lib.exists():
                    try:
                        os.add_dll_directory(str(torch_lib))
                    except Exception:
                        pass
            import llama_cpp
        except ImportError:
            raise ImportError("Cần cài: pip install llama-cpp-python")

        logger.info("Loading local LLM via llama.cpp: %s", model_path)
        
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
            
        n_gpu_layers = -1 if has_cuda else 0
        
        self.model = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False,
        )

    def invoke(self, messages: list) -> object:
        """Nhận list LangChain messages, trả về object có .content."""
        chat = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                chat.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat.append({"role": "assistant", "content": msg.content})

        response = self.model.create_chat_completion(
            messages=chat,
            max_tokens=512,
            temperature=0.2,
        )
        answer = response["choices"][0]["message"]["content"]

        # Trả về object giả lập LangChain response
        class _Resp:
            def __init__(self, content):
                self.content = content
        return _Resp(answer)


# ── Speaker detection ─────────────────────────────────────────────────────────
_SPEAKER_ALIASES: dict[str, list[str]] = {
    "Sang":   ["sang", "anh sang", "chị sang"],
    "Khang":  ["khang", "anh khang"],
    "Duy":    ["duy", "anh duy"],
    "Phuong": ["phuong", "phương", "chị phương", "anh phương"],
}

def _detect_speaker(question: str) -> Optional[str]:
    q_lower = question.lower()
    for canonical, aliases in _SPEAKER_ALIASES.items():
        for alias in aliases:
            if alias in q_lower:
                return canonical
    return None


# ── Main class ────────────────────────────────────────────────────────────────
class MeetingChatbot:
    def __init__(
        self,
        api_key: str = "",
        backend: LLMBackend = "gemini",
        local_model_name: str = DEFAULT_LOCAL_MODEL,
    ):
        """
        Args:
            api_key:          Google API key (chỉ cần khi backend="gemini")
            backend:          "gemini" | "local"
            local_model_name: HuggingFace model id khi backend="local"
                              default: "Qwen/Qwen2.5-1.5B-Instruct"
        """
        self.api_key = api_key
        self.backend: LLMBackend = backend
        self.local_model_name = local_model_name

        self._index: Optional[FAISS] = None
        self._llm = None
        self._history: list = []
        self.loaded_path: Optional[str] = None
        self._known_speakers: list[str] = []

    @property
    def is_ready(self) -> bool:
        return self._index is not None

    def _build_llm(self):
        if self.backend == "local":
            return _LocalHFLLM(self.local_model_name)
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.2,
        )

    def _build_index(
        self,
        docs: list[Document],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> FAISS:
        return build_rag_index(
            docs,
            api_key=self.api_key,
            backend=self.backend,
            local_model_name=self.local_model_name,
            on_progress=on_progress,
        )

    def load(
        self,
        summary_path: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Load summary.json từ Qwen."""
        docs = load_documents_from_summary(summary_path)
        self._index = self._build_index(docs, on_progress)
        self._llm = self._build_llm()
        self._history = []
        self.loaded_path = summary_path

    def load_transcript(
        self,
        transcript_path: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Load asr_results.json từ ASR engine."""
        docs = load_documents_from_transcript(transcript_path)
        self._known_speakers = list({
            d.metadata["speaker"]
            for d in docs
            if d.metadata.get("chunk_type") == "utterance"
        })
        self._index = self._build_index(docs, on_progress)
        self._llm = self._build_llm()
        self._history = []
        self.loaded_path = transcript_path

    # ── Retrieval ─────────────────────────────────────────────────────────
    def _retrieve(self, question: str) -> list[Document]:
        target_speaker = _detect_speaker(question)

        if target_speaker and target_speaker in self._known_speakers:
            all_results = self._index.similarity_search(question, k=50)
            speaker_utterances = [
                doc for doc in all_results
                if doc.metadata.get("speaker") == target_speaker
                and doc.metadata.get("chunk_type") == "utterance"
            ]
            window_results = [
                doc for doc in all_results
                if doc.metadata.get("chunk_type") == "window"
            ][:3]
            speaker_utterances.sort(key=lambda d: d.metadata.get("start", 0))
            combined = speaker_utterances + window_results
            seen, unique = set(), []
            for doc in combined:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique.append(doc)
            return unique if unique else self._index.similarity_search(question, k=8)

        return self._index.similarity_search(question, k=8)

    # ── ask ───────────────────────────────────────────────────────────────
    def ask(self, question: str) -> str:
        if not self.is_ready:
            return "Chưa load transcript. Vui lòng chọn session trước."

        relevant_docs = self._retrieve(question)
        target_speaker = _detect_speaker(question)

        if target_speaker:
            utterances = [d for d in relevant_docs if d.metadata.get("chunk_type") == "utterance"]
            others     = [d for d in relevant_docs if d.metadata.get("chunk_type") != "utterance"]
            speaker_block = (
                f"Toàn bộ phát ngôn của {target_speaker}:\n"
                + "\n".join(
                    f"  ({d.metadata.get('start_fmt','')}) {d.page_content}"
                    for d in utterances
                )
            )
            other_block = "\n\n".join(d.page_content for d in others)
            context = speaker_block + (f"\n\nNgữ cảnh bổ sung:\n{other_block}" if other_block else "")
        else:
            context = "\n\n".join(d.page_content for d in relevant_docs)

        messages = [
            SystemMessage(content=CHATBOT_SYSTEM_PROMPT),
            SystemMessage(content=f"Nội dung cuộc họp:\n{context}"),
            *self._history[-6:],
            HumanMessage(content=question),
        ]

        response = self._llm.invoke(messages)
        answer = response.content

        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        return answer

    def reset(self) -> None:
        self._history = []
        self._index = None
        self._llm = None
        self.loaded_path = None
        self._known_speakers = []
