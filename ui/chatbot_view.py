"""
ui/chatbot_view.py
Giao diện chatbot hỏi đáp nội dung cuộc họp.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.rag.engine import MeetingChatbot

load_dotenv()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_summary_files() -> list[str]:
    """Chỉ lấy summary.json từ output qwen."""
    output_dir = Path("output")
    return sorted([
        str(p) for p in output_dir.rglob("summary.json")
    ])


def _find_transcript_files() -> list[str]:
    """Chỉ lấy asr_results.json từ output asr."""
    output_dir = Path("output")
    return sorted([
        str(p) for p in output_dir.rglob("asr_results.json")
    ])


def _init_chatbot(api_key: str) -> MeetingChatbot:
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = MeetingChatbot(api_key=api_key)
    return st.session_state.chatbot


def _reset_chat():
    if "chatbot" in st.session_state:
        st.session_state.chatbot.reset()
    st.session_state.chat_history = []
    st.session_state.loaded_file = None


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Meeting Chatbot")
    st.caption("Ask questions about meeting content based on transcript or summary.")

    # ── API Key ───────────────────────────────────────────────────────────────
    st.subheader("Settings")

    default_key = os.getenv("GEMINI_API_KEY", "")
    if default_key:
        st.success("API key loaded from .env file")
        api_key = default_key
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
        )

    if not api_key:
        st.warning("Please enter your API key to continue.")
        return

    # ── Chọn file input ───────────────────────────────────────────────────────
    st.subheader("Select data source")

    input_type = st.radio(
        "File type",
        options=["Summary JSON (from Qwen)", "ASR Transcript JSON"],
        horizontal=True,
    )

    if input_type == "Summary JSON (from Qwen)":
        files = _find_summary_files()
        label = "Select summary.json file"
    else:
        files = _find_transcript_files()
        label = "Select asr_results.json file"

    if not files:
        st.warning("No files found in output/. Please run the pipeline first.")
        return

    selected_file = st.selectbox(label, options=files)

    # ── Load file ─────────────────────────────────────────────────────────────
    if "loaded_file" not in st.session_state:
        st.session_state.loaded_file = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Load file", use_container_width=True):
            st.info("Building embedding index... This may take a few minutes due to API quota limits.")
            progress_bar = st.progress(0, text="Preparing...")
            status_text = st.empty()

            def on_progress(done: int, total: int):
                pct = done / total
                progress_bar.progress(pct, text=f"Embedding {done}/{total} chunks ({int(pct*100)}%)")
                if done < total:
                    status_text.caption("Waiting for quota reset before next batch...")
                else:
                    status_text.caption("")

            try:
                bot = _init_chatbot(api_key)
                if input_type == "Summary JSON (from Qwen)":
                    bot.load(summary_path=selected_file, on_progress=on_progress)
                else:
                    bot.load_transcript(selected_file, on_progress=on_progress)
                st.session_state.loaded_file = selected_file
                st.session_state.chat_history = []
                progress_bar.progress(1.0, text="Done!")
                status_text.empty()
                st.success(f"Loaded: {selected_file}")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error loading file: {e}")

    with col2:
        if st.button("Reset chat", use_container_width=True):
            _reset_chat()
            st.rerun()

    # ── Chat UI ───────────────────────────────────────────────────────────────
    if not st.session_state.loaded_file:
        st.info("Select a file and click Load to begin.")
        return

    st.divider()
    st.subheader("Chat")
    st.caption(f"Using: {st.session_state.loaded_file}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about the meeting...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding answer..."):
                bot = st.session_state.chatbot
                answer = bot.ask(question)
            st.write(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
        })