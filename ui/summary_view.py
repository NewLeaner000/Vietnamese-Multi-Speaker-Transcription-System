from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from .transcript_view import read_asr_rows, render_transcript_panel, render_workspace_metrics

load_dotenv()


def load_json_data(json_path: Path):
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_summary_tab(summary_data: dict, qwen_md_path: Path, transcript_rows: list, tr):
    segments = summary_data.get("segments", []) if isinstance(summary_data, dict) else []
    speaker_main = summary_data.get("speaker_main_summaries", []) if isinstance(summary_data, dict) else []

    if summary_data:
        st.markdown(f"#### {tr('topics_heading')}")
        if segments:
            for seg in segments[:12]:
                summary_text = (seg.get("summary") or "").strip()
                if summary_text:
                    st.markdown(f"- {summary_text}")
                elif seg.get("title"):
                    st.markdown(f"- {seg.get('title')}")
        else:
            st.info(tr("no_summary_data"))

        st.markdown(f"#### {tr('assessment_heading')}")
        meeting_overview = (summary_data.get("meeting_overview") or "").strip()
        conversation_main = (summary_data.get("conversation_main_summary") or "").strip()

        if meeting_overview:
            st.markdown(f"**{tr('meeting_overview_label')}**")
            st.write(meeting_overview)

        if conversation_main:
            st.markdown(f"**{tr('conversation_main_label')}**")
            st.write(conversation_main)

        if speaker_main:
            st.markdown(f"**{tr('speaker_focus_label')}**")
            for item in speaker_main[:8]:
                speaker = item.get("speaker", "Speaker")
                main_summary = (item.get("main_summary") or "").strip()
                if main_summary:
                    st.markdown(f"- **{speaker}**: {main_summary}")
    elif qwen_md_path.exists():
        st.markdown(qwen_md_path.read_text(encoding="utf-8"))
    else:
        st.info(tr("no_summary_data"))


def render_mindmap_tab(summary_data: dict, transcript_rows: list, tr):
    if summary_data:
        lines = [f"- **{tr('mindmap_root')}**"]
        meeting_overview = (summary_data.get("meeting_overview") or "").strip()
        if meeting_overview:
            lines.append(f"  - {meeting_overview}")

        segments = summary_data.get("segments", [])
        if segments:
            lines.append(f"  - **{tr('mindmap_segments')}**")
            for seg in segments[:10]:
                title = (seg.get("title") or "").strip() or "Segment"
                summary = (seg.get("summary") or "").strip()
                if summary:
                    lines.append(f"    - **{title}**: {summary}")
                else:
                    lines.append(f"    - **{title}**")

        speaker_main = summary_data.get("speaker_main_summaries", [])
        if speaker_main:
            lines.append(f"  - **{tr('mindmap_speakers')}**")
            for item in speaker_main[:8]:
                speaker = item.get("speaker", "Speaker")
                main_summary = (item.get("main_summary") or "").strip()
                if main_summary:
                    lines.append(f"    - **{speaker}**: {main_summary}")
        st.markdown("\n".join(lines))
    else:
        speaker_counts = {}
        for row in transcript_rows:
            speaker_counts[row["speaker"]] = speaker_counts.get(row["speaker"], 0) + 1
        if not speaker_counts:
            st.info(tr("no_summary_data"))
            return
        lines = [f"- **{tr('mindmap_root')}**", f"  - **{tr('mindmap_speakers')}**"]
        for speaker, count in sorted(speaker_counts.items(), key=lambda x: (-x[1], x[0]))[:10]:
            lines.append(f"    - **{speaker}**: {count} turns")
        st.markdown("\n".join(lines))


def render_asr_workspace(display_dir: Path, tr):
    asr_csv_path = display_dir / "asr" / "asr_results.csv"
    qwen_json_path = display_dir / "qwen" / "qwen_summary.json"
    qwen_md_path = display_dir / "qwen" / "qwen_summary.md"

    transcript_rows = read_asr_rows(asr_csv_path)
    summary_data = load_json_data(qwen_json_path)

    if not transcript_rows and not summary_data:
        return

    st.header(tr("workspace_header"))
    render_workspace_metrics(transcript_rows, summary_data, tr)

    left_col, right_col = st.columns([1.0, 1.4], gap="large")

    with left_col:
        tab_summary, tab_mindmap, tab_chat = st.tabs([
            tr("summary_tab"), tr("mindmap_tab"), tr("chatbot_tab"),
        ])
        with tab_summary:
            render_summary_tab(summary_data, qwen_md_path, transcript_rows, tr)
        with tab_mindmap:
            render_mindmap_tab(summary_data, transcript_rows, tr)
        with tab_chat:
            _render_inline_chatbot(display_dir, tr)

    with right_col:
        render_transcript_panel(transcript_rows, tr, display_dir)


# ── Inline chatbot (embedded in Workspace tab) ────────────────────────────────

def _render_inline_chatbot(display_dir: Path, tr) -> None:
    """Render chatbot panel inside the Workspace tab.
    Embedding chỉ chạy khi user bấm nút Start — không auto-run khi mở tab.
    """
    from core.rag.engine import MeetingChatbot

    # Session-state keys scoped to this session dir
    _bot_key      = f"inline_chatbot_{display_dir}"
    _loaded_key   = f"inline_loaded_{display_dir}"
    _history_key  = f"inline_history_{display_dir}"
    _enabled_key  = f"inline_enabled_{display_dir}"

    if _history_key not in st.session_state:
        st.session_state[_history_key] = []
    if _enabled_key not in st.session_state:
        st.session_state[_enabled_key] = False

    bot: MeetingChatbot = st.session_state.get(_bot_key)   # type: ignore[assignment]
    loaded_path = st.session_state.get(_loaded_key)

    # ── Chưa kích hoạt → hiện config + nút Start ────────────────────────────
    if not st.session_state[_enabled_key]:
        st.info(tr("chatbot_not_started"))

        # Backend selector (hiện trước khi start)
        backend_choice_pre = st.radio(
            tr("chatbot_backend_label"),
            options=[tr("chatbot_backend_gemini"), tr("chatbot_backend_local")],
            horizontal=True,
            key=f"chatbot_backend_{display_dir}",
        )
        use_local_pre = backend_choice_pre == tr("chatbot_backend_local")

        api_key_pre = os.getenv("GEMINI_API_KEY", "")
        if not use_local_pre:
            if not api_key_pre:
                api_key_pre = st.text_input(
                    tr("chatbot_api_key_label"),
                    type="password",
                    placeholder="AIzaSy...",
                    key="chatbot_inline_api_key",
                )
            if not api_key_pre:
                st.warning(tr("chatbot_api_key_missing"))
        else:
            st.caption(tr("chatbot_local_note"))

        # Chỉ cho phép Start khi config hợp lệ
        can_start = use_local_pre or bool(api_key_pre)
        if st.button(tr("chatbot_start"), key=f"chatbot_start_{display_dir}", type="primary", disabled=not can_start):
            st.session_state[_enabled_key] = True
            st.rerun()
        return

    # ── Đã kích hoạt ─────────────────────────────────────────────────────────

    # Backend selector (vẫn hiện để cho phép đổi backend)
    backend_choice = st.radio(
        tr("chatbot_backend_label"),
        options=[tr("chatbot_backend_gemini"), tr("chatbot_backend_local")],
        horizontal=True,
        key=f"chatbot_backend_{display_dir}",
    )
    use_local = backend_choice == tr("chatbot_backend_local")
    backend   = "local" if use_local else "gemini"

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not use_local:
        if not api_key:
            api_key = st.text_input(
                tr("chatbot_api_key_label"),
                type="password",
                placeholder="AIzaSy...",
                key="chatbot_inline_api_key",
            )
        if not api_key:
            st.warning(tr("chatbot_api_key_missing"))
            return
    else:
        st.caption(tr("chatbot_local_note"))

    # Auto-detect source files
    transcript_path = display_dir / "asr" / "asr_results.json"
    summary_path    = display_dir / "qwen" / "summary.json"
    has_transcript  = transcript_path.exists()
    has_summary     = summary_path.exists()

    if not has_transcript and not has_summary:
        st.info(tr("chatbot_no_source"))
        return

    # Source selector (only when both exist)
    if has_transcript and has_summary:
        source_choice = st.radio(
            tr("chatbot_source_label"),
            options=[tr("chatbot_source_transcript"), tr("chatbot_source_summary")],
            horizontal=True,
            key=f"chatbot_source_choice_{display_dir}",
        )
        use_transcript = source_choice == tr("chatbot_source_transcript")
    else:
        use_transcript = has_transcript

    chosen_path = str(transcript_path if use_transcript else summary_path)

    # Cache key includes backend so switching backend forces reload
    chosen_path_key = f"{chosen_path}::{backend}"

    # ── Build index nếu chưa load hoặc source/backend đổi ────────────────────
    if bot is None or loaded_path != chosen_path_key:
        st.info(tr("chatbot_building_index"))
        progress_bar = st.progress(0, text=tr("chatbot_preparing"))
        status_text  = st.empty()

        def on_progress(done: int, total: int) -> None:
            pct = done / total
            progress_bar.progress(pct, text=f"{tr('chatbot_embedding')} {done}/{total} ({int(pct*100)}%)")
            if done < total:
                status_text.caption(tr("chatbot_waiting_quota"))
            else:
                status_text.caption("")

        try:
            bot = MeetingChatbot(api_key=api_key, backend=backend)
            if use_transcript:
                bot.load_transcript(chosen_path, on_progress=on_progress)
            else:
                bot.load(summary_path=chosen_path, on_progress=on_progress)

            st.session_state[_bot_key]     = bot
            st.session_state[_loaded_key]  = chosen_path_key
            st.session_state[_history_key] = []

            progress_bar.progress(1.0, text=tr("chatbot_done"))
            status_text.empty()
            st.success(tr("chatbot_ready"))
        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"{tr('chatbot_load_error')}: {exc}")
            return

    # ── Controls: Reset + Stop ────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button(tr("chatbot_reset"), key=f"chatbot_reset_{display_dir}"):
            st.session_state[_history_key] = []
            if _bot_key in st.session_state:
                st.session_state[_bot_key].reset()
            st.rerun()
    with btn_col2:
        if st.button(tr("chatbot_stop"), key=f"chatbot_stop_{display_dir}"):
            # Tắt chatbot, xoá index khỏi memory
            for k in [_bot_key, _loaded_key]:
                st.session_state.pop(k, None)
            st.session_state[_history_key] = []
            st.session_state[_enabled_key] = False
            st.rerun()

    # Chat history display
    for msg in st.session_state[_history_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    question = st.chat_input(tr("chatbot_input_placeholder"), key=f"chatbot_input_{display_dir}")
    if question and bot and bot.is_ready:
        st.session_state[_history_key].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner(tr("chatbot_thinking")):
                answer = bot.ask(question)
            st.write(answer)
        st.session_state[_history_key].append({"role": "assistant", "content": answer})