
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from .transcript_view import read_asr_rows, render_transcript_panel, render_workspace_metrics


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
        tab_summary, tab_mindmap = st.tabs([tr("summary_tab"), tr("mindmap_tab")])
        with tab_summary:
            render_summary_tab(summary_data, qwen_md_path, transcript_rows, tr)
        with tab_mindmap:
            render_mindmap_tab(summary_data, transcript_rows, tr)

    with right_col:
        render_transcript_panel(transcript_rows, tr, display_dir)
