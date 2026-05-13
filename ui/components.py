
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import streamlit as st


def save_uploaded_file(uploaded_file, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def append_to_debug_log(message: str, log_path: Path) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


def list_output_files(output_dir: Path) -> List[str]:
    if not output_dir.exists():
        return []
    return sorted(str(p.relative_to(output_dir)) for p in output_dir.glob("**/*") if p.is_file())


def preview_file(file_path: Path, tr) -> str:
    if not file_path.exists() or not file_path.is_file():
        return ""
    try:
        text = file_path.read_text(encoding="utf-8")
        return "\n".join(text.splitlines()[:200])
    except Exception:
        return tr("binary_preview_unavailable")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


def render_hero(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{title}</div>
            <div class="hero-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str) -> None:
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)


def group_files_by_top_level(display_dir: Path) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for rel in list_output_files(display_dir):
        parts = rel.split(os.path.sep)
        group = parts[0] if len(parts) > 1 else "__root__"
        grouped.setdefault(group, []).append(rel)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def render_output_files_grouped(display_dir: Path, tr) -> None:
    files_grouped = group_files_by_top_level(display_dir)
    if not files_grouped:
        st.info(tr("no_output_files"))
        return

    for group_name, rel_paths in files_grouped.items():
        pretty_group = tr("group_root") if group_name == "__root__" else group_name
        st.markdown(f'<div class="file-group-title">{pretty_group}</div>', unsafe_allow_html=True)
        for idx, rel in enumerate(rel_paths):
            file_path = display_dir / rel
            with st.expander(rel, expanded=False):
                if file_path.suffix.lower() in {".txt", ".rttm", ".csv", ".md", ".json", ".jsonl", ".log"}:
                    st.text(preview_file(file_path, tr))
                st.download_button(
                    tr("download_file", name=file_path.name),
                    file_path.read_bytes(),
                    file_name=file_path.name,
                    key=f"dl_btn_{idx}_{rel.replace(os.path.sep, '_')}",
                )
