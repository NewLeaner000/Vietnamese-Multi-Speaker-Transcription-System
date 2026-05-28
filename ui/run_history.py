from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict

import streamlit as st


_EXCLUDED_SESSION_DIRS = {"temp_uploads", "temp_enrollment", "__pycache__"}


def _is_session_dir(path: Path) -> bool:
    return path.is_dir() and path.name not in _EXCLUDED_SESSION_DIRS and not path.name.startswith(".")


def _latest_session_mtime(run_dir: Path) -> float:
    mtimes = [run_dir.stat().st_mtime]
    mtimes.extend(p.stat().st_mtime for p in run_dir.glob("**/*") if p.is_file())
    return max(mtimes) if mtimes else run_dir.stat().st_mtime


def list_saved_runs(output_root: Path, newest_first: bool = True) -> List[Path]:
    if not output_root.exists():
        return []
    runs = [p for p in output_root.iterdir() if _is_session_dir(p)]
    runs.sort(key=_latest_session_mtime, reverse=newest_first)
    return runs


def collect_run_info(run_dir: Path) -> Dict[str, object]:
    has_der = any((run_dir / "der").glob("**/*")) if (run_dir / "der").exists() else False
    has_asr = (run_dir / "asr" / "asr_results.csv").exists()
    has_qwen = (run_dir / "qwen" / "qwen_summary.json").exists() or (run_dir / "qwen" / "qwen_summary.md").exists()
    file_count = sum(1 for _ in run_dir.glob("**/*") if _.is_file())
    updated = datetime.fromtimestamp(_latest_session_mtime(run_dir)).strftime("%Y-%m-%d %H:%M:%S")
    status = "complete" if (has_der and has_asr) else "partial"
    return {
        "path": run_dir,
        "name": run_dir.name,
        "updated": updated,
        "has_der": has_der,
        "has_asr": has_asr,
        "has_qwen": has_qwen,
        "file_count": file_count,
        "status": status,
    }


def select_saved_run_sidebar(output_root: Path, current_output_dir: Path | None, tr):
    runs = list_saved_runs(output_root, newest_first=True)
    if not runs:
        st.info(tr("no_saved_runs"))
        return None

    options = [r.name for r in runs]
    current_name = current_output_dir.name if current_output_dir and current_output_dir.exists() else options[0]
    index = options.index(current_name) if current_name in options else 0
    selected_name = st.selectbox(tr("open_saved_run"), options, index=index)
    return next((r for r in runs if r.name == selected_name), None)


def _filter_runs(infos: List[Dict[str, object]], filter_value: str, tr):
    if filter_value == tr("history_filter_complete"):
        return [x for x in infos if x["status"] == "complete"]
    if filter_value == tr("history_filter_partial"):
        return [x for x in infos if x["status"] != "complete"]
    return infos


def render_history_tab(output_root: Path, tr):
    st.header(tr("history_header"))
    controls = st.columns([1.2, 1.2, 2.6])
    filter_value = controls[0].selectbox(
        tr("history_filter"),
        [tr("history_filter_all"), tr("history_filter_complete"), tr("history_filter_partial")],
        index=0,
    )
    sort_value = controls[1].selectbox(
        tr("history_sort"),
        [tr("history_sort_newest"), tr("history_sort_oldest")],
        index=0,
    )
    newest_first = sort_value == tr("history_sort_newest")

    infos = [collect_run_info(p) for p in list_saved_runs(output_root, newest_first=newest_first)]
    infos = _filter_runs(infos, filter_value, tr)

    complete_count = sum(1 for x in infos if x["status"] == "complete")
    partial_count = sum(1 for x in infos if x["status"] != "complete")
    controls[2].markdown(
        f"""
        <div class="tiny-muted">{tr("history_counts")}</div>
        <div>
            <span class="status-badge status-complete">{tr("history_badge_complete")}: {complete_count}</span>
            <span class="status-badge status-partial">{tr("history_badge_partial")}: {partial_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not infos:
        st.info(tr("history_empty"))
        return None

    opened = None
    for info in infos:
        badge_class = "status-complete" if info["status"] == "complete" else "status-partial"
        badge_text = tr("history_status_complete") if info["status"] == "complete" else tr("history_status_partial")
        der = tr("run_card_yes") if info["has_der"] else tr("run_card_no")
        asr = tr("run_card_yes") if info["has_asr"] else tr("run_card_no")
        qwen = tr("run_card_yes") if info["has_qwen"] else tr("run_card_no")
        artifact_line = tr("history_artifacts_line", der=der, asr=asr, qwen=qwen)

        st.markdown('<div class="run-row">', unsafe_allow_html=True)
        cols = st.columns([2.2, 1.5, 1.0, 2.6, 0.7, 1.0])
        cols[0].markdown(f"**{info['name']}**")
        cols[1].write(info["updated"])
        cols[2].markdown(f'<span class="status-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
        cols[3].write(artifact_line)
        cols[4].write(str(info["file_count"]))
        if cols[5].button(tr("history_open"), key=f"open_{info['name']}"):
            opened = info["path"]
        st.markdown('</div>', unsafe_allow_html=True)

    return opened
