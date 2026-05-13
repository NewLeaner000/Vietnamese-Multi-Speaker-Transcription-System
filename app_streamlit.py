
from __future__ import annotations

import inspect
import json
import shutil
import sys
import time
from pathlib import Path
import re

import streamlit as st

from audio_preprocess_input import SUPPORTED_AUDIO_EXTENSIONS, normalize_audio_to_mono16k
from der_infer_bridge import run_der_pipeline
from asr_infer_bridge import run_asr_pipeline
from qwen_infer_bridge import run_qwen_pipeline
from pipeline_config import (
    DEFAULT_OUTPUT_DIR,
    DER_SCRIPT_PATH,
    PYANNOTE_SCRIPT_PATH,
    QWEN_NORMALIZE_MODEL_DEFAULT,
    QWEN_SUMMARY_MODEL_DEFAULT,
    DER_CHECKPOINT_DEFAULT,
    PYANNOTE_CHECKPOINT_DEFAULT,
    ASR_CHECKPOINT_DEFAULT,
)
from ui.theme import get_theme_css, get_translator
from ui.components import (
    append_to_debug_log,
    format_time,
    render_hero,
    render_output_files_grouped,
    save_uploaded_file,
)
from ui.summary_view import render_asr_workspace
from ui.run_history import render_history_tab, select_saved_run_sidebar, collect_run_info, list_saved_runs


def _call_run_qwen_pipeline(*, asr_csv_path: Path, output_dir: Path, normalize_model_name: str, summary_model_name: str, no_4bit: bool, qwen_mode: str):
    sig = inspect.signature(run_qwen_pipeline)
    kwargs = {
        "asr_csv_path": asr_csv_path,
        "output_dir": output_dir,
        "normalize_model_name": normalize_model_name,
        "summary_model_name": summary_model_name,
        "no_4bit": no_4bit,
    }
    if "qwen_mode" in sig.parameters:
        kwargs["qwen_mode"] = qwen_mode
    return run_qwen_pipeline(**kwargs)




def _sanitize_session_folder_name(name: str) -> str:
    value = (name or "").strip()
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._ ")
    return value

def _render_runtime_summary(audio_input_path, final_enrollment_dir, run_mode, run_qwen, tr):
    audio_name = Path(audio_input_path).name if audio_input_path else "-"
    enrollment_name = Path(final_enrollment_dir).name if final_enrollment_dir else "-"
    qwen_state = tr("run_card_yes") if run_qwen else tr("run_card_no")
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-card-title">{tr("runtime_card")}</div>
            <div class="tiny-muted">{tr("run_card_audio")}</div>
            <div><strong>{audio_name}</strong></div>
            <div style="height:10px"></div>
            <div class="tiny-muted">{tr("run_card_enrollment")}</div>
            <div>{enrollment_name}</div>
            <div style="height:10px"></div>
            <div class="tiny-muted">{tr("run_card_mode")}</div>
            <div>{run_mode}</div>
            <div style="height:10px"></div>
            <div class="tiny-muted">{tr("run_card_qwen")}</div>
            <div>{qwen_state}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_selected_run_summary(current_output_dir: Path | None, tr):
    if current_output_dir and current_output_dir.exists():
        info = collect_run_info(current_output_dir)
        der = tr("run_card_yes") if info["has_der"] else tr("run_card_no")
        asr = tr("run_card_yes") if info["has_asr"] else tr("run_card_no")
        qwen = tr("run_card_yes") if info["has_qwen"] else tr("run_card_no")
        status_label = tr("history_status_complete") if info["status"] == "complete" else tr("history_status_partial")
        status_class = "status-complete" if info["status"] == "complete" else "status-partial"

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-card-title">{tr("execution_card")}</div>
                <div class="tiny-muted">{tr("selected_run")}</div>
                <div><strong>{info["name"]}</strong></div>
                <div style="height:10px"></div>
                <div class="tiny-muted">{tr("history_updated")}</div>
                <div>{info["updated"]}</div>
                <div style="height:10px"></div>
                <div class="tiny-muted">{tr("history_quick_summary")}</div>
                <div>
                    <span class="status-badge {status_class}">{status_label}</span>
                    <span class="status-badge">{tr("history_has_der")}: {der}</span>
                    <span class="status-badge">{tr("history_has_asr")}: {asr}</span>
                    <span class="status-badge">{tr("history_has_qwen")}: {qwen}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-card-title">{tr("execution_card")}</div>
                <div class="tiny-muted">-</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def main():
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "en"
    if "enrollment_speakers" not in st.session_state:
        st.session_state.enrollment_speakers = []
    if "current_output_dir" not in st.session_state:
        st.session_state.current_output_dir = None

    tr = get_translator(st.session_state.ui_language)

    st.set_page_config(page_title=tr("page_title"), layout="wide")
    st.markdown(get_theme_css(st.session_state.theme_mode), unsafe_allow_html=True)
    render_hero(tr("app_title"), tr("caption"))

    with st.sidebar:
        st.header(tr("settings"))
        selected_language = st.radio(
            tr("language"),
            ["English", "Tiếng Việt"],
            index=0 if st.session_state.ui_language == "en" else 1,
            horizontal=True,
        )
        next_lang = "en" if selected_language == "English" else "vi"
        if next_lang != st.session_state.ui_language:
            st.session_state.ui_language = next_lang
            st.rerun()

        tr = get_translator(st.session_state.ui_language)

        selected_theme = st.radio(
            tr("theme"),
            [tr("dark"), tr("light")],
            index=0 if st.session_state.theme_mode == "Dark" else 1,
            horizontal=True,
        )
        next_theme = "Dark" if selected_theme == tr("dark") else "Light"
        if next_theme != st.session_state.theme_mode:
            st.session_state.theme_mode = next_theme
            st.rerun()

        output_root = Path(st.text_input(tr("output_root"), str(DEFAULT_OUTPUT_DIR)))
        selected_run_from_sidebar = select_saved_run_sidebar(output_root, st.session_state.current_output_dir, tr)
        if selected_run_from_sidebar is not None and (
            st.session_state.current_output_dir is None or selected_run_from_sidebar != st.session_state.current_output_dir
        ):
            st.session_state.current_output_dir = selected_run_from_sidebar

        if st.session_state.current_output_dir:
            st.info(tr("current_session_folder", name=st.session_state.current_output_dir.name))

        st.write("---")
        st.subheader(tr("section_asr"))
        asr_fast_mode = st.checkbox(tr("asr_fast_mode"), value=False, help=tr("asr_fast_help"))

        st.write("---")
        st.subheader(tr("section_der"))
        der_step = st.slider(tr("der_step"), 0.1, 1.0, 0.1, 0.1, help=tr("der_step_help"))

        st.write("---")
        st.subheader(tr("section_qwen"))
        qwen_normalize_model = st.text_input(tr("qwen_normalizer"), QWEN_NORMALIZE_MODEL_DEFAULT)
        qwen_summary_model = st.text_input(tr("qwen_summarizer"), QWEN_SUMMARY_MODEL_DEFAULT)
        qwen_mode_label = st.selectbox(
            tr("qwen_mode"),
            [tr("qwen_mode_stable"), tr("qwen_mode_fast")],
            index=0,
            help=tr("qwen_mode_help"),
        )
        qwen_mode = "stable" if qwen_mode_label == tr("qwen_mode_stable") else "fast"
        qwen_disable_4bit = st.checkbox(tr("qwen_disable_4bit"), value=False)

        st.write("---")
        st.subheader(tr("section_system"))
        st.write(tr("python", path=sys.executable))
        if st.button(tr("reset_session")):
            st.session_state.clear()
            st.rerun()

    temp_root = output_root / "temp_enrollment"
    temp_upload_dir = output_root / "temp_uploads"

    nav_run, nav_workspace, nav_files, nav_history = st.tabs([
        tr("nav_run"), tr("nav_workspace"), tr("nav_files"), tr("nav_history")
    ])

    with nav_run:
        left_col, right_col = st.columns([1.55, 0.95], gap="large")

        # Input + options on the left
        with left_col:
            _render_section_label(tr("run_section_input"))
            with st.container(border=True):
                st.markdown(f"### {tr('audio_input_header')}")
                col_a, col_b = st.columns(2)
                with col_a:
                    uploaded_audio = st.file_uploader(tr("upload_audio"), type=SUPPORTED_AUDIO_EXTENSIONS)
                with col_b:
                    audio_path = st.text_input(tr("audio_local_path"), "")

                audio_input_path = None
                if uploaded_audio is not None:
                    audio_input_path = save_uploaded_file(uploaded_audio, temp_upload_dir / uploaded_audio.name)
                    st.success(tr("audio_loaded", name=audio_input_path.name))
                elif audio_path:
                    audio_input_path = Path(audio_path)

            with st.container(border=True):
                st.markdown(f"### {tr('speaker_enrollment_header')}")
                enrollment_options = [tr("enroll_none"), tr("enroll_existing"), tr("enroll_upload")]
                enrollment_mode = st.radio(tr("enrollment_source"), enrollment_options, horizontal=True)

                final_enrollment_dir = None
                if enrollment_mode == tr("enroll_existing"):
                    local_enroll_path = st.text_input(tr("enroll_dir_path"), "")
                    if local_enroll_path:
                        final_enrollment_dir = Path(local_enroll_path)
                elif enrollment_mode == tr("enroll_upload"):
                    st.info(tr("enroll_info"))
                    for idx, spk in enumerate(st.session_state.enrollment_speakers):
                        with st.container(border=True):
                            ec1, ec2, ec3 = st.columns([2, 3, 1])
                            with ec1:
                                new_name = st.text_input(tr("speaker_name", index=idx + 1), spk["name"], key=f"spk_name_{idx}")
                                st.session_state.enrollment_speakers[idx]["name"] = new_name
                            with ec2:
                                files = st.file_uploader(
                                    tr("samples_for", name=new_name),
                                    type=SUPPORTED_AUDIO_EXTENSIONS,
                                    accept_multiple_files=True,
                                    key=f"spk_files_{idx}",
                                )
                                st.session_state.enrollment_speakers[idx]["files"] = files
                            with ec3:
                                if st.button(tr("remove"), key=f"del_spk_{idx}"):
                                    st.session_state.enrollment_speakers.pop(idx)
                                    st.rerun()

                    if st.button(tr("add_speaker")):
                        st.session_state.enrollment_speakers.append({"name": f"Speaker_{len(st.session_state.enrollment_speakers) + 1}", "files": []})
                        st.rerun()

                    if st.session_state.enrollment_speakers:
                        if temp_root.exists():
                            shutil.rmtree(temp_root)
                        temp_root.mkdir(parents=True, exist_ok=True)

                        save_count = 0
                        for spk in st.session_state.enrollment_speakers:
                            if spk["name"] and spk["files"]:
                                spk_dir = temp_root / spk["name"]
                                for ef in spk["files"]:
                                    save_uploaded_file(ef, spk_dir / ef.name)
                                    save_count += 1
                        if save_count > 0:
                            final_enrollment_dir = temp_root
                            append_to_debug_log(
                                f"[UI] Saved {save_count} enrollment files to {temp_root}",
                                output_root / "debug" / "pipeline.log",
                            )

            _render_section_label(tr("run_section_options"))
            with st.container(border=True):
                st.markdown(f"### {tr('run_mode_header')}")
                run_mode_options = [tr("full_pipeline"), tr("full_pipeline_no_qwen"), tr("qwen_only")]
                run_mode = st.radio(tr("run_mode"), run_mode_options, horizontal=True)
                run_qwen = run_mode == tr("full_pipeline") or run_mode == tr("qwen_only")

                session_folder_name = ""
                if run_mode in (tr("full_pipeline"), tr("full_pipeline_no_qwen")):
                    session_folder_name = st.text_input(
                        tr("session_folder_name"),
                        "",
                        placeholder=tr("session_folder_placeholder"),
                        help=tr("session_folder_help"),
                    )
                    st.caption(tr("session_folder_note_default"))

                st.markdown(f"### {tr('model_config_header')}")
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    der_engine_options = [tr("der_engine_diarizen"), tr("der_engine_pyannote")]
                    der_engine_type = st.selectbox(tr("diarization_engine"), der_engine_options)
                    der_script_to_use = DER_SCRIPT_PATH if der_engine_type == tr("der_engine_diarizen") else PYANNOTE_SCRIPT_PATH
                    default_der_ckpt = DER_CHECKPOINT_DEFAULT if der_engine_type == tr("der_engine_diarizen") else PYANNOTE_CHECKPOINT_DEFAULT
                with p_col2:
                    der_checkpoint = st.text_input(tr("der_checkpoint"), default_der_ckpt)
                with p_col3:
                    n_speakers = st.number_input(tr("max_speakers"), min_value=1, max_value=32, value=4)

                if run_mode == tr("full_pipeline_no_qwen"):
                    st.caption(tr("qwen_skip_note"))

        # Summary cards and run button on the right
        with right_col:
            _render_runtime_summary(audio_input_path, final_enrollment_dir, run_mode, run_qwen, tr)
            _render_selected_run_summary(st.session_state.current_output_dir, tr)

            if st.button(tr("run_pipeline"), use_container_width=True, type="primary"):
                if run_mode in (tr("full_pipeline"), tr("full_pipeline_no_qwen")):
                    if not audio_input_path:
                        st.error(tr("audio_required"))
                        st.stop()

                    safe_folder_name = _sanitize_session_folder_name(session_folder_name)
                    if session_folder_name and not safe_folder_name:
                        st.error(tr("session_folder_error_empty"))
                        st.stop()

                    if safe_folder_name:
                        candidate_output_dir = output_root / safe_folder_name
                        if candidate_output_dir.exists():
                            st.error(tr("session_folder_error_exists"))
                            st.stop()
                        st.session_state.current_output_dir = candidate_output_dir
                    else:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        st.session_state.current_output_dir = output_root / f"run_{ts}"
                else:
                    if not st.session_state.current_output_dir or not st.session_state.current_output_dir.exists():
                        run_folders = list_saved_runs(output_root, newest_first=True)
                        if run_folders:
                            st.session_state.current_output_dir = run_folders[0]
                        else:
                            st.error(tr("no_previous_session"))
                            st.stop()

                working_output_dir = st.session_state.current_output_dir
                working_output_dir.mkdir(parents=True, exist_ok=True)

                pipeline_log_path = working_output_dir / "debug" / "pipeline.log"
                if pipeline_log_path.exists():
                    pipeline_log_path.unlink()
                pipeline_log_path.parent.mkdir(parents=True, exist_ok=True)

                if run_mode in (tr("full_pipeline"), tr("full_pipeline_no_qwen")):
                    input_audio_dir = working_output_dir / "input_audio"
                    input_audio_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        with st.status(tr("audio_norm_running"), expanded=False) as audio_status:
                            source_audio_path = Path(audio_input_path)
                            if uploaded_audio:
                                original_audio_path = input_audio_dir / f"original_{uploaded_audio.name}"
                                save_uploaded_file(uploaded_audio, original_audio_path)
                            else:
                                original_audio_path = input_audio_dir / f"original{source_audio_path.suffix.lower() or '.wav'}"
                                shutil.copy2(source_audio_path, original_audio_path)

                            normalized_audio_path = input_audio_dir / "normalized_mono16k.wav"
                            norm_meta = normalize_audio_to_mono16k(original_audio_path, normalized_audio_path)
                            (input_audio_dir / "audio_normalization_info.json").write_text(
                                json.dumps(norm_meta, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            audio_input_path = normalized_audio_path
                            append_to_debug_log(
                                f"[AUDIO] normalized source={original_audio_path} -> output={normalized_audio_path}",
                                pipeline_log_path,
                            )
                            audio_status.update(label=tr("audio_norm_done"), state="complete", expanded=False)
                    except Exception as exc:
                        st.error(tr("audio_norm_error", error=exc))
                        st.stop()

                    if st.session_state.enrollment_speakers:
                        final_enr_root = working_output_dir / "enrollment"
                        if temp_root.exists():
                            shutil.copytree(temp_root, final_enr_root, dirs_exist_ok=True)
                        final_enrollment_dir = final_enr_root

                    der_output_dir = working_output_dir / "der"
                    der_output_dir.mkdir(parents=True, exist_ok=True)
                    der_start_time = time.time()
                    try:
                        with st.status(tr("der_status_running", name=working_output_dir.name), expanded=True) as status:
                            log_placeholder = st.empty()
                            der_progress_bar = st.progress(0)
                            der_progress_text = st.empty()

                            for line in run_der_pipeline(
                                audio_path=audio_input_path,
                                enrollment_dir=final_enrollment_dir,
                                n_speakers=int(n_speakers),
                                checkpoint_path=Path(der_checkpoint) if der_checkpoint else None,
                                output_dir=der_output_dir,
                                script_path=der_script_to_use,
                                segmentation_step=der_step,
                            ):
                                append_to_debug_log(f"[DER] {line}", pipeline_log_path)
                                if line.startswith("PROGRESS:"):
                                    try:
                                        parts = line.replace("PROGRESS:", "").split("/")
                                        cur, tot = int(parts[0]), int(parts[1])
                                        percent = cur / tot
                                        elapsed = time.time() - der_start_time
                                        elapsed_str = format_time(elapsed)
                                        if cur > 0:
                                            etc = (elapsed / cur) * (tot - cur)
                                            etc_str = format_time(etc)
                                            der_progress_text.text(
                                                tr("der_progress_full", cur=cur, tot=tot, percent=int(percent * 100), elapsed=elapsed_str, remaining=etc_str)
                                            )
                                        else:
                                            der_progress_text.text(
                                                tr("der_progress_partial", cur=cur, tot=tot, percent=int(percent * 100), elapsed=elapsed_str)
                                            )
                                        der_progress_bar.progress(percent)
                                    except Exception:
                                        pass
                                else:
                                    log_placeholder.text(line)
                            status.update(label=tr("der_completed", time=format_time(time.time() - der_start_time)), state="complete", expanded=False)
                    except Exception as exc:
                        st.error(tr("der_error", error=exc))
                        st.stop()

                    rttm_files = list(der_output_dir.rglob("*.rttm"))
                    if not rttm_files:
                        st.error(tr("no_rttm"))
                        st.stop()
                    rttm_path = rttm_files[0]

                    asr_output_dir = working_output_dir / "asr"
                    asr_output_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        start_time = time.time()

                        with st.status(tr("asr_status_running"), expanded=True) as status:
                            log_placeholder = st.empty()
                            for line in run_asr_pipeline(
                                audio_path=audio_input_path,
                                rttm_path=rttm_path,
                                asr_mode="whisper_only",
                                output_dir=asr_output_dir,
                                checkpoint_path=Path(ASR_CHECKPOINT_DEFAULT),
                                use_fast_mode=asr_fast_mode,
                            ):
                                append_to_debug_log(f"[ASR] {line}", pipeline_log_path)
                                if line.startswith("PROGRESS:"):
                                    try:
                                        parts = line.replace("PROGRESS:", "").split("/")
                                        cur, tot = int(parts[0]), int(parts[1])
                                        percent = cur / tot
                                        elapsed = time.time() - start_time
                                        elapsed_str = format_time(elapsed)
                                        if cur > 0:
                                            etc = (elapsed / cur) * (tot - cur)
                                            etc_str = format_time(etc)
                                            progress_text.text(
                                                tr("asr_progress_full", cur=cur, tot=tot, percent=int(percent * 100), elapsed=elapsed_str, remaining=etc_str)
                                            )
                                        else:
                                            progress_text.text(
                                                tr("asr_progress_partial", cur=cur, tot=tot, percent=int(percent * 100), elapsed=elapsed_str)
                                            )
                                        progress_bar.progress(percent)
                                    except Exception:
                                        pass
                                else:
                                    log_placeholder.text(line)
                            status.update(label=tr("asr_completed", time=format_time(time.time() - start_time)), state="complete", expanded=False)
                        st.success(tr("asr_success"))
                    except Exception as exc:
                        st.error(tr("asr_error", error=exc))
                        st.stop()

                else:
                    asr_output_dir = st.session_state.current_output_dir / "asr"
                    target_asr = asr_output_dir / "asr_results.csv"
                    if not target_asr.exists():
                        st.error(tr("missing_asr_results", path=target_asr, name=st.session_state.current_output_dir.name))
                        st.stop()
                    st.info(tr("qwen_only_info", name=st.session_state.current_output_dir.name))

                if run_qwen:
                    qwen_output_dir = st.session_state.current_output_dir / "qwen"
                    qwen_csv_path = (st.session_state.current_output_dir / "asr" / "asr_results.csv")
                    try:
                        with st.status(tr("qwen_status_running"), expanded=True) as status:
                            log_placeholder = st.empty()
                            q_progress_bar = st.progress(0)
                            q_progress_text = st.empty()
                            q_start_time = time.time()

                            for line in _call_run_qwen_pipeline(
                                asr_csv_path=qwen_csv_path,
                                output_dir=qwen_output_dir,
                                normalize_model_name=qwen_normalize_model,
                                summary_model_name=qwen_summary_model,
                                no_4bit=qwen_disable_4bit,
                                qwen_mode=qwen_mode,
                            ):
                                append_to_debug_log(f"[QWEN] {line}", pipeline_log_path)
                                if line.startswith("PROGRESS:"):
                                    try:
                                        parts = line.replace("PROGRESS:", "").split("/")
                                        cur, tot = int(parts[0]), int(parts[1])
                                        percent = cur / tot
                                        elapsed = time.time() - q_start_time
                                        elapsed_str = format_time(elapsed)
                                        if cur > 0:
                                            etc = (elapsed / cur) * (tot - cur)
                                            etc_str = format_time(etc)
                                            q_progress_text.text(
                                                tr("qwen_progress_full", cur=cur, tot=tot, percent=int(percent * 100), elapsed=elapsed_str, remaining=etc_str)
                                            )
                                        else:
                                            q_progress_text.text(
                                                tr("qwen_progress_partial", cur=cur, tot=tot, percent=int(percent * 100), elapsed=elapsed_str)
                                            )
                                        q_progress_bar.progress(percent)
                                    except Exception:
                                        pass
                                else:
                                    log_placeholder.text(line)
                            status.update(label=tr("qwen_completed", time=format_time(time.time() - q_start_time)), state="complete", expanded=False)
                    except Exception as exc:
                        st.error(tr("qwen_error", error=exc))
                        st.stop()

                st.rerun()

    with nav_workspace:
        if st.session_state.current_output_dir and st.session_state.current_output_dir.exists():
            render_asr_workspace(st.session_state.current_output_dir, tr)
        else:
            st.info(tr("no_previous_session"))

    with nav_files:
        if st.session_state.current_output_dir and st.session_state.current_output_dir.exists():
            render_output_files_grouped(st.session_state.current_output_dir, tr)
        else:
            st.info(tr("no_previous_session"))

    with nav_history:
        opened = render_history_tab(output_root, tr)
        if opened is not None:
            st.session_state.current_output_dir = opened
            st.rerun()


if __name__ == "__main__":
    main()
