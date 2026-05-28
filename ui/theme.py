from __future__ import annotations

from typing import Callable, Dict


TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "page_title": "ViMeet",
        "app_title": "ViMeet",
        "caption": "Deep Learning Framework for Automatic Meeting Transcription in Noisy Environments End-to-end pipeline.",
        "settings": "Settings",
        "theme": "Theme",
        "language": "Language",
        "dark": "Dark",
        "light": "Light",
        "output_root": "Output root directory",
        "current_session_folder": "Current session folder: {name}",
        "saved_runs": "Saved runs",
        "open_saved_run": "Open an existing session",
        "no_saved_runs": "No saved runs were found yet.",
        "section_asr": "ASR",
        "section_der": "DER",
        "section_qwen": "Qwen",
        "section_system": "System",
        "python": "Python: `{path}`",
        "reset_session": "Reset session state",
        "audio_input_header": "Audio input",
        "upload_audio": "Upload audio file",
        "audio_local_path": "Or use a local file path",
        "audio_loaded": "Loaded: {name}",
        "audio_norm_running": "Normalizing input audio...",
        "audio_norm_done": "Audio normalized: mono / 16 kHz WAV",
        "audio_norm_error": "Audio normalization error: {error}",
        "speaker_enrollment_header": "Speaker enrollment",
        "enrollment_source": "Enrollment source",
        "enroll_none": "Do not use",
        "enroll_existing": "Use an existing local directory",
        "enroll_upload": "Upload sample audio files (multi-speaker)",
        "enroll_dir_path": "Directory path (contains one subfolder per speaker)",
        "enroll_info": "Enter a speaker name and upload at least one sample audio file for each speaker.",
        "speaker_name": "Speaker name {index}",
        "samples_for": "Samples for {name}",
        "remove": "Remove",
        "add_speaker": "Add speaker",
        "run_mode_header": "Run mode",
        "run_mode": "Processing scope",
        "full_pipeline": "Full pipeline (DER -> ASR -> Qwen)",
        "full_pipeline_no_qwen": "DER + ASR only (skip Qwen)",
        "qwen_only": "Qwen only (use existing ASR results)",
        "model_config_header": "Model configuration",
        "diarization_engine": "Diarization engine",
        "der_checkpoint": "DER checkpoint (.pth)",
        "max_speakers": "Maximum speakers",
        "run_pipeline": "Run pipeline",
        "audio_required": "Please upload an audio file or provide a local audio path before running the pipeline.",
        "no_previous_session": "No previous session was found. Please run the full pipeline first.",
        "asr_batch_size": "ASR batch size",
        "asr_batch_help": "Number of audio segments processed together.",
        "asr_fast_mode": "ASR fast mode (greedy)",
        "asr_fast_help": "Use num_beams=1. Faster, with a small quality tradeoff.",
        "der_step": "DER step size",
        "der_step_help": "Higher values run faster but reduce temporal resolution.",
        "run_qwen": "Run Qwen summarization",
        "qwen_normalizer": "Qwen normalizer model",
        "qwen_summarizer": "Qwen summarizer model",
        "qwen_disable_4bit": "Disable 4-bit mode",
        "qwen_mode": "Qwen mode",
        "qwen_mode_help": "Stable uses more conservative settings. Fast prioritizes speed.",
        "qwen_mode_stable": "Stable",
        "qwen_mode_fast": "Fast",
        "der_engine_diarizen": "DiariZen (Golden v3)",
        "der_engine_pyannote": "Pyannote (Finetuned)",
        "nav_run": "Run",
        "nav_workspace": "Workspace",
        "nav_files": "Files",
        "nav_history": "History",
        "runtime_card": "Runtime configuration",
        "execution_card": "Execution summary",
        "selected_run": "Selected session",
        "run_card_audio": "Audio",
        "run_card_enrollment": "Enrollment",
        "run_card_mode": "Mode",
        "run_card_qwen": "Qwen",
        "run_card_yes": "Yes",
        "run_card_no": "No",
        "config_note": "UI-only changes. Backend and pipeline behavior stay intact.",
        "der_status_running": "Running diarization in {name}...",
        "der_progress_full": "DER: {cur}/{tot} ({percent}%) | Elapsed: {elapsed} | Remaining: {remaining}",
        "der_progress_partial": "DER: {cur}/{tot} ({percent}%) | Elapsed: {elapsed}",
        "der_completed": "DER completed. Total time: {time}",
        "der_error": "DER error: {error}",
        "no_rttm": "No RTTM output file was found.",
        "asr_status_running": "Running ASR...",
        "asr_progress_full": "ASR: {cur}/{tot} ({percent}%) | Elapsed: {elapsed} | Remaining: {remaining}",
        "asr_progress_partial": "ASR: {cur}/{tot} ({percent}%) | Elapsed: {elapsed}",
        "asr_completed": "ASR completed. Total time: {time}",
        "asr_success": "ASR completed successfully.",
        "asr_error": "ASR error: {error}",
        "missing_asr_results": "ASR results were not found at {path}. Session: {name}",
        "qwen_only_info": "Mode: Qwen only. Using data from session: {name}",
        "qwen_status_running": "Running Qwen summarization...",
        "qwen_progress_full": "Qwen: {cur}/{tot} ({percent}%) | Elapsed: {elapsed} | Remaining: {remaining}",
        "qwen_progress_partial": "Qwen: {cur}/{tot} ({percent}%) | Elapsed: {elapsed}",
        "qwen_completed": "Qwen completed. Total time: {time}",
        "qwen_summary_results": "Qwen summary results",
        "qwen_error": "Qwen error: {error}",
        "workspace_header": "Final ASR workspace",
        "summary_tab": "Summary",
        "mindmap_tab": "Mind Map",
        "chatbot_tab": "Chatbot",
        "chatbot_api_key_label": "Gemini API Key",
        "chatbot_api_key_missing": "Please enter your Gemini API key to use the chatbot.",
        "chatbot_no_source": "No transcript or summary file found for this session.",
        "chatbot_source_label": "Data source",
        "chatbot_source_transcript": "ASR Transcript",
        "chatbot_source_summary": "Summary JSON",
        "chatbot_building_index": "Building embedding index... This may take a few minutes due to API quota limits.",
        "chatbot_preparing": "Preparing...",
        "chatbot_embedding": "Embedding",
        "chatbot_waiting_quota": "Waiting for quota reset before next batch...",
        "chatbot_done": "Done!",
        "chatbot_ready": "Chatbot is ready. Ask a question below.",
        "chatbot_load_error": "Error loading file",
        "chatbot_reset": "Reset chat",
        "chatbot_input_placeholder": "Ask a question about the meeting...",
        "chatbot_thinking": "Finding answer...",
        "chatbot_not_started": "Chatbot is not active. Click Start to build the index and enable chat.",
        "chatbot_start": "Start Chatbot",
        "chatbot_stop": "Stop & Clear",
        "chatbot_backend_label": "LLM Backend",
        "chatbot_backend_gemini": "Gemini API",
        "chatbot_backend_local": "Local (Qwen2.5-1.5B)",
        "chatbot_local_note": "Running local model — no API key needed, no quota limits.",
        "transcript_tab": "Transcript",
        "topics_heading": "Topics",
        "assessment_heading": "Assessment",
        "meeting_overview_label": "Meeting overview",
        "conversation_main_label": "Conversation main summary",
        "speaker_focus_label": "Speaker focus",
        "no_summary_data": "No Qwen summary data is available yet. The transcript can still be reviewed below.",
        "no_transcript_data": "No ASR transcript is available yet.",
        "mindmap_root": "Meeting",
        "mindmap_segments": "Segments",
        "mindmap_speakers": "Speakers",
        "transcript_source": "ASR final output",
        "plain_transcript_copy_hint": "Plain-text transcript view",
        "workspace_metric_turns": "ASR turns",
        "workspace_metric_speakers": "Speakers",
        "workspace_metric_last_time": "Last timestamp",
        "workspace_metric_qwen": "Qwen summary",
        "output_files": "Output files",
        "download_file": "Download {name}",
        "no_output_files": "No output files are available in this session yet.",
        "binary_preview_unavailable": "[Binary file preview is not available]",
        "history_header": "Saved sessions",
        "history_empty": "No saved sessions are available.",
        "history_run_name": "Session",
        "history_updated": "Updated",
        "history_files": "Files",
        "history_open": "Open session",
        "history_status": "Status",
        "history_has_der": "DER",
        "history_has_asr": "ASR",
        "history_has_qwen": "Qwen",
        "group_root": "Root files",
        "history_filter": "Filter",
        "history_filter_all": "All",
        "history_filter_complete": "Complete",
        "history_filter_partial": "Partial / Failed",
        "history_sort": "Sort",
        "history_sort_newest": "Newest first",
        "history_sort_oldest": "Oldest first",
        "history_status_complete": "Complete",
        "history_status_partial": "Partial",
        "history_badge_complete": "Complete",
        "history_badge_partial": "Partial",
        "history_quick_summary": "Quick status",
        "history_counts": "Counts",
        "run_section_input": "Input",
        "run_section_options": "Processing options",
        "session_folder_name": "Session folder name",
        "session_folder_help": "Optional. Used only when starting a new full pipeline session.",
        "session_folder_placeholder": "Example: defense_demo_run_01",
        "session_folder_note_default": "Leave empty to use an automatic timestamp folder.",
        "session_folder_error_empty": "Please enter a valid folder name.",
        "session_folder_error_exists": "This folder already exists. Please choose another name.",
        "qwen_skip_note": "Qwen will be skipped for this run.",
        "history_artifacts": "Artifacts",
        "history_artifacts_line": "DER: {der} | ASR: {asr} | Qwen: {qwen}",
        "history_files_count": "Files",
    },
    "vi": {
        "page_title": "DiariZen Pipeline Pro",
        "app_title": "DiariZen Pipeline Pro",
        "caption": "Giao diện Streamlit đã refactor. Giữ nguyên backend và logic pipeline hiện tại.",
        "settings": "Cài đặt",
        "theme": "Giao diện",
        "language": "Ngôn ngữ",
        "dark": "Tối",
        "light": "Sáng",
        "output_root": "Thư mục gốc đầu ra",
        "current_session_folder": "Thư mục phiên hiện tại: {name}",
        "saved_runs": "Các phiên đã lưu",
        "open_saved_run": "Mở phiên chạy cũ",
        "no_saved_runs": "Chưa có phiên chạy nào.",
        "section_asr": "ASR",
        "section_der": "DER",
        "section_qwen": "Qwen",
        "section_system": "Hệ thống",
        "python": "Python: `{path}`",
        "reset_session": "Đặt lại session state",
        "audio_input_header": "Đầu vào audio",
        "upload_audio": "Tải file audio",
        "audio_local_path": "Hoặc dùng đường dẫn file cục bộ",
        "audio_loaded": "Đã nạp: {name}",
        "audio_norm_running": "Đang chuẩn hóa audio đầu vào...",
        "audio_norm_done": "Đã chuẩn hóa audio: mono / 16 kHz WAV",
        "audio_norm_error": "Lỗi chuẩn hóa audio: {error}",
        "speaker_enrollment_header": "Đăng ký người nói",
        "enrollment_source": "Nguồn enrollment",
        "enroll_none": "Không dùng",
        "enroll_existing": "Dùng thư mục cục bộ có sẵn",
        "enroll_upload": "Tải mẫu audio lên (nhiều người nói)",
        "enroll_dir_path": "Đường dẫn thư mục (mỗi người nói là một thư mục con)",
        "enroll_info": "Nhập tên speaker và tải ít nhất một file mẫu cho mỗi người.",
        "speaker_name": "Tên speaker {index}",
        "samples_for": "Mẫu cho {name}",
        "remove": "Xóa",
        "add_speaker": "Thêm speaker",
        "run_mode_header": "Chế độ chạy",
        "run_mode": "Phạm vi xử lý",
        "full_pipeline": "Toàn bộ pipeline (DER -> ASR -> Qwen)",
        "full_pipeline_no_qwen": "Chỉ DER + ASR (bỏ qua Qwen)",
        "qwen_only": "Chỉ Qwen (dùng kết quả ASR sẵn có)",
        "model_config_header": "Cấu hình model",
        "diarization_engine": "Engine diarization",
        "der_checkpoint": "Checkpoint DER (.pth)",
        "max_speakers": "Số người tối đa",
        "run_pipeline": "Chạy pipeline",
        "audio_required": "Vui lòng tải file audio lên hoặc nhập đường dẫn audio cục bộ trước khi chạy pipeline.",
        "no_previous_session": "Không tìm thấy phiên chạy trước đó. Vui lòng chạy full pipeline trước.",
        "asr_batch_size": "ASR batch size",
        "asr_batch_help": "Số lượng đoạn âm thanh xử lý cùng lúc.",
        "asr_fast_mode": "ASR fast mode (greedy)",
        "asr_fast_help": "Dùng num_beams=1. Nhanh hơn, có thể giảm nhẹ chất lượng.",
        "der_step": "DER step size",
        "der_step_help": "Giá trị cao hơn chạy nhanh hơn nhưng giảm độ phân giải thời gian.",
        "run_qwen": "Chạy Qwen tóm tắt",
        "qwen_normalizer": "Model Qwen normalizer",
        "qwen_summarizer": "Model Qwen summarizer",
        "qwen_disable_4bit": "Tắt chế độ 4-bit",
        "qwen_mode": "Chế độ Qwen",
        "qwen_mode_help": "Stable dùng cấu hình bảo thủ hơn. Fast ưu tiên tốc độ.",
        "qwen_mode_stable": "Stable",
        "qwen_mode_fast": "Fast",
        "der_engine_diarizen": "DiariZen (Golden v3)",
        "der_engine_pyannote": "Pyannote (Finetuned)",
        "nav_run": "Chạy",
        "nav_workspace": "Không gian",
        "nav_files": "Tệp",
        "nav_history": "Lịch sử",
        "runtime_card": "Cấu hình chạy",
        "execution_card": "Tóm tắt thực thi",
        "selected_run": "Phiên đang chọn",
        "run_card_audio": "Audio",
        "run_card_enrollment": "Enrollment",
        "run_card_mode": "Mode",
        "run_card_qwen": "Qwen",
        "run_card_yes": "Có",
        "run_card_no": "Không",
        "config_note": "Chỉ thay đổi UI. Backend và pipeline giữ nguyên.",
        "der_status_running": "Đang chạy diarization trong {name}...",
        "der_progress_full": "DER: {cur}/{tot} ({percent}%) | Đã chạy: {elapsed} | Còn lại: {remaining}",
        "der_progress_partial": "DER: {cur}/{tot} ({percent}%) | Đã chạy: {elapsed}",
        "der_completed": "DER hoàn tất. Tổng thời gian: {time}",
        "der_error": "Lỗi DER: {error}",
        "no_rttm": "Không tìm thấy file RTTM output.",
        "asr_status_running": "Đang chạy ASR...",
        "asr_progress_full": "ASR: {cur}/{tot} ({percent}%) | Đã chạy: {elapsed} | Còn lại: {remaining}",
        "asr_progress_partial": "ASR: {cur}/{tot} ({percent}%) | Đã chạy: {elapsed}",
        "asr_completed": "ASR hoàn tất. Tổng thời gian: {time}",
        "asr_success": "ASR chạy thành công.",
        "asr_error": "Lỗi ASR: {error}",
        "missing_asr_results": "Không tìm thấy kết quả ASR tại {path}. Phiên: {name}",
        "qwen_only_info": "Chế độ: Chỉ Qwen. Dùng dữ liệu từ phiên: {name}",
        "qwen_status_running": "Đang chạy Qwen tóm tắt...",
        "qwen_progress_full": "Qwen: {cur}/{tot} ({percent}%) | Đã chạy: {elapsed} | Còn lại: {remaining}",
        "qwen_progress_partial": "Qwen: {cur}/{tot} ({percent}%) | Đã chạy: {elapsed}",
        "qwen_completed": "Qwen hoàn tất. Tổng thời gian: {time}",
        "qwen_summary_results": "Kết quả tóm tắt Qwen",
        "qwen_error": "Lỗi Qwen: {error}",
        "workspace_header": "Không gian kết quả ASR cuối",
        "summary_tab": "Tóm tắt",
        "mindmap_tab": "Sơ đồ ý",
        "chatbot_tab": "Chatbot",
        "chatbot_api_key_label": "Gemini API Key",
        "chatbot_api_key_missing": "Vui lòng nhập Gemini API key để dùng chatbot.",
        "chatbot_no_source": "Không tìm thấy transcript hoặc summary cho phiên này.",
        "chatbot_source_label": "Nguồn dữ liệu",
        "chatbot_source_transcript": "ASR Transcript",
        "chatbot_source_summary": "Summary JSON",
        "chatbot_building_index": "Đang tạo embedding index... Có thể mất vài phút do giới hạn quota API.",
        "chatbot_preparing": "Chuẩn bị...",
        "chatbot_embedding": "Đang embed",
        "chatbot_waiting_quota": "Đang đợi quota reset trước batch tiếp theo...",
        "chatbot_done": "Hoàn thành!",
        "chatbot_ready": "Chatbot sẵn sàng. Hãy đặt câu hỏi bên dưới.",
        "chatbot_load_error": "Lỗi khi load file",
        "chatbot_reset": "Xóa lịch sử chat",
        "chatbot_input_placeholder": "Đặt câu hỏi về cuộc họp...",
        "chatbot_thinking": "Đang tìm câu trả lời...",
        "chatbot_not_started": "Chatbot chưa được kích hoạt. Bấm Start để tạo index và bắt đầu chat.",
        "chatbot_start": "Start Chatbot",
        "chatbot_stop": "Dừng & Xoá",
        "chatbot_backend_label": "LLM Backend",
        "chatbot_backend_gemini": "Gemini API",
        "chatbot_backend_local": "Local (Qwen2.5-1.5B)",
        "chatbot_local_note": "Đang dùng model local — không cần API key, không bị giới hạn quota.",
        "transcript_tab": "Transcript",
        "topics_heading": "Chủ đề",
        "assessment_heading": "Đánh giá",
        "meeting_overview_label": "Tổng quan cuộc họp",
        "conversation_main_label": "Tóm tắt nội dung chính",
        "speaker_focus_label": "Trọng tâm theo người nói",
        "no_summary_data": "Chưa có dữ liệu tóm tắt từ Qwen. Bạn vẫn có thể xem transcript bên dưới.",
        "no_transcript_data": "Chưa có transcript ASR.",
        "mindmap_root": "Cuộc họp",
        "mindmap_segments": "Các đoạn",
        "mindmap_speakers": "Người nói",
        "transcript_source": "Đầu ra ASR cuối",
        "plain_transcript_copy_hint": "Transcript plain-text",
        "workspace_metric_turns": "Lượt ASR",
        "workspace_metric_speakers": "Người nói",
        "workspace_metric_last_time": "Mốc cuối",
        "workspace_metric_qwen": "Qwen summary",
        "output_files": "Các tệp đầu ra",
        "download_file": "Tải {name}",
        "no_output_files": "Chưa có file đầu ra nào.",
        "binary_preview_unavailable": "[Không thể xem trước file nhị phân]",
        "history_header": "Các phiên đã lưu",
        "history_empty": "Chưa có phiên chạy nào.",
        "history_run_name": "Phiên",
        "history_updated": "Cập nhật",
        "history_files": "Tệp",
        "history_open": "Mở phiên",
        "history_status": "Trạng thái",
        "history_has_der": "DER",
        "history_has_asr": "ASR",
        "history_has_qwen": "Qwen",
        "group_root": "Tệp gốc",
        "history_filter": "Bộ lọc",
        "history_filter_all": "Tất cả",
        "history_filter_complete": "Hoàn chỉnh",
        "history_filter_partial": "Thiếu / lỗi",
        "history_sort": "Sắp xếp",
        "history_sort_newest": "Mới nhất trước",
        "history_sort_oldest": "Cũ nhất trước",
        "history_status_complete": "Hoàn chỉnh",
        "history_status_partial": "Thiếu",
        "history_badge_complete": "Hoàn chỉnh",
        "history_badge_partial": "Thiếu",
        "history_quick_summary": "Trạng thái nhanh",
        "history_counts": "Số lượng",
        "run_section_input": "Đầu vào",
        "run_section_options": "Tùy chọn xử lý",
        "session_folder_name": "Tên thư mục phiên chạy",
        "session_folder_help": "Tùy chọn. Chỉ dùng khi bắt đầu một phiên full pipeline mới.",
        "session_folder_placeholder": "Ví dụ: bao_ve_do_an_demo_01",
        "session_folder_note_default": "Để trống nếu muốn dùng thư mục timestamp tự động.",
        "session_folder_error_empty": "Vui lòng nhập tên thư mục hợp lệ.",
        "session_folder_error_exists": "Thư mục này đã tồn tại. Vui lòng chọn tên khác.",
        "qwen_skip_note": "Qwen sẽ bị bỏ qua trong lần chạy này.",
        "history_artifacts": "Artifacts",
        "history_artifacts_line": "DER: {der} | ASR: {asr} | Qwen: {qwen}",
        "history_files_count": "Số tệp",
    },
}


def get_translator(lang: str) -> Callable[..., str]:
    def tr(key: str, **kwargs) -> str:
        template = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
        return template.format(**kwargs) if kwargs else template
    return tr


def get_theme_css(theme_mode: str) -> str:
    if theme_mode == "Dark":
        # VS Code Dark+ inspired palette
        bg = "#1e1e1e"
        secondary_bg = "#252526"
        tertiary_bg = "#2d2d30"
        border = "#3c3c3c"
        text = "#d4d4d4"
        muted = "#969696"
        accent = "#0e639c"
        accent2 = "#1177bb"
        soft = "#333333"
        input_bg = "#3c3c3c"
        success = "#388a34"
    else:
        # VS Code Light inspired palette
        bg = "#f3f3f3"
        secondary_bg = "#ffffff"
        tertiary_bg = "#f8f8f8"
        border = "#e5e5e5"
        text = "#333333"
        muted = "#616161"
        accent = "#007acc"
        accent2 = "#005fb8"
        soft = "#f0f6ff"
        input_bg = "#ffffff"
        success = "#dff3e4"

    return f"""
    <style>
    :root {{
        --app-bg: {bg};
        --app-secondary-bg: {secondary_bg};
        --app-tertiary-bg: {tertiary_bg};
        --app-border: {border};
        --app-text: {text};
        --app-muted: {muted};
        --app-accent: {accent};
        --app-accent2: {accent2};
        --app-soft: {soft};
        --app-input-bg: {input_bg};
        --app-success: {success};
    }}

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: var(--app-bg);
        color: var(--app-text);
    }}

    [data-testid="stSidebar"] {{
        background: var(--app-secondary-bg);
        border-right: 1px solid var(--app-border);
    }}

    .block-container {{
        max-width: 1280px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }}

    h1, h2, h3, h4, h5, h6, p, label, div, span {{
        color: var(--app-text);
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea, input {{
        background: var(--app-input-bg) !important;
        color: var(--app-text) !important;
        border-color: var(--app-border) !important;
    }}

    .hero {{
        background: linear-gradient(135deg, var(--app-secondary-bg) 0%, var(--app-soft) 100%);
        border: 1px solid var(--app-border);
        border-radius: 28px;
        padding: 1.4rem 1.35rem 1.15rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 16px 50px rgba(0,0,0,0.14);
    }}

    .hero-title {{
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }}

    .hero-caption {{
        color: var(--app-muted);
        font-size: 0.98rem;
    }}

    .section-label {{
        font-size: 0.82rem;
        color: var(--app-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
        font-weight: 700;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
    }}

    .metric-box {{
        background: var(--app-secondary-bg);
        border: 1px solid var(--app-border);
        border-radius: 18px;
        padding: 0.9rem 1rem;
    }}

    .metric-label {{
        color: var(--app-muted);
        font-size: 0.88rem;
        margin-bottom: 0.3rem;
    }}

    .metric-value {{
        font-size: 1.2rem;
        font-weight: 800;
    }}

    .transcript-shell {{
        background: var(--app-secondary-bg);
        border: 1px solid var(--app-border);
        border-radius: 20px;
        padding: 0.75rem;
    }}

    .file-group-title {{
        font-weight: 700;
        margin: 0.3rem 0 0.75rem 0;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--app-secondary-bg);
        border: 1px solid var(--app-border) !important;
        border-radius: 22px !important;
        padding: 0.2rem 0.2rem;
    }}

    div.stButton > button,
    [data-testid="stDownloadButton"] > button {{
        border-radius: 999px;
        font-weight: 700;
    }}

    div.stButton > button[kind="primary"] {{
        background: var(--app-accent);
        border-color: var(--app-accent);
    }}

    .tiny-muted {{
        color: var(--app-muted);
        font-size: 0.90rem;
    }}

    .summary-card {{
        background: var(--app-secondary-bg);
        border: 1px solid var(--app-border);
        border-radius: 20px;
        padding: 1rem;
    }}

    .summary-card-title {{
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }}

    .status-badge {{
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        border: 1px solid var(--app-border);
        margin-right: 0.35rem;
    }}

    .status-complete {{
        background: color-mix(in srgb, var(--app-success) 80%, transparent);
    }}

    .status-partial {{
        background: color-mix(in srgb, var(--app-soft) 90%, transparent);
    }}

    .run-row {{
        background: var(--app-secondary-bg);
        border: 1px solid var(--app-border);
        border-radius: 18px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.7rem;
    }}

    [data-testid="stStatusWidget"], [data-testid="stExpander"], [data-testid="stInfo"],
    [data-testid="stSuccess"], [data-testid="stWarning"], [data-testid="stError"] {{
        border-radius: 16px;
    }}
    </style>
    """