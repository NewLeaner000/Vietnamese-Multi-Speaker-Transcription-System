from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from .components import format_time


_MAX_EMBED_AUDIO_BYTES = 85 * 1024 * 1024


def read_asr_rows(csv_path: Path):
    rows = []
    if not csv_path.exists():
        return rows
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("predicted_text") or "").strip()
                if not text:
                    continue
                try:
                    start = float(row.get("start", 0.0))
                    end = float(row.get("end", start))
                except Exception:
                    continue
                speaker = (row.get("speaker") or row.get("data_name") or "Speaker").strip()
                rows.append({
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                    "audio_path": (row.get("audio_path") or "").strip(),
                })
    except Exception:
        return []
    return rows


def transcript_to_plaintext(transcript_rows: list) -> str:
    plain_lines = []
    for row in transcript_rows:
        time_label = f"{int(row['start']//60):02d}:{int(row['start']%60):02d}"
        plain_lines.append(f"{time_label} | {row['speaker']} | {row['text']}")
    return "\n".join(plain_lines)


def _find_workspace_audio_path(display_dir: Path | None, transcript_rows: list) -> Path | None:
    candidates: list[Path] = []
    if display_dir is not None:
        candidates.extend([
            display_dir / "input_audio" / "normalized_mono16k.wav",
            display_dir / "asr" / "_engine_input" / "data0001" / "mixture.wav",
        ])

    for row in transcript_rows:
        audio_path = (row.get("audio_path") or "").strip()
        if audio_path:
            candidates.append(Path(audio_path))

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


def _audio_to_data_uri(audio_path: Path) -> str | None:
    try:
        if audio_path.stat().st_size > _MAX_EMBED_AUDIO_BYTES:
            return None
        suffix = audio_path.suffix.lower()
        mime = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
        }.get(suffix, "audio/wav")
        payload = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"
    except Exception:
        return None


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def _theme_tokens() -> dict:
    theme_mode = st.session_state.get("theme_mode", "Dark")
    if theme_mode == "Dark":
        return {
            "bg": "#252526",
            "row_bg": "#2d2d30",
            "row_active": "#333333",
            "border": "#3c3c3c",
            "text": "#d4d4d4",
            "muted": "#969696",
            "accent": "#0e639c",
            "accent2": "#1177bb",
        }
    return {
        "bg": "#ffffff",
        "row_bg": "#f8f8f8",
        "row_active": "#f0f6ff",
        "border": "#e5e5e5",
        "text": "#333333",
        "muted": "#616161",
        "accent": "#007acc",
        "accent2": "#005fb8",
    }


def _build_interactive_transcript_html(transcript_rows: list, audio_data_uri: str) -> str:
    tokens = _theme_tokens()
    rows = []
    for idx, row in enumerate(transcript_rows):
        start = max(0.0, float(row.get("start", 0.0)))
        end = max(start, float(row.get("end", start)))
        rows.append({
            "idx": idx,
            "start": start,
            "end": end,
            "time": f"{_format_timestamp(start)} - {_format_timestamp(end)}",
            "speaker": row.get("speaker") or "Speaker",
            "text": row.get("text") or "",
        })

    rows_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    audio_json = json.dumps(audio_data_uri).replace("</", "<\\/")
    tokens_json = json.dumps(tokens).replace("</", "<\\/")

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root {{
    --bg: {html.escape(tokens['bg'])};
    --row-bg: {html.escape(tokens['row_bg'])};
    --row-active: {html.escape(tokens['row_active'])};
    --border: {html.escape(tokens['border'])};
    --text: {html.escape(tokens['text'])};
    --muted: {html.escape(tokens['muted'])};
    --accent: {html.escape(tokens['accent'])};
    --accent2: {html.escape(tokens['accent2'])};
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    color: var(--text);
    font-family: "Source Sans Pro", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .transcript-player {{
    height: 840px;
    overflow-y: auto;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 12px;
  }}
  .segment-row {{
    position: relative;
    display: grid;
    grid-template-columns: 98px 1fr;
    gap: 12px;
    padding: 12px 12px;
    margin-bottom: 10px;
    background: var(--row-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
  }}
  .segment-row:hover {{
    background: var(--row-active);
    border-color: var(--accent);
  }}
  .segment-row.active {{
    background: var(--row-active) !important;
    border-color: var(--accent) !important;
    box-shadow: inset 4px 0 0 var(--accent), 0 0 0 1px var(--accent);
  }}
  .segment-row.active .segment-text-highlight {{
    background: rgba(14, 99, 156, 0.35);
    border-radius: 8px;
    padding: 2px 5px;
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
  }}
  .segment-left {{
    min-width: 0;
  }}
  .time-label {{
    color: var(--muted);
    font-size: 12px;
    line-height: 1.3;
    white-space: nowrap;
    margin-bottom: 8px;
  }}
  .speaker-label {{
    color: var(--text);
    font-weight: 700;
    font-size: 13px;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .play-button {{
    opacity: 0;
    margin-top: 10px;
    height: 28px;
    min-width: 62px;
    border-radius: 999px;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: #ffffff;
    font-size: 12px;
    font-weight: 800;
    cursor: pointer;
    transition: opacity 0.12s ease, background 0.12s ease;
  }}
  .segment-row:hover .play-button,
  .segment-row.active .play-button {{
    opacity: 1;
  }}
  .play-button:hover {{ background: var(--accent2); }}
  .segment-text {{
    min-width: 0;
    font-size: 15px;
    line-height: 1.75;
    padding-right: 4px;
  }}
  .segment-text-highlight {{
    display: inline;
    transition: background 0.12s ease;
  }}
  .progress-line {{
    position: absolute;
    left: 0;
    bottom: 0;
    width: 0%;
    height: 3px;
    border-radius: 0 0 0 16px;
    background: var(--accent);
  }}
  .hint {{
    color: var(--muted);
    font-size: 12px;
    margin: 0 0 10px 4px;
  }}
</style>
</head>
<body>
<div class="transcript-player" id="transcriptPlayer">
  <div class="hint">Hover a segment to show Play. The active segment is highlighted while playback is running.</div>
  <audio id="workspaceAudio" preload="metadata" src=""></audio>
  <div id="rows"></div>
</div>
<script>
const rows = {rows_json};
const audioSrc = {audio_json};
const audio = document.getElementById('workspaceAudio');
const rowsRoot = document.getElementById('rows');
audio.src = audioSrc;
let activeIdx = null;
let stopTimer = null;

function escapeHtml(value) {{
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}}

function buildRows() {{
  rowsRoot.innerHTML = rows.map((row) => {{
    return `
      <div class="segment-row" id="segment-${{row.idx}}" data-idx="${{row.idx}}">
        <div class="segment-left">
          <div class="time-label">${{escapeHtml(row.time)}}</div>
          <div class="speaker-label">${{escapeHtml(row.speaker)}}</div>
          <button class="play-button" type="button" title="Play this segment">▶ Play</button>
        </div>
        <div class="segment-text"><span class="segment-text-highlight">${{escapeHtml(row.text)}}</span></div>
        <div class="progress-line"></div>
      </div>`;
  }}).join('');

  for (const el of document.querySelectorAll('.segment-row')) {{
    const idx = Number(el.dataset.idx);
    const button = el.querySelector('.play-button');
    button.addEventListener('click', (event) => {{
      event.preventDefault();
      event.stopPropagation();
      playSegment(idx);
    }});
    el.addEventListener('dblclick', () => playSegment(idx));
  }}
}}

function resetHighlights() {{
  for (const el of document.querySelectorAll('.segment-row')) {{
    el.classList.remove('active');
    el.style.background = '';
    el.style.borderColor = '';
    el.style.boxShadow = '';
    const textHighlight = el.querySelector('.segment-text-highlight');
    if (textHighlight) {{
      textHighlight.style.background = '';
      textHighlight.style.borderRadius = '';
      textHighlight.style.padding = '';
      textHighlight.style.webkitBoxDecorationBreak = '';
      textHighlight.style.boxDecorationBreak = '';
    }}
    const progress = el.querySelector('.progress-line');
    if (progress) progress.style.width = '0%';
    const button = el.querySelector('.play-button');
    if (button) button.textContent = '▶ Play';
  }}
}}

function applyActiveSegment(rowEl) {{
  if (!rowEl) return;
  rowEl.classList.add('active');
  rowEl.style.background = 'var(--row-active)';
  rowEl.style.borderColor = 'var(--accent)';
  rowEl.style.boxShadow = 'inset 4px 0 0 var(--accent), 0 0 0 1px var(--accent)';
  const textHighlight = rowEl.querySelector('.segment-text-highlight');
  if (textHighlight) {{
    textHighlight.style.background = 'rgba(14, 99, 156, 0.35)';
    textHighlight.style.borderRadius = '8px';
    textHighlight.style.padding = '2px 5px';
    textHighlight.style.webkitBoxDecorationBreak = 'clone';
    textHighlight.style.boxDecorationBreak = 'clone';
  }}
}}

function playSegment(idx) {{
  if (activeIdx === idx && !audio.paused) {{
    stopPlayback();
    return;
  }}
  const row = rows[idx];
  if (!row) return;
  if (stopTimer) clearTimeout(stopTimer);
  resetHighlights();
  activeIdx = idx;
  const rowEl = document.getElementById(`segment-${{idx}}`);
  applyActiveSegment(rowEl);
  const button = rowEl.querySelector('.play-button');
  if (button) button.textContent = '⏸ Stop';
  audio.pause();
  audio.currentTime = row.start;
  const playPromise = audio.play();
  if (playPromise && typeof playPromise.catch === 'function') {{
    playPromise.catch(() => {{
      // Keep the selected segment highlighted so the UI still confirms the clicked segment.
      // Do not clear here; some browsers briefly reject play while metadata is loading.
      applyActiveSegment(rowEl);
    }});
  }}
  stopTimer = setTimeout(() => stopPlayback(), Math.max(0, (row.end - row.start) * 1000 + 160));
}}

function stopPlayback() {{
  audio.pause();
  activeIdx = null;
  if (stopTimer) clearTimeout(stopTimer);
  stopTimer = null;
  resetHighlights();
}}

function updateHighlight() {{
  if (activeIdx === null) return;
  const row = rows[activeIdx];
  if (!row) return;
  const rowEl = document.getElementById(`segment-${{activeIdx}}`);
  if (!rowEl) return;

  const duration = Math.max(0.001, row.end - row.start);
  const elapsed = Math.max(0, Math.min(duration, audio.currentTime - row.start));
  const progressPct = Math.max(0, Math.min(100, elapsed / duration * 100));
  const progress = rowEl.querySelector('.progress-line');
  if (progress) progress.style.width = `${{progressPct}}%`;


  if (audio.currentTime >= row.end) stopPlayback();
}}

audio.addEventListener('timeupdate', updateHighlight);
audio.addEventListener('pause', () => {{
  if (activeIdx !== null) updateHighlight();
}});
audio.addEventListener('ended', stopPlayback);

buildRows();
</script>
</body>
</html>
"""


def render_transcript_panel(transcript_rows: list, tr, display_dir: Path | None = None):
    if not transcript_rows:
        st.info(tr("no_transcript_data"))
        return

    st.markdown(f"**{tr('transcript_tab')}**")
    st.caption(tr("transcript_source"))

    audio_path = _find_workspace_audio_path(display_dir, transcript_rows)
    audio_data_uri = _audio_to_data_uri(audio_path) if audio_path else None

    if audio_data_uri:
        html_doc = _build_interactive_transcript_html(transcript_rows, audio_data_uri)
        components.html(html_doc, height=880, scrolling=False)
        return

    transcript_text = transcript_to_plaintext(transcript_rows)
    st.text_area(
        label=tr("transcript_tab"),
        value=transcript_text,
        height=840,
        disabled=True,
        label_visibility="collapsed",
    )


def render_workspace_metrics(transcript_rows: list, summary_data: dict, tr):
    if not transcript_rows and not summary_data:
        return

    speaker_count = len({row["speaker"] for row in transcript_rows}) if transcript_rows else 0
    last_ts = format_time(max((row["end"] for row in transcript_rows), default=0.0))
    qwen_available = tr("run_card_yes") if summary_data else tr("run_card_no")

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">{tr("workspace_metric_turns")}</div>
                <div class="metric-value">{len(transcript_rows)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">{tr("workspace_metric_speakers")}</div>
                <div class="metric-value">{speaker_count}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">{tr("workspace_metric_last_time")}</div>
                <div class="metric-value">{last_ts}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">{tr("workspace_metric_qwen")}</div>
                <div class="metric-value">{qwen_available}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
