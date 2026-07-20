import os
import io
import queue
import shutil
import subprocess
import tempfile
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

MODEL_PATH = "PPE-YOLOv8-best.pt"
HISTORY_FILE = "detection_history.csv"
HISTORY_COLUMNS = ["timestamp", "filename", "detections", "compliance_pct", "violations"]

st.set_page_config(
    page_title="IronSight Industrial Safety Intelligence Hub",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #0c0e10;
    --bg-2: #15181d;
    --panel: rgba(21, 24, 29, 0.85);
    --panel-border: rgba(251, 191, 36, 0.18);
    --accent: #fbbf24;      /* hazard yellow */
    --accent-2: #f97316;    /* high-vis orange */
    --good: #059669;
    --warn: #f59e0b;
    --bad: #dc2626;
    --text: #e8e6e3;
    --muted: #9aa0a6;
}

html, body, [class*="stApp"] {
    background:
        radial-gradient(1100px 550px at 15% -10%, rgba(251, 191, 36, 0.12), transparent 55%),
        radial-gradient(900px 500px at 85% 0%, rgba(249, 115, 22, 0.09), transparent 60%),
        radial-gradient(800px 600px at 50% 115%, rgba(251, 191, 36, 0.06), transparent 60%),
        linear-gradient(rgba(251, 191, 36, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(251, 191, 36, 0.03) 1px, transparent 1px),
        var(--bg);
    background-size: auto, auto, auto, 44px 44px, 44px 44px, auto;
    color: var(--text);
}

header { visibility: hidden; }
footer { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14161a 0%, #0c0e10 100%);
    border-right: 1px solid rgba(251, 191, 36, 0.18);
    box-shadow: inset -14px 0 28px rgba(0, 0, 0, 0.35);
}

.hazard-tape {
    height: 10px;
    border-radius: 999px;
    background: repeating-linear-gradient(-45deg, #fbbf24 0 14px, #16181c 14px 28px);
    box-shadow: 0 0 14px rgba(251, 191, 36, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.25);
    margin: 0.4rem 0 1rem 0;
    animation: tape-crawl 18s linear infinite;
}

@keyframes tape-crawl {
    to { background-position: 396px 0; }
}

.app-hero {
    position: relative;
    overflow: hidden;
    padding: 1.8rem 2.2rem;
    border-radius: 18px;
    border: 1px solid var(--panel-border);
    border-left: 6px solid var(--accent);
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.05), transparent 40%),
                linear-gradient(120deg, rgba(251, 191, 36, 0.10), rgba(249, 115, 22, 0.05)),
                var(--bg-2);
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.07);
    animation: drop-in 0.7s cubic-bezier(0.2, 0.9, 0.3, 1.15) backwards,
               hero-glow 5s 1.2s ease-in-out infinite;
}

.app-hero::after {
    content: "";
    position: absolute;
    bottom: -10px;
    left: 0;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    box-shadow:
        8vw 0 0 rgba(251, 191, 36, 0.8), 20vw -6px 0 rgba(251, 191, 36, 0.5),
        32vw 2px 0 rgba(249, 115, 22, 0.7), 44vw -4px 0 rgba(251, 191, 36, 0.6),
        56vw 0 0 rgba(249, 115, 22, 0.5), 68vw -8px 0 rgba(251, 191, 36, 0.7),
        80vw -2px 0 rgba(251, 191, 36, 0.45);
    animation: sparks 7s linear infinite;
    opacity: 0;
    pointer-events: none;
}

@keyframes drop-in {
    from { opacity: 0; transform: translateY(-18px) scale(0.98); }
}

@keyframes rise-in {
    from { opacity: 0; transform: translateY(16px); }
}

@keyframes hero-glow {
    0%, 100% { box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5), 0 0 22px rgba(251, 191, 36, 0.10); }
    50% { box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5), 0 0 44px rgba(251, 191, 36, 0.22); }
}

@keyframes sparks {
    0% { transform: translateY(0); opacity: 0; }
    15% { opacity: 0.9; }
    100% { transform: translateY(-150px); opacity: 0; }
}

.hero-flex {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
}

.badge-scene {
    position: relative;
    perspective: 700px;
    width: 110px;
    height: 110px;
    flex-shrink: 0;
}

.badge-scene::before {
    content: "";
    position: absolute;
    inset: -26px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, rgba(251, 191, 36, 0.45), transparent 75deg);
    box-shadow: inset 0 0 0 1px rgba(251, 191, 36, 0.2);
    animation: radar 4s linear infinite;
    filter: blur(1px);
    opacity: 0.6;
    pointer-events: none;
}

@keyframes radar {
    to { transform: rotate(360deg); }
}

.badge-3d {
    width: 100%;
    height: 100%;
    position: relative;
    transform-style: preserve-3d;
    animation: spin3d 9s linear infinite;
}

.badge-face {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.2rem;
    border-radius: 18px;
    border: 2px solid rgba(251, 191, 36, 0.5);
    background: linear-gradient(160deg, rgba(251, 191, 36, 0.18), rgba(21, 24, 29, 0.92));
    box-shadow: 0 0 26px rgba(251, 191, 36, 0.28);
    backface-visibility: hidden;
}

.badge-face.back { transform: rotateY(180deg); }

@keyframes spin3d {
    0% { transform: rotateY(0deg) rotateX(8deg); }
    100% { transform: rotateY(360deg) rotateX(8deg); }
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 700;
    letter-spacing: 0.4px;
    background: linear-gradient(90deg, #e8e6e3 0%, #fbbf24 35%, #fde68a 50%, #fbbf24 65%, #e8e6e3 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 7s linear infinite;
}

@keyframes shimmer {
    to { background-position: 200% center; }
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(251, 191, 36, 0.45);
    background: rgba(251, 191, 36, 0.08);
    color: var(--accent);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-top: 1.4rem;
    perspective: 900px;
}

.kpi-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.06), transparent 38%), var(--panel);
    border: 1px solid var(--panel-border);
    border-left: 4px solid var(--accent);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(12px);
    transform-style: preserve-3d;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    animation: rise-in 0.6s ease backwards;
}

.kpi-grid .kpi-card:nth-child(1) { animation-delay: 0.05s; }
.kpi-grid .kpi-card:nth-child(2) { animation-delay: 0.15s; }
.kpi-grid .kpi-card:nth-child(3) { animation-delay: 0.25s; }
.kpi-grid .kpi-card:nth-child(4) { animation-delay: 0.35s; }

.kpi-card:hover {
    transform: translateY(-6px) rotateX(6deg) rotateY(-4deg) scale(1.02);
    border-color: rgba(251, 191, 36, 0.45);
    box-shadow: 0 0 28px rgba(251, 191, 36, 0.2), 0 26px 60px rgba(0, 0, 0, 0.55);
}

.kpi-card::after {
    content: "";
    position: absolute;
    top: 0;
    left: -80%;
    width: 50%;
    height: 100%;
    background: linear-gradient(105deg, transparent, rgba(251, 191, 36, 0.16), transparent);
    transform: skewX(-20deg);
    transition: left 0.6s ease;
}

.kpi-card:hover::after { left: 130%; }

.kpi-label {
    color: var(--muted);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

.glass-panel {
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.04), transparent 40%), var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    animation: rise-in 0.6s ease backwards;
}

.glass-panel:hover {
    border-color: rgba(251, 191, 36, 0.4);
    box-shadow: 0 0 22px rgba(251, 191, 36, 0.12), 0 18px 40px rgba(0, 0, 0, 0.5);
}

div.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #fcd34d, #f59e0b);
    color: #1a1a19;
    font-weight: 700;
    border: 0;
    box-shadow: 0 5px 0 #92650a, 0 12px 24px rgba(0, 0, 0, 0.45);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}

div.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
    color: #1a1a19;
    box-shadow: 0 7px 0 #92650a, 0 16px 30px rgba(0, 0, 0, 0.5);
}

div.stButton > button[kind="primary"]:active,
button[data-testid="stBaseButton-primary"]:active {
    transform: translateY(3px);
    box-shadow: 0 2px 0 #92650a, 0 6px 14px rgba(0, 0, 0, 0.45);
}

.status-good { color: #34d399; font-weight: 600; animation: pulse-glow 2.2s ease-in-out infinite; }
.status-warn { color: var(--warn); font-weight: 600; animation: pulse-glow 2.2s ease-in-out infinite; }
.status-bad { color: #f87171; font-weight: 600; animation: pulse-glow 1.4s ease-in-out infinite; }

@keyframes pulse-glow {
    0%, 100% { text-shadow: 0 0 4px currentColor; }
    50% { text-shadow: 0 0 12px currentColor; opacity: 0.8; }
}

.stPlotlyChart, [data-testid="stImage"], [data-testid="stAlert"], .stDataFrame {
    animation: rise-in 0.5s ease backwards;
}

[data-testid="stImage"] img {
    border-radius: 12px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.012);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55);
}

section[data-testid="stSidebar"] h2 {
    background: linear-gradient(90deg, #fbbf24, #fde68a, #f97316);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}

section[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.3rem;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100%;
    margin: 0;
    padding: 0.55rem 0.9rem;
    border-radius: 10px;
    border: 1px solid transparent;
    background: rgba(255, 255, 255, 0.025);
    cursor: pointer;
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(251, 191, 36, 0.10);
    border-color: rgba(251, 191, 36, 0.30);
    transform: translateX(4px);
}

section[data-testid="stSidebar"] [role="radiogroup"] label > div:not(:has(p)),
section[data-testid="stSidebar"] [role="radiogroup"] label > span:not(:has(p)) {
    display: none;
}

section[data-testid="stSidebar"] [role="radiogroup"] label > div:has(p) {
    display: block;
}

section[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #cfd2d6;
    font-weight: 500;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(180deg, #fcd34d, #f59e0b);
    border-color: transparent;
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.35);
}

section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked):hover {
    transform: none;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #1a1a19;
    font-weight: 700;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] details {
    border: 1px solid rgba(251, 191, 36, 0.22);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.02);
}

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-title::before {
    content: "";
    width: 10px;
    height: 10px;
    border-radius: 2px;
    background: var(--accent);
    box-shadow: 0 0 8px rgba(251, 191, 36, 0.6);
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state() -> None:
    if "history" not in st.session_state:
        history = pd.DataFrame(columns=HISTORY_COLUMNS)
        if os.path.exists(HISTORY_FILE):
            try:
                loaded = pd.read_csv(HISTORY_FILE)
                if list(loaded.columns) == HISTORY_COLUMNS:
                    history = loaded
            except Exception:
                pass
        st.session_state.history = history


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(model_path)


def is_violation_label(label: str) -> bool:
    keywords = ["no ", "without", "missing", "not", "violation"]
    label_lower = label.lower()
    return any(keyword in label_lower for keyword in keywords)


def summarize_detections(results) -> dict:
    result = results[0]
    names = result.names or {}
    counts = {}
    if result.boxes is not None and len(result.boxes) > 0:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for cls in classes:
            name = names.get(cls, f"class_{cls}")
            counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    violation_count = sum(count for label, count in counts.items() if is_violation_label(label))
    ppe_count = total - violation_count
    compliance_pct = 100.0 if total == 0 else round((ppe_count / total) * 100, 2)
    return {
        "counts": counts,
        "total": total,
        "ppe_count": ppe_count,
        "violation_count": violation_count,
        "compliance_pct": compliance_pct,
    }


def page_header(title: str) -> None:
    st.markdown(f"## {title}")
    st.markdown("<div class='hazard-tape'></div>", unsafe_allow_html=True)


def render_kpi_grid(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div></div>"
        for label, value in items
    )
    st.markdown(f"<div class='kpi-grid'>{cards}</div>", unsafe_allow_html=True)


def add_history_entry(filename: str, detections: int, compliance_pct: float, violations: int) -> None:
    entry = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename,
                "detections": detections,
                "compliance_pct": compliance_pct,
                "violations": violations,
            }
        ]
    )
    st.session_state.history = pd.concat([st.session_state.history, entry], ignore_index=True)
    try:
        st.session_state.history.to_csv(HISTORY_FILE, index=False)
    except OSError:
        pass


def flush_live_history() -> None:
    """Log the aggregate of a finished webcam session to history."""
    agg = st.session_state.get("live_agg")
    if not agg or not agg["frames"]:
        st.session_state.live_agg = None
        return
    total = agg["ppe"] + agg["violations"]
    compliance_pct = 100.0 if total == 0 else round((agg["ppe"] / total) * 100, 2)
    add_history_entry("Webcam Live", total, compliance_pct, agg["violations"])
    st.session_state.live_agg = None
    st.toast("Live session saved to Detection History", icon="💾")


def render_analytics_panels(summary: dict) -> None:
    counts = summary["counts"]
    ppe_count = summary["ppe_count"]
    violation_count = summary["violation_count"]
    compliance_pct = summary["compliance_pct"]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=compliance_pct,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#fbbf24"},
                    "steps": [
                        {"range": [0, 70], "color": "rgba(220,38,38,0.40)"},
                        {"range": [70, 90], "color": "rgba(245,158,11,0.35)"},
                        {"range": [90, 100], "color": "rgba(5,150,105,0.45)"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e6e3"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        pie_df = pd.DataFrame(
            {
                "Category": ["Compliant PPE", "Violations"],
                "Count": [ppe_count, violation_count],
            }
        )
        fig_pie = px.pie(
            pie_df,
            names="Category",
            values="Count",
            color_discrete_sequence=["#059669", "#dc2626"],
            hole=0.45,
        )
        fig_pie.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e6e3"),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col3:
        bar_df = pd.DataFrame(
            {"Class": list(counts.keys()) or ["No detections"], "Count": list(counts.values()) or [0]}
        )
        fig_bar = px.bar(
            bar_df,
            x="Class",
            y="Count",
            color="Count",
            color_continuous_scale=["#4a3b0a", "#fbbf24"],
        )
        fig_bar.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e6e3"),
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def run_image_inference(model: YOLO, image: Image.Image, conf: float, iou: float):
    image_array = np.array(image.convert("RGB"))
    results = model.predict(image_array, conf=conf, iou=iou, verbose=False)
    summary = summarize_detections(results)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, summary, results


def convert_to_h264(path: str) -> str:
    """Re-encode to H.264 so browsers can play the video; falls back to the
    original file when ffmpeg is unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return path
    converted = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    result = subprocess.run(
        [ffmpeg, "-y", "-i", path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-an", converted],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.getsize(converted):
        if os.path.exists(converted):
            os.remove(converted)
        return path
    os.remove(path)
    return converted


def process_video(
    model: YOLO,
    video_path: str,
    conf: float,
    iou: float,
    frame_stride: int,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Video contains no frames.")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    output_fps = max(fps / frame_stride, 1.0)
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    frame_index = 0
    processed_frames = 0
    agg_counts = {}
    agg_violations = 0
    agg_ppe = 0

    progress = st.progress(0)
    status = st.empty()

    while True:
        if frame_index == 0:
            frame = first_frame
        else:
            ret, frame = cap.read()
            if not ret:
                break
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        summary = summarize_detections(results)
        annotated = results[0].plot()
        writer.write(annotated)

        for label, count in summary["counts"].items():
            agg_counts[label] = agg_counts.get(label, 0) + count
        agg_violations += summary["violation_count"]
        agg_ppe += summary["ppe_count"]

        processed_frames += 1
        frame_index += 1

        if total_frames:
            progress.progress(min(frame_index / total_frames, 1.0))
        status.markdown(
            f"<span class='status-good'>Processing frames:</span> {processed_frames}",
            unsafe_allow_html=True,
        )

    cap.release()
    writer.release()
    progress.empty()
    status.empty()

    output_path = convert_to_h264(output_path)

    total = sum(agg_counts.values())
    compliance_pct = 100.0 if total == 0 else round((agg_ppe / total) * 100, 2)
    summary = {
        "counts": agg_counts,
        "total": total,
        "ppe_count": agg_ppe,
        "violation_count": agg_violations,
        "compliance_pct": compliance_pct,
    }
    return output_path, summary


init_state()

with st.sidebar:
    st.markdown("## 🦺 IronSight")
    st.caption("Industrial Safety Intelligence Hub")
    st.markdown("<div class='hazard-tape'></div>", unsafe_allow_html=True)

    PAGE_ICONS = {
        "Dashboard": "🏠",
        "Image Detection": "🖼️",
        "Video Detection": "🎥",
        "Webcam Live": "📡",
        "Analytics": "📊",
        "Dataset Overview": "🗂️",
        "Detection History": "📜",
    }
    page = st.radio(
        "Navigate",
        list(PAGE_ICONS.keys()),
        format_func=lambda name: f"{PAGE_ICONS[name]}  {name}",
        label_visibility="collapsed",
    )

    with st.expander("Inference Settings", expanded=True):
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        iou = st.slider("IoU Threshold", 0.2, 0.9, 0.45, 0.05)
        frame_stride = st.slider("Video Frame Stride", 1, 5, 1)

    st.divider()
    st.caption("Model: **PPE-YOLOv8-best.pt**")

with st.spinner("Initializing vision engine..."):
    try:
        model = load_model(MODEL_PATH)
        st.sidebar.success("Model loaded")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

if page != "Webcam Live":
    flush_live_history()


if page == "Dashboard":
    st.markdown(
        """
        <div class="app-hero">
            <div class="hero-flex">
                <div>
                    <div class="pill">🦺 AI Surveillance • PPE Compliance</div>
                    <div class="hero-title">IronSight Industrial Safety Intelligence Hub</div>
                    <div class="hero-subtitle">Real-time detection, compliance analytics, and safety enforcement for industrial operations.</div>
                </div>
                <div class="badge-scene">
                    <div class="badge-3d">
                        <div class="badge-face">⛑️</div>
                        <div class="badge-face back">🦺</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hazard-tape'></div>", unsafe_allow_html=True)

    total_sessions = len(st.session_state.history)
    avg_compliance = (
        st.session_state.history["compliance_pct"].mean() if total_sessions else 100.0
    )
    total_violations = (
        int(st.session_state.history["violations"].sum()) if total_sessions else 0
    )
    total_detections = (
        int(st.session_state.history["detections"].sum()) if total_sessions else 0
    )

    render_kpi_grid(
        [
            ("Total Sessions", f"{total_sessions}"),
            ("Avg Compliance", f"{avg_compliance:.1f}%"),
            ("Total Detections", f"{total_detections}"),
            ("Total Violations", f"{total_violations}"),
        ]
    )

    st.markdown("### Live Ops Snapshot")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="section-title">Operational Highlights</div>
                <ul>
                    <li>Multi-stream analytics for high-risk zones</li>
                    <li>Automated PPE compliance scoring</li>
                    <li>Enterprise-grade audit trail & reporting</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="section-title">System Status</div>
                <p><span class="status-good">●</span> Inference Engine Online</p>
                <p><span class="status-good">●</span> Surveillance Nodes Connected</p>
                <p><span class="status-warn">●</span> Alert Queue: Low</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "Image Detection":
    page_header("Image PPE Compliance")
    uploader = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploader:
        image = Image.open(io.BytesIO(uploader.read()))
        with st.spinner("Running detection on image..."):
            try:
                annotated, summary, _ = run_image_inference(model, image, confidence, iou)
                st.toast("Image inference complete", icon="✅")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.stop()

        tabs = st.tabs(["Annotated Output", "Analytics", "Detections"])
        with tabs[0]:
            st.image(annotated, caption="Annotated Output", use_container_width=True)

        with tabs[1]:
            render_analytics_panels(summary)

        with tabs[2]:
            st.dataframe(pd.DataFrame(list(summary["counts"].items()), columns=["Class", "Count"]))

        run_key = f"{getattr(uploader, 'file_id', uploader.name)}-{confidence}-{iou}"
        if st.session_state.get("last_image_run") != run_key:
            add_history_entry(
                uploader.name,
                summary["total"],
                summary["compliance_pct"],
                summary["violation_count"],
            )
            st.session_state.last_image_run = run_key

        if summary["violation_count"] > 0:
            st.error("Violation alert: Missing PPE detected.")
        else:
            st.success("All personnel compliant.")


elif page == "Video Detection":
    page_header("Video PPE Compliance")
    uploader = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploader:
        with st.spinner("Preparing video pipeline..."):
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploader.getvalue())
            temp_input.close()
            temp_input_path = temp_input.name

        if st.button("Run Video Inference", type="primary"):
            with st.spinner("Processing video frames..."):
                output_path = None
                output_bytes = None
                try:
                    output_path, summary = process_video(
                        model=model,
                        video_path=temp_input_path,
                        conf=confidence,
                        iou=iou,
                        frame_stride=frame_stride,
                    )
                    with open(output_path, "rb") as video_file:
                        output_bytes = video_file.read()
                    st.toast("Video inference complete", icon="🎥")
                except Exception as exc:
                    st.error(f"Video inference failed: {exc}")
                    st.stop()
                finally:
                    for path in [temp_input_path, output_path]:
                        if path and os.path.exists(path):
                            os.remove(path)

            tabs = st.tabs(["Annotated Output", "Analytics", "Detections"])
            with tabs[0]:
                if output_bytes:
                    st.video(output_bytes)

            with tabs[1]:
                render_analytics_panels(summary)

            with tabs[2]:
                st.dataframe(
                    pd.DataFrame(list(summary["counts"].items()), columns=["Class", "Count"])
                )

            add_history_entry(
                uploader.name,
                summary["total"],
                summary["compliance_pct"],
                summary["violation_count"],
            )

            if summary["violation_count"] > 0:
                st.warning("Violation alert: Missing PPE detected.")
            else:
                st.success("Video fully compliant.")


elif page == "Webcam Live":
    page_header("Live Webcam Compliance")
    st.info(
        "Click START and allow camera access. Frames stream from your browser, "
        "so live detection works both locally and on a deployed app."
    )

    result_queue: "queue.Queue[dict]" = queue.Queue()

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        results = model.predict(image, conf=confidence, iou=iou, verbose=False)
        result_queue.put(summarize_detections(results))
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="ppe-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        live_status = st.empty()
        if not st.session_state.get("live_agg"):
            st.session_state.live_agg = {"counts": {}, "ppe": 0, "violations": 0, "frames": 0}
        agg = st.session_state.live_agg
        while webrtc_ctx.state.playing:
            try:
                summary = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            agg["frames"] += 1
            agg["ppe"] += summary["ppe_count"]
            agg["violations"] += summary["violation_count"]
            for label, count in summary["counts"].items():
                agg["counts"][label] = agg["counts"].get(label, 0) + count
            badge = (
                "<span class='status-good'>● Compliant</span>"
                if summary["violation_count"] == 0
                else "<span class='status-bad'>● Violations detected</span>"
            )
            live_status.markdown(
                f"{badge} &nbsp; Detections: {summary['total']} &nbsp;|&nbsp; "
                f"Compliance: {summary['compliance_pct']}% &nbsp;|&nbsp; "
                f"Session violations: {agg['violations']}",
                unsafe_allow_html=True,
            )
    else:
        flush_live_history()


elif page == "Analytics":
    page_header("PPE Compliance Analytics")
    if st.session_state.history.empty:
        st.info("No analytics yet. Run detections to populate the dashboard.")
    else:
        summary = {
            "counts": {},
            "total": int(st.session_state.history["detections"].sum()),
            "ppe_count": int(
                st.session_state.history["detections"].sum()
                - st.session_state.history["violations"].sum()
            ),
            "violation_count": int(st.session_state.history["violations"].sum()),
            "compliance_pct": float(st.session_state.history["compliance_pct"].mean()),
        }

        render_analytics_panels(summary)

        st.markdown("### Compliance Trend")
        trend_df = st.session_state.history.copy()
        trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
        trend_chart = px.line(
            trend_df,
            x="timestamp",
            y="compliance_pct",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#fbbf24"],
        )
        trend_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e6e3"),
        )
        st.plotly_chart(trend_chart, use_container_width=True)


elif page == "Dataset Overview":
    page_header("Dataset Overview")
    st.markdown(
        """
        <div class="glass-panel">
            <div class="section-title">Industrial PPE Detection Dataset</div>
            <p>
                Annotated construction and industrial workplace imagery optimized for real-time PPE compliance
                monitoring. The dataset includes train/validation/test splits with multi-worker scenes, varying
                lighting, and YOLO-format bounding boxes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_grid(
        [
            ("Total Classes", "18"),
            ("Compliance Classes", "9"),
            ("Violation Classes", "9"),
            ("Annotation Format", "YOLO"),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="section-title">PPE Compliance Classes</div>
                <ol>
                    <li>Ear Protectors</li>
                    <li>Full Body Suit</li>
                    <li>Glasses</li>
                    <li>Gloves</li>
                    <li>Helmet</li>
                    <li>Mask</li>
                    <li>Safety Harness</li>
                    <li>Safety Shoes</li>
                    <li>Safety Vest</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="section-title">PPE Non-Compliance Classes</div>
                <ol>
                    <li>Without Ear Protectors</li>
                    <li>Without Full Body Suit</li>
                    <li>Without Glasses</li>
                    <li>Without Gloves</li>
                    <li>Without Helmet</li>
                    <li>Without Mask</li>
                    <li>Without Safety Harness</li>
                    <li>Without Safety Shoes</li>
                    <li>Without Safety Vest</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Annotation Details", expanded=True):
        st.markdown(
            """
            **YOLO Format:** `(class_id, x_center, y_center, width, height)` using normalized coordinates.  
            **Dataset Coverage:** Multi-object workplace scenes, construction sites, factories, manufacturing plants,
            and hazardous work zones with diverse lighting and viewpoints.
            """
        )

    with st.expander("Augmentation Strategy", expanded=True):
        st.markdown(
            """
            Mosaic • MixUp • HSV color jitter • Scaling • Translation • Shearing • Horizontal flipping
            """
        )

    with st.expander("Supported Use-Cases", expanded=True):
        st.markdown(
            """
            Real-time safety monitoring • Compliance analytics • Smart CCTV surveillance • Industrial AI automation
            """
        )




elif page == "Detection History":
    page_header("Detection History")
    if st.session_state.history.empty:
        st.info("No inference logs available yet.")
    else:
        hist = st.session_state.history
        render_kpi_grid(
            [
                ("Logged Runs", f"{len(hist)}"),
                ("Avg Compliance", f"{hist['compliance_pct'].mean():.1f}%"),
                ("Total Violations", f"{int(hist['violations'].sum())}"),
                ("Last Run", str(hist["timestamp"].iloc[-1])),
            ]
        )
        st.dataframe(
            hist,
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Time"),
                "filename": st.column_config.TextColumn("Source"),
                "detections": st.column_config.NumberColumn("Detections"),
                "compliance_pct": st.column_config.ProgressColumn(
                    "Compliance", format="%.1f%%", min_value=0, max_value=100
                ),
                "violations": st.column_config.NumberColumn("Violations"),
            },
        )
        col_download, col_clear = st.columns(2)
        with col_download:
            csv = hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name="ppe_detection_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clear:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = pd.DataFrame(columns=HISTORY_COLUMNS)
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()
