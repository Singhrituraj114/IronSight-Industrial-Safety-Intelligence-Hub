import os
import io
import time
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "PPE-YOLOv8-best.pt"

st.set_page_config(
    page_title="IronSight Industrial Safety Intelligence Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #0b1020;
    --bg-2: #0f172a;
    --panel: rgba(17, 25, 40, 0.72);
    --panel-border: rgba(148, 163, 184, 0.15);
    --accent: #6ee7ff;
    --accent-2: #8b5cf6;
    --good: #22c55e;
    --warn: #f59e0b;
    --bad: #ef4444;
    --text: #e2e8f0;
    --muted: #94a3b8;
}

html, body, [class*="stApp"] {
    background: radial-gradient(1200px 600px at 15% -10%, rgba(59, 130, 246, 0.15), transparent 55%),
                radial-gradient(900px 500px at 85% 0%, rgba(139, 92, 246, 0.15), transparent 60%),
                var(--bg);
    color: var(--text);
}

header { visibility: hidden; }
footer { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1324 0%, #0b1020 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

.app-hero {
    padding: 1.8rem 2.2rem;
    border-radius: 20px;
    background: linear-gradient(120deg, rgba(59,130,246,0.15), rgba(139,92,246,0.12));
    border: 1px solid var(--panel-border);
    box-shadow: 0 30px 80px rgba(15, 23, 42, 0.35);
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 700;
    letter-spacing: 0.4px;
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
    border: 1px solid rgba(110, 231, 255, 0.35);
    background: rgba(15, 23, 42, 0.6);
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
}

.kpi-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 18px 40px rgba(2, 6, 23, 0.35);
    backdrop-filter: blur(12px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: float 6s ease-in-out infinite;
}

.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.45);
}

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
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 18px 40px rgba(2, 6, 23, 0.35);
    backdrop-filter: blur(10px);
}

.status-good { color: var(--good); font-weight: 600; }
.status-warn { color: var(--warn); font-weight: 600; }
.status-bad { color: var(--bad); font-weight: 600; }

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame(
            columns=["timestamp", "filename", "detections", "compliance_pct", "violations"]
        )


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


def render_kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
                    "bar": {"color": "#6ee7ff"},
                    "steps": [
                        {"range": [0, 70], "color": "rgba(239,68,68,0.35)"},
                        {"range": [70, 90], "color": "rgba(245,158,11,0.35)"},
                        {"range": [90, 100], "color": "rgba(34,197,94,0.35)"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
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
            color_discrete_sequence=["#22c55e", "#ef4444"],
            hole=0.45,
        )
        fig_pie.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
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
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
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
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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


def run_webcam_stream(model: YOLO, conf: float, iou: float, max_frames: int):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam.")

    frame_placeholder = st.empty()
    progress = st.progress(0)
    summary_placeholder = st.empty()

    agg_counts = {}
    agg_violations = 0
    agg_ppe = 0

    for idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        summary = summarize_detections(results)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

        for label, count in summary["counts"].items():
            agg_counts[label] = agg_counts.get(label, 0) + count
        agg_violations += summary["violation_count"]
        agg_ppe += summary["ppe_count"]

        progress.progress((idx + 1) / max_frames)
        summary_placeholder.info(
            f"Live detections: {summary['total']} | Compliance: {summary['compliance_pct']}%"
        )
        time.sleep(0.02)

    cap.release()
    progress.empty()
    summary_placeholder.empty()

    total = sum(agg_counts.values())
    compliance_pct = 100.0 if total == 0 else round((agg_ppe / total) * 100, 2)
    return {
        "counts": agg_counts,
        "total": total,
        "ppe_count": agg_ppe,
        "violation_count": agg_violations,
        "compliance_pct": compliance_pct,
    }


init_state()

with st.sidebar:
    st.markdown("## 🛡️ IronSight")
    st.caption("Industrial Safety Intelligence Hub")

    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Image Detection",
            "Video Detection",
            "Webcam Live",
            "Analytics",
            "Dataset Overview",
            "Detection History",
        ],
        label_visibility="collapsed",
    )

    with st.expander("Inference Settings", expanded=True):
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        iou = st.slider("IoU Threshold", 0.2, 0.9, 0.45, 0.05)
        frame_stride = st.slider("Video Frame Stride", 1, 5, 1)
        webcam_frames = st.slider("Webcam Frames per Run", 30, 300, 120, 10)

    st.divider()
    st.caption("Model: **PPE-YOLOv8-best.pt**")

with st.spinner("Initializing vision engine..."):
    try:
        model = load_model(MODEL_PATH)
        st.sidebar.success("Model loaded")
    except Exception as exc:
        st.error(str(exc))
        st.stop()


if page == "Dashboard":
    st.markdown(
        """
        <div class="app-hero">
            <div class="pill">AI Surveillance • PPE Compliance</div>
            <div class="hero-title">IronSight Industrial Safety Intelligence Hub</div>
            <div class="hero-subtitle">Real-time detection, compliance analytics, and safety enforcement for industrial operations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
    render_kpi("Total Sessions", f"{total_sessions}")
    render_kpi("Avg Compliance", f"{avg_compliance:.1f}%")
    render_kpi("Total Detections", f"{total_detections}")
    render_kpi("Total Violations", f"{total_violations}")
    st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("## Image PPE Compliance")
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

        add_history_entry(
            uploader.name,
            summary["total"],
            summary["compliance_pct"],
            summary["violation_count"],
        )

        if summary["violation_count"] > 0:
            st.error("Violation alert: Missing PPE detected.")
        else:
            st.success("All personnel compliant.")


elif page == "Video Detection":
    st.markdown("## Video PPE Compliance")
    uploader = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploader:
        with st.spinner("Preparing video pipeline..."):
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploader.read())
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
    st.markdown("## Live Webcam Compliance")
    st.info(
        "Start a live run to process a burst of frames. Re-run for continuous monitoring."
    )
    if st.button("Start Live Detection", type="primary"):
        with st.spinner("Activating webcam..."):
            try:
                summary = run_webcam_stream(model, confidence, iou, webcam_frames)
                st.toast("Live run complete", icon="📡")
            except Exception as exc:
                st.error(f"Webcam error: {exc}")
                st.stop()

        render_analytics_panels(summary)
        add_history_entry("Webcam Stream", summary["total"], summary["compliance_pct"], summary["violation_count"])


elif page == "Analytics":
    st.markdown("## PPE Compliance Analytics")
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
        )
        trend_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(trend_chart, use_container_width=True)


elif page == "Dataset Overview":
    st.markdown("## Dataset Overview")
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

    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
    render_kpi("Total Classes", "18")
    render_kpi("Compliance Classes", "9")
    render_kpi("Violation Classes", "9")
    render_kpi("Annotation Format", "YOLO")
    st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("## Detection History")
    if st.session_state.history.empty:
        st.info("No inference logs available yet.")
    else:
        st.dataframe(st.session_state.history, use_container_width=True)
        csv = st.session_state.history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="ppe_detection_history.csv",
            mime="text/csv",
        )
