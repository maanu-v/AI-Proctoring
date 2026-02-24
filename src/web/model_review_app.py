"""
AI Proctoring — Video Review Dashboard

A premium Streamlit application for reviewing batch-processed proctoring
videos with flagged violation regions, interactive timelines, and
detailed analytics.

Usage:
    streamlit run src/web/review_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import os
import sys
import base64
from pathlib import Path

# Project root
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "model_processing", "results")

# --- Page Config ---
st.set_page_config(
    page_title="AI Proctoring — Video Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium Dark Theme CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a15 0%, #12122a 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0ff;
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Subject cards */
    .subject-card {
        background: linear-gradient(145deg, rgba(30,30,60,0.8), rgba(20,20,45,0.9));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .subject-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    .subject-card.risk-low::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .subject-card.risk-medium::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .subject-card.risk-high::before { background: linear-gradient(90deg, #ef4444, #f87171); }
    
    .subject-card:hover {
        border-color: rgba(59,130,246,0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59,130,246,0.15);
    }
    .subject-card h3 {
        color: #e0e0ff;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .subject-card .meta {
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
        margin: 4px 0;
    }
    
    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .risk-badge.low { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
    .risk-badge.medium { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .risk-badge.high { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30,30,60,0.7), rgba(20,20,45,0.8));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Timeline */
    .timeline-container {
        background: rgba(15,15,30,0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }
    .timeline-bar {
        position: relative;
        height: 36px;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        overflow: hidden;
        margin: 10px 0;
    }
    .timeline-segment {
        position: absolute;
        top: 0;
        height: 100%;
        border-radius: 4px;
        opacity: 0.8;
        transition: opacity 0.2s;
        cursor: pointer;
    }
    .timeline-segment:hover {
        opacity: 1;
        z-index: 10;
    }
    .timeline-segment.violation { background: linear-gradient(180deg, #ef4444, #dc2626); }
    .timeline-segment.gt { background: linear-gradient(180deg, #f59e0b, #d97706); }
    
    .timeline-label {
        display: flex;
        justify-content: space-between;
        color: rgba(255,255,255,0.4);
        font-size: 0.75rem;
        margin-top: 6px;
    }
    
    /* Legend */
    .legend {
        display: flex;
        gap: 20px;
        margin: 8px 0;
        flex-wrap: wrap;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        color: rgba(255,255,255,0.6);
        font-size: 0.8rem;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 3px;
        display: inline-block;
    }
    
    /* Violation table */
    .violation-row {
        background: rgba(20,20,45,0.6);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.2s;
        cursor: pointer;
    }
    .violation-row:hover {
        background: rgba(59,130,246,0.1);
        border-color: rgba(59,130,246,0.3);
    }
    .violation-type-badge {
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        white-space: nowrap;
        background: rgba(59,130,246,0.2); 
        color: #60a5fa;
    }
    
    .violation-time {
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        min-width: 120px;
    }
    .violation-msg {
        color: rgba(255,255,255,0.7);
        font-size: 0.85rem;
        flex: 1;
    }
    .violation-severity {
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .sev-critical { background: rgba(239,68,68,0.3); color: #fca5a5; }
    .sev-high { background: rgba(249,115,22,0.2); color: #fdba74; }
    .sev-medium { background: rgba(245,158,11,0.15); color: #fcd34d; }
    .sev-low { background: rgba(34,197,94,0.15); color: #86efac; }
    
    /* Section headers */
    .section-header {
        color: #93c5fd;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 24px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Ground truth */
    .gt-row {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.15);
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .gt-type {
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        background: rgba(245,158,11,0.2);
        color: #fbbf24;
    }
    .gt-time {
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        font-family: monospace;
        min-width: 100px;
    }
    .gt-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        flex: 1;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #60a5fa, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
    ::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# --- Utility Functions ---

def load_index():
    """Load the aggregate index.json."""
    index_path = os.path.join(RESULTS_DIR, "index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path, 'r') as f:
        return json.load(f)

def load_subject_results(subject_id: str):
    """Load detailed results for a specific subject."""
    path = os.path.join(RESULTS_DIR, f"{subject_id}_results.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def format_time(seconds: float) -> str:
    """Format seconds to MM:SS."""
    if seconds is None:
        return "00:00"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def get_risk_class(score: float) -> str:
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    return "high"

def get_risk_label(score: float) -> str:
    if score < 0.3:
        return "Low Risk"
    elif score < 0.6:
        return "Medium Risk"
    return "High Risk"

def get_video_base64(video_path: str) -> str:
    """Read video file and return base64-encoded data URI."""
    abs_path = os.path.join(PROJECT_ROOT, video_path) if not os.path.isabs(video_path) else video_path
    if not os.path.exists(abs_path):
        return ""
    with open(abs_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


# --- Dashboard View ---

def render_dashboard(index_data):
    """Render the main dashboard / gallery view."""
    
    st.markdown("""
    <div class="main-header">
        <h1>AI Proctoring Review Dashboard</h1>
        <p>Review batch-processed proctoring videos with flagged violation regions</p>
    </div>
    """, unsafe_allow_html=True)
    
    subjects = index_data.get("subjects", [])
    
    if not subjects:
        st.warning("No processed results found. Run the batch processor first.")
        st.code("python -m src.batch.run_batch", language="bash")
        return
    
    # Summary metrics
    total_subjects = len(subjects)
    total_violations = sum(s.get("total_violations", 0) for s in subjects if "error" not in s)
    avg_risk = sum(s.get("risk_score", 0) for s in subjects if "error" not in s) / max(
        sum(1 for s in subjects if "error" not in s), 1
    )
    high_risk_count = sum(1 for s in subjects if s.get("risk_score", 0) >= 0.6 and "error" not in s)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{total_subjects}</div>
            <div class="label">Subjects</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{total_violations}</div>
            <div class="label">Total Violations</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{avg_risk:.2f}</div>
            <div class="label">Avg Risk Score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="-webkit-text-fill-color:#f87171;">{high_risk_count}</div>
            <div class="label">High Risk</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sorting / filtering
    col_sort, col_filter = st.columns([1, 1])
    with col_sort:
        sort_by = st.selectbox("Sort by", ["Risk Score (High→Low)", "Risk Score (Low→High)", "Subject ID", "Violations"], label_visibility="collapsed")
    with col_filter:
        filter_risk = st.selectbox("Filter", ["All", "High Risk", "Medium Risk", "Low Risk"], label_visibility="collapsed")
    
    # Apply sort
    valid_subjects = [s for s in subjects if "error" not in s]
    if sort_by == "Risk Score (High→Low)":
        valid_subjects.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
    elif sort_by == "Risk Score (Low→High)":
        valid_subjects.sort(key=lambda x: x.get("risk_score", 0))
    elif sort_by == "Subject ID":
        valid_subjects.sort(key=lambda x: int(x["subject_id"].replace("subject", "")))
    elif sort_by == "Violations":
        valid_subjects.sort(key=lambda x: x.get("total_violations", 0), reverse=True)
    
    # Apply filter
    if filter_risk == "High Risk":
        valid_subjects = [s for s in valid_subjects if s.get("risk_score", 0) >= 0.6]
    elif filter_risk == "Medium Risk":
        valid_subjects = [s for s in valid_subjects if 0.3 <= s.get("risk_score", 0) < 0.6]
    elif filter_risk == "Low Risk":
        valid_subjects = [s for s in valid_subjects if s.get("risk_score", 0) < 0.3]
    
    # Subject grid
    cols = st.columns(3)
    for i, subject in enumerate(valid_subjects):
        with cols[i % 3]:
            sid = subject["subject_id"]
            risk = subject.get("risk_score", 0)
            risk_cls = get_risk_class(risk)
            violations = subject.get("total_violations", 0)
            duration = subject.get("duration_seconds", 0)
            vtypes = subject.get("violation_types", {})
            
            # Violation type breakdown
            vtype_html = ""
            for vt, count in sorted(vtypes.items(), key=lambda x: x[1], reverse=True)[:3]:
                display_vt = vt.replace("Label_", "Class ")
                vtype_html += f'<span class="violation-type-badge">{display_vt}: {count}</span> '
            
            st.markdown(f"""
            <div class="subject-card risk-{risk_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3>{sid.replace('subject', 'Subject ')}</h3>
                    <span class="risk-badge {risk_cls}">{get_risk_label(risk)} ({risk:.2f})</span>
                </div>
                <p class="meta">Duration: {format_time(duration)} &nbsp;|&nbsp; Violations: {violations}</p>
                <div style="margin-top:8px;">{vtype_html}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Review →", key=f"btn_{sid}", use_container_width=True):
                st.session_state.selected_subject = sid
                st.session_state.page = "review"
                st.rerun()


# --- Video Review View ---

def render_review(subject_id: str):
    """Render detailed review for a specific subject."""
    
    data = load_subject_results(subject_id)
    if data is None:
        st.error(f"No results found for {subject_id}")
        return
    
    # Back button
    if st.button("← Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()
    
    # Header
    risk = data.get("summary", {}).get("risk_score", 0)
    risk_cls = get_risk_class(risk)
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{subject_id.replace('subject', 'Subject ')} — Video Review</h1>
        <p>
            <span class="risk-badge {risk_cls}" style="font-size:0.85rem;">
                {get_risk_label(risk)} — Score: {risk:.3f}
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    meta = data.get("video_metadata", {})
    summary = data.get("summary", {})
    violations = data.get("violations", [])
    ground_truth = data.get("ground_truth", [])
    duration = meta.get("duration_seconds", 0)
    
    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{summary.get('total_violations', 0)}</div>
            <div class="label">Violations</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{format_time(duration)}</div>
            <div class="label">Duration</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{meta.get('fps', 0):.0f}</div>
            <div class="label">FPS</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        viol_dur = summary.get('total_violation_duration_seconds', 0)
        st.markdown(f"""<div class="metric-card">
            <div class="value">{format_time(viol_dur)}</div>
            <div class="label">Violation Time</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{len(ground_truth)}</div>
            <div class="label">GT Annotations</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Video player + timeline
    video_col, info_col = st.columns([2, 1])
    
    with video_col:
        # Toggle overlays
        show_overlays = st.toggle("Show Behavior Overlay (CNN-BiLSTM)", value=True)

        # Video player
        video_path = data.get("video_path", "")
        overlaid_video_path = data.get("overlaid_video_path", "")
        
        abs_base_video = os.path.join(PROJECT_ROOT, video_path) if not os.path.isabs(video_path) else video_path
        abs_mp4_path = abs_base_video.rsplit('.', 1)[0] + '.mp4'
        abs_overlaid_path = os.path.join(PROJECT_ROOT, overlaid_video_path) if overlaid_video_path else abs_base_video.rsplit('.', 1)[0] + '_overlaid.mp4'
        
        playback_path = abs_overlaid_path if show_overlays else abs_mp4_path
        
        if not os.path.exists(playback_path):
            playback_path = abs_base_video if os.path.exists(abs_base_video) else ""
            
        if os.path.exists(playback_path):
            st.video(playback_path, format="video/mp4" if playback_path.endswith('.mp4') else "video/avi")
        else:
            st.warning(f"Video file not found: {video_path}")
            
    # Combine Timeline and Log into a single Interactive HTML Component
    html_template = """
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            color: #e0e0ff;
            margin: 0;
            padding: 0;
            background: transparent;
        }
        .section-header { color: #93c5fd; font-size: 1.1rem; font-weight: 700; margin: 24px 0 12px 0; display: flex; align-items: center; gap: 8px; }
        .timeline-container { background: rgba(15,15,30,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin: 16px 0; }
        .timeline-bar { position: relative; height: 36px; background: rgba(255,255,255,0.05); border-radius: 8px; overflow: hidden; margin: 10px 0; }
        .timeline-segment { position: absolute; top: 0; height: 100%; border-radius: 4px; opacity: 0.8; transition: opacity 0.2s; cursor: pointer; }
        .timeline-segment:hover { opacity: 1; z-index: 10; border: 1px solid white; box-sizing: border-box; }
        .timeline-segment.gt { background: linear-gradient(180deg, #f59e0b, #d97706); }
        .timeline-label { display: flex; justify-content: space-between; color: rgba(255,255,255,0.4); font-size: 0.75rem; margin-top: 6px; }
        .legend { display: flex; gap: 20px; margin: 8px 0; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 6px; color: rgba(255,255,255,0.6); font-size: 0.8rem; }
        .legend-dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
        .violation-row { background: rgba(20,20,45,0.6); border: 1px solid rgba(255,255,255,0.04); border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; display: flex; align-items: center; gap: 12px; transition: all 0.2s; cursor: pointer; }
        .violation-row:hover { background: rgba(59,130,246,0.1); border-color: rgba(59,130,246,0.3); }
        .violation-type-badge { padding: 2px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; white-space: nowrap; background: rgba(239,68,68,0.2); color: #f87171;}
        .violation-time { color: rgba(255,255,255,0.4); font-size: 0.8rem; font-family: 'Courier New', monospace; min-width: 120px; }
        .violation-msg { color: rgba(255,255,255,0.7); font-size: 0.85rem; flex: 1; }
        .violation-severity { font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; }
        .sev-critical { background: rgba(239,68,68,0.3); color: #fca5a5; }
        .sev-high { background: rgba(249,115,22,0.2); color: #fdba74; }
        .sev-medium { background: rgba(245,158,11,0.15); color: #fcd34d; }
        .sev-low { background: rgba(34,197,94,0.15); color: #86efac; }
        .gt-row { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.15); border-radius: 8px; padding: 8px 14px; margin-bottom: 6px; display: flex; align-items: center; gap: 12px; cursor: pointer; }
        .gt-row:hover { background: rgba(245,158,11,0.15); border-color: rgba(245,158,11,0.3); }
        .gt-type { padding: 2px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; background: rgba(245,158,11,0.2); color: #fbbf24; }
        .gt-time { color: rgba(255,255,255,0.4); font-size: 0.8rem; font-family: monospace; min-width: 100px; }
        .gt-label { color: rgba(255,255,255,0.6); font-size: 0.85rem; flex: 1; }
    </style>
    <script>
        function seekVideo(time) {
            try {
                const vid = window.parent.document.querySelector('video');
                if (vid) {
                    vid.currentTime = time;
                    vid.play();
                } else {
                    console.log("Video element not found in parent.");
                }
            } catch(e) {
                console.error("Cross-origin or element access error: ", e);
            }
        }
    </script>
    </head>
    <body>
    """
    
    # 1. Timeline Building
    html_template += '<div class="section-header">📊 Interactive Violation Timeline</div>'
    
    if duration > 0:
        violation_segments_html = ""
        for v in violations:
            left = (v["start_time"] / duration) * 100
            width = max(((v["end_time"] - v["start_time"]) / duration) * 100, 0.5)
            vtype = v.get("type", "unknown")
            msg = v.get("message", "")
            
            colors = {
                "Label_1": "#3b82f6",
                "Label_2": "#38bdf8",
                "Label_3": "#ef4444",
                "Label_4": "#f59e0b",
                "Label_5": "#2dd4bf",
                "Label_6": "#f97316",
            }
            color = colors.get(vtype, "#ef4444")
            tooltip = f"{msg} : {format_time(v['start_time'])} → {format_time(v['end_time'])} ({v.get('duration', 0):.1f}s)"
            violation_segments_html += f'<div class="timeline-segment" onclick="seekVideo({v["start_time"]})" style="left:{left:.2f}%;width:{width:.2f}%;background:{color};" title="{tooltip}"></div>'
        
        gt_segments_html = ""
        for gt in ground_truth:
            left = (gt["start_time"] / duration) * 100
            width = max(((gt["end_time"] - gt["start_time"]) / duration) * 100, 0.5)
            tooltip = f"GT: {gt['type_label']} ({gt['start_str']} → {gt['end_str']})"
            gt_segments_html += f'<div class="timeline-segment gt" onclick="seekVideo({gt["start_time"]})" style="left:{left:.2f}%;width:{width:.2f}%;" title="{tooltip}"></div>'
            
        html_template += f"""
        <div class="timeline-container">
            <div class="legend">
                <div class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span> Model Detected Violation</div>
                <div class="legend-item"><span class="legend-dot" style="background:#f59e0b;"></span> Ground Truth Box</div>
            </div>
            <p style="color:rgba(255,255,255,0.5);font-size:0.75rem;margin:8px 0 2px 0;">▼ Detected Violations (Click to Seek)</p>
            <div class="timeline-bar">{violation_segments_html}</div>
            <p style="color:rgba(255,255,255,0.5);font-size:0.75rem;margin:8px 0 2px 0;">▼ Ground Truth Annotations (Click to Seek)</p>
            <div class="timeline-bar">{gt_segments_html}</div>
            <div class="timeline-label">
                <span>0:00</span>
                <span>{format_time(duration / 4)}</span>
                <span>{format_time(duration / 2)}</span>
                <span>{format_time(duration * 3 / 4)}</span>
                <span>{format_time(duration)}</span>
            </div>
        </div>
        """

    # 2. Violation Log
    html_template += '<div class="section-header">🚨 Interactive Violation Log</div>'
    if violations:
        for v in violations:
            vtype = v.get("type", "unknown")
            sev = v.get("severity", "medium")
            html_template += f"""
            <div class="violation-row" onclick="seekVideo({v['start_time']})">
                <span class="violation-type-badge">{vtype.replace("Label_", "Class ")}</span>
                <span class="violation-time">{format_time(v['start_time'])} → {format_time(v['end_time'])}</span>
                <span class="violation-msg">{v.get('message', '')}</span>
                <span class="violation-severity sev-{sev}">{sev}</span>
                <span style="color:rgba(255,255,255,0.3);font-size:0.75rem;">{v.get('duration', 0):.1f}s</span>
            </div>
            """
    else:
        html_template += '<p style="color:rgba(255,255,255,0.4);">No violations detected for this video.</p>'

    # 3. Ground truth Log
    if ground_truth:
        html_template += '<div class="section-header">📋 Ground Truth Annotations</div>'
        for gt in ground_truth:
            html_template += f"""
            <div class="gt-row" onclick="seekVideo({gt['start_time']})">
                <span class="gt-type">Type {gt['type']}</span>
                <span class="gt-time">{gt['start_str']} → {gt['end_str']}</span>
                <span class="gt-label">{gt['type_label']}</span>
            </div>
            """
            
    html_template += """
    <br><br>
    </body>
    </html>
    """

    with video_col:
        st.markdown("<br>", unsafe_allow_html=True)
        components.html(html_template, height=800, scrolling=True)

    with info_col:
        # Violation type breakdown
        st.markdown('<div class="section-header">Violation Breakdown</div>', unsafe_allow_html=True)
        
        vtypes = summary.get("violation_types", {})
        if vtypes:
            for vtype, count in sorted(vtypes.items(), key=lambda x: x[1], reverse=True):
                pct = (count / max(summary.get("total_violations", 1), 1)) * 100
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;color:rgba(255,255,255,0.7);font-size:0.85rem;margin-bottom:4px;">
                        <span>{vtype.replace('Label_', 'Class ')}</span>
                        <span>{count} ({pct:.0f}%)</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:#ef4444;border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:rgba(255,255,255,0.4);">No violations detected.</p>', unsafe_allow_html=True)
        
        # Processing info
        proc = data.get("processing", {})
        st.markdown(f"""
        <div style="margin-top:20px;padding:12px;background:rgba(20,20,45,0.5);border-radius:8px;border:1px solid rgba(255,255,255,0.04);">
            <p style="color:rgba(255,255,255,0.5);font-size:0.75rem;margin:0;">
                Processed: {proc.get('timestamp', 'N/A')}<br>
                Processing time: {proc.get('processing_time_seconds', 0):.1f}s<br>
                Resolution: {meta.get('width', 0)}×{meta.get('height', 0)}<br>
                Frames processed: {meta.get('frames_processed', 0):,} / {meta.get('total_frames', 0):,}
            </p>
        </div>
        """, unsafe_allow_html=True)


# --- Main App ---

def main():
    # Initialize state
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    if "selected_subject" not in st.session_state:
        st.session_state.selected_subject = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### AI Proctoring")
        st.markdown("---")
        
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
        
        st.markdown("---")
        
        # Quick jump to subject
        index_data = load_index()
        if index_data:
            subjects = [s["subject_id"] for s in index_data.get("subjects", []) if "error" not in s]
            if subjects:
                st.markdown("**Quick Jump**")
                selected = st.selectbox(
                    "Select Subject",
                    options=subjects,
                    index=subjects.index(st.session_state.selected_subject) if st.session_state.selected_subject in subjects else 0,
                    label_visibility="collapsed",
                )
                if st.button("Go →", use_container_width=True):
                    st.session_state.selected_subject = selected
                    st.session_state.page = "review"
                    st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="color:rgba(255,255,255,0.3);font-size:0.7rem;padding:8px;">
            AI Proctoring Review System<br>
            Batch Video Analysis v1.0
        </div>
        """, unsafe_allow_html=True)
    
    # Route to page
    if st.session_state.page == "review" and st.session_state.selected_subject:
        render_review(st.session_state.selected_subject)
    else:
        if index_data:
            render_dashboard(index_data)
        else:
            st.markdown("""
            <div class="main-header">
                <h1>AI Proctoring Review Dashboard</h1>
                <p>No processed results found. Run the batch processor first.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Getting Started")
            st.markdown("""
            1. **Run the batch processor** to analyze all videos using the CNN-BiLSTM:
            ```bash
            python -m src.batch.run_model_batch
            ```
            
            2. **Process specific subjects** (faster for testing):
            ```bash
            python -m src.batch.run_model_batch --subjects subject1 subject2
            ```
            
            3. **Refresh this page** after processing completes.
            """)


if __name__ == "__main__":
    main()
