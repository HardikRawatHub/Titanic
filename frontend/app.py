"""
app.py  –  Streamlit Frontend
------------------------------
A sleek, dark-themed chat interface for the Titanic Agent.

Run with:
    streamlit run frontend/app.py
"""

import base64
import io
import os
import time
from datetime import datetime

import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ═══════════════════════════════════════════════════════════════════════════
# Page config & global CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🚢 Titanic Chat Agent",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

/* ── Root palette ───────────────────────────────────────────────────────── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #21262d;
    --accent:    #e94560;
    --accent2:   #4fc3f7;
    --text:      #c9d1d9;
    --text-dim:  #8b949e;
    --user-bg:   #1c2733;
    --bot-bg:    #161b22;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Chat bubbles ───────────────────────────────────────────────────────── */
.user-bubble {
    background: var(--user-bg);
    border: 1px solid var(--accent);
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 15%;
    font-size: 0.95rem;
    line-height: 1.6;
}
.bot-bubble {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 2px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 15% 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.bot-bubble strong { color: var(--accent2); }

/* ── Avatar chips ───────────────────────────────────────────────────────── */
.avatar-user { color: var(--accent);  font-family: 'Space Mono', monospace; font-size:0.75rem; text-align:right; margin-bottom:2px; }
.avatar-bot  { color: var(--accent2); font-family: 'Space Mono', monospace; font-size:0.75rem; margin-bottom:2px; }

/* ── Quick question buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    padding: 0.35rem 0.7rem !important;
    transition: border-color 0.2s, color 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Input ──────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(233,69,96,0.15) !important;
}

/* ── Stat cards ─────────────────────────────────────────────────────────── */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.stat-value { font-size: 1.6rem; font-weight: 700; color: var(--accent2); font-family: 'Space Mono', monospace; }
.stat-label { font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Section titles ─────────────────────────────────────────────────────── */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
    margin: 1rem 0 0.5rem;
}

/* ── Chart container ────────────────────────────────────────────────────── */
.chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem;
    margin-top: 0.5rem;
}

/* ── Hero title ─────────────────────────────────────────────────────────── */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.02em;
}
.hero-sub {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-top: -0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def get_backend_info() -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}/dataset/info", timeout=5)
        return r.json()
    except Exception:
        return None


def send_message(message: str, history: list) -> dict:
    """POST to /chat and return the response dict."""
    payload = {
        "message": message,
        "chat_history": [
            {"role": h["role"], "content": h["content"]}
            for h in history
        ],
    }
    r = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        detail = None
        try:
            body = r.json()
            detail = body.get("detail")
        except Exception:
            detail = r.text or str(exc)
        raise RuntimeError(f"{r.status_code} {detail}") from exc
    return r.json()


def get_chart(chart_type: str) -> str | None:
    """Directly fetch a chart from the backend."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/dataset/chart",
            json={"chart_type": chart_type},
            timeout=30,
        )
        return r.json().get("image_b64")
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = get_backend_info()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="hero-title">🚢 TitanicBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-powered passenger analysis</div>', unsafe_allow_html=True)

    # ── Backend status ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Backend status</div>', unsafe_allow_html=True)
    info = st.session_state.dataset_info
    if info:
        st.success(f"✅ Connected — {info['rows']} passengers loaded")
    else:
        st.error("❌ Backend offline. Start FastAPI first.")
        st.code("cd backend\nuvicorn main:app --reload", language="bash")

    # ── Dataset stats ───────────────────────────────────────────────────────
    if info:
        st.markdown('<div class="section-title">Dataset at a glance</div>', unsafe_allow_html=True)
        s = info.get("summary", {})
        cols = st.columns(2)
        metrics = [
            ("Passengers",  info["rows"]),
            ("Survival rate", f"{s.get('survival_rate_pct', '')}%"),
            ("Avg age",     f"{s.get('avg_age', '')} yr"),
            ("Avg fare",    f"£{s.get('avg_fare', '')}"),
        ]
        for i, (label, val) in enumerate(metrics):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-value">{val}</div>'
                    f'<div class="stat-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Quick visualisations ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Quick visualisations</div>', unsafe_allow_html=True)
    chart_options = {
        "📊 Overview Dashboard":    "overview_dashboard",
        "🎂 Age Histogram":         "age_histogram",
        "⚥ Survival by Sex":        "survival_by_sex",
        "💷 Fare Distribution":     "fare_distribution",
        "🗺️ Embarkation Ports":     "embarkation_counts",
        "🔥 Class × Sex Heatmap":   "class_survival_heatmap",
        "👨‍👩‍👧 Family Size Effect":  "family_size_survival",
        "🎻 Age vs Survival":       "age_survival_violin",
    }
    for label, chart_key in chart_options.items():
        if st.button(label, use_container_width=True):
            with st.spinner("Generating chart…"):
                b64 = get_chart(chart_key)
            if b64:
                st.session_state.messages.append({
                    "role": "user", "content": f"Show me the {label.split(' ', 1)[1]} chart.",
                })
                st.session_state.messages.append({
                    "role": "ai",
                    "content": f"Here's the **{label.split(' ', 1)[1]}** visualisation:",
                    "image_b64": b64,
                    "chart_type": chart_key,
                })
                st.rerun()

    # ── Suggested questions ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Suggested questions</div>', unsafe_allow_html=True)
    suggestions = [
        "What percentage of passengers were male?",
        "What was the average ticket fare?",
        "How many passengers embarked from each port?",
        "What was the survival rate by class?",
        "How did family size affect survival?",
        "Tell me about the youngest passengers.",
        "Which class had the most expensive fares?",
        "What was the overall survival rate?",
    ]
    for q in suggestions:
        if st.button(f"💬 {q}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    # ── Clear chat ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Main chat area
# ═══════════════════════════════════════════════════════════════════════════

# Header
col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">Titanic Chat Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Ask anything about the 891 passengers aboard RMS Titanic — '
        'powered by LangChain + GPT-4o-mini</div>',
        unsafe_allow_html=True,
    )
with col_time:
    st.markdown(
        f'<div style="text-align:right; color:#8b949e; font-size:0.75rem; margin-top:0.5rem;">'
        f'{datetime.now().strftime("%d %b %Y · %H:%M")}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Welcome message ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="bot-bubble">
    <div class="avatar-bot">🤖 TitanicBot</div>
    Welcome aboard! 🚢 I'm TitanicBot, your AI analyst for the famous Titanic dataset.
    I can answer questions in plain English and generate charts on the fly.<br><br>
    Try asking:
    <ul>
        <li>📊 <em>"Show me a histogram of passenger ages"</em></li>
        <li>💬 <em>"What percentage of passengers survived?"</em></li>
        <li>🔥 <em>"How did class and gender affect survival rates?"</em></li>
    </ul>
    Or click a <strong>Quick Visualisation</strong> or <strong>Suggested Question</strong> in the sidebar!
    </div>
    """, unsafe_allow_html=True)

# ── Render chat history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="avatar-user">You</div>'
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="avatar-bot">🤖 TitanicBot</div>'
            f'<div class="bot-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("image_b64"):
            with st.container():
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                img = b64_to_image(msg["image_b64"])
                st.image(img, use_container_width=True)

                # Download button for the chart
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download chart",
                    data=buf.getvalue(),
                    file_name=f"{msg.get('chart_type', 'chart')}.png",
                    mime="image/png",
                    use_container_width=False,
                )
                st.markdown("</div>", unsafe_allow_html=True)

# ── Input bar ───────────────────────────────────────────────────────────────
st.markdown("---")
input_col, send_col = st.columns([6, 1])
with input_col:
    user_input = st.text_input(
        label="Chat input",
        label_visibility="collapsed",
        placeholder="Ask anything about the Titanic dataset…",
        value=st.session_state.pending_question,
        key="chat_input",
    )
with send_col:
    send_btn = st.button("Send ➤", type="primary", use_container_width=True)

# Clear pending question after it's been populated into the input
if st.session_state.pending_question:
    st.session_state.pending_question = ""

# ── Process submission ──────────────────────────────────────────────────────
if (send_btn or (user_input and user_input.endswith("\n"))) and user_input.strip():
    question = user_input.strip()

    # Check backend connectivity
    if not st.session_state.dataset_info:
        st.error("⚠️ Backend is not reachable. Please start the FastAPI server first.")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(
        f'<div class="avatar-user">You</div>'
        f'<div class="user-bubble">{question}</div>',
        unsafe_allow_html=True,
    )

    # Call agent
    with st.spinner("🤖 Thinking…"):
        try:
            history = [
                m for m in st.session_state.messages[:-1]  # exclude just-added user msg
                if m["role"] in ("user", "ai")
            ]
            response = send_message(question, history)
            answer   = response.get("text", "Sorry, I couldn't process that.")
            img_b64  = response.get("image_b64")
            chart_tp = response.get("chart_type")
            latency  = response.get("latency_ms", 0)
        except requests.exceptions.ConnectionError:
            answer  = "❌ Could not connect to backend. Is `uvicorn main:app` running?"
            img_b64 = None
            chart_tp = None
            latency  = 0
        except Exception as exc:
            answer  = f"❌ Error: {exc}"
            img_b64 = None
            chart_tp = None
            latency  = 0

    # Append bot message
    bot_msg = {
        "role":       "ai",
        "content":    answer,
        "image_b64":  img_b64,
        "chart_type": chart_tp,
    }
    st.session_state.messages.append(bot_msg)

    # Render answer immediately
    st.markdown(
        f'<div class="avatar-bot">🤖 TitanicBot</div>'
        f'<div class="bot-bubble">{answer}</div>',
        unsafe_allow_html=True,
    )
    if img_b64:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        img = b64_to_image(img_b64)
        st.image(img, use_container_width=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download chart",
            data=buf.getvalue(),
            file_name=f"{chart_tp or 'chart'}.png",
            mime="image/png",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"⏱️ Response time: {latency} ms")
    st.rerun()
