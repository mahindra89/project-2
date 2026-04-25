import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import time
import random
from transformers import AutoTokenizer, BertModel
import plotly.graph_objects as go
from collections import Counter

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}
.stApp { background: #080c14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0b0f1a;
    border-right: 1px solid #1a2235;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d1829 0%, #0a1020 50%, #0f1525 100%);
    border: 1px solid #1a2d4a;
    border-radius: 18px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px; height: 1px;
    background: #38bdf8;
    opacity: 0.6;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.03em;
    color: #f1f5f9;
    margin-bottom: 0.9rem;
}
.hero-title span {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-desc {
    color: #64748b;
    font-size: 0.95rem;
    max-width: 520px;
    line-height: 1.65;
}
.badge-row { display: flex; gap: 0.5rem; margin-top: 1.4rem; flex-wrap: wrap; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.28rem 0.7rem;
    border-radius: 6px;
    border: 1px solid;
    letter-spacing: 0.03em;
}
.badge-sky     { color: #7dd3fc; border-color: #1e3a52; background: rgba(56,189,248,0.06); }
.badge-violet  { color: #c4b5fd; border-color: #2d2060; background: rgba(167,139,250,0.06); }
.badge-emerald { color: #6ee7b7; border-color: #1a4032; background: rgba(52,211,153,0.06); }

/* ── Stats ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: #0d1117;
    border: 1px solid #1a2235;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.stat-val {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #38bdf8;
    line-height: 1;
}
.stat-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.3rem;
}

/* ── Card ── */
.card {
    background: #0d1117;
    border: 1px solid #1a2235;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.card-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.card-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a2235;
}

/* ── Inputs ── */
textarea {
    background: #0b0f1a !important;
    border: 1px solid #1a2d4a !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    caret-color: #38bdf8 !important;
    resize: vertical !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
    outline: none !important;
}
.char-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding-top: 0.6rem;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: #0f1e38;
    color: #7dd3fc;
    border: 1px solid #1e3a5a;
    border-radius: 9px;
    padding: 0.7rem 1.4rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    cursor: pointer;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #162a4a;
    border-color: #38bdf8;
    color: #e0f2fe;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Result ── */
.winner {
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1.2rem;
    border: 1px solid;
}
.winner-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    margin-bottom: 0.5rem;
}
.winner-label {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.winner-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #64748b;
}
.score-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 0.5rem;
}
.score-card {
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.score-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 10px 10px 0 0;
}
.sc-pos { background: rgba(16,185,129,0.05); border-color: rgba(16,185,129,0.2); }
.sc-pos::after { background: #10b981; }
.sc-neg { background: rgba(239,68,68,0.05); border-color: rgba(239,68,68,0.2); }
.sc-neg::after { background: #ef4444; }
.sc-neu { background: rgba(245,158,11,0.05); border-color: rgba(245,158,11,0.2); }
.sc-neu::after { background: #f59e0b; }
.sc-irr { background: rgba(100,116,139,0.05); border-color: rgba(100,116,139,0.2); }
.sc-irr::after { background: #64748b; }
.score-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 0.4rem;
}
.score-pct {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1;
}
.bar-track {
    margin-top: 0.6rem;
    height: 3px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 999px; }

/* ── Sidebar ── */
.sidebar-block {
    background: #0f1420;
    border: 1px solid #1a2235;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}
.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}
.status-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

/* ── History ── */
.hist-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem 0.9rem;
    background: #0b0f1a;
    border: 1px solid #1a2235;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    gap: 0.8rem;
}
.hist-text { font-size: 0.82rem; color: #64748b; flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.hist-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    padding: 0.18rem 0.55rem;
    border-radius: 5px;
    white-space: nowrap;
    font-weight: 500;
}
.hc-pos { background: rgba(16,185,129,0.12); color: #34d399; }
.hc-neg { background: rgba(239,68,68,0.12); color: #f87171; }
.hc-neu { background: rgba(245,158,11,0.12); color: #fbbf24; }
.hc-irr { background: rgba(100,116,139,0.12); color: #94a3b8; }

hr { border-color: #1a2235 !important; }
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Definition ────────────────────────────────────────────────────────
class BERT_LSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            input_size=768, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(input_ids=sent_id, attention_mask=mask,
                           return_dict=False, output_hidden_states=True)
        x = self.dropout(cls_hs[0])
        lstm_out, _ = self.lstm(x)
        return self.softmax(self.fc(lstm_out[:, -1, :]))


# ─── Constants ───────────────────────────────────────────────────────────────
LABELS = ["Irrelevant", "Negative", "Neutral", "Positive"]

LABEL_META = {
    "Positive":   {"color": "#10b981", "bar": "linear-gradient(90deg,#10b981,#34d399)", "css": "sc-pos", "chip": "hc-pos", "win_border": "rgba(16,185,129,0.25)",  "win_bg": "rgba(16,185,129,0.05)"},
    "Negative":   {"color": "#ef4444", "bar": "linear-gradient(90deg,#ef4444,#f87171)", "css": "sc-neg", "chip": "hc-neg", "win_border": "rgba(239,68,68,0.25)",   "win_bg": "rgba(239,68,68,0.05)"},
    "Neutral":    {"color": "#f59e0b", "bar": "linear-gradient(90deg,#f59e0b,#fbbf24)", "css": "sc-neu", "chip": "hc-neu", "win_border": "rgba(245,158,11,0.25)",  "win_bg": "rgba(245,158,11,0.05)"},
    "Irrelevant": {"color": "#64748b", "bar": "linear-gradient(90deg,#475569,#64748b)", "css": "sc-irr", "chip": "hc-irr", "win_border": "rgba(100,116,139,0.25)", "win_bg": "rgba(100,116,139,0.05)"},
}

HF_REPO_ID = "puneethas26/sentiment-bert-bilstm"

SAMPLE_TWEETS = [
    "I absolutely love this new update! Best thing ever! 🎉",
    "This is the worst product I have ever purchased. Total waste of money.",
    "Just had a meeting about quarterly results. Numbers look okay.",
    "Watching the game tonight, not sure who'll win.",
    "OMG the customer support team is incredible! Fixed my issue instantly! 💯",
    "I can't believe they removed my favorite feature. So disappointed 😤",
    "New album drops next Friday. Haven't heard it yet.",
    "Traffic was bad this morning. Finally made it to work.",
]

# ─── Session State ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "tweet_area" not in st.session_state:
    st.session_state.tweet_area = ""


# ─── Model — auto-load on startup ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="best_model.pth")
    tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
    model      = BERT_LSTM(num_classes=4, hidden_size=128, num_layers=2, bidirectional=True)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer

# Trigger load immediately — result cached, so only runs once
@st.cache_resource(show_spinner=False)
def get_model_status():
    try:
        load_model_and_tokenizer()
        return True, None
    except Exception as e:
        return False, str(e)

model_ok, model_err = get_model_status()


# ─── Inference ───────────────────────────────────────────────────────────────
def predict(text, model, tokenizer):
    enc = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        probs = torch.exp(model(enc["input_ids"], enc["attention_mask"])).squeeze().numpy()
    idx = int(np.argmax(probs))
    return LABELS[idx], float(np.max(probs)), {LABELS[i]: float(probs[i]) for i in range(4)}


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.4rem;'>
      <div style='font-size:1.05rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.01em;'>SentimentAI</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#334155;letter-spacing:0.1em;margin-top:0.25rem;'>BERT + BiLSTM · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model status (auto) ──
    st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    if model_ok:
        st.markdown("""
        <div class="status-row">
          <span class="status-dot" style="background:#10b981;"></span>
          <span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#34d399;'>Ready</span>
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#1e3a52;margin-top:0.4rem;'>bert-base-uncased + BiLSTM</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-row">
          <span class="status-dot" style="background:#ef4444;"></span>
          <span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#f87171;'>Failed to load</span>
        </div>
        <div style='font-size:0.72rem;color:#475569;margin-top:0.4rem;'>{model_err}</div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Sample tweets ──
    st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Try a Sample</div>', unsafe_allow_html=True)
    for tw in SAMPLE_TWEETS[:6]:
        preview = tw[:42] + ("…" if len(tw) > 42 else "")
        if st.button(preview, key=f"s_{tw[:15]}"):
            st.session_state.tweet_area = tw
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Main Layout ─────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Deep Learning · NLP Pipeline</div>
  <div class="hero-title">Twitter Sentiment<br><span>Intelligence</span></div>
  <div class="hero-desc">
    BERT-base-uncased encoder stacked with a bidirectional LSTM —
    classifying tweets across 4 sentiment categories with ~90% accuracy.
  </div>
  <div class="badge-row">
    <span class="badge badge-sky">BERT-base-uncased</span>
    <span class="badge badge-violet">BiLSTM · 2 Layers</span>
    <span class="badge badge-emerald">88.4% Accuracy</span>
    <span class="badge badge-sky">4-Class</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stat-row">
  <div class="stat-card"><div class="stat-val">74,681</div><div class="stat-lbl">Training Tweets</div></div>
  <div class="stat-card"><div class="stat-val">88.4%</div><div class="stat-lbl">Test Accuracy</div></div>
  <div class="stat-card"><div class="stat-val">4</div><div class="stat-lbl">Sentiment Classes</div></div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Tweet Input</div>', unsafe_allow_html=True)

    tweet_text = st.text_area(
        label="tweet",
        label_visibility="collapsed",
        placeholder="Type or paste a tweet here…",
        height=120,
        key="tweet_area"
    )

    char_count = len(tweet_text)
    c1, c2, c3 = st.columns([1, 1.6, 1.3])
    with c1:
        cc = "#ef4444" if char_count > 280 else "#334155"
        st.markdown(f'<div class="char-count" style="color:{cc};">{char_count}/280</div>', unsafe_allow_html=True)
    with c2:
        analyse_btn = st.button("Analyse Sentiment", key="analyse")
    with c3:
        if st.button("Random Sample", key="random"):
            st.session_state.tweet_area = random.choice(SAMPLE_TWEETS)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Inference ──
    if analyse_btn and tweet_text.strip():
        if not model_ok:
            st.error(f"Model unavailable: {model_err}")
        else:
            try:
                model, tokenizer = load_model_and_tokenizer()
                with st.spinner("Analysing…"):
                    time.sleep(0.1)
                    label, confidence, probs = predict(tweet_text, model, tokenizer)

                st.session_state.history.insert(0, {
                    "text": tweet_text, "label": label,
                    "confidence": confidence, "probs": probs,
                })

                meta = LABEL_META[label]
                st.markdown(f"""
                <div class="winner" style="background:{meta['win_bg']};border-color:{meta['win_border']};">
                  <div class="winner-eyebrow">Predicted Sentiment</div>
                  <div class="winner-label" style="color:{meta['color']};">{label}</div>
                  <div class="winner-conf">Confidence · {confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="score-grid">', unsafe_allow_html=True)
                for lbl in LABELS:
                    m   = LABEL_META[lbl]
                    pct = probs[lbl] * 100
                    st.markdown(f"""
                    <div class="score-card {m['css']}">
                      <div class="score-name">{lbl}</div>
                      <div class="score-pct" style="color:{m['color']};">{pct:.1f}%</div>
                      <div class="bar-track">
                        <div class="bar-fill" style="width:{pct:.1f}%;background:{m['bar']};"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inference error: {e}")

    elif analyse_btn:
        st.warning("Please enter some tweet text first.")

with col_right:
    if st.session_state.history:
        latest = st.session_state.history[0]
        probs  = latest["probs"]
        cats   = list(probs.keys())
        vals   = [probs[k] * 100 for k in cats]

        # Radar
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Probability Radar</div>', unsafe_allow_html=True)
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill='toself',
            fillcolor='rgba(56,189,248,0.08)',
            line=dict(color='#38bdf8', width=1.8),
            marker=dict(color='#38bdf8', size=5),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(color='#334155', size=9, family='JetBrains Mono'),
                                gridcolor='#1a2235', linecolor='#1a2235'),
                angularaxis=dict(tickfont=dict(color='#64748b', size=10, family='Inter'),
                                 gridcolor='#1a2235', linecolor='#1a2235'),
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=10, l=40, r=40), height=260,
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Bar
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Score Breakdown</div>', unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            x=cats, y=vals,
            marker=dict(color=[LABEL_META[k]["color"] for k in cats], opacity=0.75,
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            text=[f"{v:.1f}%" for v in vals],
            textposition='outside',
            textfont=dict(color='#64748b', size=10, family='JetBrains Mono'),
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=15, l=10, r=10), height=220,
            yaxis=dict(visible=False, range=[0, max(vals) * 1.28]),
            xaxis=dict(tickfont=dict(color='#475569', size=10, family='Inter'),
                       gridcolor='#1a2235', linecolor='#1a2235'),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # How it works
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">How It Works</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("1", "Tokenize",  "BERT WordPiece tokenizer converts text → token IDs (max 128)"),
            ("2", "Encode",    "BERT extracts deep contextual embeddings from all transformer layers"),
            ("3", "Sequence",  "BiLSTM processes the sequence forward and backward simultaneously"),
            ("4", "Classify",  "Linear layer maps BiLSTM output → 4 sentiment logits"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:0.9rem;margin-bottom:0.9rem;align-items:flex-start;'>
              <div style='min-width:24px;height:24px;background:rgba(56,189,248,0.1);
                          border:1px solid rgba(56,189,248,0.2);border-radius:6px;
                          display:flex;align-items:center;justify-content:center;
                          font-family:JetBrains Mono,monospace;font-weight:600;font-size:0.72rem;
                          color:#38bdf8;flex-shrink:0;'>{num}</div>
              <div>
                <div style='font-size:0.85rem;font-weight:600;color:#cbd5e1;margin-bottom:0.2rem;'>{title}</div>
                <div style='font-size:0.78rem;color:#475569;line-height:1.5;'>{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Sentiment Classes</div>', unsafe_allow_html=True)
        for lbl, desc in [
            ("Positive",   "Happiness, excitement, praise"),
            ("Negative",   "Anger, sadness, criticism"),
            ("Neutral",    "Factual, no strong emotion"),
            ("Irrelevant", "Off-topic or spam content"),
        ]:
            m = LABEL_META[lbl]
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.75rem;padding:0.55rem 0;border-bottom:1px solid #111827;'>
              <div style='width:7px;height:7px;border-radius:50%;background:{m["color"]};flex-shrink:0;'></div>
              <div>
                <div style='font-size:0.83rem;font-weight:600;color:{m["color"]};'>{lbl}</div>
                <div style='font-size:0.74rem;color:#334155;margin-top:0.1rem;'>{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─── History ─────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;font-weight:600;color:#334155;text-transform:uppercase;
                letter-spacing:0.09em;margin-bottom:0.9rem;'>Analysis History</div>
    """, unsafe_allow_html=True)

    h1, h2 = st.columns(2, gap="medium")
    for i, item in enumerate(st.session_state.history[:10]):
        m       = LABEL_META[item["label"]]
        preview = item["text"][:65] + ("…" if len(item["text"]) > 65 else "")
        html = f"""
        <div class="hist-row">
          <div class="hist-text">{preview}</div>
          <span class="hist-chip {m['chip']}">{item['label']} · {item['confidence']*100:.0f}%</span>
        </div>
        """
        (h1 if i % 2 == 0 else h2).markdown(html, unsafe_allow_html=True)

    if len(st.session_state.history) >= 3:
        counts = Counter(h["label"] for h in st.session_state.history)
        st.markdown('<div class="card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Session Distribution</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            marker=dict(colors=[LABEL_META[k]["color"] for k in counts.keys()],
                        line=dict(color='#080c14', width=3)),
            textfont=dict(family='JetBrains Mono', size=11, color='#e2e8f0'),
            hole=0.58,
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(font=dict(color='#64748b', family='JetBrains Mono', size=11),
                        bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=10, b=10, l=10, r=10), height=240,
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Clear History", key="clear_hist"):
        st.session_state.history = []
        st.rerun()


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;font-family:JetBrains Mono,monospace;font-size:0.65rem;
            color:#1a2235;padding:2rem 0 1rem;'>
  SentimentAI · BERT + BiLSTM · 4-Class Twitter Sentiment Analysis
</div>
""", unsafe_allow_html=True)
