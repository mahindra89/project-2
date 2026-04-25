import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import time
import random
import html
from transformers import AutoTokenizer, BertModel
import plotly.graph_objects as go


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Navy Blue UI Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #061a33;
    color: #f8fafc;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.25), transparent 30%),
        radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.14), transparent 32%),
        linear-gradient(180deg, #071f3d 0%, #061a33 48%, #041326 100%);
}

/* Do not hide Streamlit header completely.
   The sidebar reopen button lives there after sidebar collapse. */
#MainMenu, footer {
    visibility: hidden;
}

header {
    visibility: visible !important;
    background: rgba(6, 26, 51, 0.88) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(148, 163, 184, 0.22);
}

/* Sidebar reopen control */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    color: #f8fafc !important;
}

.block-container {
    padding: 2.5rem 2.5rem 3rem;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #082345;
    border-right: 1px solid rgba(148, 163, 184, 0.22);
}

[data-testid="stSidebar"] .block-container {
    padding: 1.8rem 1.1rem;
}

/* General text */
p, span, label {
    color: inherit;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: rgba(255, 255, 255, 0.08);
    color: #e2e8f0;
    border: 1px solid rgba(203, 213, 225, 0.22);
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    text-align: left;
    cursor: pointer;
    transition: all 0.15s ease;
    margin-bottom: 0.25rem;
}

.stButton > button:hover {
    background: rgba(96, 165, 250, 0.20);
    border-color: #60a5fa;
    color: #ffffff;
}

/* Analyse button */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #3b82f6;
    color: #ffffff;
    border-color: #3b82f6;
    font-weight: 700;
    font-size: 0.85rem;
    text-align: center;
    padding: 0.65rem 1rem;
}

div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
}

/* Textarea */
textarea {
    background: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid rgba(191, 219, 254, 0.9) !important;
    border-radius: 14px !important;
    color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    caret-color: #2563eb !important;
    resize: none !important;
    transition: all 0.15s ease !important;
}

textarea:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.22) !important;
    outline: none !important;
}

/* Result banner */
.result-banner {
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    border: 1px solid;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1rem 0 0.8rem;
    box-shadow: 0 16px 34px rgba(2, 6, 23, 0.22);
}

.result-label {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.result-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #cbd5e1;
    margin-top: 0.2rem;
}

.result-pct {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
}

/* Score grid */
.score-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem;
}

.score-item {
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    border: 1px solid;
    box-shadow: 0 12px 26px rgba(2, 6, 23, 0.18);
}

.score-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #cbd5e1;
    margin-bottom: 0.35rem;
}

.score-val {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1;
}

.bar-track {
    margin-top: 0.5rem;
    height: 4px;
    background: rgba(226, 232, 240, 0.22);
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.sc-pos {
    background: rgba(16,185,129,0.13);
    border-color: rgba(16,185,129,0.30);
}

.sc-neg {
    background: rgba(239,68,68,0.13);
    border-color: rgba(239,68,68,0.30);
}

.sc-neu {
    background: rgba(245,158,11,0.15);
    border-color: rgba(245,158,11,0.32);
}

.sc-irr {
    background: rgba(148,163,184,0.14);
    border-color: rgba(148,163,184,0.30);
}

/* History */
.hist-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(226, 232, 240, 0.16);
    gap: 1rem;
}

.hist-text {
    font-size: 0.78rem;
    color: #cbd5e1;
    flex: 1;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}

.hist-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    padding: 0.16rem 0.5rem;
    border-radius: 999px;
    white-space: nowrap;
}

.hc-pos {
    background: rgba(16,185,129,0.18);
    color: #6ee7b7;
}

.hc-neg {
    background: rgba(239,68,68,0.18);
    color: #fca5a5;
}

.hc-neu {
    background: rgba(245,158,11,0.20);
    color: #fcd34d;
}

.hc-irr {
    background: rgba(148,163,184,0.18);
    color: #cbd5e1;
}

.mono-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    color: #93c5fd;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}

hr {
    border-color: rgba(226, 232, 240, 0.18) !important;
    margin: 1.5rem 0 1rem !important;
}

.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
class BERT_LSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size,
            num_classes
        )

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        bert_output = self.bert(
            input_ids=sent_id,
            attention_mask=mask,
            return_dict=False,
            output_hidden_states=True
        )[0]

        x = self.dropout(bert_output)
        lstm_out, _ = self.lstm(x)

        final_hidden = lstm_out[:, -1, :]
        logits = self.fc(final_hidden)

        return self.softmax(logits)


# ── Constants ─────────────────────────────────────────────────────────────────
LABELS = ["Irrelevant", "Negative", "Neutral", "Positive"]

META = {
    "Positive": {
        "color": "#34d399",
        "bar": "linear-gradient(90deg,#10b981,#6ee7b7)",
        "css": "sc-pos",
        "chip": "hc-pos",
        "border": "rgba(16,185,129,0.34)",
        "bg": "rgba(16,185,129,0.13)"
    },
    "Negative": {
        "color": "#f87171",
        "bar": "linear-gradient(90deg,#ef4444,#fca5a5)",
        "css": "sc-neg",
        "chip": "hc-neg",
        "border": "rgba(239,68,68,0.34)",
        "bg": "rgba(239,68,68,0.13)"
    },
    "Neutral": {
        "color": "#fbbf24",
        "bar": "linear-gradient(90deg,#f59e0b,#fcd34d)",
        "css": "sc-neu",
        "chip": "hc-neu",
        "border": "rgba(245,158,11,0.36)",
        "bg": "rgba(245,158,11,0.15)"
    },
    "Irrelevant": {
        "color": "#cbd5e1",
        "bar": "linear-gradient(90deg,#94a3b8,#cbd5e1)",
        "css": "sc-irr",
        "chip": "hc-irr",
        "border": "rgba(148,163,184,0.36)",
        "bg": "rgba(148,163,184,0.14)"
    },
}

HF_REPO = "puneethas26/sentiment-bert-bilstm"

SAMPLES = [
    "I absolutely love this new update! Best thing ever! 🎉",
    "This is the worst product I have ever purchased. Total waste of money.",
    "Just had a meeting about quarterly results. Numbers look okay.",
    "Watching the game tonight, not sure who'll win.",
    "OMG the customer support team is incredible! Fixed my issue instantly!",
    "I can't believe they removed my favorite feature. So disappointed.",
    "New album drops next Friday. Haven't heard it yet.",
    "Traffic was bad this morning. Finally made it to work.",
]


# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "tweet_area" not in st.session_state:
    st.session_state.tweet_area = ""


# ── Button Callback Functions ─────────────────────────────────────────────────
def set_sample_text(sample_text):
    st.session_state.tweet_area = sample_text


def set_random_sample():
    st.session_state.tweet_area = random.choice(SAMPLES)


def clear_history():
    st.session_state.history = []


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename="best_model.pth"
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = BERT_LSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    return model, tokenizer


@st.cache_resource(show_spinner=False)
def check_model():
    try:
        load_model()
        return True, None
    except Exception as e:
        return False, str(e)


model_ok, model_err = check_model()


# ── Prediction Function ───────────────────────────────────────────────────────
def predict(text):
    model, tokenizer = load_model()

    encoded = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        log_probs = model(
            encoded["input_ids"],
            encoded["attention_mask"]
        )

        probs = torch.exp(log_probs).squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    confidence = float(np.max(probs))

    prob_dict = {
        LABELS[i]: float(probs[i])
        for i in range(len(LABELS))
    }

    return pred_label, confidence, prob_dict


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.4rem;'>
      <div style='font-size:0.95rem;font-weight:700;color:#f8fafc;letter-spacing:-0.01em;'>
        SentimentAI
      </div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#93c5fd;
                  letter-spacing:0.1em;margin-top:0.2rem;'>
        BERT + BiLSTM
      </div>
    </div>
    """, unsafe_allow_html=True)

    dot = "#34d399" if model_ok else "#f87171"
    msg = "Ready" if model_ok else "Failed"
    col = "#6ee7b7" if model_ok else "#fca5a5"

    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:0.45rem;margin-bottom:1.8rem;'>
      <span style='width:7px;height:7px;border-radius:50%;background:{dot};display:inline-block;'></span>
      <span style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:{col};'>
        {msg}
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="mono-label">Samples</div>', unsafe_allow_html=True)

    for i, sample in enumerate(SAMPLES[:6]):
        label = sample[:44] + ("…" if len(sample) > 44 else "")
        st.button(
            label,
            key=f"sample_{i}",
            on_click=set_sample_text,
            args=(sample,)
        )


# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.8rem;'>
  <div style='font-size:1.45rem;font-weight:700;color:#f8fafc;letter-spacing:-0.02em;'>
    Twitter Sentiment Analysis
  </div>
  <div style='font-size:0.72rem;color:#93c5fd;margin-top:0.35rem;
              font-family:JetBrains Mono,monospace;letter-spacing:0.06em;'>
    BERT-base-uncased · BiLSTM · 4-class · 88.4% accuracy
  </div>
</div>
""", unsafe_allow_html=True)


# ── Main Layout ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1.1, 1], gap="large")


with col_l:
    tweet = st.text_area(
        label="tweet",
        label_visibility="collapsed",
        placeholder="Paste or type a tweet…",
        height=100,
        key="tweet_area"
    )

    n = len(tweet)

    c1, c2, c3 = st.columns([1, 1.8, 1])

    with c1:
        clr = "#f87171" if n > 280 else "#cbd5e1"
        st.markdown(
            f"""
            <div style="font-family:JetBrains Mono,monospace;
                        font-size:0.68rem;
                        color:{clr};
                        padding-top:0.55rem;">
                {n}/280
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        go_btn = st.button("Analyse Sentiment", key="analyse")

    with c3:
        st.button(
            "Random",
            key="random_sample",
            on_click=set_random_sample
        )

    if go_btn and tweet.strip():
        if not model_ok:
            st.error(f"Model unavailable: {model_err}")
        else:
            try:
                with st.spinner("Analysing..."):
                    time.sleep(0.05)
                    label, conf, probs = predict(tweet)

                st.session_state.history.insert(0, {
                    "text": tweet,
                    "label": label,
                    "confidence": conf,
                    "probs": probs
                })

                m = META[label]

                st.markdown(f"""
                <div class="result-banner" style="background:{m['bg']};border-color:{m['border']};">
                  <div>
                    <div class="result-label" style="color:{m['color']};">{label}</div>
                    <div class="result-conf">{conf * 100:.1f}% confidence</div>
                  </div>
                  <div class="result-pct" style="color:{m['color']};">{conf * 100:.0f}%</div>
                </div>
                <div class="score-grid">
                """, unsafe_allow_html=True)

                for lbl in LABELS:
                    pm = META[lbl]
                    pct = probs[lbl] * 100

                    st.markdown(f"""
                    <div class="score-item {pm['css']}">
                      <div class="score-name">{lbl}</div>
                      <div class="score-val" style="color:{pm['color']};">{pct:.1f}%</div>
                      <div class="bar-track">
                        <div class="bar-fill" style="width:{pct:.1f}%;background:{pm['bar']};"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

    elif go_btn:
        st.warning("Enter some text first.")


with col_r:
    if st.session_state.history:
        latest = st.session_state.history[0]

        cats = list(latest["probs"].keys())
        vals = [latest["probs"][k] * 100 for k in cats]

        fig = go.Figure(
            go.Bar(
                x=cats,
                y=vals,
                marker=dict(
                    color=[META[k]["color"] for k in cats],
                    opacity=0.85,
                    line=dict(width=0)
                ),
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                textfont=dict(
                    color="#e2e8f0",
                    size=10,
                    family="JetBrains Mono"
                ),
            )
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=15, b=10, l=0, r=0),
            height=240,
            yaxis=dict(
                visible=False,
                range=[0, max(vals) * 1.3 if max(vals) > 0 else 100]
            ),
            xaxis=dict(
                tickfont=dict(
                    color="#e2e8f0",
                    size=10,
                    family="Inter"
                ),
                gridcolor="rgba(226,232,240,0.18)",
                linecolor="rgba(226,232,240,0.18)"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    else:
        for lbl, desc in [
            ("Positive", "Happiness, excitement, praise"),
            ("Negative", "Anger, sadness, criticism"),
            ("Neutral", "Factual, no strong emotion"),
            ("Irrelevant", "Off-topic or spam"),
        ]:
            m = META[lbl]

            st.markdown(f"""
            <div style='display:flex;
                        align-items:center;
                        gap:0.7rem;
                        padding:0.65rem 0;
                        border-bottom:1px solid rgba(226,232,240,0.16);'>
              <div style='width:7px;
                          height:7px;
                          border-radius:50%;
                          background:{m["color"]};
                          flex-shrink:0;'>
              </div>
              <div style='font-size:0.8rem;
                          font-weight:700;
                          color:{m["color"]};
                          width:76px;
                          flex-shrink:0;'>
                {lbl}
              </div>
              <div style='font-size:0.75rem;color:#cbd5e1;'>
                {desc}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="mono-label">Recent</div>', unsafe_allow_html=True)

    h1, h2 = st.columns(2, gap="medium")

    for i, item in enumerate(st.session_state.history[:8]):
        m = META[item["label"]]

        safe_preview = html.escape(
            item["text"][:72] + ("…" if len(item["text"]) > 72 else "")
        )

        block = f"""
        <div class="hist-item">
          <div class="hist-text">{safe_preview}</div>
          <span class="hist-chip {m['chip']}">
            {item['label']} {item['confidence'] * 100:.0f}%
          </span>
        </div>
        """

        if i % 2 == 0:
            h1.markdown(block, unsafe_allow_html=True)
        else:
            h2.markdown(block, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.button(
        "Clear",
        key="clear_history",
        on_click=clear_history
    )
