import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import time
import random
from transformers import AutoTokenizer, BertModel
import plotly.graph_objects as go

st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #07090f;
    color: #e2e8f0;
}
.stApp { background: #07090f; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2.5rem 3rem; max-width: 1200px; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0b0e18; border-right: 1px solid #0f1420; }
[data-testid="stSidebar"] .block-container { padding: 1.8rem 1.1rem; }

/* All buttons default — sidebar samples */
.stButton > button {
    width: 100%;
    background: transparent;
    color: #334155;
    border: 1px solid #0f1420;
    border-radius: 7px;
    padding: 0.5rem 0.8rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 400;
    text-align: left;
    cursor: pointer;
    transition: all 0.12s;
    margin-bottom: 0.25rem;
}
.stButton > button:hover {
    background: #0f1420;
    border-color: #1a2d4a;
    color: #64748b;
}

/* Analyse button — second column */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #0f1e38;
    color: #7dd3fc;
    border-color: #1a3050;
    font-weight: 600;
    font-size: 0.85rem;
    text-align: center;
    padding: 0.62rem 1rem;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: #142540;
    border-color: #2563a8;
    color: #bae6fd;
}

/* Textarea */
textarea {
    background: #0b0e18 !important;
    border: 1px solid #0f1420 !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: #38bdf8 !important;
    resize: none !important;
    transition: border-color 0.15s !important;
}
textarea:focus {
    border-color: #1a3050 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Result banner */
.result-banner {
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border: 1px solid;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1rem 0 0.8rem;
}
.result-label { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em; }
.result-conf  { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #334155; margin-top: 0.2rem; }
.result-pct   { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; }

/* Score grid */
.score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.score-item { border-radius: 8px; padding: 0.8rem 0.9rem; border: 1px solid; }
.score-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1e2d42;
    margin-bottom: 0.3rem;
}
.score-val { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
.bar-track { margin-top: 0.45rem; height: 2px; background: rgba(255,255,255,0.04); border-radius: 2px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 2px; }

.sc-pos { background: rgba(16,185,129,0.04);  border-color: rgba(16,185,129,0.12); }
.sc-neg { background: rgba(239,68,68,0.04);   border-color: rgba(239,68,68,0.12); }
.sc-neu { background: rgba(245,158,11,0.04);  border-color: rgba(245,158,11,0.12); }
.sc-irr { background: rgba(100,116,139,0.04); border-color: rgba(100,116,139,0.12); }

/* History */
.hist-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid #0b0e18;
    gap: 1rem;
}
.hist-text { font-size: 0.78rem; color: #1e2d42; flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.hist-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    padding: 0.12rem 0.45rem;
    border-radius: 4px;
    white-space: nowrap;
}
.hc-pos { background: rgba(16,185,129,0.1);  color: #34d399; }
.hc-neg { background: rgba(239,68,68,0.1);   color: #f87171; }
.hc-neu { background: rgba(245,158,11,0.1);  color: #fbbf24; }
.hc-irr { background: rgba(100,116,139,0.1); color: #475569; }

.mono-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    color: #1a2535;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}

hr { border-color: #0b0e18 !important; margin: 1.5rem 0 1rem !important; }
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
class BERT_LSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.bert    = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.lstm    = nn.LSTM(768, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc      = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        x = self.dropout(self.bert(input_ids=sent_id, attention_mask=mask,
                                   return_dict=False, output_hidden_states=True)[0])
        lstm_out, _ = self.lstm(x)
        return self.softmax(self.fc(lstm_out[:, -1, :]))


LABELS = ["Irrelevant", "Negative", "Neutral", "Positive"]
META   = {
    "Positive":   {"color": "#10b981", "bar": "linear-gradient(90deg,#10b981,#34d399)", "css": "sc-pos", "chip": "hc-pos", "border": "rgba(16,185,129,0.18)",  "bg": "rgba(16,185,129,0.04)"},
    "Negative":   {"color": "#ef4444", "bar": "linear-gradient(90deg,#ef4444,#f87171)", "css": "sc-neg", "chip": "hc-neg", "border": "rgba(239,68,68,0.18)",   "bg": "rgba(239,68,68,0.04)"},
    "Neutral":    {"color": "#f59e0b", "bar": "linear-gradient(90deg,#f59e0b,#fbbf24)", "css": "sc-neu", "chip": "hc-neu", "border": "rgba(245,158,11,0.18)",  "bg": "rgba(245,158,11,0.04)"},
    "Irrelevant": {"color": "#475569", "bar": "linear-gradient(90deg,#334155,#475569)", "css": "sc-irr", "chip": "hc-irr", "border": "rgba(100,116,139,0.18)", "bg": "rgba(100,116,139,0.04)"},
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

# ── State ─────────────────────────────────────────────────────────────────────
if "history"    not in st.session_state: st.session_state.history    = []
if "tweet_area" not in st.session_state: st.session_state.tweet_area = ""


# ── Load (auto, cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_REPO, filename="best_model.pth")
    tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    m    = BERT_LSTM()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m, tok

@st.cache_resource(show_spinner=False)
def check_model():
    try:    load_model(); return True, None
    except Exception as e: return False, str(e)

model_ok, model_err = check_model()


def predict(text):
    model, tok = load_model()
    enc = tok(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        probs = torch.exp(model(enc["input_ids"], enc["attention_mask"])).squeeze().numpy()
    idx = int(np.argmax(probs))
    return LABELS[idx], float(np.max(probs)), {LABELS[i]: float(probs[i]) for i in range(4)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.4rem;'>
      <div style='font-size:0.9rem;font-weight:700;color:#94a3b8;letter-spacing:-0.01em;'>SentimentAI</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#1a2535;
                  letter-spacing:0.1em;margin-top:0.2rem;'>BERT + BiLSTM</div>
    </div>
    """, unsafe_allow_html=True)

    dot = "#10b981" if model_ok else "#ef4444"
    msg = "Ready" if model_ok else "Failed"
    col = "#1e3a2e" if model_ok else "#3b1212"
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:0.45rem;margin-bottom:1.8rem;'>
      <span style='width:5px;height:5px;border-radius:50%;background:{dot};display:inline-block;'></span>
      <span style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:{col};'>{msg}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="mono-label">Samples</div>', unsafe_allow_html=True)
    for tw in SAMPLES[:6]:
        label = tw[:44] + ("…" if len(tw) > 44 else "")
        if st.button(label, key=f"s_{tw[:12]}"):
            st.session_state.tweet_area = tw
            st.rerun()


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.8rem;'>
  <div style='font-size:1.35rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.02em;'>
    Twitter Sentiment Analysis
  </div>
  <div style='font-size:0.72rem;color:#1a2535;margin-top:0.3rem;font-family:JetBrains Mono,monospace;letter-spacing:0.06em;'>
    BERT-base-uncased · BiLSTM · 4-class · 88.4% accuracy
  </div>
</div>
""", unsafe_allow_html=True)

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
        clr = "#ef4444" if n > 280 else "#1a2535"
        st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:{clr};padding-top:0.55rem;">{n}/280</div>', unsafe_allow_html=True)
    with c2:
        go_btn = st.button("Analyse Sentiment", key="analyse")
    with c3:
        if st.button("Random", key="rnd"):
            st.session_state.tweet_area = random.choice(SAMPLES)
            st.rerun()

    if go_btn and tweet.strip():
        if not model_ok:
            st.error(f"Model unavailable: {model_err}")
        else:
            try:
                with st.spinner(""):
                    time.sleep(0.05)
                    label, conf, probs = predict(tweet)

                st.session_state.history.insert(0, {
                    "text": tweet, "label": label, "confidence": conf, "probs": probs
                })

                m = META[label]
                st.markdown(f"""
                <div class="result-banner" style="background:{m['bg']};border-color:{m['border']};">
                  <div>
                    <div class="result-label" style="color:{m['color']};">{label}</div>
                    <div class="result-conf">{conf*100:.1f}% confidence</div>
                  </div>
                  <div class="result-pct" style="color:{m['color']};">{conf*100:.0f}%</div>
                </div>
                <div class="score-grid">
                """, unsafe_allow_html=True)
                for lbl in LABELS:
                    pm  = META[lbl]
                    pct = probs[lbl] * 100
                    st.markdown(f"""
                    <div class="score-item {pm['css']}">
                      <div class="score-name">{lbl}</div>
                      <div class="score-val" style="color:{pm['color']};">{pct:.1f}%</div>
                      <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{pm['bar']};"></div></div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

    elif go_btn:
        st.warning("Enter some text first.")


with col_r:
    if st.session_state.history:
        latest = st.session_state.history[0]
        cats   = list(latest["probs"].keys())
        vals   = [latest["probs"][k] * 100 for k in cats]

        fig = go.Figure(go.Bar(
            x=cats, y=vals,
            marker=dict(color=[META[k]["color"] for k in cats], opacity=0.65, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color="#1e2d42", size=10, family="JetBrains Mono"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=15, b=10, l=0, r=0), height=240,
            yaxis=dict(visible=False, range=[0, max(vals) * 1.3]),
            xaxis=dict(tickfont=dict(color="#1e2d42", size=10, family="Inter"),
                       gridcolor="#0b0e18", linecolor="#0b0e18"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        for lbl, desc in [
            ("Positive",   "Happiness, excitement, praise"),
            ("Negative",   "Anger, sadness, criticism"),
            ("Neutral",    "Factual, no strong emotion"),
            ("Irrelevant", "Off-topic or spam"),
        ]:
            m = META[lbl]
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.7rem;padding:0.55rem 0;border-bottom:1px solid #0b0e18;'>
              <div style='width:5px;height:5px;border-radius:50%;background:{m["color"]};flex-shrink:0;'></div>
              <div style='font-size:0.8rem;font-weight:600;color:{m["color"]};width:76px;flex-shrink:0;'>{lbl}</div>
              <div style='font-size:0.75rem;color:#1a2535;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="mono-label">Recent</div>', unsafe_allow_html=True)

    h1, h2 = st.columns(2, gap="medium")
    for i, item in enumerate(st.session_state.history[:8]):
        m       = META[item["label"]]
        preview = item["text"][:72] + ("…" if len(item["text"]) > 72 else "")
        block   = f"""
        <div class="hist-item">
          <div class="hist-text">{preview}</div>
          <span class="hist-chip {m['chip']}">{item['label']} {item['confidence']*100:.0f}%</span>
        </div>"""
        (h1 if i % 2 == 0 else h2).markdown(block, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Clear", key="clear"):
        st.session_state.history = []
        st.rerun()
