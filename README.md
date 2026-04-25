# SentimentAI · Twitter Sentiment Analysis

A deep learning web app that classifies tweets into **4 sentiment categories** using a hybrid **BERT + BiLSTM** architecture.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://twitter-sentiment-analysis-vvwqu3ohssprazxy7nf7q6.streamlit.app/)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/puneethas26/sentiment-bert-bilstm)

---

## Model Architecture

| Layer | Details |
|---|---|
| BERT Encoder | bert-base-uncased |
| Dropout | p = 0.10 |
| BiLSTM | 2 layers · hidden size = 128 |
| Linear | 256 → 4 classes |
| Log-Softmax | Output activation |

## Sentiment Classes

| Class | Description |
|---|---|
| Positive | Happiness, excitement, praise |
| Negative | Anger, sadness, criticism |
| Neutral | Informational, no strong emotion |
| Irrelevant | Spam or off-topic content |

## Performance

| Metric | Score |
|---|---|
| Accuracy | 88.4% |
| Macro Precision | 89% |
| Macro Recall | 87% |
| Macro F1 | 88% |

## Dataset

- 74,681 tweets across train/val/test splits
- 4 balanced sentiment classes
- Real-world Twitter data

## Tech Stack

- **Frontend**: Streamlit
- **Model**: PyTorch + HuggingFace Transformers
- **Visualization**: Plotly
- **Deployment**: Streamlit Community Cloud
- **Model Hosting**: HuggingFace Hub

## Run Locally

```bash
git clone https://github.com/puneethas26/sentiment-bert-bilstm.git
cd sentiment-bert-bilstm
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

The model weights are fetched automatically from HuggingFace Hub on first load — no manual setup required.
