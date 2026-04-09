# Legum AI v2 — Supreme Court of Pakistan Dataset
### Samsung AI Training Program — Final Project (Updated)

Real AI-powered legal recommendation engine using **1,414 actual Supreme Court of Pakistan judgments**
from HuggingFace: `Ibtehaj10/supreme-court-of-pak-judgments`

---

## What Changed from v1

| Component | v1 | v2 |
|---|---|---|
| Dataset | 10 handcrafted sample cases | 1,414 real Supreme Court judgments |
| Embeddings | Sentence-BERT re-encoded | Pre-computed HF embeddings (1024-dim, mxbai-embed-large-v1) |
| TF-IDF vocab | 5,000 terms | 30,000 terms |
| BiLSTM seq length | 200 tokens | 512 tokens |
| BiLSTM vocab | 10,000 | 30,000 |
| Risk model features | 500 | 5,000 |
| Data augmentation | Required (small dataset) | Removed (1,414 real samples) |

---

## Project Structure

```
legum-ai-v2/
├── data/
│   ├── hf_dataset_loader.py   ← NEW: HuggingFace dataset loader
│   ├── cases_dataset.py       ← UPDATED: uses real HF data
│   └── preprocessor.py       ← unchanged
├── models/
│   ├── tfidf_retriever.py     ← UPDATED: 30k vocab, min_df=2
│   ├── bert_embedder.py       ← UPDATED: uses HF pre-computed embeddings
│   ├── bilstm_classifier.py   ← UPDATED: 512 seq len, 30k vocab, no augmentation
│   ├── risk_predictor.py      ← UPDATED: 5k features, deeper trees
│   └── recommendation_engine.py ← unchanged
├── utils/
│   ├── file_parser.py         ← unchanged
│   ├── helpers.py             ← unchanged
│   └── evaluator.py           ← UPDATED: new charts for real dataset
├── api/app.py                 ← unchanged
├── static/index.html          ← unchanged
├── main.py                    ← unchanged
├── train.py                   ← UPDATED: HF dataset pipeline
├── demo.py                    ← UPDATED: real case examples
├── colab_runner.ipynb         ← NEW: complete Colab notebook
└── requirements.txt           ← UPDATED: added datasets library
```
