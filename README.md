# Verdict AI — Legal Intelligence & Case Recommendation System

Verdict AI is a full-stack legal intelligence platform designed to assist legal professionals with case understanding, prioritization, and legal drafting support. It combines a modern React + Node.js frontend with a Python-based machine learning backend and integrates Google Gemini for natural language reasoning.

The system is built around three core capabilities:
- Legal case prioritization
- Case recommendation and similarity matching
- AI-assisted legal reasoning using LLMs

It is deployed as a cloud-based system and structured as a modular monorepo.

---

# System Architecture

Verdict AI follows a multi-layer architecture:

## Frontend (Node.js + React + Vite)

A modern TypeScript-based UI layer responsible for:
- Chat-based interaction with AI assistant
- Case input and visualization
- Recommendation and analysis display
- Integration with backend services

Key modules:
- services/chatbot
- services/legal
- services/recommendation
- services/judiciary
- services/transcription

---

## Backend (Python ML & Legal Engines)

A dedicated Python backend responsible for legal intelligence processing.

### Case Priority Engine
- Extracts legal signals from case documents
- Scores cases based on urgency and relevance
- Hybrid rule-based + ML scoring system

Core files:
- backend/case-priority-engine/scorer.py
- backend/case-priority-engine/signal_extractor.py
- backend/case-priority-engine/pdf_extractor.py

---

### Case Recommendation Engine
- Retrieves similar legal cases
- Uses embedding-based similarity search
- Ranks results using ML models

Models used:
- TF-IDF Retriever
- BERT Embedder
- BiLSTM Classifier
- Risk Predictor

---

## AI Layer (Gemini Integration)

Google Gemini is used for:
- Natural language legal reasoning
- Chat-based legal assistance
- Explanation of case insights
- Enhancing ML-generated outputs

File:
- gemini code/lib/services/gemini.ts

---

# Core Features

- Case priority scoring based on urgency signals
- AI-powered legal case recommendation system
- Semantic similarity search across legal datasets
- Hybrid ML + LLM reasoning pipeline
- PDF case ingestion and parsing
- Chat-based legal assistant interface
- Multi-domain legal intelligence (judiciary, recommendation, transcription)

---

# Tech Stack

## Frontend
- React (TypeScript)
- Vite
- Node.js
- CSS

## Backend
- Python
- Scikit-learn
- PyTorch
- Pandas
- NLP models

## AI / ML Layer
- Google Gemini API
- BERT embeddings
- BiLSTM classifier
- TF-IDF retrieval

## Infrastructure
- Monorepo architecture
- REST API communication
- Modular service-based system

---

# Project Structure

```text
Verdict-AI/
│
├── backend/
│   ├── case-priority-engine/
│   ├── case-recommendation-engine/
│   ├── utils/
│   ├── train.py
│   ├── main.py
│   └── requirements.txt
│
├── gemini code/
│   ├── lib/
│   │   ├── priorityEngine.ts
│   │   ├── utils.ts
│   │   └── services/
│   │       ├── chatbot/
│   │       ├── legal/
│   │       ├── recommendation/
│   │       ├── transcription/
│   │       └── gemini.ts
│
├── server.ts
├── vite.config.ts
├── tsconfig.json
├── package.json
└── index.html
