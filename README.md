# 🧠 AI MINDS — Intelligent Personal Knowledge System

> An AI-powered cognitive assistant that converts raw personal data into structured, searchable memory with natural language interaction. Built for the AI MINDS hackathon.

## What It Does

1. **Automatically ingests** personal data from multiple sources — files (PDF, DOCX, TXT, images), web pages (via browser extension), clipboard, and watched directories
2. **Understands content** using local embedding models (sentence-transformers + CLIP) — no cloud APIs needed
3. **Organizes information** into meaningful categories (work, learning, finance, health, personal, code, news)
4. **Extracts action items** and produces concise summaries
5. **Answers questions** with grounded, referenced answers using Ollama LLM (≤4B params)
6. **Handles uncertainty** — reports confidence scores and explicitly states when evidence is weak
7. **Runs continuously** with persistent memory, directory watching, and background automations

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Chrome Extension                       │
│  • Element selection (Ctrl+Shift+F)                      │
│  • Full page capture (Ctrl+Shift+S)                      │
│  • Popup: query memory, daily digest                     │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP (JSON)
┌────────────────────────▼─────────────────────────────────┐
│                  Flask Backend (port 5000)                │
│  • /api/extract — element extraction ingestion           │
│  • /api/auto-capture — full page ingestion               │
│  • /index — memex-style web clip ingestion               │
│  • /query — RAG: search + Ollama LLM answer              │
│  • /chat/stream — streaming chat (SSE)                   │
│  • /ingest/file, /ingest/text, /ingest/scan              │
│  • /stats, /actions, /digest                             │
│  • /config — runtime configuration                       │
├──────────────────────────────────────────────────────────┤
│                    AI MINDS Engine                        │
│  • Hybrid search: semantic (cosine) + lexical (BM25)     │
│  • MMR reranking for diversity                           │
│  • ChromaDB vector store + SQLite metadata               │
│  • Auto-categorization, summaries, action extraction     │
│  • Directory watcher (background thread)                 │
│  • Ollama LLM integration (qwen2.5:3b default)          │
├──────────────────────────────────────────────────────────┤
│                    Automations                            │
│  • Clipboard monitor (auto-ingest copied text)           │
│  • Scheduled daily digests                               │
│  • Smart reminders from action items                     │
└──────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              Streamlit Dashboard (port 8501)              │
│  • Chat with memory (full RAG UI)                        │
│  • Knowledge dashboard (stats, categories, charts)       │
│  • Action items manager                                  │
│  • Manual ingestion (text, files, directory scan)        │
│  • Settings panel                                        │
└──────────────────────────────────────────────────────────┘
```

## Models Used (All Local, ≤4B Params)

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Text Embeddings | `BAAI/bge-small-en-v1.5` | 33M | Semantic text search |
| Image Embeddings | `clip-ViT-B-32` | 151M | Image understanding |
| LLM (Q&A) | `qwen2.5:3b` (via Ollama) | 3B | Grounded answering |

**Alternative LLMs** (all ≤4B): `phi3.5:3.8b`, `gemma2:2b`, `llama3.2:3b`, `smollm2:1.7b`

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** — [install from ollama.com](https://ollama.com)
- **Chrome/Edge** browser (for extension)

### 1. Install Dependencies

```bash
cd ai-minds
pip install -r requirements.txt
```

### 2. Pull Ollama Model

```bash
ollama pull qwen2.5:3b
```

### 3. Start the Backend

```bash
cd backend
python server.py
```

The server starts on `http://127.0.0.1:5000`.

### 4. Start the Frontend Dashboard

```bash
cd frontend
streamlit run app.py
```

Opens at `http://127.0.0.1:8501`.

### 5. Load the Chrome Extension

1. Open Chrome → `chrome://extensions/`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** → select the `ai-minds/extension/` folder
4. The 🧠 AI MINDS icon appears in your toolbar

### Or Use the Start Script (Windows)

```bash
start.bat
```

This launches both backend and frontend automatically.

## Usage

### Browser Extension

- **Ctrl+Shift+F** — Toggle element selection mode. Click elements on any page to capture them.
- **Ctrl+Shift+S** — Capture entire page content.
- **Popup** — Click the extension icon to ask questions or generate a daily digest.

### Streamlit Dashboard

- **Chat** — Ask natural language questions. Get grounded answers with confidence scores and source references.
- **Dashboard** — View stats, categories, recent records, and generate digests.
- **Actions** — View and manage action items extracted from your data.
- **Ingest** — Manually paste text, specify file paths, or trigger directory scans.
- **Settings** — Configure Ollama model, embedding models, watch directories, chunk size, system prompt, etc.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Ask a question (RAG) |
| `/chat/stream` | POST | Streaming chat (SSE) |
| `/index` | POST | Ingest web clip text |
| `/api/extract` | POST | Ingest selected elements |
| `/api/auto-capture` | POST | Ingest full page |
| `/ingest/file` | POST | Ingest file by path |
| `/ingest/text` | POST | Ingest raw text |
| `/ingest/scan` | POST | Scan watch directories |
| `/stats` | GET | Knowledge base stats |
| `/actions` | GET | List action items |
| `/digest` | GET | Generate daily digest |
| `/config` | GET/POST | Read/update config |

## Project Structure

```
ai-minds/
├── backend/
│   ├── server.py           # Flask API server
│   ├── engine.py           # Core AI engine (ingest, search, answer)
│   ├── file_parsers.py     # PDF, DOCX, TXT, image parsers
│   ├── automations.py      # Clipboard monitor, digests, reminders
│   ├── config.json         # Default configuration
│   └── data/               # Runtime data (ChromaDB, SQLite, BM25)
├── extension/
│   ├── manifest.json       # Chrome extension manifest v3
│   ├── background.js       # Service worker
│   ├── contentScript.js    # Content script entry
│   ├── constants.js        # Config constants
│   ├── metadata.js         # Content type detection
│   ├── selection.js        # Element selection + page capture
│   ├── ui.js               # Floating UI widget
│   ├── utils.js            # DOM utilities
│   ├── styles.css          # Extension styles
│   ├── popup.html          # Extension popup
│   ├── popup.js            # Popup logic
│   └── icons/              # Extension icons
├── frontend/
│   └── app.py              # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── setup.py                # Automated setup script
├── start.bat               # Windows startup script
└── README.md               # This file
```

## Key Features Addressing Hackathon Requirements

| Requirement | Implementation |
|-------------|---------------|
| Multimodal ingestion | Text (PDF, DOCX, TXT), images (CLIP), web pages (extension) |
| No manual uploads | Directory watcher, browser auto-capture, clipboard monitor |
| Meaningful categories | Rule-based + LLM-assisted categorization |
| Summaries & actions | Auto-extracted on ingestion |
| Natural language Q&A | Hybrid search + Ollama RAG |
| Uncertainty handling | Confidence scores, explicit uncertainty messaging |
| Continuous operation | Background watcher threads, persistent ChromaDB + SQLite |
| Persistent memory | ChromaDB vectors + SQLite records survive restarts |
| Semantic reasoning | Hybrid search (cosine similarity + BM25) with MMR diversity |
| Self-verification | Confidence thresholds, grounded-only answers |
| No external APIs | All models run locally (sentence-transformers + Ollama) |
| ≤4B param LLM | Default: qwen2.5:3b via Ollama |

## License

MIT
