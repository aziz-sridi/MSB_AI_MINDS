"""
GigaMind Engine — Core intelligence layer.

Combines the best of second-brain (hybrid search, embeddings, BM25, MMR)
and ai_minds_engine (categorization, summaries, actions, SQLite records)
with Ollama-powered LLM (≤4B params) for grounded Q&A.
"""

import json
import math
import os
import pickle
import re
import sqlite3
import string
import threading
import time
import unicodedata
import uuid
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from PIL import Image

from file_parsers import (
    parse_txt, parse_pdf, parse_docx, parse_media,
    SUPPORTED_TEXT_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

STOP_WORDS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the "
    "and but if or because as until while of at by for with about against "
    "between into through during before after above below to from up down "
    "in out on off over under again further then once here there when where "
    "why how all any both each few more most other some such no nor not only "
    "own same so than too very s t can will just don should now".split()
)

ACTION_VERBS = {
    "follow", "remind", "send", "call", "email", "review", "check", "prepare",
    "schedule", "fix", "update", "submit", "complete", "create", "write", "plan",
    "buy", "book", "draft", "organize", "finalize", "share", "finish",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _epoch_now() -> float:
    return time.time()


def _parse_iso_to_epoch(iso_str: str) -> float:
    """Parse an ISO timestamp string to epoch seconds."""
    try:
        return time.mktime(time.strptime(iso_str[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return 0.0


def _time_decay_weight(created_epoch: float, half_life_days: float = 30.0) -> float:
    """Exponential time-decay: score multiplier in (0, 1].
    Half-life = number of days until the weight drops to 0.5."""
    if created_epoch <= 0:
        return 0.5
    age_days = max(0, (time.time() - created_epoch)) / 86400.0
    return math.pow(0.5, age_days / half_life_days)


def _normalize_tokens(text: str) -> List[str]:
    lowered = re.sub(r"[\W_]+", " ", text.lower()).strip()
    return [t for t in lowered.split() if t and t not in STOP_WORDS]


def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """Split text into chunks by character count, respecting word boundaries."""
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            split = cleaned.rfind(" ", start, end)
            if split > start + 100:
                end = split
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _is_gibberish(text: str) -> bool:
    if not text or len(text) < 15:
        return True
    try:
        normalized = unicodedata.normalize("NFKC", text)
    except Exception:
        normalized = text
    allowed = set(string.printable)
    total = len(normalized)
    if total == 0:
        return True
    non_standard = sum(ch not in allowed for ch in normalized)
    if non_standard / total > 0.15:
        return True
    try:
        text_bytes = normalized.encode("utf-8", "ignore")
        if not text_bytes:
            return True
        compressed = len(zlib.compress(text_bytes, 9))
        ratio = compressed / len(text_bytes)
        if ratio < 0.1:
            return True
        if len(text_bytes) > 100 and ratio > 0.95:
            return True
    except Exception:
        pass
    return False


def _sentence_summary(text: str, max_sentences: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences]).strip() or text[:240]


def _is_noisy_snippet(text: str) -> bool:
    if _is_gibberish(text):
        return True
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) < 25:
        return True
    alpha_count = sum(1 for c in cleaned if c.isalpha())
    return (alpha_count / max(1, len(cleaned))) < 0.35


def _extract_actions(text: str) -> List[str]:
    patterns = [
        r"\b(todo|to do|action item|follow up|next step|remind me to|need to)\b",
        r"\b(deadline|due|by\s+\w+day|next week|tomorrow|tonight|asap)\b",
        r"\b(i|we|you)\s+(should|must|need to)\b",
    ]
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    def _looks_like_noise(candidate: str) -> bool:
        c = re.sub(r"\s+", " ", candidate).strip()
        if len(c) < 10 or len(c) > 200:
            return True
        words = c.split()
        if len(words) < 3 or len(words) > 24:
            return True
        lower = c.lower()
        alpha_tokens = [w for w in re.findall(r"[a-zA-Z]+", c)]
        if not alpha_tokens:
            return True
        digit_ratio = sum(ch.isdigit() for ch in c) / max(1, len(c))
        if digit_ratio > 0.18:
            return True
        if lower.count("|") > 1 or lower.count("\t") > 1:
            return True
        has_action_verb = any(v in lower for v in ACTION_VERBS)
        if not has_action_verb and any(ch in lower for ch in ["median", "price", "range", "value insight"]):
            return True
        return False

    hits = []
    seen = set()
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if any(re.search(p, s, re.IGNORECASE) for p in patterns):
            if _looks_like_noise(s):
                continue
            snippet = re.sub(r"\s+", " ", s)[:200].strip(" -•\t")
            key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", snippet.lower())).strip()
            if key and key not in seen:
                seen.add(key)
                hits.append(snippet)
    return hits[:5]


VALID_CATEGORIES = ["work", "learning", "finance", "health", "personal", "news", "code", "general"]


def _categorize_rules(text: str, metadata: Dict[str, Any]) -> str:
    """Fast rule-based fallback when Ollama is unavailable."""
    joined = f"{text} {metadata.get('url', '')} {metadata.get('source_path', '')}".lower()
    rules = {
        "work": ["meeting", "project", "jira", "deadline", "client", "sprint", "task"],
        "learning": ["tutorial", "course", "research", "paper", "lesson", "study"],
        "finance": ["invoice", "bank", "budget", "payment", "price", "cost"],
        "health": ["workout", "nutrition", "doctor", "sleep", "health", "exercise"],
        "personal": ["family", "travel", "hobby", "journal", "shopping", "recipe"],
        "news": ["breaking", "report", "update", "election", "crisis"],
        "code": ["function", "class", "import", "def ", "const ", "var ", "git"],
    }
    for category, keywords in rules.items():
        if any(kw in joined for kw in keywords):
            return category
    return "general"


def _intent_from_query(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["summar", "overview", "tl;dr", "recap"]):
        return "summary"
    if any(w in q for w in ["action", "todo", "next step", "follow up", "task"]):
        return "actions"
    if any(w in q for w in ["where", "source", "link", "reference", "from"]):
        return "trace"
    if any(w in q for w in ["automat", "schedule", "remind", "alert", "notify"]):
        return "automation"
    return "qa"


IMAGE_INTENTS = ["find_similar", "describe", "ingest", "text_query_with_image"]


def _image_intent_fallback(prompt: str) -> str:
    """Fast rule-based image intent when LLM is unavailable."""
    p = prompt.lower()
    if any(w in p for w in ["similar", "find", "match", "look like", "resembl", "same"]):
        return "find_similar"
    if any(w in p for w in ["describe", "what is", "what's in", "analyze", "explain", "caption", "identify", "recogni"]):
        return "describe"
    if any(w in p for w in ["save", "store", "ingest", "remember", "keep"]):
        return "ingest"
    return "find_similar"


# ── Main Engine ──────────────────────────────────────────────────────────────

class AIMindsEngine:
    """
    Unified knowledge engine that:
    - Ingests text/images/web clips
    - Embeds with sentence-transformers / CLIP
    - Stores in ChromaDB + SQLite
    - Hybrid search: semantic (cosine) + lexical (BM25) with MMR reranking
    - Answers via Ollama (≤4B param model) with grounded context
    - Tracks categories, summaries, action items
    - Watches directories for auto-ingestion
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config.get("data_dir", "./data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "memory.db"
        self.bm25_path = self.data_dir / "bm25.pkl"
        self.file_state_path = self.data_dir / "file_state.json"

        # ChromaDB
        chroma_path = self.data_dir / "chroma"
        self.chroma = chromadb.PersistentClient(path=str(chroma_path))
        self.text_collection = self.chroma.get_or_create_collection(
            name="ai_minds_text", metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.chroma.get_or_create_collection(
            name="ai_minds_image", metadata={"hnsw:space": "cosine"}
        )

        # Embedding models
        text_model_name = self.config.get("text_model_name", "BAAI/bge-small-en-v1.5")
        print(f"[engine] Loading text model: {text_model_name}")
        self.text_model = SentenceTransformer(text_model_name)

        image_model_name = self.config.get("image_model_name", "clip-ViT-B-32")
        self.image_model: Optional[SentenceTransformer] = None
        if image_model_name:
            try:
                print(f"[engine] Loading image model: {image_model_name}")
                self.image_model = SentenceTransformer(image_model_name)
            except Exception as e:
                print(f"[engine] Could not load image model: {e}")

        # SQLite
        self._init_db()

        # BM25
        self.bm25_index: Optional[BM25Okapi] = None
        self.doc_id_order: List[str] = []
        self.doc_token_corpus: List[List[str]] = []
        self._bm25_rebuild()

        # Watcher
        self._watch_stop = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None

        # Lock for thread safety
        self.lock = threading.Lock()

    # ── Config ─────────────────────────────────────────────────────────────

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    # ── SQLite ─────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    modality TEXT NOT NULL,
                    category TEXT NOT NULL,
                    summary TEXT,
                    source TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    action_text TEXT NOT NULL,
                    done INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES records(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS automations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    last_run TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL,
                    references_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                )
            """)
            conn.commit()

    def _store_record(self, record_id: str, modality: str, category: str,
                      summary: str, source: str, metadata: Dict, actions: List[str]):
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?)",
                (record_id, modality, category, summary, source,
                 json.dumps(metadata), now),
            )
            conn.execute("DELETE FROM actions WHERE record_id = ?", (record_id,))
            for action in actions:
                conn.execute(
                    "INSERT INTO actions VALUES (?,?,?,0,?)",
                    (str(uuid.uuid4()), record_id, action, _now_iso()),
                )
            conn.commit()

    # ── BM25 ───────────────────────────────────────────────────────────────

    def _bm25_rebuild(self):
        if self.text_collection.count() == 0:
            self.bm25_index = None
            return
        records = self.text_collection.get(include=["documents"])
        self.doc_id_order = records.get("ids", []) or []
        docs = records.get("documents", []) or []
        self.doc_token_corpus = [_normalize_tokens(d) for d in docs]
        if self.doc_token_corpus:
            self.bm25_index = BM25Okapi(self.doc_token_corpus)
            with open(self.bm25_path, "wb") as f:
                pickle.dump({"ids": self.doc_id_order, "tokens": self.doc_token_corpus}, f)
        else:
            self.bm25_index = None

    # ── LLM Categorization ──────────────────────────────────────────────

    def _categorize_llm(self, full_text: str, metadata: Dict[str, Any]) -> str:
        """Use Ollama LLM to classify content into a category.

        Sends the full source text (truncated) + metadata so the model
        understands overall context rather than judging isolated chunks.
        Falls back to rule-based classification on any failure.
        """
        llm_cfg = self.config.get("ollama", {})
        if not llm_cfg.get("enabled", False):
            return _categorize_rules(full_text, metadata)

        base_url = llm_cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")
        model = llm_cfg.get("model", "qwen2.5:3b")

        # Build a concise preview the LLM can judge (first ~1500 chars)
        preview = re.sub(r"\s+", " ", full_text).strip()[:1500]
        source_hint = metadata.get("url") or metadata.get("source_path") or metadata.get("source", "")
        source_type = metadata.get("source_type", "unknown")

        prompt = (
            "Classify the following content into exactly ONE category.\n"
            f"Valid categories: {', '.join(VALID_CATEGORIES)}\n\n"
            f"Source: {source_hint}\n"
            f"Source type: {source_type}\n\n"
            f"Content preview:\n{preview}\n\n"
            "Reply with ONLY the single category word, nothing else."
        )

        try:
            import requests as _req
            resp = _req.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 12},
                },
                timeout=10,
            )
            if resp.ok:
                raw = resp.json().get("message", {}).get("content", "").strip().lower()
                # Extract first valid category word from the response
                for token in re.split(r"[\s,.:;]+", raw):
                    if token in VALID_CATEGORIES:
                        return token
        except Exception as e:
            print(f"[engine] LLM categorization failed, using rules: {e}")

        return _categorize_rules(full_text, metadata)

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_text(self, text: str, source: str, modality: str,
                    metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a block of text: chunk → embed → store → index."""
        chunk_size = int(self.config.get("chunk_size_chars", 800))
        chunks = _chunk_text(text, max_chars=chunk_size)
        chunks = [c for c in chunks if not _is_gibberish(c)]
        if not chunks:
            return {"status": "skipped", "reason": "empty_or_gibberish", "source": source}

        # LLM-based categorization using full source text (not individual chunks)
        category = self._categorize_llm(text, metadata)
        summary = _sentence_summary(text)
        actions_enabled = bool(self.config.get("action_extraction_enabled", False))
        actions = _extract_actions(text) if actions_enabled else []

        # Store raw chunks (no prefix in documents — prefix hurts semantic matching)
        prefixed = chunks

        embeddings = self.text_model.encode(
            prefixed, convert_to_numpy=True, normalize_embeddings=True, batch_size=16
        ).tolist()

        now_iso = _now_iso()
        now_epoch = _epoch_now()

        # Extract file timestamps if available
        file_created_at = metadata.get("file_created_at", now_iso)
        file_modified_at = metadata.get("file_modified_at", now_iso)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{
            "source": source, "modality": modality, "category": category,
            "summary": summary, "chunk_index": i,
            "ingested_at": now_iso,
            "ingested_at_epoch": now_epoch,
            "file_created_at": file_created_at,
            "file_modified_at": file_modified_at,
            **metadata,
        } for i in range(len(chunks))]

        with self.lock:
            self.text_collection.upsert(
                ids=ids, documents=prefixed, embeddings=embeddings, metadatas=metadatas
            )
            for idx, cid in enumerate(ids):
                self._store_record(
                    cid,
                    modality,
                    category,
                    summary,
                    source,
                    metadata,
                    actions if idx == 0 else [],
                )
            self._bm25_rebuild()

        return {
            "status": "ok", "source": source, "modality": modality,
            "category": category, "summary": summary, "actions": actions,
            "chunks_added": len(chunks),
        }

    def ingest_web_clip(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest content captured by the browser extension."""
        url = metadata.get("url", metadata.get("pageUrl", "web"))
        return self.ingest_text(
            text=text, source=url, modality="web_clip",
            metadata={**metadata, "source_type": "web", "timestamp": time.time()},
        )

    def _get_file_timestamps(self, path: Path) -> Dict[str, str]:
        """Extract file creation and modification timestamps."""
        try:
            stat = path.stat()
            created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_ctime))
            modified = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime))
            return {
                "file_created_at": created,
                "file_modified_at": modified,
                "file_created_epoch": stat.st_ctime,
                "file_modified_epoch": stat.st_mtime,
            }
        except Exception:
            now = _now_iso()
            return {"file_created_at": now, "file_modified_at": now,
                    "file_created_epoch": _epoch_now(), "file_modified_epoch": _epoch_now()}

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a local file (text, image, audio, or video)."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return {"status": "error", "reason": "file_not_found"}

        # Skip files still being downloaded
        if path.suffix.lower() in {".crdownload", ".partial", ".tmp", ".download"}:
            return {"status": "skipped", "reason": "download_in_progress"}

        ext = path.suffix.lower()
        file_ts = self._get_file_timestamps(path)

        # Build a filename header so the file is discoverable by name
        filename_header = f"File: {path.name}\nPath: {path.parent}\n\n"

        # Text-based files
        if ext in SUPPORTED_TEXT_EXTENSIONS:
            parsers = {
                ".txt": parse_txt, ".md": parse_txt, ".log": parse_txt,
                ".csv": parse_txt, ".json": parse_txt,
                ".pdf": parse_pdf, ".docx": parse_docx,
            }
            parser = parsers.get(ext, parse_txt)
            text = parser(path)
            if not text:
                return {"status": "skipped", "reason": "empty_content"}
            # Prepend filename so users can search by file name
            text = filename_header + text
            return self.ingest_text(
                text=text, source=str(path), modality=f"file_{ext.strip('.')}",
                metadata={"source_path": str(path), "filename": path.name, "extension": ext, "source_type": "file", **file_ts},
            )

        # Images
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            return self._ingest_image(path)

        # Audio / Video (metadata-level ingestion)
        if ext in (SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS):
            media_text = parse_media(path, self.config.get("transcription", {}))
            media_kind = "audio" if ext in SUPPORTED_AUDIO_EXTENSIONS else "video"
            return self.ingest_text(
                text=filename_header + media_text,
                source=str(path),
                modality=f"file_{media_kind}",
                metadata={
                    "source_path": str(path),
                    "filename": path.name,
                    "extension": ext,
                    "source_type": "file",
                    "media_type": media_kind,
                    **file_ts,
                },
            )

        return {"status": "skipped", "reason": "unsupported_extension", "path": file_path}

    def _ingest_image(self, path: Path) -> Dict[str, Any]:
        filename = path.name
        folder = path.parent.name
        surrogate = f"File: {filename}\nImage file: {filename} located in folder {folder}"
        meta = {
            "source_path": str(path), "filename": filename,
            "extension": path.suffix.lower(),
            "modality": "image", "source": str(path), "source_type": "file",
            "category": _categorize_rules(surrogate, {"source_path": str(path)}),
            "summary": f"Image file: {filename}",
        }
        doc_id = str(uuid.uuid4())

        if self.image_model is not None:
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((2048, 2048))
                emb = self.image_model.encode([img], normalize_embeddings=True).tolist()
                with self.lock:
                    self.image_collection.upsert(
                        ids=[doc_id], documents=[surrogate],
                        embeddings=emb, metadatas=[meta],
                    )
            except Exception as e:
                print(f"[engine] Image embed failed for {path}: {e}")
                emb = self.text_model.encode([surrogate], normalize_embeddings=True).tolist()
                with self.lock:
                    self.text_collection.upsert(
                        ids=[doc_id], documents=[surrogate],
                        embeddings=emb, metadatas=[meta],
                    )
                    self._bm25_rebuild()
        else:
            emb = self.text_model.encode([surrogate], normalize_embeddings=True).tolist()
            with self.lock:
                self.text_collection.upsert(
                    ids=[doc_id], documents=[surrogate],
                    embeddings=emb, metadatas=[meta],
                )
                self._bm25_rebuild()

        self._store_record(doc_id, "image", meta["category"], meta["summary"],
                           str(path), meta, [])
        return {"status": "ok", "source": str(path), "modality": "image",
                "category": meta["category"], "chunks_added": 1}

    # ── Search ─────────────────────────────────────────────────────────────

    def _semantic_search(self, query: str, top_k: int, category: Optional[str] = None) -> List[Dict]:
        emb = self.text_model.encode([query], normalize_embeddings=True).tolist()
        n_available = self.text_collection.count()
        if n_available == 0:
            return []
        actual_k = min(top_k, n_available)
        query_kwargs = dict(
            query_embeddings=emb, n_results=actual_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        if category:
            query_kwargs["where"] = {"category": category}
        result = self.text_collection.query(**query_kwargs)
        out = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        embs = result.get("embeddings", [[]])[0]
        for i, doc_id in enumerate(ids):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - distance (already in [0,1] for normalized vecs)
            raw_sim = max(0.0, float(1 - dists[i]))
            out.append({
                "id": doc_id, "documents": docs[i], "metadata": metas[i],
                "embedding": np.array(embs[i]),
                "score": raw_sim, "raw_score": raw_sim, "result_type": "semantic",
            })
        return out

    def _lexical_search(self, query: str, top_k: int, category: Optional[str] = None) -> List[Dict]:
        if not self.bm25_index or not self.doc_id_order:
            return []
        tokens = _normalize_tokens(query)
        if not tokens:
            return []
        scores = self.bm25_index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            doc_id = self.doc_id_order[idx]
            # If category filter is set, check the doc's metadata in ChromaDB
            if category:
                try:
                    meta = self.text_collection.get(ids=[doc_id], include=["metadatas"])
                    doc_cat = (meta.get("metadatas") or [{}])[0].get("category", "")
                    if doc_cat != category:
                        continue
                except Exception:
                    continue
            results.append({"id": doc_id, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def _filename_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for chunks that belong to a file whose name appears in the query.
        Instead of regex-parsing filenames from the query, checks each ingested
        file's name against the query text — more reliable with special chars."""
        query_lower = query.lower()
        results = []

        try:
            # Get all records (no where filter — some old records lack source_type)
            all_records = self.text_collection.get(
                include=["documents", "metadatas", "embeddings"],
            )
            ids = all_records.get("ids", [])
            docs = all_records.get("documents", [])
            metas = all_records.get("metadatas", [])
            embs = all_records.get("embeddings", [])

            for i, doc_id in enumerate(ids):
                meta = metas[i] or {}
                # Only look at file-sourced records
                if meta.get("source_type") != "file":
                    continue
                fname = meta.get("filename", "")
                if not fname:
                    # Fallback: extract filename from source_path
                    sp = meta.get("source_path", "")
                    if sp:
                        fname = Path(sp).name
                if not fname:
                    continue
                # Check if this file's name appears in the query
                if fname.lower() in query_lower:
                    results.append({
                        "id": doc_id,
                        "documents": docs[i],
                        "metadata": meta,
                        "embedding": np.array(embs[i]),
                        "score": 1.0,
                        "raw_score": 0.95,
                        "result_type": "filename_match",
                    })
                else:
                    # Also try matching without extension
                    stem = Path(fname).stem.lower()
                    if len(stem) > 3 and stem in query_lower:
                        results.append({
                            "id": doc_id,
                            "documents": docs[i],
                            "metadata": meta,
                            "embedding": np.array(embs[i]),
                            "score": 0.9,
                            "raw_score": 0.85,
                            "result_type": "filename_match",
                        })
        except Exception as e:
            print(f"[engine] Filename search error: {e}")

        return results[:top_k]

    def _hybrid_search(self, query: str, top_k: int = 6, category: Optional[str] = None,
                       date_from: Optional[str] = None, date_to: Optional[str] = None) -> List[Dict]:
        """Combine semantic + lexical search with MMR reranking.
        If category is provided, only results from that category are returned.
        date_from/date_to: ISO date strings to filter by ingestion date."""
        fetch_k = top_k * 5

        sem_results = self._semantic_search(query, fetch_k, category=category)
        lex_results = self._lexical_search(query, fetch_k, category=category)
        file_results = self._filename_search(query, top_k=fetch_k)

        # Keep raw semantic scores for confidence (before normalization)
        # Normalize semantic scores for ranking
        sem_scores = np.array([r["score"] for r in sem_results]) if sem_results else np.array([])
        if sem_scores.size > 0:
            mn, mx = sem_scores.min(), sem_scores.max()
            for i, r in enumerate(sem_results):
                r["score"] = r["raw_score"] if mx == mn else (r["score"] - mn) / (mx - mn)

        # Normalize lexical scores
        lex_score_map = {}
        lex_scores = np.array([r["score"] for r in lex_results]) if lex_results else np.array([])
        if lex_scores.size > 0:
            mn, mx = lex_scores.min(), lex_scores.max()
            for i, r in enumerate(lex_results):
                norm = r["score"] if mx == mn else (r["score"] - mn) / (mx - mn)
                lex_score_map[r["id"]] = norm

        # Merge: semantic 60% + lexical 40%
        combined = {r["id"]: r.copy() for r in sem_results}
        for r in lex_results:
            if r["id"] in combined:
                combined[r["id"]]["score"] = (
                    combined[r["id"]]["score"] * 0.6 + lex_score_map.get(r["id"], 0) * 0.4
                )
                combined[r["id"]]["result_type"] = "hybrid"
            else:
                # Fetch full data from ChromaDB
                try:
                    doc = self.text_collection.get(
                        ids=[r["id"]], include=["documents", "metadatas", "embeddings"]
                    )
                    if not doc.get("ids"):
                        continue
                    combined[r["id"]] = {
                        "id": r["id"], "documents": doc["documents"][0],
                        "metadata": doc["metadatas"][0],
                        "embedding": np.array(doc["embeddings"][0]),
                        "score": lex_score_map.get(r["id"], 0) * 0.4,
                        "raw_score": 0.0,
                        "result_type": "lexical",
                    }
                except Exception:
                    continue

        # Merge filename matches with high priority
        for r in file_results:
            if r["id"] in combined:
                # Boost existing result if it also matches by filename
                combined[r["id"]]["score"] = max(combined[r["id"]]["score"], 0.95)
                combined[r["id"]]["raw_score"] = max(combined[r["id"]].get("raw_score", 0), 0.95)
                combined[r["id"]]["result_type"] = "filename_match"
            else:
                combined[r["id"]] = r.copy()

        # ── Date-period filtering ──
        if date_from or date_to:
            from_epoch = _parse_iso_to_epoch(date_from) if date_from else 0.0
            to_epoch = _parse_iso_to_epoch(date_to) if date_to else float("inf")
            filtered = {}
            for doc_id, item in combined.items():
                meta = item.get("metadata", {})
                item_epoch = float(meta.get("ingested_at_epoch", 0) or meta.get("file_modified_epoch", 0) or 0)
                if item_epoch == 0:
                    # Fallback: try parsing ingested_at string
                    item_epoch = _parse_iso_to_epoch(meta.get("ingested_at", ""))
                if from_epoch <= item_epoch <= to_epoch:
                    filtered[doc_id] = item
            combined = filtered

        # ── Time-decay boost ──
        decay_enabled = self.config.get("time_decay_enabled", True)
        half_life = float(self.config.get("time_decay_half_life_days", 30))
        if decay_enabled:
            decay_weight = float(self.config.get("time_decay_weight", 0.15))
            for doc_id, item in combined.items():
                meta = item.get("metadata", {})
                created_epoch = float(meta.get("ingested_at_epoch", 0) or meta.get("file_modified_epoch", 0) or 0)
                if created_epoch <= 0:
                    created_epoch = _parse_iso_to_epoch(meta.get("ingested_at", ""))
                decay = _time_decay_weight(created_epoch, half_life)
                # Blend: (1 - decay_weight) * relevance_score + decay_weight * decay
                item["score"] = (1 - decay_weight) * item["score"] + decay_weight * decay

        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

        # MMR reranking for diversity
        return self._mmr_rerank(ranked[:fetch_k], top_k)

    def _mmr_rerank(self, results: List[Dict], n_results: int,
                    mmr_lambda: float = 0.5) -> List[Dict]:
        """Maximal Marginal Relevance for diversity."""
        if not results or len(results) <= 1:
            return results[:n_results]

        embeddings = np.array([r["embedding"] for r in results])
        scores = np.array([r["score"] for r in results])
        sim_matrix = 1 - cdist(embeddings, embeddings, metric="cosine")
        np.fill_diagonal(sim_matrix, -1)

        selected = []
        remaining = list(range(len(results)))

        first = int(np.argmax(scores))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < n_results and remaining:
            mmr = (mmr_lambda * scores[remaining] -
                   (1 - mmr_lambda) * np.max(sim_matrix[remaining][:, selected], axis=1))
            best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)

        return [results[i] for i in selected]

    # ── Smart Query Pre-processing ─────────────────────────────────────────

    def extract_query_params(self, query: str, has_image: bool = False) -> Dict[str, Any]:
        """Use the LLM to semantically extract date range, category, and image intent
        from a natural language query. Falls back to empty/defaults if LLM unavailable.

        Returns dict with keys:
          - category: str or None
          - date_from: ISO string or None
          - date_to: ISO string or None
          - image_intent: str or None  (find_similar | describe | ingest | text_query_with_image)
          - cleaned_query: str (the actual question without date/category noise)
        """
        result = {
            "category": None,
            "date_from": None,
            "date_to": None,
            "image_intent": None,
            "cleaned_query": query,
        }

        llm_cfg = self.config.get("ollama", {})
        if not llm_cfg.get("enabled", False):
            # Rule-based fallback for image intent
            if has_image:
                result["image_intent"] = _image_intent_fallback(query)
            return result

        base_url = llm_cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")
        model = llm_cfg.get("model", "qwen2.5:3b")

        today = time.strftime("%Y-%m-%d")
        image_section = ""
        if has_image:
            image_section = (
                '\n- "image_intent": one of "find_similar", "describe", "ingest", "text_query_with_image"'
                "\n  find_similar = user wants to find visually similar images in memory"
                "\n  describe = user wants a description/analysis of what's in the image"
                "\n  ingest = user just wants to save/store the image"
                "\n  text_query_with_image = user is asking a text question and the image is supplementary context"
            )

        prompt = (
            f"Today's date is {today}.\n"
            "Extract structured parameters from the user's query below.\n"
            "Return ONLY a valid JSON object with these keys:\n"
            '- "category": one of [work, learning, finance, health, personal, news, code] or null if not implied\n'
            '- "date_from": ISO date string (YYYY-MM-DD) or null. Convert relative dates like "3 months ago", "last week", "yesterday" to absolute dates.\n'
            '- "date_to": ISO date string (YYYY-MM-DD) or null. If the user says "3 months ago" this is today\'s date.\n'
            f'{image_section}\n'
            '- "cleaned_query": the core question without date/category qualifiers, keep the user\'s intent\n\n'
            "IMPORTANT: Output ONLY the JSON, no explanation, no markdown fences.\n\n"
            f"User query: {query}"
        )

        try:
            import requests as _req
            resp = _req.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 200},
                },
                timeout=12,
            )
            if resp.ok:
                raw = resp.json().get("message", {}).get("content", "").strip()
                # Try to extract JSON from response (handle markdown fences)
                # First try the full response, then look for JSON block
                json_str = None
                # Remove markdown fences if present
                clean = re.sub(r"```json\s*", "", raw)
                clean = re.sub(r"```\s*", "", clean).strip()
                try:
                    parsed = json.loads(clean)
                    json_str = clean
                except (json.JSONDecodeError, ValueError):
                    # Try to find a JSON object in the response
                    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()

                if json_str:
                    try:
                        parsed = json.loads(json_str)
                    except (json.JSONDecodeError, ValueError):
                        parsed = {}

                    # Validate category
                    cat = parsed.get("category")
                    if cat and cat in VALID_CATEGORIES and cat != "general":
                        result["category"] = cat

                    # Validate dates
                    df = parsed.get("date_from")
                    if df and re.match(r"\d{4}-\d{2}-\d{2}", str(df)):
                        result["date_from"] = str(df) + "T00:00:00Z"

                    dt = parsed.get("date_to")
                    if dt and re.match(r"\d{4}-\d{2}-\d{2}", str(dt)):
                        result["date_to"] = str(dt) + "T23:59:59Z"

                    # Validate image intent
                    img_intent = parsed.get("image_intent")
                    if img_intent and img_intent in IMAGE_INTENTS:
                        result["image_intent"] = img_intent

                    cq = parsed.get("cleaned_query")
                    if cq and isinstance(cq, str) and len(cq.strip()) > 2:
                        result["cleaned_query"] = cq.strip()

        except Exception as e:
            print(f"[engine] Query param extraction failed: {e}")

        # Fallback for image intent if LLM didn't return one
        if has_image and not result["image_intent"]:
            result["image_intent"] = _image_intent_fallback(query)

        return result

    def smart_query(self, query: str, image_path: Optional[str] = None,
                    persist_image: bool = False) -> Dict[str, Any]:
        """Smart query endpoint — uses LLM to understand the full intent:
        - Extracts date range, category, and image intent semantically
        - Routes image operations based on detected intent
        - Returns unified answer with all results merged
        """
        has_image = bool(image_path)

        # Step 1: LLM extracts structured params from natural language
        params = self.extract_query_params(query, has_image=has_image)

        # Step 2: Run text RAG with extracted params
        text_result = self.answer(
            query=params["cleaned_query"],
            category=params.get("category"),
            date_from=params.get("date_from"),
            date_to=params.get("date_to"),
        )

        # Step 3: Handle image based on detected intent
        image_result = None
        image_intent = params.get("image_intent")

        if has_image and image_path:
            if image_intent == "find_similar":
                image_result = self.find_similar_images(image_path=image_path, top_k=6)
            elif image_intent == "describe":
                # Use CLIP to find similar images + use text model to provide context
                image_result = self.find_similar_images(image_path=image_path, top_k=3)
                if image_result.get("status") == "ok":
                    image_result["answer"] = (
                        "Image analysis (via visual similarity search):\n"
                        + image_result.get("answer", "")
                        + "\n\nNote: For detailed image descriptions, a vision-language model "
                        "(e.g. llava) is needed. Currently using CLIP similarity matching."
                    )
            elif image_intent == "ingest":
                ingest_result = self.ingest_file(image_path)
                image_result = {
                    "status": ingest_result.get("status", "ok"),
                    "answer": f"Image saved to memory. Category: {ingest_result.get('category', 'general')}",
                    "references": [],
                }
            elif image_intent == "text_query_with_image":
                # Image is supplementary — do both text query and visual search
                image_result = self.find_similar_images(image_path=image_path, top_k=3)

            # Optionally persist image
            if persist_image and image_intent != "ingest":
                self.ingest_file(image_path)

        # Step 4: Merge results
        answer_parts = []
        all_refs = []
        confidence = 0.0
        uncertainty = None

        if text_result:
            answer_parts.append(text_result.get("answer", ""))
            all_refs.extend(text_result.get("references", []) or [])
            confidence = max(confidence, float(text_result.get("confidence", 0) or 0))
            uncertainty = text_result.get("uncertainty")

        if isinstance(image_result, dict) and image_result.get("status") == "ok":
            img_answer = image_result.get("answer", "")
            if img_answer:
                answer_parts.append(img_answer)
            all_refs.extend(image_result.get("references", []) or [])
            confidence = max(confidence, float(image_result.get("confidence", 0) or 0))

        final_answer = "\n\n".join([p for p in answer_parts if p]).strip() or "No answer found."

        return {
            "query": query,
            "cleaned_query": params["cleaned_query"],
            "extracted_params": {
                "category": params.get("category"),
                "date_from": params.get("date_from"),
                "date_to": params.get("date_to"),
                "image_intent": image_intent,
            },
            "answer": final_answer,
            "confidence": round(confidence, 4),
            "uncertainty": uncertainty,
            "references": all_refs,
        }

    # ── Answering ──────────────────────────────────────────────────────────

    def answer(self, query: str, top_k: Optional[int] = None, category: Optional[str] = None,
               date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
        """Full RAG pipeline: search → context → Ollama LLM → grounded answer.
        If category is provided, retrieval is scoped to that category only.
        date_from/date_to: ISO date strings for time filtering."""
        k = int(top_k or self.config.get("top_k", 6))
        intent = _intent_from_query(query)

        results = self._hybrid_search(query, k, category=category,
                                       date_from=date_from, date_to=date_to)
        confidence = self._confidence(results)

        # Try Ollama LLM
        llm_answer, llm_note = self._ollama_answer(query, intent, results)
        if not llm_answer:
            llm_answer = self._build_grounded_answer(query, intent, results)

        uncertainty = None
        if confidence < float(self.config.get("low_confidence_threshold", 0.35)):
            uncertainty = "Low confidence — consider refining your query or ingesting more data."
        if llm_note:
            uncertainty = f"{uncertainty} {llm_note}".strip() if uncertainty else llm_note

        source_threshold = float(self.config.get("source_score_threshold", 0.2))
        references = []
        for item in results:
            if float(item.get("score", 0)) < source_threshold:
                continue
            meta = item.get("metadata", {})
            references.append({
                "source": meta.get("source", "unknown"),
                "source_type": meta.get("source_type", "unknown"),
                "category": meta.get("category", "general"),
                "modality": meta.get("modality", "text"),
                "snippet": (item.get("documents") or "")[:240],
                "score": round(item.get("score", 0), 4),
                "ingested_at": meta.get("ingested_at", ""),
                "file_created_at": meta.get("file_created_at", ""),
                "file_modified_at": meta.get("file_modified_at", ""),
            })

        return {
            "query": query, "intent": intent, "answer": llm_answer,
            "confidence": round(confidence, 4), "uncertainty": uncertainty,
            "references": references,
        }

    def _confidence(self, results: List[Dict]) -> float:
        """Compute answer confidence from raw semantic similarity scores."""
        if not results:
            return 0.0
        # Use raw_score (pre-normalization cosine similarity) if available
        raw = [r.get("raw_score", r.get("score", 0)) for r in results]
        # Weighted: top result counts more
        if len(raw) == 1:
            return float(max(0, min(1, raw[0])))
        weights = [0.5, 0.3, 0.2] if len(raw) >= 3 else [0.6, 0.4]
        weighted = sum(w * s for w, s in zip(weights, raw[:len(weights)]))
        return float(max(0, min(1, weighted)))

    def _build_grounded_answer(self, query: str, intent: str, results: List[Dict]) -> str:
        if not results:
            return "I couldn't find relevant information in your memory for that query."

        if intent == "summary":
            points = []
            for r in results[:4]:
                summary = r.get("metadata", {}).get("summary") or _sentence_summary(r.get("documents", ""), 1)
                points.append(f"- {summary}")
            return "Summary from your memory:\n" + "\n".join(points)

        if intent == "actions":
            action_lines = []
            with sqlite3.connect(self.db_path) as conn:
                for r in results[:8]:
                    rows = conn.execute(
                        "SELECT action_text FROM actions WHERE record_id = ? AND done = 0 LIMIT 2",
                        (r["id"],)
                    ).fetchall()
                    for row in rows:
                        action_lines.append(f"- {row[0]}")
            if action_lines:
                return "Action items found:\n" + "\n".join(action_lines[:8])
            return "Related items found, but no explicit action items detected."

        clean_results = [r for r in results if not _is_noisy_snippet(r.get("documents", ""))]
        if not clean_results:
            return (
                "I found potential matches, but the extracted text is too noisy to answer reliably. "
                "Try a more specific question or ingest cleaner source files."
            )

        snippets = [f"- {r.get('documents', '')[:200].strip()}" for r in clean_results[:4]]
        return (
            f"Most relevant results for '{query}':\n" + "\n".join(snippets)
        )

    def _ollama_answer(self, query: str, intent: str, results: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """Query Ollama for a grounded answer."""
        llm_cfg = self.config.get("ollama", {})
        if not llm_cfg.get("enabled", False):
            return None, None

        model = llm_cfg.get("model", "qwen2.5:3b")
        base_url = llm_cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")

        # Build context from search results (include category + timestamp for grounding)
        context_parts = []
        for i, r in enumerate(results[:6]):
            meta = r.get("metadata", {})
            source = meta.get("source", "unknown")
            cat = meta.get("category", "general")
            ingested = meta.get("ingested_at", meta.get("file_modified_at", "unknown"))
            text = r.get("documents", "")[:400]
            context_parts.append(f"[{i+1}] Source: {source} | Category: {cat} | Date: {ingested}\n{text}")
        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are GigaMind, a personal knowledge assistant. "
            "Answer the user's question using the provided context from their stored memories. "
            "The context snippets come from different sources (web pages, notes, files) and have categories. "
            "Be helpful: if context is available, synthesize a clear answer. "
            "Reference which source number(s) support your answer using [1], [2] etc. "
            "If the context doesn't contain enough info, say so honestly. "
            "If asked for actions/todos, extract them from context. "
            "If asked for a summary, be brief and structured."
        )

        # Override with user's custom system prompt if set
        custom_prompt = self.config.get("system_prompt")
        if custom_prompt:
            system_prompt = custom_prompt

        user_msg = f"Question: {query}\n\nContext from memory:\n{context}"

        try:
            import requests
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 512,
                },
            }
            resp = requests.post(
                f"{base_url}/api/chat", json=payload,
                timeout=float(llm_cfg.get("timeout_seconds", 30)),
            )
            if resp.ok:
                content = resp.json().get("message", {}).get("content", "").strip()
                if content:
                    return content, None
                return None, "LLM returned empty response."
            return None, f"LLM request failed (HTTP {resp.status_code})."
        except Exception as e:
            return None, f"Ollama not reachable: {e}"

    def ollama_stream(self, query: str, top_k: Optional[int] = None, category: Optional[str] = None,
                      date_from: Optional[str] = None, date_to: Optional[str] = None):
        """Streaming version of answer — yields chunks for real-time UI."""
        k = int(top_k or self.config.get("top_k", 6))
        results = self._hybrid_search(query, k, category=category,
                                       date_from=date_from, date_to=date_to)

        llm_cfg = self.config.get("ollama", {})
        if not llm_cfg.get("enabled", False):
            yield self._build_grounded_answer(query, _intent_from_query(query), results)
            return

        model = llm_cfg.get("model", "qwen2.5:3b")
        base_url = llm_cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")

        context_parts = []
        for i, r in enumerate(results[:6]):
            meta = r.get("metadata", {})
            source = meta.get("source", "unknown")
            cat = meta.get("category", "general")
            ingested = meta.get("ingested_at", meta.get("file_modified_at", "unknown"))
            text = r.get("documents", "")[:400]
            context_parts.append(f"[{i+1}] Source: {source} | Category: {cat} | Date: {ingested}\n{text}")
        context = "\n\n".join(context_parts)

        system_prompt = self.config.get("system_prompt", (
            "You are GigaMind, a personal knowledge assistant. "
            "Answer using ONLY the provided context. Never fabricate. "
            "Be concise, reference sources."
        ))

        try:
            import requests
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
                ],
                "stream": True,
                "options": {"temperature": 0.3, "num_predict": 512},
            }
            resp = requests.post(f"{base_url}/api/chat", json=payload, stream=True, timeout=60)
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except Exception as e:
            yield f"[Error: Ollama unreachable — {e}]"

    # ── Directory Watching ─────────────────────────────────────────────────

    def ingest_watch_dirs(self) -> Dict[str, Any]:
        """Scan all watched directories for new/modified files.
        Uses non-recursive scan and filters by supported extensions.
        Saves state incrementally so progress isn't lost on crash."""
        from file_parsers import ALL_SUPPORTED

        watch_dirs = [Path(p) for p in self.config.get("watch_dirs", []) if p]
        if not watch_dirs:
            return {"status": "skipped", "reason": "no_watch_dirs"}

        previous_state: Dict[str, float] = {}
        if self.file_state_path.exists():
            try:
                previous_state = json.loads(self.file_state_path.read_text(encoding="utf-8"))
            except Exception:
                previous_state = {}

        current_state: Dict[str, float] = dict(previous_state)  # preserve known files
        processed = 0
        errors = 0

        for directory in watch_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            # Use iterdir (non-recursive) — faster and avoids scanning extracted archives
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                # Skip unsupported extensions and temp download files early
                if ext in {".crdownload", ".partial", ".tmp", ".download"}:
                    continue
                if ext not in ALL_SUPPORTED:
                    continue
                try:
                    as_str = str(path)
                    mtime = path.stat().st_mtime
                    current_state[as_str] = mtime
                    if previous_state.get(as_str) == mtime:
                        continue
                    # Ensure file is fully written (size stable)
                    size1 = path.stat().st_size
                    if size1 == 0:
                        continue
                    time.sleep(0.3)
                    size2 = path.stat().st_size
                    if size1 != size2:
                        # File still being written, skip for now
                        current_state.pop(as_str, None)
                        continue
                    result = self.ingest_file(as_str)
                    if result.get("status") == "ok":
                        processed += 1
                        print(f"[watcher] Ingested: {path.name} ({result.get('chunks_added', 0)} chunks)")
                    # Save state after each file so progress isn't lost
                    self.file_state_path.write_text(json.dumps(current_state), encoding="utf-8")
                except Exception as e:
                    print(f"[engine] Error ingesting {path.name}: {e}")
                    errors += 1

        # Final state save
        self.file_state_path.write_text(json.dumps(current_state), encoding="utf-8")
        return {"status": "ok", "processed": processed, "errors": errors}

    def start_watcher(self):
        if self._watch_thread and self._watch_thread.is_alive():
            return
        interval = float(self.config.get("watch_interval_seconds", 30))
        self._watch_stop.clear()

        def _loop():
            while not self._watch_stop.is_set():
                try:
                    result = self.ingest_watch_dirs()
                    if result.get("processed", 0) > 0:
                        print(f"[watcher] Ingested {result['processed']} files")
                except Exception as e:
                    print(f"[watcher] Error: {e}")
                self._watch_stop.wait(interval)

        self._watch_thread = threading.Thread(target=_loop, daemon=True)
        self._watch_thread.start()
        print(f"[watcher] Started (interval: {interval}s)")

    def stop_watcher(self):
        self._watch_stop.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=3)

    # ── Stats & Overview ───────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        text_count = self.text_collection.count()
        image_count = self.image_collection.count()

        categories = defaultdict(int)
        modalities = defaultdict(int)
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute("SELECT category, modality FROM records"):
                categories[row[0]] += 1
                modalities[row[1]] += 1

        return {
            "text_chunks": text_count,
            "image_items": image_count,
            "total_records": text_count + image_count,
            "categories": dict(categories),
            "modalities": dict(modalities),
        }

    def recent_records(self, limit: int = 20) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, modality, category, summary, source, created_at "
                "FROM records ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "modality": r[1], "category": r[2],
             "summary": r[3], "source": r[4], "created_at": r[5]}
            for r in rows
        ]

    def get_actions(self, done: bool = False) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT a.id, a.action_text, a.done, a.created_at, r.source "
                "FROM actions a JOIN records r ON a.record_id = r.id "
                "WHERE a.done = ? ORDER BY a.created_at DESC",
                (1 if done else 0,),
            ).fetchall()
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in rows:
            action_id, text, done_value, created_at, source = r
            norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", (text or "").lower())).strip()
            norm = norm[:120]
            key = (norm, source or "")
            if key not in grouped:
                grouped[key] = {
                    "id": action_id,
                    "text": text,
                    "done": bool(done_value),
                    "created_at": created_at,
                    "source": source,
                    "duplicate_count": 1,
                }
            else:
                grouped[key]["duplicate_count"] += 1

        return sorted(grouped.values(), key=lambda x: x.get("created_at", ""), reverse=True)

    def mark_action_done(self, action_id: str):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT a.action_text, r.source FROM actions a "
                "JOIN records r ON a.record_id = r.id WHERE a.id = ?",
                (action_id,),
            ).fetchone()
            if not row:
                return 0
            action_text, source = row
            cur = conn.execute(
                "UPDATE actions SET done = 1 WHERE done = 0 AND id IN ("
                "SELECT a2.id FROM actions a2 JOIN records r2 ON a2.record_id = r2.id "
                "WHERE a2.action_text = ? AND COALESCE(r2.source, '') = COALESCE(?, '')"
                ")",
                (action_text, source),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def cleanup_actions(self) -> Dict[str, Any]:
        """Remove noisy actions and collapse exact duplicates in DB."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT a.id, a.action_text, a.done, r.source FROM actions a "
                "JOIN records r ON a.record_id = r.id"
            ).fetchall()

            removed_noisy = 0
            seen = set()
            removed_duplicates = 0

            for action_id, action_text, done_value, source in rows:
                normalized = re.sub(r"\s+", " ", (action_text or "")).strip()
                key_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", normalized.lower())).strip()[:120]
                if len(key_norm) < 8:
                    conn.execute("DELETE FROM actions WHERE id = ?", (action_id,))
                    removed_noisy += 1
                    continue

                if any(term in key_norm for term in ["median price", "price range", "value insight", "property types"]):
                    conn.execute("DELETE FROM actions WHERE id = ?", (action_id,))
                    removed_noisy += 1
                    continue

                dedup_key = (key_norm, source or "", int(done_value or 0))
                if dedup_key in seen:
                    conn.execute("DELETE FROM actions WHERE id = ?", (action_id,))
                    removed_duplicates += 1
                    continue
                seen.add(dedup_key)

            conn.commit()

        return {
            "status": "ok",
            "removed_noisy": removed_noisy,
            "removed_duplicates": removed_duplicates,
            "remaining": len(self.get_actions(done=False)) + len(self.get_actions(done=True)),
        }

    def purge_actions(self) -> Dict[str, Any]:
        """Delete all action items (pending + done)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM actions")
            conn.commit()
            deleted = int(cur.rowcount or 0)
        return {"status": "ok", "deleted": deleted}

    # ── Image Query ────────────────────────────────────────────────────────

    def find_similar_images(self, image_path: str, top_k: int = 6) -> Dict[str, Any]:
        """Search for similar ingested images using image embeddings."""
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return {"status": "error", "reason": "file_not_found"}
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return {"status": "error", "reason": "not_an_image"}

        if self.image_collection.count() == 0:
            return {
                "status": "ok",
                "answer": "No indexed images yet. Ingest images first, then try again.",
                "references": [],
            }

        if self.image_model is None:
            return {
                "status": "error",
                "reason": "image_model_unavailable",
                "message": "Image model is not available. Configure a CLIP model to enable similar image search.",
            }

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((2048, 2048))
            emb = self.image_model.encode([img], normalize_embeddings=True).tolist()
        except Exception as e:
            return {"status": "error", "reason": f"image_read_failed: {e}"}

        n_results = min(max(1, int(top_k)), self.image_collection.count())
        result = self.image_collection.query(
            query_embeddings=emb,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        score_threshold = float(self.config.get("image_similarity_threshold", 0.2))
        references: List[Dict[str, Any]] = []

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        query_resolved = str(path.resolve())

        for i, image_id in enumerate(ids):
            score = max(0.0, float(1 - dists[i]))
            if score < score_threshold:
                continue
            meta = metas[i] or {}
            source_candidate = meta.get("source_path") or meta.get("source") or ""
            try:
                if source_candidate and str(Path(source_candidate).resolve()) == query_resolved:
                    continue
            except Exception:
                pass
            references.append({
                "id": image_id,
                "source": meta.get("source", meta.get("source_path", "unknown")),
                "source_type": meta.get("source_type", "file"),
                "category": meta.get("category", "general"),
                "modality": "image",
                "snippet": (docs[i] or "")[:240],
                "score": round(score, 4),
            })

        if not references:
            return {
                "status": "ok",
                "answer": "No similar images found in memory above the similarity threshold.",
                "references": [],
            }

        lines = [f"- {ref['source']} (score: {ref['score']:.2f})" for ref in references[:8]]
        return {
            "status": "ok",
            "answer": "Closest visual matches from memory:\n" + "\n".join(lines),
            "references": references,
            "confidence": references[0].get("score", 0),
        }

    # ── Chat Sessions (Persistent) ────────────────────────────────────────

    def create_chat_session(self, title: Optional[str] = None) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = _now_iso()
        safe_title = (title or "New Chat").strip() or "New Chat"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (session_id, safe_title[:120], now, now),
            )
            conn.commit()
        return {"id": session_id, "title": safe_title[:120], "created_at": now, "updated_at": now}

    def list_chat_sessions(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(m.id) as message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON m.session_id = s.id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [
            {
                "id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
                "message_count": int(r[4] or 0),
            }
            for r in rows
        ]

    def ensure_chat_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        if session_id:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
            if row:
                return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}
        sessions = self.list_chat_sessions()
        if sessions:
            return sessions[0]
        return self.create_chat_session("New Chat")

    def get_chat_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, confidence, references_json, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            refs = []
            if r[4]:
                try:
                    refs = json.loads(r[4])
                except Exception:
                    refs = []
            out.append({
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "confidence": r[3],
                "references": refs,
                "created_at": r[5],
            })
        return out

    def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        confidence: Optional[float] = None,
        references: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        session = self.ensure_chat_session(session_id)
        message_id = str(uuid.uuid4())
        now = _now_iso()
        refs_json = json.dumps(references or [])
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_messages (id, session_id, role, content, confidence, references_json, created_at) VALUES (?,?,?,?,?,?,?)",
                (message_id, session["id"], role, content, confidence, refs_json, now),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session["id"]),
            )
            conn.commit()
        return {
            "id": message_id,
            "session_id": session["id"],
            "role": role,
            "content": content,
            "confidence": confidence,
            "references": references or [],
            "created_at": now,
        }

    def delete_chat_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a chat session and all of its messages."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return {"status": "error", "reason": "session_not_found"}

            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()

        return {"status": "ok", "deleted_session_id": session_id}

    # ── Automation Helpers ─────────────────────────────────────────────────

    def daily_digest(self) -> str:
        """Generate a summary of recent memories using Ollama."""
        recent = self.recent_records(limit=15)
        if not recent:
            return "No recent records to summarize."

        summaries = [f"- [{r['category']}] {r['summary']}" for r in recent if r.get("summary")]
        context = "\n".join(summaries)

        llm_cfg = self.config.get("ollama", {})
        if not llm_cfg.get("enabled", False):
            return f"Recent activity:\n{context}"

        try:
            import requests
            payload = {
                "model": llm_cfg.get("model", "qwen2.5:3b"),
                "messages": [
                    {"role": "system", "content": "Summarize the user's recent knowledge entries into a concise daily digest. Group by category."},
                    {"role": "user", "content": f"Here are my recent entries:\n{context}"},
                ],
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": 400},
            }
            resp = requests.post(
                f"{llm_cfg.get('base_url', 'http://127.0.0.1:11434')}/api/chat",
                json=payload, timeout=30,
            )
            if resp.ok:
                return resp.json().get("message", {}).get("content", context)
        except Exception:
            pass
        return f"Recent activity:\n{context}"
    # ── Purge All Data ─────────────────────────────────────────────────────

    def purge_all_data(self) -> Dict[str, Any]:
        """Delete EVERYTHING the model has learned: ChromaDB collections,
        SQLite records/actions/chats, BM25 cache, file state.
        This is a full factory reset of all knowledge."""
        counts = {
            "text_chunks_deleted": 0,
            "image_items_deleted": 0,
            "records_deleted": 0,
            "actions_deleted": 0,
            "chat_sessions_deleted": 0,
            "chat_messages_deleted": 0,
        }
        with self.lock:
            # 1) Wipe ChromaDB collections
            try:
                counts["text_chunks_deleted"] = self.text_collection.count()
                # Try deleting all IDs instead of dropping collection (more robust)
                all_ids = self.text_collection.get(include=[])["ids"]
                if all_ids:
                    # ChromaDB delete in batches (max ~40k per call)
                    for i in range(0, len(all_ids), 5000):
                        self.text_collection.delete(ids=all_ids[i:i+5000])
            except Exception as e:
                print(f"[purge] Error wiping text collection: {e}")
                # Fallback: try drop + recreate
                try:
                    self.chroma.delete_collection("ai_minds_text")
                    self.text_collection = self.chroma.get_or_create_collection(
                        name="ai_minds_text", metadata={"hnsw:space": "cosine"}
                    )
                except Exception as e2:
                    print(f"[purge] Fallback also failed for text: {e2}")

            try:
                counts["image_items_deleted"] = self.image_collection.count()
                all_ids = self.image_collection.get(include=[])["ids"]
                if all_ids:
                    for i in range(0, len(all_ids), 5000):
                        self.image_collection.delete(ids=all_ids[i:i+5000])
            except Exception as e:
                print(f"[purge] Error wiping image collection: {e}")
                try:
                    self.chroma.delete_collection("ai_minds_image")
                    self.image_collection = self.chroma.get_or_create_collection(
                        name="ai_minds_image", metadata={"hnsw:space": "cosine"}
                    )
                except Exception as e2:
                    print(f"[purge] Fallback also failed for images: {e2}")

            # 2) Wipe SQLite tables
            try:
                with sqlite3.connect(self.db_path) as conn:
                    for table in ["actions", "chat_messages", "chat_sessions", "records", "automations"]:
                        cur = conn.execute(f"DELETE FROM {table}")
                        key = f"{table}_deleted"
                        if key in counts:
                            counts[key] = int(cur.rowcount or 0)
                    conn.commit()
            except Exception as e:
                print(f"[purge] Error wiping SQLite: {e}")

            # 3) Delete BM25 cache
            try:
                if self.bm25_path.exists():
                    self.bm25_path.unlink()
                self.bm25_index = None
                self.doc_id_order = []
                self.doc_token_corpus = []
            except Exception as e:
                print(f"[purge] Error deleting BM25 cache: {e}")

            # 4) Reset file state
            try:
                if self.file_state_path.exists():
                    self.file_state_path.unlink()
            except Exception as e:
                print(f"[purge] Error resetting file state: {e}")

        print("[purge] All data has been wiped.")
        return {"status": "ok", **counts}