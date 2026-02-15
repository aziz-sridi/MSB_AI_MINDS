"""
GigaMind Engine — Core intelligence layer.

Combines the best of second-brain (hybrid search, embeddings, BM25, MMR)
and ai_minds_engine (categorization, summaries, actions, SQLite records)
with Ollama-powered LLM (≤4B params) for grounded Q&A.
"""

import json
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


def _categorize(text: str, metadata: Dict[str, Any]) -> str:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?)",
                (record_id, modality, category, summary, source,
                 json.dumps(metadata), _now_iso()),
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

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_text(self, text: str, source: str, modality: str,
                    metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a block of text: chunk → embed → store → index."""
        chunk_size = int(self.config.get("chunk_size_chars", 800))
        chunks = _chunk_text(text, max_chars=chunk_size)
        chunks = [c for c in chunks if not _is_gibberish(c)]
        if not chunks:
            return {"status": "skipped", "reason": "empty_or_gibberish", "source": source}

        category = _categorize(text, metadata)
        summary = _sentence_summary(text)
        actions_enabled = bool(self.config.get("action_extraction_enabled", False))
        actions = _extract_actions(text) if actions_enabled else []

        # Store raw chunks (no prefix in documents — prefix hurts semantic matching)
        prefixed = chunks

        embeddings = self.text_model.encode(
            prefixed, convert_to_numpy=True, normalize_embeddings=True, batch_size=16
        ).tolist()

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{
            "source": source, "modality": modality, "category": category,
            "summary": summary, "chunk_index": i, **metadata,
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

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a local file (text, image, audio, or video)."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return {"status": "error", "reason": "file_not_found"}

        ext = path.suffix.lower()

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
            return self.ingest_text(
                text=text, source=str(path), modality=f"file_{ext.strip('.')}",
                metadata={"source_path": str(path), "extension": ext, "source_type": "file"},
            )

        # Images
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            return self._ingest_image(path)

        # Audio / Video (metadata-level ingestion)
        if ext in (SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS):
            media_text = parse_media(path, self.config.get("transcription", {}))
            media_kind = "audio" if ext in SUPPORTED_AUDIO_EXTENSIONS else "video"
            return self.ingest_text(
                text=media_text,
                source=str(path),
                modality=f"file_{media_kind}",
                metadata={
                    "source_path": str(path),
                    "extension": ext,
                    "source_type": "file",
                    "media_type": media_kind,
                },
            )

        return {"status": "skipped", "reason": "unsupported_extension", "path": file_path}

    def _ingest_image(self, path: Path) -> Dict[str, Any]:
        filename = path.name
        folder = path.parent.name
        surrogate = f"image {filename} in folder {folder}"
        meta = {
            "source_path": str(path), "extension": path.suffix.lower(),
            "modality": "image", "source": str(path), "source_type": "file",
            "category": _categorize(surrogate, {"source_path": str(path)}),
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
        else:
            emb = self.text_model.encode([surrogate], normalize_embeddings=True).tolist()
            with self.lock:
                self.text_collection.upsert(
                    ids=[doc_id], documents=[surrogate],
                    embeddings=emb, metadatas=[meta],
                )

        self._store_record(doc_id, "image", meta["category"], meta["summary"],
                           str(path), meta, [])
        return {"status": "ok", "source": str(path), "modality": "image",
                "category": meta["category"], "chunks_added": 1}

    # ── Search ─────────────────────────────────────────────────────────────

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        emb = self.text_model.encode([query], normalize_embeddings=True).tolist()
        n_available = self.text_collection.count()
        if n_available == 0:
            return []
        actual_k = min(top_k, n_available)
        result = self.text_collection.query(
            query_embeddings=emb, n_results=actual_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
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

    def _lexical_search(self, query: str, top_k: int) -> List[Dict]:
        if not self.bm25_index or not self.doc_id_order:
            return []
        tokens = _normalize_tokens(query)
        if not tokens:
            return []
        scores = self.bm25_index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            doc_id = self.doc_id_order[idx]
            results.append({"id": doc_id, "score": float(score)})
        return results

    def _hybrid_search(self, query: str, top_k: int = 6) -> List[Dict]:
        """Combine semantic + lexical search with MMR reranking."""
        fetch_k = top_k * 5

        sem_results = self._semantic_search(query, fetch_k)
        lex_results = self._lexical_search(query, fetch_k)

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

    # ── Answering ──────────────────────────────────────────────────────────

    def answer(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Full RAG pipeline: search → context → Ollama LLM → grounded answer."""
        k = int(top_k or self.config.get("top_k", 6))
        intent = _intent_from_query(query)

        results = self._hybrid_search(query, k)
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
            references.append({
                "source": item.get("metadata", {}).get("source", "unknown"),
                "source_type": item.get("metadata", {}).get("source_type", "unknown"),
                "category": item.get("metadata", {}).get("category", "general"),
                "modality": item.get("metadata", {}).get("modality", "text"),
                "snippet": (item.get("documents") or "")[:240],
                "score": round(item.get("score", 0), 4),
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

        # Build context from search results
        context_parts = []
        for i, r in enumerate(results[:6]):
            source = r.get("metadata", {}).get("source", "unknown")
            text = r.get("documents", "")[:400]
            context_parts.append(f"[{i+1}] Source: {source}\n{text}")
        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are GigaMind, a personal knowledge assistant. "
            "Answer the user's question using the provided context from their stored memories. "
            "The context snippets come from different sources (web pages, notes, files). "
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

    def ollama_stream(self, query: str, top_k: Optional[int] = None):
        """Streaming version of answer — yields chunks for real-time UI."""
        k = int(top_k or self.config.get("top_k", 6))
        results = self._hybrid_search(query, k)

        llm_cfg = self.config.get("ollama", {})
        if not llm_cfg.get("enabled", False):
            yield self._build_grounded_answer(query, _intent_from_query(query), results)
            return

        model = llm_cfg.get("model", "qwen2.5:3b")
        base_url = llm_cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")

        context_parts = []
        for i, r in enumerate(results[:6]):
            source = r.get("metadata", {}).get("source", "unknown")
            text = r.get("documents", "")[:400]
            context_parts.append(f"[{i+1}] Source: {source}\n{text}")
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
        """Scan all watched directories for new/modified files."""
        watch_dirs = [Path(p) for p in self.config.get("watch_dirs", []) if p]
        if not watch_dirs:
            return {"status": "skipped", "reason": "no_watch_dirs"}

        previous_state: Dict[str, float] = {}
        if self.file_state_path.exists():
            try:
                previous_state = json.loads(self.file_state_path.read_text(encoding="utf-8"))
            except Exception:
                previous_state = {}

        current_state: Dict[str, float] = {}
        processed = 0
        errors = 0

        for directory in watch_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                as_str = str(path)
                mtime = path.stat().st_mtime
                current_state[as_str] = mtime
                if previous_state.get(as_str) == mtime:
                    continue
                try:
                    result = self.ingest_file(as_str)
                    if result.get("status") == "ok":
                        processed += 1
                except Exception as e:
                    print(f"[engine] Error ingesting {as_str}: {e}")
                    errors += 1

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
