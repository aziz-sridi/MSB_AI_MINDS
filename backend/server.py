"""
GigaMind — Unified Flask API Server.

Serves:
  - Browser extension endpoints (web clip ingestion, element extraction)
  - Query/answer endpoint (RAG with Ollama)
  - File ingestion and directory watch
  - Stats, actions, automations
  - Streaming chat endpoint
"""

import json
import os
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from engine import AIMindsEngine
from automations import AutomationManager

# ── Setup ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"

app = Flask(__name__)
CORS(app)  # Allow extension & frontend requests

engine = AIMindsEngine(CONFIG_PATH)
engine.start_watcher()

automations = AutomationManager(engine)
automations.start_all()


# ── Global error handler — always return JSON, never HTML ────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    """Catch all unhandled exceptions and return JSON error."""
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ── Health ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ai-minds-backend", "timestamp": time.time()})


# ── Web Clip Ingestion (from browser extension) ─────────────────────────────

@app.route("/index", methods=["POST"])
def index_web_clip():
    """Accept text content + metadata from the browser extension (memex-style)."""
    payload = request.get_json(force=True)
    text = (payload.get("textContent") or "").strip()
    metadata = payload.get("metadata", {})
    if not text:
        return jsonify({"status": "ignored", "reason": "empty_text"})
    result = engine.ingest_web_clip(text, metadata)
    return jsonify(result)


@app.route("/api/extract", methods=["POST"])
def extract_elements():
    """Accept structured element data from the Newe-style extension.
    Converts element selections into text and ingests them."""
    payload = request.get_json(force=True)
    items = payload.get("items", [])
    meta = payload.get("meta", {})
    note = payload.get("note", "")

    if not items:
        return jsonify({"status": "ignored", "reason": "no_items"})

    # Combine all selected elements into a text block
    text_parts = []
    if note:
        text_parts.append(f"User note: {note}")
    text_parts.append(f"Page: {meta.get('pageTitle', '')} ({meta.get('pageUrl', '')})")

    for item in items:
        elem = item.get("element", {})
        elem_text = elem.get("text", "")
        sel_text = item.get("selectionText", "")
        content = sel_text or elem_text
        if content:
            text_parts.append(content)

        # Also capture any link/image info
        href = elem.get("href")
        if href:
            text_parts.append(f"Link: {href}")
        src = elem.get("src")
        if src:
            text_parts.append(f"Media: {src}")

    combined_text = "\n\n".join(text_parts)

    result = engine.ingest_web_clip(combined_text, {
        "url": meta.get("pageUrl", ""),
        "pageTitle": meta.get("pageTitle", ""),
        "extractedAt": meta.get("extractedAt", ""),
        "source_type": "web_extraction",
        "items_count": len(items),
    })

    return jsonify(result)


# ── Auto-capture (full page from extension content script) ───────────────────

@app.route("/api/auto-capture", methods=["POST"])
def auto_capture():
    """Auto-capture page content sent by content script on navigation."""
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    url = payload.get("url", "")
    title = payload.get("title", "")

    if not text or len(text) < 50:
        return jsonify({"status": "ignored", "reason": "too_short"})

    result = engine.ingest_web_clip(text, {
        "url": url, "pageTitle": title,
        "source_type": "auto_capture",
        "timestamp": time.time(),
    })
    return jsonify(result)


# ── Query / Answer ───────────────────────────────────────────────────────────

@app.route("/query", methods=["POST"])
def query():
    """Ask a question → get grounded answer with references."""
    payload = request.get_json(force=True)
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    top_k = payload.get("top_k")
    category = (payload.get("category") or "").strip() or None
    date_from = (payload.get("date_from") or "").strip() or None
    date_to = (payload.get("date_to") or "").strip() or None
    result = engine.answer(question, top_k=top_k, category=category,
                           date_from=date_from, date_to=date_to)
    return jsonify(result)


@app.route("/query/smart", methods=["POST"])
def query_smart():
    """Smart query — LLM extracts date, category, image intent from natural language.
    Accepts: question, image_path (optional), persist_image (optional bool).
    The LLM parses 'what was I working on 3 months ago' into
    category=work, date_from=3mo ago, date_to=today automatically."""
    try:
        payload = request.get_json(force=True)
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400

        image_path = (payload.get("image_path") or "").strip() or None
        persist_image = bool(payload.get("persist_image", False))

        result = engine.smart_query(
            query=question,
            image_path=image_path,
            persist_image=persist_image,
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Smart query failed: {str(e)}"}), 500


@app.route("/query/image", methods=["POST"])
def query_image():
    """Search for images similar to the provided query image."""
    payload = request.get_json(force=True)
    image_path = (payload.get("image_path") or "").strip()
    if not image_path:
        return jsonify({"error": "image_path is required"}), 400
    top_k = int(payload.get("top_k", 6))
    result = engine.find_similar_images(image_path=image_path, top_k=top_k)
    code = 200 if result.get("status") != "error" else 400
    return jsonify(result), code


@app.route("/answer", methods=["GET", "POST"])
def answer_compat():
    """Legacy-compatible answer endpoint."""
    if request.method == "POST":
        question = (request.get_json(force=True).get("question") or "").strip()
    else:
        question = (request.args.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    result = engine.answer(question)
    return jsonify(result)


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming chat endpoint — yields tokens in real-time."""
    payload = request.get_json(force=True)
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    category = (payload.get("category") or "").strip() or None
    date_from = (payload.get("date_from") or "").strip() or None
    date_to = (payload.get("date_to") or "").strip() or None

    def generate():
        for chunk in engine.ollama_stream(question, category=category,
                                           date_from=date_from, date_to=date_to):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: {\"done\": true}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── File Ingestion ───────────────────────────────────────────────────────────

@app.route("/ingest/file", methods=["POST"])
def ingest_file():
    """Ingest a specific file by path."""
    payload = request.get_json(force=True)
    path = payload.get("path", "").strip()
    if not path:
        return jsonify({"error": "path is required"}), 400
    result = engine.ingest_file(path)
    return jsonify(result)


@app.route("/ingest/text", methods=["POST"])
def ingest_text():
    """Ingest raw text directly (for clipboard, notes, etc.)."""
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    source = payload.get("source", "manual_input")
    if not text:
        return jsonify({"error": "text is required"}), 400
    result = engine.ingest_text(text, source, "manual_input", {
        "source_type": "manual", "timestamp": time.time()
    })
    return jsonify(result)


@app.route("/ingest/scan", methods=["POST"])
def scan_dirs():
    """Trigger a manual scan of watched directories."""
    result = engine.ingest_watch_dirs()
    return jsonify(result)


# ── Config ───────────────────────────────────────────────────────────────────

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify(engine.config)


@app.route("/config", methods=["POST"])
def update_config():
    payload = request.get_json(force=True)
    engine.config.update(payload)
    engine.save_config()
    return jsonify({"status": "ok", "config": engine.config})


@app.route("/config/watch_dirs", methods=["POST"])
def update_watch_dirs():
    payload = request.get_json(force=True)
    dirs = payload.get("watch_dirs", [])
    if not isinstance(dirs, list):
        return jsonify({"error": "watch_dirs must be a list"}), 400
    engine.config["watch_dirs"] = dirs
    engine.save_config()
    return jsonify({"status": "ok", "watch_dirs": dirs})


# ── Stats & Records ──────────────────────────────────────────────────────────

@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(engine.get_stats())


@app.route("/records/recent", methods=["GET"])
def recent_records():
    limit = int(request.args.get("limit", 20))
    return jsonify(engine.recent_records(limit))


@app.route("/actions", methods=["GET"])
def actions():
    done = request.args.get("done", "false").lower() == "true"
    return jsonify(engine.get_actions(done))


@app.route("/actions/<action_id>/done", methods=["POST"])
def mark_done(action_id):
    updated = engine.mark_action_done(action_id)
    return jsonify({"status": "ok", "updated": updated})


@app.route("/actions/cleanup", methods=["POST"])
def cleanup_actions():
    result = engine.cleanup_actions()
    return jsonify(result)


@app.route("/actions/purge", methods=["POST"])
def purge_actions():
    result = engine.purge_actions()
    return jsonify(result)


@app.route("/purge/all", methods=["POST"])
def purge_all():
    """Delete ALL data: vector stores, SQLite records, BM25 cache, chat history.
    This is a full factory reset of everything the system has learned."""
    try:
        payload = request.get_json(silent=True) or {}
        confirm = payload.get("confirm", False)
        if not confirm:
            return jsonify({"error": "Must send {\"confirm\": true} to purge all data."}), 400
        result = engine.purge_all_data()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Purge failed: {str(e)}"}), 500


# ── Chat Sessions (Persistent) ─────────────────────────────────────────────

@app.route("/chat/sessions", methods=["GET"])
def chat_sessions_list():
    sessions = engine.list_chat_sessions()
    return jsonify(sessions)


@app.route("/chat/sessions", methods=["POST"])
def chat_sessions_create():
    payload = request.get_json(silent=True) or {}
    title = payload.get("title")
    session = engine.create_chat_session(title=title)
    return jsonify(session)


@app.route("/chat/sessions/<session_id>", methods=["DELETE"])
def chat_sessions_delete(session_id):
    result = engine.delete_chat_session(session_id)
    code = 200 if result.get("status") == "ok" else 404
    return jsonify(result), code


@app.route("/chat/sessions/<session_id>/messages", methods=["GET"])
def chat_messages_get(session_id):
    session = engine.ensure_chat_session(session_id)
    messages = engine.get_chat_messages(session["id"])
    return jsonify({"session": session, "messages": messages})


@app.route("/chat/sessions/<session_id>/messages", methods=["POST"])
def chat_messages_add(session_id):
    payload = request.get_json(force=True)
    role = (payload.get("role") or "").strip()
    content = (payload.get("content") or "").strip()
    if role not in {"user", "assistant", "system"}:
        return jsonify({"error": "role must be one of user, assistant, system"}), 400
    if not content:
        return jsonify({"error": "content is required"}), 400

    confidence = payload.get("confidence")
    references = payload.get("references") if isinstance(payload.get("references"), list) else []
    saved = engine.save_chat_message(
        session_id=session_id,
        role=role,
        content=content,
        confidence=confidence,
        references=references,
    )
    return jsonify(saved)


# ── Automations ──────────────────────────────────────────────────────────────

@app.route("/digest", methods=["GET"])
def daily_digest():
    """Generate an AI-powered daily digest of recent activity."""
    return jsonify({"digest": engine.daily_digest()})


@app.route("/automations/status", methods=["GET"])
def automations_status():
    """Get running status of all automations."""
    return jsonify(automations.status())


@app.route("/automations/toggle", methods=["POST"])
def automations_toggle():
    """Toggle a specific automation on/off."""
    payload = request.get_json(force=True)
    name = payload.get("name", "").strip()
    enabled = payload.get("enabled", True)

    valid = ["clipboard", "browser_history", "screenshots", "downloads", "digest", "reminders"]
    if name not in valid:
        return jsonify({"error": f"Unknown automation: {name}. Valid: {valid}"}), 400

    # Update config
    auto_cfg = engine.config.setdefault("automations", {})

    # Map automation names to config keys
    config_key_map = {
        "clipboard": "clipboard_monitor",
        "browser_history": "browser_history",
        "screenshots": "screenshot_watcher",
        "downloads": "downloads_watcher",
        "digest": "digest_scheduler",
        "reminders": "reminders",
    }
    cfg_key = config_key_map.get(name, name)
    auto_cfg[cfg_key] = enabled
    engine.save_config()

    if enabled:
        automations.restart(name)
    else:
        automations.stop(name)

    return jsonify({"status": "ok", "name": name, "enabled": enabled, "running": automations._is_running(name)})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  GigaMind Backend — Starting on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
