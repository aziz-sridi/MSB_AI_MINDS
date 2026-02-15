"""
GigaMind — Streamlit Dashboard Frontend.

A web-based UI for:
  - Chatting with your knowledge base (RAG with Ollama)
  - Viewing stats, categories, recent records
  - Managing action items / todos
  - Configuring watch directories and settings
  - Ingesting files and text manually
  - Daily digest generation
"""

import json
import os
import re
import time
from pathlib import Path

import requests
import streamlit as st

API = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="GigaMind",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0f1218 0%, #111827 100%); }
    .block-container { padding-top: 1rem; max-width: 1300px; }
    h1, h2, h3 { color: #ecf1ff; letter-spacing: 0.2px; }
    .stMetric label { color: #9fb0cb !important; }
    .stMetric [data-testid="stMetricValue"] { color: #80eaff !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1622 0%, #121b2b 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] img {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.35rem 0.6rem;
        margin-bottom: 0.55rem;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
        border-radius: 10px;
    }
    .chat-panel-title {
        font-size: 1rem;
        color: #cfe2ff;
        margin-bottom: 0.45rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper ───────────────────────────────────────────────────────────────────

def api(method, endpoint, **kwargs):
    """Make API call to backend."""
    try:
        resp = getattr(requests, method)(f"{API}{endpoint}", timeout=60, **kwargs)
        return resp.json()
    except requests.ConnectionError:
        return {"error": "Backend not reachable. Start the server first."}
    except Exception as e:
        return {"error": str(e)}


def check_backend():
    try:
        r = requests.get(f"{API}/health", timeout=3)
        return r.ok
    except Exception:
        return False


def save_uploaded_file(uploaded_file, folder_name: str) -> str:
    """Persist uploaded file into backend data folder and return saved path."""
    upload_dir = Path(__file__).parent.parent / "backend" / "data" / folder_name
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{int(time.time() * 1000)}_{uploaded_file.name}"
    save_path = upload_dir / safe_name
    save_path.write_bytes(uploaded_file.getvalue())
    return str(save_path.resolve())


def resolve_mascot_path() -> Path | None:
    """Find mascot image from common runtime locations."""
    candidates = [
        Path(__file__).parent.parent / "GigaMaid.png",
        Path.cwd() / "GigaMaid.png",
        Path.cwd().parent / "GigaMaid.png",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def is_image_description_prompt(prompt: str) -> bool:
    """Detect prompts that mainly ask to describe/analyze an image."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", (prompt or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    describe_words = ["describe", "what is in", "what s in", "analyze", "explain image", "caption"]
    image_words = ["image", "img", "picture", "photo", "pic", "qge"]
    has_describe = any(word in normalized for word in describe_words)
    has_image_word = any(word in normalized for word in image_words)
    return has_describe and has_image_word


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("GigaMind")
    st.caption("Personal Knowledge Assistant")
    mascot_path = resolve_mascot_path()
    if mascot_path is not None:
        st.image(str(mascot_path), caption="GigaMaid", use_container_width=True)
    else:
        st.info("Mascot not found. Put `GigaMaid.png` in the `ai-minds/` folder.")

    online = check_backend()
    if online:
        st.success("Backend connected")
        stats = api("get", "/stats")
        if "error" not in stats:
            st.metric("Total Records", stats.get("total_records", 0))
            st.metric("Text Chunks", stats.get("text_chunks", 0))
            st.metric("Image Items", stats.get("image_items", 0))
    else:
        st.error("Backend offline — run `python server.py`")

    st.divider()
    page = st.radio("Navigation", [
        "Chat",
        "Dashboard",
        "Automations",
        "Manual Ingest",
        "Settings",
    ], label_visibility="collapsed")


# ── Chat Page ────────────────────────────────────────────────────────────────

if page == "Chat":
    chat_title_col, chat_mascot_col = st.columns([5, 1])
    with chat_title_col:
        st.header("Chat with Your Memory")
        st.caption("Ask about your memory, or upload an image to find similar visuals.")
    with chat_mascot_col:
        mascot_path = resolve_mascot_path()
        if mascot_path is not None:
            st.image(str(mascot_path), width=110)

    tips_col1, tips_col2 = st.columns(2)
    with tips_col1:
        st.info("Tip: For image-only prompts, ask like: **Describe this image** with an uploaded file.")
    with tips_col2:
        st.info("Tip: Start a new chat per topic for cleaner context and better answers.")
    if not online:
        st.warning("Backend offline")
    else:
        sessions = api("get", "/chat/sessions")
        if not isinstance(sessions, list):
            st.error("Could not load chat sessions.")
            sessions = []

        if not sessions:
            created = api("post", "/chat/sessions", json={"title": "New Chat"})
            if created.get("id"):
                sessions = [created]

        session_ids = [s.get("id") for s in sessions if s.get("id")]
        if session_ids:
            if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in session_ids:
                st.session_state.current_chat_id = session_ids[0]

            chats_col, convo_col = st.columns([1, 3])

            with chats_col:
                st.markdown("### Chats")
                if st.button("New Chat", use_container_width=True):
                    created = api("post", "/chat/sessions", json={"title": "New Chat"})
                    if created.get("id"):
                        st.session_state.current_chat_id = created["id"]
                        st.rerun()

                with st.container(height=640, border=True):
                    for s in sessions:
                        sid = s.get("id")
                        if not sid:
                            continue
                        label = f"{s.get('title', 'Chat')} ({s.get('message_count', 0)})"
                        is_current = sid == st.session_state.current_chat_id
                        row_l, row_r = st.columns([5, 1])
                        with row_l:
                            button_type = "primary" if is_current else "secondary"
                            if st.button(label, key=f"chat_btn_{sid}", use_container_width=True, type=button_type):
                                st.session_state.current_chat_id = sid
                                st.rerun()
                        with row_r:
                            if st.button("🗑️", key=f"chat_del_{sid}", use_container_width=True, help="Delete chat"):
                                deleted = api("delete", f"/chat/sessions/{sid}")
                                if deleted.get("status") == "ok":
                                    if st.session_state.current_chat_id == sid:
                                        st.session_state.pop("current_chat_id", None)
                                st.rerun()

            selected_chat_id = st.session_state.current_chat_id

            msg_resp = api("get", f"/chat/sessions/{selected_chat_id}/messages")
            messages = msg_resp.get("messages", []) if isinstance(msg_resp, dict) else []

            with convo_col:
                st.markdown("#### Conversation")
                with st.container(height=640, border=True):
                    for msg in messages:
                        with st.chat_message(msg.get("role", "assistant")):
                            st.markdown(msg.get("content", ""))
                            if msg.get("confidence") is not None:
                                conf = float(msg.get("confidence", 0))
                                color = "green" if conf > 0.5 else "orange" if conf > 0.3 else "red"
                                st.caption(f"Confidence: :{color}[{conf:.0%}]")
                            refs = msg.get("references", [])
                            if refs:
                                with st.expander("Sources", expanded=False):
                                    for ref in refs:
                                        st.caption(
                                            f"**{ref.get('source', 'unknown')}** "
                                            f"(score: {ref.get('score', 0):.2f}, "
                                            f"category: {ref.get('category', '')})"
                                        )

                uploaded_query_image = st.file_uploader(
                    "Optional image prompt",
                    type=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
                    key=f"chat_image_prompt_{selected_chat_id}",
                    help="Upload an image to search visually similar images in memory.",
                )
                if uploaded_query_image is not None:
                    st.image(uploaded_query_image, caption=uploaded_query_image.name, width=220)

                persist_image = st.checkbox(
                    "Also save uploaded image to memory",
                    value=False,
                    key=f"save_chat_image_{selected_chat_id}",
                    help="Keep this off for pure query images. Turn on if you want this image indexed permanently.",
                )

                if prompt := st.chat_input("Ask anything about your stored memories..."):
                    image_path = None
                    if uploaded_query_image is not None:
                        image_path = save_uploaded_file(uploaded_query_image, "chat_queries")

                    user_content = prompt
                    if image_path and uploaded_query_image is not None:
                        user_content = f"{prompt}\n\n[Image prompt: {uploaded_query_image.name}]"

                    api(
                        "post",
                        f"/chat/sessions/{selected_chat_id}/messages",
                        json={"role": "user", "content": user_content},
                    )

                    with st.spinner("Searching memory & thinking..."):
                        image_only_prompt = bool(image_path) and is_image_description_prompt(prompt)
                        text_result = None
                        if not image_only_prompt:
                            text_result = api("post", "/query", json={"question": prompt})

                        image_result = None

                        if image_path:
                            image_result = api("post", "/query/image", json={"image_path": image_path, "top_k": 6})
                            if persist_image:
                                api("post", "/ingest/file", json={"path": image_path})

                    answer_parts = []
                    refs = []
                    confidence = 0.0
                    uncertainty = None

                    if isinstance(text_result, dict) and "error" not in text_result:
                        answer_parts.append(text_result.get("answer", ""))
                        refs.extend(text_result.get("references", []) or [])
                        confidence = max(confidence, float(text_result.get("confidence", 0) or 0))
                        uncertainty = text_result.get("uncertainty")
                    elif isinstance(text_result, dict) and text_result.get("error"):
                        answer_parts.append(f"Text query error: {text_result.get('error')}")

                    if isinstance(image_result, dict) and image_result.get("status") == "ok":
                        answer_parts.append(image_result.get("answer", ""))
                        refs.extend(image_result.get("references", []) or [])
                        confidence = max(confidence, float(image_result.get("confidence", 0) or 0))
                    elif isinstance(image_result, dict) and image_result.get("status") == "error":
                        answer_parts.append(f"Image query error: {image_result.get('reason', 'failed')}")

                    final_answer = "\n\n".join([p for p in answer_parts if p]).strip() or "No answer found."

                    api(
                        "post",
                        f"/chat/sessions/{selected_chat_id}/messages",
                        json={
                            "role": "assistant",
                            "content": final_answer,
                            "confidence": confidence,
                            "references": refs,
                        },
                    )

                    if uncertainty:
                        st.caption(f"Warning: {uncertainty}")
                    st.rerun()


# ── Dashboard Page ───────────────────────────────────────────────────────────

elif page == "Dashboard":
    st.header("Knowledge Dashboard")

    if not online:
        st.warning("Backend offline")
    else:
        stats = api("get", "/stats")
        if "error" not in stats:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", stats.get("total_records", 0))
            col2.metric("Text Chunks", stats.get("text_chunks", 0))
            col3.metric("Image Items", stats.get("image_items", 0))

            st.subheader("Categories")
            cats = stats.get("categories", {})
            if cats:
                import pandas as pd
                df = pd.DataFrame(list(cats.items()), columns=["Category", "Count"])
                st.bar_chart(df.set_index("Category"))
            else:
                st.info("No categories yet — ingest some data first.")

            st.subheader("Modalities")
            mods = stats.get("modalities", {})
            if mods:
                for mod, count in mods.items():
                    st.caption(f"**{mod}**: {count} records")

        st.divider()
        st.subheader("Recent Records")
        records = api("get", "/records/recent?limit=15")
        if isinstance(records, list):
            for r in records:
                with st.expander(f"[{r.get('category', '')}] {r.get('summary', 'No summary')[:80]}"):
                    st.caption(f"Source: {r.get('source', 'unknown')}")
                    st.caption(f"Modality: {r.get('modality', '')}")
                    st.caption(f"Created: {r.get('created_at', '')}")

        st.divider()
        st.subheader("Daily Digest")
        if st.button("Generate Digest"):
            with st.spinner("Generating..."):
                digest = api("get", "/digest")
            st.markdown(digest.get("digest", "No data."))


# ── Automations Page ─────────────────────────────────────────────────────────

elif page == "Automations":
    st.header("Automations — Hands-Free Data Ingestion")
    st.caption("These run in the background, automatically feeding data into your knowledge base.")

    if not online:
        st.warning("Backend offline")
    else:
        auto_status = api("get", "/automations/status")
        if "error" in auto_status:
            st.error(auto_status["error"])
        else:
            AUTOMATION_INFO = {
                "clipboard": {
                    "label": "Clipboard Monitor",
                    "desc": "Auto-ingests anything you copy (Ctrl+C) that's longer than 20 characters. "
                            "Skips passwords and random tokens.",
                    "config_key": "clipboard_monitor",
                },
                "browser_history": {
                    "label": "Browser History",
                    "desc": "Periodically scans Chrome/Edge history and ingests page titles + URLs you visit. "
                            "Skips internal browser pages.",
                    "config_key": "browser_history",
                },
                "screenshots": {
                    "label": "Screenshot Watcher",
                    "desc": "Watches your Screenshots folder and auto-ingests new screenshots using CLIP embeddings.",
                    "config_key": "screenshot_watcher",
                },
                "downloads": {
                    "label": "Downloads Watcher",
                    "desc": "Watches your Downloads folder and auto-ingests new documents (PDF, DOCX, TXT) and images.",
                    "config_key": "downloads_watcher",
                },
                "digest": {
                    "label": "Daily Digest",
                    "desc": "Generates a periodic AI summary of your recent knowledge entries.",
                    "config_key": "digest_scheduler",
                },
                "reminders": {
                    "label": "Smart Reminders",
                    "desc": "Checks for pending action items extracted from your data.",
                    "config_key": "reminders",
                },
            }

            auto_cfg = api("get", "/config").get("automations", {})

            for name, info in AUTOMATION_INFO.items():
                status = auto_status.get(name, {})
                running = status.get("running", False)

                with st.container():
                    col1, col2, col3 = st.columns([4, 1, 1])
                    with col1:
                        st.markdown(f"### {info['label']}")
                        st.caption(info["desc"])
                    with col2:
                        if running:
                            st.success("Running")
                        else:
                            st.error("Stopped")
                    with col3:
                        if running:
                            if st.button("Stop", key=f"stop_{name}"):
                                api("post", "/automations/toggle", json={"name": name, "enabled": False})
                                st.rerun()
                        else:
                            if st.button("Start", key=f"start_{name}"):
                                api("post", "/automations/toggle", json={"name": name, "enabled": True})
                                st.rerun()
                    st.divider()

            # Watch directories section
            st.subheader("Watched Folders")
            st.caption("The directory watcher auto-ingests new/modified files from these folders every 30 seconds.")
            cfg = api("get", "/config")
            if "error" not in cfg:
                wd = cfg.get("watch_dirs", [])
                if wd:
                    for i, d in enumerate(wd):
                        exists = os.path.isdir(d)
                        col_d, col_r = st.columns([6, 1])
                        col_d.markdown(f"{'Folder' if exists else 'Missing'}: `{d}`")
                        if col_r.button("Remove", key=f"rm_auto_dir_{i}"):
                            wd.pop(i)
                            cfg["watch_dirs"] = wd
                            api("post", "/config", json=cfg)
                            st.rerun()
                else:
                    st.info("No folders watched yet. Add one below.")

                new_dir = st.text_input(
                    "Add folder to watch",
                    placeholder="e.g. C:\\Users\\you\\Documents\\Notes",
                    key="auto_new_dir"
                )
                col_add, col_quick = st.columns(2)
                with col_add:
                    if st.button("Add Folder", key="auto_add_folder"):
                        if new_dir and new_dir.strip():
                            d = new_dir.strip()
                            if os.path.isdir(d):
                                if d not in wd:
                                    wd.append(d)
                                    cfg["watch_dirs"] = wd
                                    api("post", "/config", json=cfg)
                                    st.success(f"Now watching: {d}")
                                    st.rerun()
                                else:
                                    st.warning("Already watching this folder.")
                            else:
                                st.error(f"Folder not found: {d}")
                with col_quick:
                    if st.button("Scan Now", key="auto_scan_now"):
                        with st.spinner("Scanning all watched folders..."):
                            result = api("post", "/ingest/scan")
                        if result.get("status") == "ok":
                            st.success(f"Found {result.get('processed', 0)} new files")
                        else:
                            st.warning(f"Result: {result}")

                # Quick-add common folders
                with st.expander("Quick-add common folders"):
                    home = Path.home()
                    common = [
                        ("Documents", home / "Documents"),
                        ("Desktop", home / "Desktop"),
                        ("Downloads", home / "Downloads"),
                        ("OneDrive Documents", home / "OneDrive" / "Documents"),
                        ("Notes", home / "Notes"),
                        ("Pictures", home / "Pictures"),
                    ]
                    for label, folder in common:
                        if folder.exists() and str(folder) not in wd:
                            if st.button(f"{label} — {folder}", key=f"quick_auto_{label}"):
                                wd.append(str(folder))
                                cfg["watch_dirs"] = wd
                                api("post", "/config", json=cfg)
                                st.rerun()


# ── Actions Page ─────────────────────────────────────────────────────────────

elif page == "Actions":
    st.header("Action Items & Todos")
    cfg_actions = api("get", "/config") if online else {}
    actions_enabled = bool(cfg_actions.get("action_extraction_enabled", False)) if isinstance(cfg_actions, dict) else False
    st.caption("Auto-detected tasks from your ingested content.")

    if not online:
        st.warning("Backend offline")
    else:
        action_top_left, action_top_right = st.columns([3, 1])
        with action_top_left:
            if actions_enabled:
                st.info("Action extraction is ON. Use cleanup if you see noisy/repeated items.")
            else:
                st.warning("Action extraction is OFF. New ingested content will not create new todos.")
        with action_top_right:
            clean_btn_col, purge_btn_col = st.columns(2)
            with clean_btn_col:
                if st.button("🧹", use_container_width=True, help="Remove noisy and duplicate action items"):
                    cleaned = api("post", "/actions/cleanup")
                    if cleaned.get("status") == "ok":
                        st.success(
                            f"Removed noisy: {cleaned.get('removed_noisy', 0)} | "
                            f"duplicates: {cleaned.get('removed_duplicates', 0)}"
                        )
                    else:
                        st.warning(f"Cleanup result: {cleaned}")
                    st.rerun()
            with purge_btn_col:
                if st.button("🗑️", use_container_width=True, help="Delete all action items"):
                    purged = api("post", "/actions/purge")
                    if purged.get("status") == "ok":
                        st.success(f"Deleted {purged.get('deleted', 0)} action items")
                    else:
                        st.warning(f"Purge result: {purged}")
                    st.rerun()

        tab1, tab2 = st.tabs(["Pending", "Completed"])

        with tab1:
            actions = api("get", "/actions?done=false")
            if isinstance(actions, list) and actions:
                for a in actions:
                    col1, col2 = st.columns([5, 1])
                    text = (a.get("text", "") or "").strip()
                    if len(text) > 180:
                        text = text[:180].rstrip() + " ..."
                    col1.markdown(f"- {text}")
                    count = int(a.get("duplicate_count", 1) or 1)
                    count_label = f" | repeats: x{count}" if count > 1 else ""
                    col1.caption(f"Source: {a.get('source', '')} | {a.get('created_at', '')}{count_label}")
                    if col2.button("✅", key=f"done_{a['id']}", help="Mark done"):
                        api("post", f"/actions/{a['id']}/done")
                        st.rerun()
            else:
                st.info("No pending actions.")

        with tab2:
            done = api("get", "/actions?done=true")
            if isinstance(done, list) and done:
                for a in done:
                    text = (a.get("text", "") or "").strip()
                    if len(text) > 180:
                        text = text[:180].rstrip() + " ..."
                    st.markdown(f"- ~~{text}~~")
                    st.caption(f"Source: {a.get('source', '')}")
            else:
                st.info("No completed actions.")


# ── Ingest Page ──────────────────────────────────────────────────────────────

elif page == "Manual Ingest":
    st.header("Manual Ingest")
    st.caption("For manually adding specific content. For automatic ingestion, use Automations.")

    if not online:
        st.warning("Backend offline")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Text Input", "Upload Files", "File Path", "Scan Directories"])

        with tab1:
            st.subheader("Paste text to ingest")
            text_input = st.text_area("Text content", height=200,
                                      placeholder="Paste notes, articles, or any text...")
            source = st.text_input("Source label", value="manual_input")
            if st.button("Ingest Text"):
                if text_input.strip():
                    with st.spinner("Ingesting..."):
                        result = api("post", "/ingest/text", json={
                            "text": text_input, "source": source
                        })
                    if result.get("status") == "ok":
                        st.success(f"Ingested {result.get('chunks_added', 0)} chunks "
                                   f"(category: {result.get('category', '')})")
                    else:
                        st.warning(f"Result: {result}")
                else:
                    st.warning("Enter some text first.")

        with tab2:
            st.subheader("Upload files to ingest")
            st.caption("Supports: PDF, DOCX, TXT, MD, CSV, JSON, images, audio, and video files.")
            uploaded = st.file_uploader(
                "Choose files", accept_multiple_files=True,
                type=["txt", "md", "pdf", "docx", "csv", "json", "log",
                      "png", "jpg", "jpeg", "gif", "bmp", "webp",
                      "mp3", "wav", "m4a", "aac", "flac", "ogg", "wma",
                      "mp4", "mov", "mkv", "avi", "webm", "m4v", "wmv"],
            )
            if uploaded and st.button("Ingest Uploaded Files"):
                upload_dir = Path(__file__).parent.parent / "backend" / "data" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                total_chunks = 0
                for f in uploaded:
                    save_path = upload_dir / f.name
                    save_path.write_bytes(f.getvalue())
                    with st.spinner(f"Ingesting {f.name}..."):
                        result = api("post", "/ingest/file", json={"path": str(save_path)})
                    if result.get("status") == "ok":
                        chunks = result.get("chunks_added", 0)
                        total_chunks += chunks
                        st.caption(f"OK: {f.name}: {chunks} chunks ({result.get('category', '')})")
                    else:
                        st.caption(f"Warning: {f.name}: {result.get('reason', result.get('error', 'failed'))}")
                if total_chunks > 0:
                    st.success(f"Done! {total_chunks} total chunks ingested from {len(uploaded)} file(s).")

        with tab3:
            st.subheader("Ingest a file by path")
            file_path = st.text_input("File path", placeholder="C:\\Users\\...\\document.pdf")
            if st.button("Ingest File"):
                if file_path.strip():
                    with st.spinner("Ingesting..."):
                        result = api("post", "/ingest/file", json={"path": file_path})
                    if result.get("status") == "ok":
                        st.success(f"Ingested! {result.get('chunks_added', 0)} chunks")
                    else:
                        st.warning(f"Result: {result}")

        with tab4:
            st.subheader("Scan watched directories")
            st.caption("Scans all configured watch directories for new/modified files.")
            # Show current watch dirs
            cfg = api("get", "/config")
            if "error" not in cfg:
                wd = cfg.get("watch_dirs", [])
                if wd:
                    st.markdown("**Monitored folders:**")
                    for d in wd:
                        exists = os.path.isdir(d)
                        st.caption(f"{'Folder' if exists else 'Missing'}: {d}")
                else:
                    st.info("No watch dirs configured — go to Settings to add folders.")
            if st.button("Trigger Scan Now"):
                with st.spinner("Scanning directories..."):
                    result = api("post", "/ingest/scan")
                if result.get("status") == "ok":
                    st.success(f"Processed {result.get('processed', 0)} files "
                               f"({result.get('errors', 0)} errors)")
                else:
                    st.warning(f"Result: {result}")


# ── Settings Page ────────────────────────────────────────────────────────────

elif page == "Settings":
    st.header("Settings")

    if not online:
        st.warning("Backend offline")
    else:
        config = api("get", "/config")
        if "error" in config:
            st.error(config["error"])
        else:
            st.subheader("Ollama LLM")
            ollama = config.get("ollama", {})
            ollama_enabled = st.checkbox("Enable Ollama LLM", value=ollama.get("enabled", True))
            ollama_model = st.text_input("Model (≤4B params)", value=ollama.get("model", "qwen2.5:3b"))
            ollama_url = st.text_input("Ollama URL", value=ollama.get("base_url", "http://127.0.0.1:11434"))
            ollama_timeout = st.slider("Timeout (seconds)", 5, 120, int(ollama.get("timeout_seconds", 30)))

            st.subheader("Search & Embedding")
            chunk_size = st.slider("Chunk size (chars)", 200, 2000, int(config.get("chunk_size_chars", 800)))
            top_k = st.slider("Results per query", 1, 20, int(config.get("top_k", 6)))
            text_model = st.selectbox("Text embedding model", [
                "BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5", "BAAI/bge-m3"
            ], index=0)
            image_model = st.selectbox("Image embedding model", [
                "clip-ViT-B-32", "clip-ViT-B-16", "clip-ViT-L-14"
            ], index=0)

            st.subheader("Watch Directories")
            st.caption("Folders to auto-monitor for new documents, images, notes, etc.")
            dirs = config.get("watch_dirs", [])

            # Show existing directories with remove buttons
            if dirs:
                for i, d in enumerate(dirs):
                    col_d, col_r = st.columns([6, 1])
                    exists = os.path.isdir(d)
                    icon = "Folder" if exists else "Missing"
                    col_d.markdown(f"{icon}: `{d}`")
                    if col_r.button("Remove", key=f"rm_dir_{i}"):
                        dirs.pop(i)
                        config["watch_dirs"] = dirs
                        api("post", "/config", json=config)
                        st.rerun()
            else:
                st.info("No watch directories configured yet.")

            # Add new directory - text input
            new_dir = st.text_input(
                "Add a folder path",
                placeholder="e.g. C:\\Users\\you\\Documents\\Notes",
                key="new_watch_dir"
            )
            if st.button("Add Folder"):
                if new_dir and new_dir.strip():
                    d = new_dir.strip()
                    if os.path.isdir(d):
                        if d not in dirs:
                            dirs.append(d)
                            config["watch_dirs"] = dirs
                            api("post", "/config", json=config)
                            st.success(f"Added: {d}")
                            st.rerun()
                        else:
                            st.warning("Already in the list.")
                    else:
                        st.error(f"Folder not found: {d}")
                else:
                    st.warning("Enter a folder path first.")

            # Quick-add common folders
            with st.expander("Quick-add common folders"):
                home = Path.home()
                common = [
                    home / "Documents",
                    home / "Desktop",
                    home / "Downloads",
                    home / "OneDrive" / "Documents",
                    home / "Notes",
                ]
                for folder in common:
                    if folder.exists() and str(folder) not in dirs:
                        if st.button(f"Add {folder}", key=f"quick_{folder}"):
                            dirs.append(str(folder))
                            config["watch_dirs"] = dirs
                            api("post", "/config", json=config)
                            st.rerun()

            watch_interval = st.slider("Watch interval (seconds)", 10, 300,
                                       int(config.get("watch_interval_seconds", 30)))

            st.subheader("Source & Similarity Thresholds")
            source_score_threshold = st.slider(
                "Minimum source score to attach references",
                0.0,
                1.0,
                float(config.get("source_score_threshold", 0.2)),
                0.01,
            )
            image_similarity_threshold = st.slider(
                "Minimum image similarity score",
                0.0,
                1.0,
                float(config.get("image_similarity_threshold", 0.2)),
                0.01,
            )

            st.caption("Supported files: `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, `.json`, `.log`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`, `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.wma`, `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`, `.wmv`")

            st.subheader("Action Extraction")
            action_extraction_enabled = st.checkbox(
                "Enable Action Items extraction",
                value=bool(config.get("action_extraction_enabled", False)),
                help="When disabled, newly ingested content will not create Action Items/Todos.",
            )

            st.subheader("Media Transcription (Whisper)")
            st.caption("Transcribes audio/video during ingestion using local faster-whisper.")
            tx_cfg = config.get("transcription", {})
            tx_enabled = st.checkbox("Enable media transcription", value=tx_cfg.get("enabled", True))
            tx_model = st.selectbox(
                "Whisper model",
                options=["tiny", "base", "small", "medium", "large-v3"],
                index=["tiny", "base", "small", "medium", "large-v3"].index(tx_cfg.get("model", "small"))
                if tx_cfg.get("model", "small") in ["tiny", "base", "small", "medium", "large-v3"] else 2,
            )
            tx_device = st.selectbox(
                "Device",
                options=["cpu", "cuda"],
                index=0 if tx_cfg.get("device", "cpu") == "cpu" else 1,
            )
            tx_compute = st.selectbox(
                "Compute type",
                options=["int8", "float16", "float32"],
                index=["int8", "float16", "float32"].index(tx_cfg.get("compute_type", "int8"))
                if tx_cfg.get("compute_type", "int8") in ["int8", "float16", "float32"] else 0,
            )
            tx_language = st.text_input(
                "Language (optional, e.g. en, fr, ar)",
                value=tx_cfg.get("language", ""),
            )
            tx_beam = st.slider("Beam size", 1, 8, int(tx_cfg.get("beam_size", 2)))
            tx_max_chars = st.slider("Max transcript chars", 1000, 40000, int(tx_cfg.get("max_chars", 12000)), 500)

            st.subheader("System Prompt")
            persona_prompts = {
                "default": "You are GigaMind, a personal knowledge assistant. Answer using ONLY the provided context from the user's stored memories. If the context is insufficient, say so explicitly. Never fabricate information. Be concise and reference which source(s) support your answer.",
                "gigamaid": "You are Gigamaid, the GigaMind assistant persona: a giga chad in a maid costume. Keep a confident, playful-but-respectful tone while staying helpful and grounded. Answer using ONLY the provided memory context. If context is insufficient, explicitly say so. Never fabricate facts and cite supporting sources when available."
            }
            selected_persona = st.selectbox(
                "Persona",
                options=["default", "gigamaid"],
                index=0 if config.get("persona", "default") == "default" else 1,
                format_func=lambda p: "Default" if p == "default" else "Gigamaid",
            )
            use_persona_prompt = st.checkbox("Use selected persona preset prompt", value=True)

            default_prompt_value = (
                persona_prompts[selected_persona]
                if use_persona_prompt
                else config.get("system_prompt", persona_prompts["default"])
            )
            sys_prompt = st.text_area(
                "System prompt for LLM",
                value=default_prompt_value,
                height=120,
                key=f"system_prompt_{selected_persona}_{'preset' if use_persona_prompt else 'custom'}",
            )

            if st.button("Save Settings"):
                new_config = {
                    "ollama": {
                        "enabled": ollama_enabled,
                        "model": ollama_model,
                        "base_url": ollama_url,
                        "timeout_seconds": ollama_timeout,
                    },
                    "chunk_size_chars": chunk_size,
                    "top_k": top_k,
                    "text_model_name": text_model,
                    "image_model_name": image_model,
                    "watch_dirs": dirs,
                    "watch_interval_seconds": watch_interval,
                    "source_score_threshold": source_score_threshold,
                    "image_similarity_threshold": image_similarity_threshold,
                    "action_extraction_enabled": action_extraction_enabled,
                    "transcription": {
                        "enabled": tx_enabled,
                        "model": tx_model,
                        "device": tx_device,
                        "compute_type": tx_compute,
                        "language": tx_language,
                        "beam_size": tx_beam,
                        "max_chars": tx_max_chars,
                    },
                    "persona": selected_persona,
                    "system_prompt": sys_prompt,
                }
                result = api("post", "/config", json=new_config)
                if result.get("status") == "ok":
                    st.success("Settings saved!")
                else:
                    st.error(f"Failed: {result}")

            st.divider()
            st.subheader("Suggested Ollama Models (≤4B params)")
            st.markdown("""
            | Model | Size | Best For |
            |-------|------|----------|
            | `qwen2.5:3b` | 3B | General purpose, great quality |
            | `phi3.5:3.8b` | 3.8B | Reasoning, code |
            | `gemma2:2b` | 2B | Fast, lightweight |
            | `llama3.2:3b` | 3B | Balanced performance |
            | `smollm2:1.7b` | 1.7B | Ultra-fast, basic tasks |

            Install with: `ollama pull qwen2.5:3b`
            """)
