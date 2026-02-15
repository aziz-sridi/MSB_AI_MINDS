"""
AI MINDS — Streamlit Dashboard Frontend.

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
import time
from pathlib import Path

import requests
import streamlit as st

API = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="AI MINDS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1218; }
    .block-container { padding-top: 1rem; }
    h1 { color: #e8eaf0; }
    .stMetric label { color: #8899b0 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #6ee7ff !important; }
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


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 AI MINDS")
    st.caption("Personal Knowledge Assistant")

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
        "💬 Chat",
        "📊 Dashboard",
        "🤖 Automations",
        "📋 Actions",
        "📁 Manual Ingest",
        "⚙️ Settings",
    ], label_visibility="collapsed")


# ── Chat Page ────────────────────────────────────────────────────────────────

if page == "💬 Chat":
    st.header("💬 Chat with Your Memory")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("confidence") is not None:
                conf = msg["confidence"]
                color = "green" if conf > 0.5 else "orange" if conf > 0.3 else "red"
                st.caption(f"Confidence: :{color}[{conf:.0%}]")
            if msg.get("references"):
                with st.expander("📎 Sources", expanded=False):
                    for ref in msg["references"]:
                        src = ref.get("source", "unknown")
                        score = ref.get("score", 0)
                        cat = ref.get("category", "")
                        st.caption(f"**{src}** (score: {score:.2f}, category: {cat})")
                        snippet = ref.get("snippet", "")
                        if snippet:
                            st.text(snippet[:200])

    # Chat input
    if prompt := st.chat_input("Ask anything about your stored memories..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching memory & thinking..."):
                result = api("post", "/query", json={"question": prompt})

            if "error" in result:
                st.error(result["error"])
                st.session_state.messages.append({
                    "role": "assistant", "content": f"Error: {result['error']}"
                })
            else:
                answer = result.get("answer", "No answer found.")
                confidence = result.get("confidence", 0)
                refs = result.get("references", [])
                uncertainty = result.get("uncertainty")

                st.markdown(answer)

                color = "green" if confidence > 0.5 else "orange" if confidence > 0.3 else "red"
                st.caption(f"Confidence: :{color}[{confidence:.0%}]")
                if uncertainty:
                    st.caption(f"⚠️ {uncertainty}")

                if refs:
                    with st.expander("📎 Sources", expanded=False):
                        for ref in refs:
                            st.caption(
                                f"**{ref.get('source', 'unknown')}** "
                                f"(score: {ref.get('score', 0):.2f}, "
                                f"category: {ref.get('category', '')})"
                            )

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "confidence": confidence, "references": refs,
                })


# ── Dashboard Page ───────────────────────────────────────────────────────────

elif page == "📊 Dashboard":
    st.header("📊 Knowledge Dashboard")

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
        st.subheader("📝 Daily Digest")
        if st.button("Generate Digest"):
            with st.spinner("Generating..."):
                digest = api("get", "/digest")
            st.markdown(digest.get("digest", "No data."))


# ── Automations Page ─────────────────────────────────────────────────────────

elif page == "🤖 Automations":
    st.header("🤖 Automations — Hands-Free Data Ingestion")
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
                    "label": "📋 Clipboard Monitor",
                    "desc": "Auto-ingests anything you copy (Ctrl+C) that's longer than 20 characters. "
                            "Skips passwords and random tokens.",
                    "config_key": "clipboard_monitor",
                },
                "browser_history": {
                    "label": "🌐 Browser History",
                    "desc": "Periodically scans Chrome/Edge history and ingests page titles + URLs you visit. "
                            "Skips internal browser pages.",
                    "config_key": "browser_history",
                },
                "screenshots": {
                    "label": "📸 Screenshot Watcher",
                    "desc": "Watches your Screenshots folder and auto-ingests new screenshots using CLIP embeddings.",
                    "config_key": "screenshot_watcher",
                },
                "downloads": {
                    "label": "📥 Downloads Watcher",
                    "desc": "Watches your Downloads folder and auto-ingests new documents (PDF, DOCX, TXT) and images.",
                    "config_key": "downloads_watcher",
                },
                "digest": {
                    "label": "📝 Daily Digest",
                    "desc": "Generates a periodic AI summary of your recent knowledge entries.",
                    "config_key": "digest_scheduler",
                },
                "reminders": {
                    "label": "🔔 Smart Reminders",
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
            st.subheader("📂 Watched Folders")
            st.caption("The directory watcher auto-ingests new/modified files from these folders every 30 seconds.")
            cfg = api("get", "/config")
            if "error" not in cfg:
                wd = cfg.get("watch_dirs", [])
                if wd:
                    for i, d in enumerate(wd):
                        exists = os.path.isdir(d)
                        col_d, col_r = st.columns([6, 1])
                        col_d.markdown(f"{'📂' if exists else '❌'} `{d}`")
                        if col_r.button("🗑️", key=f"rm_auto_dir_{i}"):
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
                    if st.button("➕ Add Folder", key="auto_add_folder"):
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
                    if st.button("🔄 Scan Now", key="auto_scan_now"):
                        with st.spinner("Scanning all watched folders..."):
                            result = api("post", "/ingest/scan")
                        if result.get("status") == "ok":
                            st.success(f"Found {result.get('processed', 0)} new files")
                        else:
                            st.warning(f"Result: {result}")

                # Quick-add common folders
                with st.expander("📌 Quick-add common folders"):
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
                            if st.button(f"📂 {label} — {folder}", key=f"quick_auto_{label}"):
                                wd.append(str(folder))
                                cfg["watch_dirs"] = wd
                                api("post", "/config", json=cfg)
                                st.rerun()


# ── Actions Page ─────────────────────────────────────────────────────────────

elif page == "📋 Actions":
    st.header("📋 Action Items & Todos")

    if not online:
        st.warning("Backend offline")
    else:
        tab1, tab2 = st.tabs(["Pending", "Completed"])

        with tab1:
            actions = api("get", "/actions?done=false")
            if isinstance(actions, list) and actions:
                for a in actions:
                    col1, col2 = st.columns([5, 1])
                    col1.markdown(f"- {a.get('text', '')}")
                    col1.caption(f"Source: {a.get('source', '')} | {a.get('created_at', '')}")
                    if col2.button("✅", key=f"done_{a['id']}"):
                        api("post", f"/actions/{a['id']}/done")
                        st.rerun()
            else:
                st.info("No pending actions.")

        with tab2:
            done = api("get", "/actions?done=true")
            if isinstance(done, list) and done:
                for a in done:
                    st.markdown(f"- ~~{a.get('text', '')}~~")
                    st.caption(f"Source: {a.get('source', '')}")
            else:
                st.info("No completed actions.")


# ── Ingest Page ──────────────────────────────────────────────────────────────

elif page == "📁 Manual Ingest":
    st.header("📁 Manual Ingest")
    st.caption("For when you want to manually add specific content. For automated ingestion, use 🤖 Automations.")

    if not online:
        st.warning("Backend offline")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Input", "📤 Upload Files", "📁 File Path", "🔄 Scan Directories"])

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
            st.caption("Supports: PDF, DOCX, TXT, MD, CSV, JSON, images (PNG, JPG, etc.)")
            uploaded = st.file_uploader(
                "Choose files", accept_multiple_files=True,
                type=["txt", "md", "pdf", "docx", "csv", "json", "log",
                      "png", "jpg", "jpeg", "gif", "bmp", "webp"],
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
                        st.caption(f"✅ {f.name}: {chunks} chunks ({result.get('category', '')})")
                    else:
                        st.caption(f"⚠️ {f.name}: {result.get('reason', result.get('error', 'failed'))}")
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
                        st.caption(f"{'📂' if exists else '❌'} {d}")
                else:
                    st.info("No watch dirs configured — go to ⚙️ Settings to add folders.")
            if st.button("🔄 Trigger Scan Now"):
                with st.spinner("Scanning directories..."):
                    result = api("post", "/ingest/scan")
                if result.get("status") == "ok":
                    st.success(f"Processed {result.get('processed', 0)} files "
                               f"({result.get('errors', 0)} errors)")
                else:
                    st.warning(f"Result: {result}")


# ── Settings Page ────────────────────────────────────────────────────────────

elif page == "⚙️ Settings":
    st.header("⚙️ Settings")

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
                    icon = "📂" if exists else "❌"
                    col_d.markdown(f"{icon} `{d}`")
                    if col_r.button("🗑️", key=f"rm_dir_{i}"):
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
            if st.button("➕ Add Folder"):
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
            with st.expander("📌 Quick-add common folders"):
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

            st.caption("Supported files: `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, `.json`, `.log`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`")

            st.subheader("System Prompt")
            sys_prompt = st.text_area("System prompt for LLM",
                                      value=config.get("system_prompt", ""), height=120)

            if st.button("💾 Save Settings"):
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
