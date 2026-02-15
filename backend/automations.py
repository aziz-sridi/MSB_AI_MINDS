"""
GigaMind — Automation module.

Background services that run autonomously:
  - Clipboard monitor: auto-ingest any copied text (>20 chars)
  - Browser history scanner: ingest page titles + URLs from Chrome/Edge
  - Screenshot watcher: watch screenshot folder for new images
  - Digest scheduler: periodic summaries of recent activity
  - Smart reminders: surface pending action items
  - Download folder watcher: auto-ingest new downloads
"""

import json
import os
import platform
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class AutomationManager:
    """Manages background automations that run alongside the engine."""

    def __init__(self, engine):
        self.engine = engine
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._clipboard_history: Set[int] = set()  # hashes of ingested clips
        self._browser_last_ts: float = 0.0
        self._seen_screenshots: Set[str] = set()

    # ── Clipboard Monitor ──────────────────────────────────────────────────

    def start_clipboard_monitor(self, min_length: int = 20, interval: float = 2.0):
        """Monitor clipboard for new text and auto-ingest."""
        if self._is_running("clipboard"):
            return

        stop = threading.Event()
        self._stop_events["clipboard"] = stop

        def _monitor():
            try:
                import pyperclip
            except ImportError:
                print("[clipboard] pyperclip not installed — run: pip install pyperclip")
                return

            last_text = ""
            try:
                last_text = pyperclip.paste() or ""
            except Exception:
                pass

            print("[clipboard] Monitor started")
            while not stop.is_set():
                try:
                    current = pyperclip.paste()
                    if (current
                            and current != last_text
                            and len(current.strip()) >= min_length
                            and hash(current) not in self._clipboard_history):

                        last_text = current
                        self._clipboard_history.add(hash(current))

                        # Skip if it looks like a password or random noise
                        text = current.strip()
                        if len(text) < 500 and not any(c == ' ' for c in text):
                            # Single long word — likely a token/hash, skip
                            pass
                        else:
                            result = self.engine.ingest_text(
                                text=text,
                                source="clipboard",
                                modality="clipboard",
                                metadata={"source_type": "clipboard", "timestamp": time.time()},
                            )
                            if result.get("status") == "ok":
                                print(f"[clipboard] Ingested ({result.get('chunks_added', 0)} chunks)")

                except Exception as e:
                    if "could not find" not in str(e).lower():
                        pass  # Silent on clipboard access errors
                stop.wait(interval)

        t = threading.Thread(target=_monitor, daemon=True)
        self._threads["clipboard"] = t
        t.start()

    # ── Browser History Scanner ────────────────────────────────────────────

    def start_browser_history_monitor(self, interval: float = 120.0):
        """Periodically scan Chrome/Edge history for new pages visited."""
        if self._is_running("browser_history"):
            return

        stop = threading.Event()
        self._stop_events["browser_history"] = stop

        # Initialize timestamp to now (don't ingest old history on first run)
        self._browser_last_ts = time.time()

        def _scan():
            print("[browser_history] Monitor started")
            while not stop.is_set():
                try:
                    entries = self._read_browser_history(since_ts=self._browser_last_ts)
                    if entries:
                        self._browser_last_ts = time.time()
                        for entry in entries:
                            title = entry.get("title", "")
                            url = entry.get("url", "")
                            if not title or len(title) < 5:
                                continue
                            # Skip common non-content pages
                            skip_patterns = [
                                "new tab", "extensions", "settings", "chrome://",
                                "edge://", "about:blank", "localhost",
                            ]
                            if any(p in (title + url).lower() for p in skip_patterns):
                                continue

                            text = f"Visited: {title}\nURL: {url}"
                            self.engine.ingest_text(
                                text=text,
                                source=url or "browser_history",
                                modality="browser_history",
                                metadata={
                                    "source_type": "browser_history",
                                    "url": url,
                                    "pageTitle": title,
                                    "visit_time": entry.get("visit_time", ""),
                                    "timestamp": time.time(),
                                },
                            )
                        if entries:
                            print(f"[browser_history] Ingested {len(entries)} new pages")
                except Exception as e:
                    print(f"[browser_history] Error: {e}")
                stop.wait(interval)

        t = threading.Thread(target=_scan, daemon=True)
        self._threads["browser_history"] = t
        t.start()

    def _read_browser_history(self, since_ts: float, limit: int = 50) -> List[Dict]:
        """Read recent history from Chrome or Edge (whichever is found)."""
        entries = []
        home = Path.home()

        # Chrome epoch starts at 1601-01-01, offset from Unix epoch
        chrome_epoch_offset = 11644473600
        chrome_ts = int((since_ts + chrome_epoch_offset) * 1_000_000)

        # Try Chrome and Edge history DBs
        possible_paths = []
        if platform.system() == "Windows":
            local = home / "AppData" / "Local"
            possible_paths = [
                local / "Google" / "Chrome" / "User Data" / "Default" / "History",
                local / "Microsoft" / "Edge" / "User Data" / "Default" / "History",
            ]
        elif platform.system() == "Darwin":
            possible_paths = [
                home / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "History",
            ]
        else:
            possible_paths = [
                home / ".config" / "google-chrome" / "Default" / "History",
                home / ".config" / "chromium" / "Default" / "History",
            ]

        for db_path in possible_paths:
            if not db_path.exists():
                continue
            try:
                # Copy DB to avoid lock issues (browser locks it)
                tmp_path = self.engine.data_dir / "browser_history_tmp.db"
                shutil.copy2(str(db_path), str(tmp_path))

                conn = sqlite3.connect(str(tmp_path))
                cursor = conn.execute(
                    "SELECT url, title, last_visit_time FROM urls "
                    "WHERE last_visit_time > ? "
                    "ORDER BY last_visit_time DESC LIMIT ?",
                    (chrome_ts, limit),
                )
                for row in cursor:
                    entries.append({
                        "url": row[0],
                        "title": row[1] or "",
                        "visit_time": row[2],
                    })
                conn.close()

                try:
                    tmp_path.unlink()
                except Exception:
                    pass

                if entries:
                    break  # Found history in this browser
            except Exception as e:
                print(f"[browser_history] Could not read {db_path.name}: {e}")
                continue

        return entries

    # ── Screenshot Watcher ─────────────────────────────────────────────────

    def start_screenshot_watcher(self, interval: float = 10.0):
        """Watch common screenshot folders for new images."""
        if self._is_running("screenshots"):
            return

        stop = threading.Event()
        self._stop_events["screenshots"] = stop

        screenshot_dirs = self._get_screenshot_dirs()
        if not screenshot_dirs:
            print("[screenshots] No screenshot directories found")
            return

        # Snapshot existing files so we don't ingest old ones
        for d in screenshot_dirs:
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        self._seen_screenshots.add(str(f))

        def _watch():
            print(f"[screenshots] Watching {len(screenshot_dirs)} dirs")
            while not stop.is_set():
                try:
                    for d in screenshot_dirs:
                        if not d.exists():
                            continue
                        for f in d.iterdir():
                            if not f.is_file():
                                continue
                            fstr = str(f)
                            if fstr in self._seen_screenshots:
                                continue
                            ext = f.suffix.lower()
                            if ext not in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                                continue
                            # Wait a moment for the file to finish writing
                            try:
                                size1 = f.stat().st_size
                                time.sleep(0.5)
                                size2 = f.stat().st_size
                                if size1 != size2:
                                    continue  # Still being written
                            except Exception:
                                continue

                            self._seen_screenshots.add(fstr)
                            result = self.engine.ingest_file(fstr)
                            if result.get("status") == "ok":
                                print(f"[screenshots] Ingested: {f.name}")
                except Exception as e:
                    print(f"[screenshots] Error: {e}")
                stop.wait(interval)

        t = threading.Thread(target=_watch, daemon=True)
        self._threads["screenshots"] = t
        t.start()

    def _get_screenshot_dirs(self) -> List[Path]:
        """Find common screenshot directories on this system."""
        home = Path.home()
        candidates = [
            home / "Pictures" / "Screenshots",
            home / "OneDrive" / "Pictures" / "Screenshots",
            home / "Screenshots",
        ]
        if platform.system() == "Windows":
            candidates.append(home / "Videos" / "Captures")  # Win+G captures
        return [d for d in candidates if d.exists() and d.is_dir()]

    # ── Download Folder Watcher ────────────────────────────────────────────

    def start_downloads_watcher(self, interval: float = 15.0):
        """Watch Downloads folder for new documents/images."""
        if self._is_running("downloads"):
            return

        stop = threading.Event()
        self._stop_events["downloads"] = stop

        from file_parsers import ALL_SUPPORTED
        dl_dir = Path.home() / "Downloads"
        if not dl_dir.exists():
            print("[downloads] Downloads folder not found")
            return

        # Snapshot existing files
        seen: Set[str] = set()
        for f in dl_dir.iterdir():
            if f.is_file():
                seen.add(str(f))

        def _watch():
            print(f"[downloads] Watching: {dl_dir}")
            while not stop.is_set():
                try:
                    for f in dl_dir.iterdir():
                        if not f.is_file():
                            continue
                        fstr = str(f)
                        if fstr in seen:
                            continue
                        ext = f.suffix.lower()
                        if ext not in ALL_SUPPORTED:
                            continue
                        # Wait for file to finish downloading
                        if f.suffix.lower() in {".crdownload", ".partial", ".tmp"}:
                            continue
                        try:
                            size1 = f.stat().st_size
                            time.sleep(1)
                            size2 = f.stat().st_size
                            if size1 != size2 or size2 == 0:
                                continue
                        except Exception:
                            continue

                        seen.add(fstr)
                        result = self.engine.ingest_file(fstr)
                        if result.get("status") == "ok":
                            print(f"[downloads] Ingested: {f.name} ({result.get('chunks_added', 0)} chunks)")
                except Exception as e:
                    print(f"[downloads] Error: {e}")
                stop.wait(interval)

        t = threading.Thread(target=_watch, daemon=True)
        self._threads["downloads"] = t
        t.start()

    # ── Digest Scheduler ───────────────────────────────────────────────────

    def start_digest_scheduler(self, interval_hours: float = 24):
        """Schedule periodic AI-generated summaries of recent activity."""
        if self._is_running("digest"):
            return

        stop = threading.Event()
        self._stop_events["digest"] = stop

        def _scheduler():
            print(f"[digest] Scheduler started (every {interval_hours}h)")
            while not stop.is_set():
                stop.wait(interval_hours * 3600)
                if stop.is_set():
                    break
                try:
                    digest = self.engine.daily_digest()
                    if digest and "No recent" not in digest:
                        self.engine.ingest_text(
                            text=f"Daily Digest:\n{digest}",
                            source="automation/daily_digest",
                            modality="digest",
                            metadata={"source_type": "automation", "automation": "daily_digest"},
                        )
                        print("[digest] Generated and saved daily digest")
                except Exception as e:
                    print(f"[digest] Error: {e}")

        t = threading.Thread(target=_scheduler, daemon=True)
        self._threads["digest"] = t
        t.start()

    # ── Smart Reminders ────────────────────────────────────────────────────

    def start_smart_reminders(self, check_interval_minutes: float = 60):
        """Periodically check for pending action items."""
        if self._is_running("reminders"):
            return

        stop = threading.Event()
        self._stop_events["reminders"] = stop

        def _check():
            print(f"[reminders] Started (every {check_interval_minutes}min)")
            while not stop.is_set():
                stop.wait(check_interval_minutes * 60)
                if stop.is_set():
                    break
                try:
                    actions = self.engine.get_actions(done=False)
                    if actions:
                        print(f"[reminders] {len(actions)} pending action items")
                except Exception as e:
                    print(f"[reminders] Error: {e}")

        t = threading.Thread(target=_check, daemon=True)
        self._threads["reminders"] = t
        t.start()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start_all(self, config: Optional[Dict] = None):
        """Start all enabled automations based on config."""
        cfg = config or self.engine.config.get("automations", {})

        if cfg.get("clipboard_monitor", False):
            self.start_clipboard_monitor()

        if cfg.get("browser_history", False):
            interval = float(cfg.get("browser_history_interval", 120))
            self.start_browser_history_monitor(interval=interval)

        if cfg.get("screenshot_watcher", False):
            self.start_screenshot_watcher()

        if cfg.get("downloads_watcher", False):
            self.start_downloads_watcher()

        if cfg.get("digest_scheduler", True):
            hours = float(cfg.get("digest_interval_hours", 24))
            self.start_digest_scheduler(interval_hours=hours)

        if cfg.get("reminders", False):
            mins = float(cfg.get("reminder_interval_minutes", 60))
            self.start_smart_reminders(check_interval_minutes=mins)

    def stop(self, name: str):
        """Stop a specific automation."""
        event = self._stop_events.get(name)
        if event:
            event.set()
        thread = self._threads.get(name)
        if thread and thread.is_alive():
            thread.join(timeout=3)
        self._threads.pop(name, None)
        self._stop_events.pop(name, None)

    def stop_all(self):
        """Stop all running automations."""
        for event in self._stop_events.values():
            event.set()
        for thread in self._threads.values():
            if thread.is_alive():
                thread.join(timeout=2)
        self._threads.clear()
        self._stop_events.clear()
        print("[automation] All automations stopped")

    def restart(self, name: str):
        """Restart a specific automation."""
        self.stop(name)
        time.sleep(0.2)
        starters = {
            "clipboard": self.start_clipboard_monitor,
            "browser_history": self.start_browser_history_monitor,
            "screenshots": self.start_screenshot_watcher,
            "downloads": self.start_downloads_watcher,
            "digest": self.start_digest_scheduler,
            "reminders": self.start_smart_reminders,
        }
        starter = starters.get(name)
        if starter:
            starter()

    def status(self) -> Dict[str, Any]:
        """Get status of all automations."""
        all_names = [
            "clipboard", "browser_history", "screenshots",
            "downloads", "digest", "reminders",
        ]
        result = {}
        for name in all_names:
            thread = self._threads.get(name)
            result[name] = {
                "running": thread is not None and thread.is_alive(),
            }
        return result

    def _is_running(self, name: str) -> bool:
        t = self._threads.get(name)
        return t is not None and t.is_alive()
