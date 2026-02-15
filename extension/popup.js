const API = "http://localhost:5000";
const chat = document.getElementById("chat");
const input = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statsText = document.getElementById("statsText");

// ── Health check ────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const resp = await fetch(`${API}/health`, { method: "GET" });
    if (resp.ok) {
      statusDot.classList.add("online");
      statusDot.classList.remove("offline");
      statusText.textContent = "Backend connected";
      // Fetch stats
      const stats = await fetch(`${API}/stats`).then((r) => r.json());
      statsText.textContent = `${stats.total_records || 0} records`;
    } else throw new Error();
  } catch {
    statusDot.classList.add("offline");
    statusDot.classList.remove("online");
    statusText.textContent = "Backend offline";
    statsText.textContent = "";
  }
}

checkHealth();

// ── Chat ─────────────────────────────────────────────────────────────────────

function addMessage(text, role, extra) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  if (extra) {
    const sub = document.createElement("div");
    sub.className = extra.className || "ref";
    sub.textContent = extra.text;
    div.appendChild(sub);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function askQuestion() {
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";

  const thinking = addMessage("Thinking...", "system");

  try {
    const resp = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const data = await resp.json();
    thinking.remove();

    if (data.error) {
      addMessage(data.error, "system");
      return;
    }

    const confClass = (data.confidence || 0) < 0.35 ? "low" : "";
    addMessage(data.answer || "No answer found.", "assistant", {
      className: `confidence ${confClass}`,
      text: `Confidence: ${((data.confidence || 0) * 100).toFixed(0)}%` +
        (data.uncertainty ? ` — ${data.uncertainty}` : "")
    });

    // Show references
    if (data.references && data.references.length > 0) {
      const sources = data.references.map((r) => r.source).filter(Boolean).slice(0, 3);
      if (sources.length) {
        addMessage("", "system", {
          className: "ref",
          text: `Sources: ${sources.join(", ")}`
        });
      }
    }
  } catch (err) {
    thinking.remove();
    addMessage("Error: Could not reach backend.", "system");
  }
}

sendBtn.addEventListener("click", askQuestion);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

// ── Action buttons ──────────────────────────────────────────────────────────

const SCRIPT_FILES = [
  "constants.js", "utils.js", "metadata.js", "ui.js", "selection.js", "contentScript.js"
];
const CSS_FILES = ["styles.css"];

function isInjectableUrl(url) {
  if (!url) return false;
  const blocked = ["chrome://", "edge://", "about:", "chrome-extension://", "devtools://"];
  return !blocked.some((s) => url.startsWith(s));
}

async function ensureContentScript(tab) {
  if (!tab || !tab.id || !isInjectableUrl(tab.url)) return false;
  try {
    await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: CSS_FILES });
  } catch (_) {}
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: SCRIPT_FILES });
  } catch (_) {}
  return true;
}

async function sendActionToTab(action) {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  if (!tab || !tab.id) return;

  try {
    await chrome.tabs.sendMessage(tab.id, { action });
  } catch (_) {
    // Content script not loaded yet — inject and retry
    const ok = await ensureContentScript(tab);
    if (ok) {
      // Small delay to let scripts initialize
      await new Promise((r) => setTimeout(r, 150));
      try {
        await chrome.tabs.sendMessage(tab.id, { action });
      } catch (e) {
        console.error("[AI MINDS popup] Could not send message after inject:", e);
      }
    }
  }
  window.close();
}

document.getElementById("btnSelect").addEventListener("click", () => {
  sendActionToTab("EXTRACTOR_TOGGLE");
});

document.getElementById("btnCapture").addEventListener("click", () => {
  sendActionToTab("EXTRACTOR_CAPTURE");
});

document.getElementById("btnDigest").addEventListener("click", async () => {
  const thinking = addMessage("Generating digest...", "system");
  try {
    const resp = await fetch(`${API}/digest`);
    const data = await resp.json();
    thinking.remove();
    addMessage(data.digest || "No recent activity.", "assistant");
  } catch {
    thinking.remove();
    addMessage("Could not generate digest.", "system");
  }
});
