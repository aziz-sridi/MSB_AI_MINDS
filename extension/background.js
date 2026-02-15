const SCRIPT_FILES = [
  "constants.js", "utils.js", "metadata.js", "ui.js", "selection.js", "contentScript.js"
];
const CSS_FILES = ["styles.css"];

function isInjectableUrl(url) {
  if (!url) return false;
  const blocked = ["chrome://", "edge://", "about:", "chrome-extension://", "devtools://"];
  return !blocked.some((s) => url.startsWith(s));
}

async function sendMessage(tabId, action) {
  await chrome.tabs.sendMessage(tabId, { action });
}

async function ensureContentScript(tab) {
  if (!tab || !tab.id || !isInjectableUrl(tab.url)) return false;
  try {
    await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: CSS_FILES });
  } catch (_) {}
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: SCRIPT_FILES });
  } catch (e) {
    console.error("[AI MINDS] Script injection failed:", e);
    return false;
  }
  return true;
}

chrome.commands.onCommand.addListener(async (command) => {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  if (!tab || !tab.id) return;

  let action = null;
  if (command === "toggle-selection-mode") action = "EXTRACTOR_TOGGLE";
  if (command === "capture-page") action = "EXTRACTOR_CAPTURE";
  if (!action) return;

  try {
    await sendMessage(tab.id, action);
  } catch (_) {
    const ok = await ensureContentScript(tab);
    if (ok) {
      // Wait for scripts to initialize before retrying
      await new Promise((r) => setTimeout(r, 150));
      try { await sendMessage(tab.id, action); } catch (e) {
        console.error("[AI MINDS] Could not send message after inject:", e);
      }
    }
  }
});

// Listen for messages from popup
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "QUERY_BACKEND") {
    fetch("http://localhost:5000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: msg.question })
    })
      .then((r) => r.json())
      .then((data) => sendResponse(data))
      .catch((err) => sendResponse({ error: err.message }));
    return true; // async sendResponse
  }
});
