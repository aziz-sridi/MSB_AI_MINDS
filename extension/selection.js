(() => {
  const { endpoints, classes, hotkey } = window.ExtractorConfig;
  const { getElementText, isEditableTarget } = window.ExtractorUtils;
  const { detectContentType, detectDirection, findTimeInfo } = window.ExtractorMetadata;
  const { ensureUI, removeUI, updateCount, setStatus } = window.ExtractorUI;
  const selectedAttr = "data-extractor-selection-id";

  const state = {
    active: false,
    selections: [],
    note: "",
    hoveredElement: null
  };

  function isInsideUI(el) {
    return el && el.closest && Boolean(el.closest("#extractor-ui-root"));
  }

  function start() {
    if (state.active) return;
    state.active = true;
    document.body.classList.add(classes.activeBody);
    ensureUI(save, cancel, capturePage, (v) => { state.note = v; });
    updateCount(state.selections.length);
    setStatus("Selection mode ON — click elements to capture.");
    document.addEventListener("mousemove", onMouseMove, true);
    document.addEventListener("click", onClickSelect, true);
    document.addEventListener("keydown", onEscape, true);
  }

  function stop() {
    state.active = false;
    document.body.classList.remove(classes.activeBody);
    if (state.hoveredElement) {
      state.hoveredElement.classList.remove(classes.hover);
      state.hoveredElement = null;
    }
    document.removeEventListener("mousemove", onMouseMove, true);
    document.removeEventListener("click", onClickSelect, true);
    document.removeEventListener("keydown", onEscape, true);
    removeUI();
  }

  function toggle() {
    state.active ? save() : start();
  }

  function cancel() {
    clearSelectedMarks();
    state.selections = [];
    state.note = "";
    stop();
  }

  function onEscape(e) {
    if (e.key === "Escape") { e.preventDefault(); cancel(); }
  }

  function onMouseMove(e) {
    if (!state.active) return;
    const target = e.target;
    if (!target || isInsideUI(target)) return;
    if (state.hoveredElement && state.hoveredElement !== target) {
      state.hoveredElement.classList.remove(classes.hover);
    }
    if (target.classList) {
      target.classList.add(classes.hover);
      state.hoveredElement = target;
    }
  }

  function onClickSelect(e) {
    if (!state.active) return;
    const el = e.target;
    if (!el || isInsideUI(el)) return;
    e.preventDefault();
    e.stopPropagation();

    const ancestor = el.closest(`.${classes.selected}`);
    if (ancestor) {
      removeSelectionByElement(ancestor);
      return;
    }

    const selection = window.getSelection();
    const selText = selection ? selection.toString() : "";
    const item = buildItem(el, selText, selection);
    state.selections.push(item);
    el.classList.add(classes.selected);
    el.setAttribute(selectedAttr, item.id);
    updateCount(state.selections.length);
    setStatus("Added element.");
  }

  function removeSelectionByElement(el) {
    if (!el) return;
    const sid = el.getAttribute(selectedAttr);
    if (!sid) return;
    state.selections = state.selections.filter((i) => i.id !== sid);
    el.classList.remove(classes.selected);
    el.removeAttribute(selectedAttr);
    updateCount(state.selections.length);
    setStatus("Removed element.");
  }

  function clearSelectedMarks() {
    document.querySelectorAll(`.${classes.selected}`).forEach((el) => {
      el.classList.remove(classes.selected);
      el.removeAttribute(selectedAttr);
    });
  }

  function buildItem(element, selText, selection) {
    const rect = element.getBoundingClientRect();
    return {
      id: `item-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      pageTitle: document.title || "",
      pageUrl: location.href,
      extractedAt: new Date().toISOString(),
      contentType: detectContentType(element),
      direction: detectDirection(element),
      time: findTimeInfo(element),
      selectionText: selText || null,
      element: {
        id: element.id || null,
        text: getElementText(element),
        value: element.value || null,
        href: element.href || null,
        src: element.src || null,
        alt: element.alt || null,
        role: element.getAttribute("role") || null,
        ariaLabel: element.getAttribute("aria-label") || null
      },
      boundingRect: { top: rect.top, left: rect.left, width: rect.width, height: rect.height }
    };
  }

  // ── Save selections to backend ────────────────────────────────────────

  async function save() {
    if (state.selections.length === 0) {
      setStatus("No selections yet.");
      return;
    }

    const payload = {
      note: state.note,
      meta: {
        pageTitle: document.title || "",
        pageUrl: location.href,
        extractedAt: new Date().toISOString()
      },
      items: state.selections
    };

    try {
      setStatus("Sending to AI MINDS...");
      const resp = await fetch(endpoints.extract, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setStatus(`Saved! ${data.chunks_added || 0} chunks added.`);
      clearSelectedMarks();
      state.selections = [];
      state.note = "";
      setTimeout(() => stop(), 800);
    } catch (err) {
      setStatus("Save failed — is the backend running?");
      console.error("[AI MINDS]", err);
    }
  }

  // ── Capture full page ─────────────────────────────────────────────────

  async function capturePage() {
    setStatus("Capturing page...");
    const text = document.body.innerText || "";
    if (!text || text.length < 50) {
      setStatus("Not enough content on this page.");
      return;
    }

    try {
      const resp = await fetch(endpoints.autoCapture, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text.slice(0, 50000), // Limit size
          url: location.href,
          title: document.title || ""
        })
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setStatus(`Page captured! ${data.chunks_added || 0} chunks.`);
      setTimeout(() => stop(), 800);
    } catch (err) {
      setStatus("Capture failed — is the backend running?");
      console.error("[AI MINDS]", err);
    }
  }

  // ── Keyboard / runtime listeners ──────────────────────────────────────

  function setupKeyboardFallback() {
    document.addEventListener("keydown", (e) => {
      if (!e.ctrlKey || !e.shiftKey) return;
      if (e.key.toLowerCase() !== hotkey.key) return;
      e.preventDefault();
      toggle();
    }, true);
  }

  function setupRuntimeListener() {
    if (!chrome.runtime || !chrome.runtime.onMessage) return;
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg && msg.action === window.ExtractorConfig.messageAction.toggle) toggle();
      if (msg && msg.action === window.ExtractorConfig.messageAction.capture) capturePage();
    });
  }

  window.ExtractorSelection = { toggle, setupKeyboardFallback, setupRuntimeListener };
})();
