(() => {
  const { ids } = window.ExtractorConfig;

  function makeDraggable(root, handle) {
    let dragging = false;
    let pointerId = null;
    let offsetX = 0;
    let offsetY = 0;

    function onPointerDown(e) {
      if (e.button !== 0) return;
      dragging = true;
      pointerId = e.pointerId;
      root.classList.add("dragging");
      const rect = root.getBoundingClientRect();
      offsetX = e.clientX - rect.left;
      offsetY = e.clientY - rect.top;
      root.style.right = "auto";
      root.style.bottom = "auto";
      root.style.left = `${rect.left}px`;
      root.style.top = `${rect.top}px`;
      handle.setPointerCapture(pointerId);
      e.preventDefault();
      e.stopPropagation();
    }

    function onPointerMove(e) {
      if (!dragging || e.pointerId !== pointerId) return;
      const w = root.offsetWidth;
      const h = root.offsetHeight;
      const nextLeft = Math.min(Math.max(e.clientX - offsetX, 8), window.innerWidth - w - 8);
      const nextTop = Math.min(Math.max(e.clientY - offsetY, 8), window.innerHeight - h - 8);
      root.style.left = `${nextLeft}px`;
      root.style.top = `${nextTop}px`;
    }

    function onPointerEnd(e) {
      if (!dragging || e.pointerId !== pointerId) return;
      dragging = false;
      root.classList.remove("dragging");
      handle.releasePointerCapture(pointerId);
      pointerId = null;
    }

    handle.addEventListener("pointerdown", onPointerDown);
    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onPointerEnd);
    handle.addEventListener("pointercancel", onPointerEnd);
  }

  function ensureUI(onSave, onCancel, onCapture, onNoteChange) {
    let root = document.getElementById(ids.root);
    if (root) return root;

    root = document.createElement("div");
    root.id = ids.root;
    root.innerHTML = `
      <div class="header" id="extractor-drag-handle">
        <div class="title">🧠 AI MINDS</div>
        <div id="${ids.count}" class="count">0</div>
      </div>
      <input id="${ids.note}" type="text" placeholder="Add note (optional)" />
      <div class="row actions">
        <button id="${ids.save}">Save Selected</button>
        <button id="${ids.capture}" class="accent">Capture Page</button>
        <button id="${ids.cancel}" class="secondary">Cancel</button>
      </div>
      <div id="${ids.status}" class="status"></div>
    `;

    document.documentElement.appendChild(root);

    root.querySelector(`#${ids.save}`).addEventListener("click", onSave);
    root.querySelector(`#${ids.cancel}`).addEventListener("click", onCancel);
    root.querySelector(`#${ids.capture}`).addEventListener("click", onCapture);
    root.querySelector(`#${ids.note}`).addEventListener("input", (e) => {
      onNoteChange(e.target.value || "");
    });

    const handle = root.querySelector("#extractor-drag-handle");
    makeDraggable(root, handle);
    return root;
  }

  function removeUI() {
    const root = document.getElementById(ids.root);
    if (root) root.remove();
  }

  function updateCount(count) {
    const el = document.getElementById(ids.count);
    if (el) el.textContent = String(count);
  }

  function setStatus(message) {
    const el = document.getElementById(ids.status);
    if (el) el.textContent = message || "";
  }

  window.ExtractorUI = { ensureUI, removeUI, updateCount, setStatus };
})();
