(() => {
  const { getAllAttrs } = window.ExtractorUtils;

  function detectContentType(element) {
    const tag = element.tagName;
    if (tag === "IMG") return "image";
    if (tag === "VIDEO") return "video";
    if (tag === "AUDIO") return "audio";
    if (tag === "INPUT" || tag === "TEXTAREA") return "text";
    const text = (element.innerText || "").toLowerCase();
    const attrs = getAllAttrs(element).toLowerCase();
    if (text.includes("email") || attrs.includes("email") || attrs.includes("mailto:")) return "email";
    if (attrs.includes("message") || text.includes("message")) return "message";
    return "text";
  }

  function detectDirection(element) {
    const attrs = getAllAttrs(element).toLowerCase();
    const cls = (element.className || "").toString().toLowerCase();
    const combined = `${attrs} ${cls}`;
    if (combined.includes("sent") || combined.includes("outgoing") || combined.includes("from-me")) return "sent";
    if (combined.includes("received") || combined.includes("incoming") || combined.includes("from-them")) return "received";
    return null;
  }

  function findTimeInfo(element) {
    let node = element;
    for (let i = 0; i < 6 && node; i++) {
      if (node.querySelector) {
        const timeEl = node.querySelector("time");
        if (timeEl) {
          return {
            datetime: timeEl.getAttribute("datetime") || null,
            text: (timeEl.innerText || "").trim() || null
          };
        }
      }
      node = node.parentElement;
    }
    return null;
  }

  window.ExtractorMetadata = { detectContentType, detectDirection, findTimeInfo };
})();
