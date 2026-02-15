(() => {
  if (!window.ExtractorSelection) {
    console.error("[AI MINDS] ExtractorSelection module not loaded");
    return;
  }
  window.ExtractorSelection.setupKeyboardFallback();
  window.ExtractorSelection.setupRuntimeListener();
  console.log("[AI MINDS] Content script loaded");
})();
