(() => {
  window.ExtractorConfig = {
    // Backend endpoints
    endpoints: {
      extract: "http://localhost:5000/api/extract",
      autoCapture: "http://localhost:5000/api/auto-capture",
      index: "http://localhost:5000/index",
      query: "http://localhost:5000/query",
      health: "http://localhost:5000/health"
    },
    hotkey: { key: "f", ctrl: true, shift: true },
    messageAction: {
      toggle: "EXTRACTOR_TOGGLE",
      capture: "EXTRACTOR_CAPTURE",
      query: "EXTRACTOR_QUERY"
    },
    classes: {
      activeBody: "extractor-active-mode",
      hover: "extractor-hover",
      selected: "extractor-selected"
    },
    ids: {
      root: "extractor-ui-root",
      note: "extractor-note-input",
      count: "extractor-count",
      status: "extractor-status",
      save: "extractor-save",
      cancel: "extractor-cancel",
      capture: "extractor-capture-page"
    }
  };
})();
