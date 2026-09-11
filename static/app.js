(function() {
  const previewN = 200;
  const output = document.getElementById("output");
  const latencyEl = document.getElementById("latency");
  const timerDisplay = document.getElementById("timerDisplay");
  const returnedCountEl = document.getElementById("returnedCount");
  const engineUsedEl = document.getElementById("engineUsed");
  const extractBtn = document.getElementById("extractBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  const filterBox = document.getElementById("filterBox");
  const previewLimitEl = document.getElementById("previewLimit");
  if (previewLimitEl) previewLimitEl.textContent = String(previewN);
  const fileInput = document.getElementById("pdfFile");
  const fileInfoRow = document.getElementById("fileInfoRow");
  const pageCountEl = document.getElementById("pageCount");
  const docTypeEl = document.getElementById("docType");
  const estTimeEl = document.getElementById("estTime");

  let lastAllSentences = null;
  let timerInterval = null;
  let currentEstSeconds = null;

  function fmtMs(x) {
    if (x === null || x === undefined) return "—";
    return Math.round(x * 10) / 10;
    if (x < 1000) return `${Math.round(x)} ms`;
    return `${(x / 1000).toFixed(2)} s`;
  }

  function renderPreview(sentences) {
    const filt = (filterBox.value || "").trim().toLowerCase();
    let s = sentences;
    if (filt.length > 0) {
      s = sentences.filter(x => x.toLowerCase().includes(filt));
    }

    const shown = s.slice(0, previewN);
    const lines = shown.map((x, i) => String(i + 1).padStart(4, " ") + ". " + x);
    let header = "";
    if (filt.length > 0) {
      header = `Filter: "${filt}"\n`;
    }
    header += `Showing ${shown.length} / ${s.length} (filtered) / ${sentences.length} (total)\n\n`;
    output.textContent = header + lines.join("\n");
  }

  function makeTxtDownload(sentences) {
    const text = sentences.map(s => s + "\n").join("");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    return URL.createObjectURL(blob);
  }

  // Pre-inspect file when selected to display page count and estimated time
  if (fileInput) {
    fileInput.addEventListener("change", async () => {
      if (!fileInput.files || fileInput.files.length === 0) {
        if (fileInfoRow) fileInfoRow.style.display = "none";
        currentEstSeconds = null;
        return;
      }

      const file = fileInput.files[0];
      const fd = new FormData();
      fd.append("pdf_file", file, file.name);

      if (fileInfoRow) fileInfoRow.style.display = "flex";
      if (pageCountEl) pageCountEl.textContent = "analyzing...";
      if (docTypeEl) docTypeEl.textContent = "analyzing...";
      if (estTimeEl) estTimeEl.textContent = "calculating...";

      try {
        const resp = await fetch("api/inspect", { method: "POST", body: fd });
        if (resp.ok) {
          const data = await resp.json();
          currentEstSeconds = data.estimated_seconds;
          if (pageCountEl) pageCountEl.textContent = String(data.page_count);
          if (docTypeEl) {
            docTypeEl.innerHTML = data.has_text_layer
              ? '<span class="ok">Digital (Fast-Path)</span>'
              : '<span class="bad">Scanned Image (OCR)</span>';
          }
          if (estTimeEl) {
            estTimeEl.textContent = `~${data.estimated_seconds} s`;
          }
        }
      } catch (e) {
        if (estTimeEl) estTimeEl.textContent = "unknown";
      }
    });
  }

  async function refreshMetrics() {
    try {
      const r = await fetch("api/metrics");
      if (!r.ok) return;
      const m = await r.json();
      const mTotal = document.getElementById("m_total");
      const mOk = document.getElementById("m_ok");
      const mFail = document.getElementById("m_fail");
      const mAvg = document.getElementById("m_avg");
      const mLast = document.getElementById("m_last");
      const mErr = document.getElementById("m_err");

      if (mTotal) mTotal.textContent = m.total_requests ?? "—";
      if (mOk) mOk.textContent = m.success_requests ?? "—";
      if (mFail) mFail.textContent = m.failed_requests ?? "—";
      if (mAvg) mAvg.textContent = fmtMs(m.avg_latency_ms);
      if (mLast) mLast.textContent = fmtMs(m.last_latency_ms);
      if (mErr) mErr.textContent = m.last_error ? m.last_error.replace(/\s+/g, " ").slice(0, 140) : "—";
    } catch (e) {
      // If metrics fail, fail silently
      // Ignore network errors on metrics poll
    }
  }

  async function runSelftest() {
    const btn = document.getElementById("selftestBtn");
    const status = document.getElementById("selftestStatus");
    const out = document.getElementById("selftestOut");

    if (btn) btn.disabled = true;
    if (status) status.textContent = "running...";
    if (out) out.textContent = "Running self-test...";
    if (out) out.textContent = "Running self-test on reference datasets...";

    try {
      const r = await fetch("api/selftest", { method: "POST" });
      const data = await r.json();

      let lines = [];
      lines.push(`Service URL: ${data.service_url}`);
      lines.push(`Passed: ${data.passed} / ${data.total}`);
      lines.push("");

      for (const it of (data.results || [])) {
        const badge = it.ok ? "OK" : "FAIL";
        lines.push(`[${badge}] ${it.filename}`);
        lines.push(`[${badge}] ${it.filename} (${it.method || 'default'})`);
        if (it.latency_ms !== null && it.latency_ms !== undefined) {
          lines.push(`  latency_ms: ${fmtMs(it.latency_ms)}`);
          lines.push(`  latency: ${fmtMs(it.latency_ms)}`);
        }
        if (it.num_returned_sentences !== null && it.num_returned_sentences !== undefined) {
          lines.push(`  returned_sentences: ${it.num_returned_sentences}`);
        }
        if (!it.ok) {
          if (it.error) {
            lines.push("  error:");
            lines.push("  " + String(it.error).split("\n").join("\n  "));
          }
          if (it.missing_sentences && it.missing_sentences.length > 0) {
            lines.push("  missing_sentences:");
            for (const s of it.missing_sentences) {
              lines.push("   - " + s);
            }
          }
        }
        lines.push("");
      }

      if (out) out.textContent = lines.join("\n");
      if (status) {
        status.innerHTML = (data.passed === data.total)
          ? '<span class="ok">passed</span>'
          : '<span class="bad">failed</span>';
      }
    } catch (e) {
      if (out) out.textContent = "Self-test failed to run.\n\n" + String(e);
      if (status) status.innerHTML = '<span class="bad">error</span>';
    } finally {
      if (btn) btn.disabled = false;
      refreshMetrics();
    }
  }

  const selftestBtn = document.getElementById("selftestBtn");
  if (selftestBtn) {
    selftestBtn.addEventListener("click", runSelftest);
  }

  if (filterBox) {
    filterBox.addEventListener("input", () => {
      if (lastAllSentences) renderPreview(lastAllSentences);
    });
  }

  const uploadForm = document.getElementById("uploadForm");
  if (uploadForm) {
    uploadForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();

      const fileInput = document.getElementById("pdfFile");
      if (!fileInput.files || fileInput.files.length === 0) {
        output.textContent = "Please choose a PDF file first.";
        return;
      }

      const pdf = fileInput.files[0];
      const fd = new FormData();
      fd.append("pdf_file", pdf, pdf.name);

      if (extractBtn) extractBtn.disabled = true;
      if (downloadBtn) downloadBtn.disabled = true;
      lastAllSentences = null;
      if (output) output.textContent = "Uploading PDF and extracting sentences via PaddleOCR...";
      if (latencyEl) latencyEl.textContent = "—";
      if (output) output.textContent = "Processing PDF and extracting sentences...";
      if (engineUsedEl) engineUsedEl.textContent = "detecting...";
      if (returnedCountEl) returnedCountEl.textContent = "—";

      // Live stopwatch timer
      const startTime = performance.now();
      if (timerInterval) clearInterval(timerInterval);
      timerInterval = setInterval(() => {
        const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(1);
        const estStr = currentEstSeconds ? ` / ~${currentEstSeconds}s` : "";
        if (timerDisplay) timerDisplay.textContent = `⏱️ ${elapsedSec}s${estStr}`;
      }, 100);

      try {
        const r = await fetch("api/extract", {
          method: "POST",
          body: fd
        });

        const data = await r.json();
        clearInterval(timerInterval);

        if (!r.ok) {
          const msg = data && data.error ? data.error : ("HTTP " + r.status);
          output.textContent = "Request failed.\n\n" + msg;
          if (timerDisplay) timerDisplay.textContent = "error";
          return;
        }

        const sentences = data.sentences || [];
        lastAllSentences = sentences;

        if (latencyEl) latencyEl.textContent = fmtMs(data.latency_ms);
        const totalTimeStr = fmtMs(data.latency_ms);
        if (timerDisplay) timerDisplay.textContent = `✅ ${totalTimeStr}`;
        if (returnedCountEl) returnedCountEl.textContent = String(sentences.length);

        if (engineUsedEl) {
          if (data.method === "fast_path") {
            engineUsedEl.innerHTML = '<span class="ok">⚡ Fast-Path</span>';
          } else if (data.method === "rapidocr_onnx" || data.method === "paddleocr_onnx" || data.method === "paddleocr") {
            engineUsedEl.innerHTML = '<span class="bad">👁️ RapidOCR ONNX</span>';
          } else {
            engineUsedEl.textContent = data.method || "default";
          }
        }

        renderPreview(sentences);

        if (downloadBtn) {
          downloadBtn.disabled = false;
          downloadBtn.onclick = () => {
            const url = makeTxtDownload(sentences);
            const a = document.createElement("a");
            a.href = url;
            a.download = (pdf.name.replace(/\.pdf$/i, "") || "sentences") + "_sentences.txt";
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 3000);
          };
        }
      } catch (e) {
        clearInterval(timerInterval);
        if (output) output.textContent = "Unexpected frontend error.\n\n" + String(e);
        if (timerDisplay) timerDisplay.textContent = "failed";
      } finally {
        if (extractBtn) extractBtn.disabled = false;
        refreshMetrics();
      }
    });
  }

  refreshMetrics();
  setInterval(refreshMetrics, 3000);
})();