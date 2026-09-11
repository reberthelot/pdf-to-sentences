(function() {
  const previewN = 200;
  const output = document.getElementById("output");
  const latencyEl = document.getElementById("latency");
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
  let currentEstSeconds = null;

  function fmtMs(x) {
    if (x === null || x === undefined) return "—";
    const num = Number(x);
    if (isNaN(num)) return "—";
    if (num > 100) {
      return `${(num / 1000).toFixed(2)} s`;
    }
    return `${Math.round(num * 10) / 10} ms`;
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
      if (latencyEl) latencyEl.textContent = "—";
      if (output) output.textContent = "Processing PDF and extracting sentences...";
      if (engineUsedEl) engineUsedEl.textContent = "detecting...";
      if (returnedCountEl) returnedCountEl.textContent = "—";

      try {
        const r = await fetch("api/extract", {
          method: "POST",
          body: fd
        });

        const data = await r.json();

        if (!r.ok) {
          const msg = data && data.error ? data.error : ("HTTP " + r.status);
          output.textContent = "Request failed.\n\n" + msg;
          if (latencyEl) latencyEl.textContent = "error";
          return;
        }

        const sentences = data.sentences || [];
        lastAllSentences = sentences;

        if (latencyEl) latencyEl.textContent = fmtMs(data.latency_ms);
        if (returnedCountEl) returnedCountEl.textContent = String(sentences.length);

        if (engineUsedEl) {
          if (data.method === "fast_path") {
            engineUsedEl.innerHTML = '<span class="ok">Fast-Path Extraction</span>';
          } else if (data.method === "rapidocr_onnx" || data.method === "paddleocr_onnx" || data.method === "paddleocr") {
            engineUsedEl.innerHTML = '<span class="bad">RapidOCR ONNX</span>';
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
        if (output) output.textContent = "Unexpected frontend error.\n\n" + String(e);
        if (latencyEl) latencyEl.textContent = "failed";
      } finally {
        if (extractBtn) extractBtn.disabled = false;
        refreshMetrics();
      }
    });
  }

  // --- Asynchronous Jobs Management ---
  const extractAsyncBtn = document.getElementById("extractAsyncBtn");
  const jobsContainer = document.getElementById("jobsContainer");
  const jobsCountBadge = document.getElementById("jobsCountBadge");
  const trackedJobs = new Map(); // job_id -> { id, filename, postTime, endTime, status, progress, currentPage, totalPages, message, result, error, expanded }
  let jobsPollInterval = null;

  function updateJobsListUI() {
    if (!jobsContainer) return;

    if (trackedJobs.size === 0) {
      jobsContainer.innerHTML = `
        <div class="small" id="noJobsMsg" style="padding: 12px; color: var(--muted); text-align: center; background: white; border: 1px dashed var(--border); border-radius: 8px;">
          No asynchronous jobs submitted yet. Click <b>"Extract sentences (asynchrone)"</b> to start one.
        </div>`;
      if (jobsCountBadge) jobsCountBadge.textContent = "0 job(s)";
      return;
    }

    if (jobsCountBadge) jobsCountBadge.textContent = `${trackedJobs.size} job(s)`;

    // Save scroll position of expanded pre blocks if any
    const scrollMap = new Map();
    const preEls = jobsContainer.querySelectorAll("pre[data-job-pre]");
    preEls.forEach(pre => {
      scrollMap.set(pre.getAttribute("data-job-pre"), pre.scrollTop);
    });

    const now = Date.now();
    const sortedJobs = Array.from(trackedJobs.values()).sort((a, b) => b.postTime - a.postTime);

    let html = `
      <table class="jobs-table">
        <thead>
          <tr>
            <th>Document</th>
            <th>Job ID</th>
            <th>Duration</th>
            <th>Pages</th>
            <th>Status</th>
            <th>Progress</th>
            <th>Engine</th>
            <th style="text-align: right;">Action</th>
          </tr>
        </thead>
        <tbody>
    `;

    for (const job of sortedJobs) {
      // Freezes timer when job reaches completed or failed
      const endTimestamp = job.endTime || (job.status === "completed" || job.status === "failed" ? (job.updatedAt ? job.updatedAt * 1000 : now) : now);
      const elapsedSec = Math.max(0, Math.floor((endTimestamp - job.postTime) / 1000));
      const elapsedStr = elapsedSec < 60 ? `${elapsedSec}s` : `${Math.floor(elapsedSec / 60)}m ${elapsedSec % 60}s`;
      const pct = Math.round((job.progress || 0) * 100);

      let statusPillClass = "status-pending";
      if (job.status === "processing") statusPillClass = "status-processing";
      else if (job.status === "completed") statusPillClass = "status-completed";
      else if (job.status === "failed") statusPillClass = "status-failed";

      const engineName = (job.result && job.result.method) ? job.result.method : "—";
      const isExpanded = !!job.expanded;
      const toggleLabel = isExpanded ? "Hide ▲" : "View ▼";

      html += `
        <tr class="job-row" id="row-job-${job.id}">
          <td class="job-filename-cell" title="${job.filename || 'Document'}">${job.filename || 'Document'}</td>
          <td><code style="font-size: 11px;">${job.id.slice(0, 8)}...</code></td>
          <td style="color: var(--muted);">${elapsedStr}</td>
          <td>${job.currentPage || 0} / ${job.totalPages || '—'}</td>
          <td><span class="job-status-pill ${statusPillClass}">${job.status}</span></td>
          <td style="min-width: 100px;">
            <div style="display:flex; align-items:center; gap:6px;">
              <div style="flex:1; height:6px; background:#eee; border-radius:3px; overflow:hidden;">
                <div style="height:100%; width:${pct}%; background:${job.status === 'completed' ? 'var(--ok)' : (job.status === 'failed' ? 'var(--bad)' : '#b25e00')};"></div>
              </div>
              <span style="font-size:11px; font-weight:600;">${pct}%</span>
            </div>
          </td>
          <td style="font-size: 11.5px; color: var(--muted);">${engineName}</td>
          <td style="text-align: right;">
            <button class="job-toggle-btn" onclick="window.toggleJobDrawer('${job.id}')">${toggleLabel}</button>
          </td>
        </tr>
      `;

      if (isExpanded) {
        html += `
          <tr id="drawer-job-${job.id}">
            <td colspan="8" class="job-drawer-cell">
              <div class="job-drawer-content">
                <div class="job-drawer-meta">
                  <span><b>Full Job ID:</b> <code style="font-size: 11px;">${job.id}</code></span>
                  <span><b>Message:</b> ${job.message || '—'}</span>
                  ${job.result && job.result.processing_time_ms ? `<span><b>Processing Latency:</b> ${fmtMs(job.result.processing_time_ms)}</span>` : ''}
                </div>
                ${job.error ? `<div style="color:var(--bad); margin-top:6px;"><b>Error:</b> ${job.error}</div>` : ''}
                ${job.status === "completed" && job.result && job.result.sentences ? `
                  <div style="margin-top: 8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                      <b>Extracted Sentences (${job.result.sentences.length}) :</b>
                      <button class="btn secondary" style="padding: 4px 10px; font-size:11px;" onclick="window.downloadJobSentences('${job.id}', event)">Download all (.txt)</button>
                    </div>
                    <pre data-job-pre="${job.id}" style="max-height: 180px; margin: 0;">${job.result.sentences.map((s, i) => String(i + 1).padStart(3, ' ') + '. ' + s).join('\n')}</pre>
                  </div>
                ` : ''}
              </div>
            </td>
          </tr>
        `;
      }
    }

    html += `
        </tbody>
      </table>
    `;

    jobsContainer.innerHTML = html;

    // Restore scroll positions of open pre blocks
    scrollMap.forEach((top, jobId) => {
      const el = jobsContainer.querySelector(`pre[data-job-pre="${jobId}"]`);
      if (el) el.scrollTop = top;
    });
  }

  // Global handlers for drawer collapse/expand and download
  window.toggleJobDrawer = function(jobId) {
    const job = trackedJobs.get(jobId);
    if (job) {
      job.expanded = !job.expanded;
      updateJobsListUI();
    }
  };

  window.downloadJobSentences = function(jobId, ev) {
    if (ev) ev.stopPropagation();
    const job = trackedJobs.get(jobId);
    if (!job || !job.result || !job.result.sentences) return;
    const url = makeTxtDownload(job.result.sentences);
    const a = document.createElement("a");
    a.href = url;
    a.download = (job.filename.replace(/\.pdf$/i, "") || "job") + "_sentences.txt";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
  };

  async function pollTrackedJobs() {
    let hasActiveJobs = false;
    let stateChanged = false;

    for (const [jobId, job] of trackedJobs.entries()) {
      if (job.status === "completed" || job.status === "failed") {
        continue;
      }
      hasActiveJobs = true;

      try {
        const resp = await fetch(`api/jobs/${jobId}`);
        if (!resp.ok) continue;
        const data = await resp.json();

        job.status = data.status;
        job.progress = data.progress;
        job.currentPage = data.current_page;
        job.totalPages = data.total_pages;
        job.message = data.message;
        job.result = data.result;
        job.error = data.error;
        if (data.updated_at) job.updatedAt = data.updated_at;

        if (data.status === "completed" || data.status === "failed") {
          if (!job.endTime) job.endTime = Date.now();
        }
        stateChanged = true;
      } catch (e) {
        // Ignore network polling error
      }
    }

    if (stateChanged) {
      updateJobsListUI();
      refreshMetrics();
    }
    return hasActiveJobs;
  }

  function startJobsPolling() {
    if (jobsPollInterval) return;
    jobsPollInterval = setInterval(async () => {
      await pollTrackedJobs();
    }, 1000);
  }

  if (extractAsyncBtn) {
    extractAsyncBtn.addEventListener("click", async () => {
      const fileInput = document.getElementById("pdfFile");
      if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please choose a PDF file first.");
        return;
      }

      const pdf = fileInput.files[0];
      const fd = new FormData();
      fd.append("pdf_file", pdf, pdf.name);

      extractAsyncBtn.disabled = true;

      try {
        const resp = await fetch("api/jobs/submit", {
          method: "POST",
          body: fd,
        });

        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          alert("Job submission failed: " + (errData.error || resp.statusText));
          return;
        }

        const data = await resp.json();
        const jobId = data.job_id;

        // Register job in tracking map
        trackedJobs.set(jobId, {
          id: jobId,
          filename: pdf.name,
          postTime: Date.now(),
          status: data.status || "pending",
          progress: 0.0,
          currentPage: 0,
          totalPages: data.total_pages || 0,
          message: "Queued for processing...",
          result: null,
          error: null,
          expanded: false,
        });

        updateJobsListUI();
        startJobsPolling();
      } catch (err) {
        alert("Could not submit asynchronous job: " + String(err));
      } finally {
        extractAsyncBtn.disabled = false;
      }
    });
  }

  // Periodic UI refresh for elapsed time counters (only if running jobs exist)
  setInterval(() => {
    let hasRunning = false;
    for (const job of trackedJobs.values()) {
      if (job.status !== "completed" && job.status !== "failed") {
        hasRunning = true;
        break;
      }
    }
    if (hasRunning) {
      updateJobsListUI();
    }
  }, 3000);

  refreshMetrics();
  setInterval(refreshMetrics, 3000);
})();