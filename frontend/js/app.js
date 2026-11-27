/**
 * pasarvision — Lightweight Zero-Lag Frontend Controller (`app.js`)
 */

(function () {
  const state = {
    activeTab: "live",
    inputMode: "image",
    activeSource: null,
    batchSources: [],
    samples: { images: [], videos: [] },
    classes: [],
    classColorMap: {},
    enabledIds: new Set(),

    enhancement: {
      method: "none",
      clahe_clip_limit: 2.5,
      clahe_tile_grid: 8,
      cs_low_percentile: 2.0,
      cs_high_percentile: 98.0,
      brightness: 18,
      contrast_alpha: 1.15,
      gamma: 1.25,
    },

    ws: null,
    prevBlobUrl: null,
    videoPaused: false,
    isSeekingSlider: false,
    lastAnnotatedB64: null,
    settingsTimer: null,
    lastChartSyncMs: 0,
  };

  let charts = null;
  let roiEditor = null;

  function showToast(msg) {
    const box = document.getElementById("toast-container");
    if (!box) return;
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => el.remove(), 2600);
  }

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
      ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
    return data;
  }

  // ---------------------------------------------------------------------------
  // Status Sync
  // ---------------------------------------------------------------------------
  function applyModelStatus(m) {
    if (!m) return;
    const dot = document.getElementById("model-status-dot");
    dot.className = m.is_production ? "status-dot prod" : "status-dot";
    document.getElementById("active-model-short").textContent = (m.active_model || "").replace(
      /\.(pt|onnx)$/i,
      ""
    );

    // Fallback Warning Banner
    const warnBanner = document.getElementById("fallback-warning-banner");
    const warnText = document.getElementById("fallback-warning-text");
    if (warnBanner && warnText) {
      if (m.is_fallback && m.fallback_warning) {
        warnText.textContent = m.fallback_warning;
        warnBanner.classList.remove("hidden");
      } else {
        warnBanner.classList.add("hidden");
      }
    }

    const sel = document.getElementById("model-selector");
    sel.innerHTML = "";
    (m.available_models || []).forEach((item) => {
      const o = document.createElement("option");
      o.value = item.name;
      o.textContent = item.is_fallback ? `${item.name} (demo)` : item.name;
      if (item.is_active) o.selected = true;
      sel.appendChild(o);
    });

    document.getElementById("slider-conf").value = m.conf_threshold;
    document.getElementById("val-conf").textContent = Number(m.conf_threshold).toFixed(2);
    document.getElementById("slider-iou").value = m.iou_threshold;
    document.getElementById("val-iou").textContent = Number(m.iou_threshold).toFixed(2);

    state.classes = m.classes || [];
    state.classColorMap = {};
    state.classes.forEach((c) => {
      state.classColorMap[c.name] = c.color_hex;
    });
    state.enabledIds = new Set(
      state.classes.filter((c) => c.enabled).map((c) => Number(c.id))
    );
    renderClassChips();
  }

  function applyStatus(payload) {
    if (!payload) return;
    if (payload.model) applyModelStatus(payload.model);

    if (payload.enhancement) {
      state.enhancement = { ...state.enhancement, ...payload.enhancement };
      document.getElementById("enhancement-method").value = state.enhancement.method;
      document.getElementById("slider-clahe-clip").value = state.enhancement.clahe_clip_limit;
      document.getElementById("val-clahe-clip").textContent = Number(
        state.enhancement.clahe_clip_limit
      ).toFixed(1);
      document.getElementById("select-clahe-grid").value = String(
        state.enhancement.clahe_tile_grid
      );

      document.getElementById("slider-cs-low").value = state.enhancement.cs_low_percentile;
      document.getElementById("val-cs-low").textContent = `${Number(
        state.enhancement.cs_low_percentile
      ).toFixed(1)}%`;
      document.getElementById("slider-cs-high").value = state.enhancement.cs_high_percentile;
      document.getElementById("val-cs-high").textContent = `${Number(
        state.enhancement.cs_high_percentile
      ).toFixed(1)}%`;

      const b = Number(state.enhancement.brightness);
      document.getElementById("slider-brightness").value = b;
      document.getElementById("val-brightness").textContent = b >= 0 ? `+${b}` : `${b}`;
      document.getElementById("slider-contrast").value = state.enhancement.contrast_alpha;
      document.getElementById("val-contrast").textContent = Number(
        state.enhancement.contrast_alpha
      ).toFixed(2);
      document.getElementById("slider-gamma").value = state.enhancement.gamma;
      document.getElementById("val-gamma").textContent = Number(state.enhancement.gamma).toFixed(2);

      syncEnhanceSections();
    }

    if (payload.render_options) {
      document.getElementById("chk-show-boxes").checked = Boolean(payload.render_options.show_boxes);
      document.getElementById("chk-show-ids").checked = Boolean(
        payload.render_options.show_track_ids
      );
      document.getElementById("chk-show-trails").checked = Boolean(
        payload.render_options.show_trails
      );
      document.getElementById("chk-show-roi").checked = Boolean(payload.render_options.show_roi);
    }

    if (payload.roi_polygon && roiEditor) {
      roiEditor.setPoints(payload.roi_polygon);
    }
    if (payload.samples) {
      state.samples = payload.samples;
      populateSamples();
    }
    if (payload.reports) {
      renderArchive(payload.reports);
    }
  }

  function updateClassFilterCounter() {
    const sumEl = document.getElementById("class-filter-summary");
    if (sumEl) {
      sumEl.textContent = `Class Filter (${state.enabledIds.size}/${state.classes.length})`;
    }
  }

  function renderClassChips() {
    const el = document.getElementById("class-filter-list");
    updateClassFilterCounter();
    el.innerHTML = state.classes
      .map((c) => {
        const on = state.enabledIds.has(Number(c.id));
        return `<span data-cid="${c.id}" class="cls-chip ${on ? "on" : ""}">
          <i class="cls-dot" style="background:${c.color_hex}"></i>${esc(c.name)}
        </span>`;
      })
      .join("");

    el.querySelectorAll(".cls-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const id = Number(chip.dataset.cid);
        if (state.enabledIds.has(id)) state.enabledIds.delete(id);
        else state.enabledIds.add(id);
        chip.classList.toggle("on", state.enabledIds.has(id));
        updateClassFilterCounter();
        scheduleSettings();
      });
    });
  }

  function populateSamples() {
    const sel = document.getElementById("sample-selector");
    sel.innerHTML = "";
    const list =
      state.inputMode === "video" ? state.samples.videos || [] : state.samples.images || [];
    if (!list.length) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = "No samples";
      sel.appendChild(o);
      return;
    }
    list.forEach((item) => {
      const o = document.createElement("option");
      o.value = item.path;
      o.textContent = item.name;
      if (state.activeSource === item.path) o.selected = true;
      sel.appendChild(o);
    });
  }

  function syncEnhanceSections() {
    const m = document.getElementById("enhancement-method").value;
    document.getElementById("params-clahe").classList.toggle("hidden", m !== "clahe");
    document.getElementById("params-cs").classList.toggle("hidden", m !== "cs");
    document.getElementById("params-bg").classList.toggle("hidden", m !== "brightness_gamma");

    const quickClahe = document.getElementById("btn-quick-clahe");
    if (quickClahe) quickClahe.classList.toggle("hidden", m !== "none");
  }

  // ---------------------------------------------------------------------------
  // Navigation & Modes
  // ---------------------------------------------------------------------------
  function switchTab(tab) {
    state.activeTab = tab;
    document.querySelectorAll(".nav-tab").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tab);
    });

    const titles = { live: "My Dashboard", compare: "Compare", reports: "Reports" };
    document.getElementById("page-heading").textContent = titles[tab] || "My Dashboard";

    document.getElementById("tab-panel-live").classList.toggle("hidden", tab !== "live");
    document.getElementById("tab-panel-compare").classList.toggle("hidden", tab !== "compare");
    document.getElementById("tab-panel-reports").classList.toggle("hidden", tab !== "reports");

    if (tab === "live" && roiEditor) setTimeout(() => roiEditor.syncSize(), 30);
    if (tab === "compare") runCompare();
    if (tab === "reports") refreshReports();
  }

  function switchMode(mode) {
    if (mode !== "video") stopVideo();
    state.inputMode = mode;

    document.getElementById("btn-mode-image").classList.toggle("active", mode === "image");
    document.getElementById("btn-mode-video").classList.toggle("active", mode === "video");
    document.getElementById("btn-mode-batch").classList.toggle("active", mode === "batch");

    document.getElementById("image-controls-bar").classList.toggle("hidden", mode === "video");
    document.getElementById("video-controls-bar").classList.toggle("hidden", mode !== "video");
    document.getElementById("batch-filmstrip-panel").classList.toggle("hidden", mode !== "batch");

    populateSamples();
    const selVal = document.getElementById("sample-selector").value;

    if (mode === "image" && selVal) {
      runImage(selVal);
    } else if (mode === "video" && selVal) {
      startVideo(selVal);
    } else if (mode === "batch") {
      const imgs = (state.samples.images || []).map((i) => i.path);
      if (imgs.length) {
        state.batchSources = imgs;
        runBatch(imgs);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Throttled Zero-Lag Telemetry Updates
  // ---------------------------------------------------------------------------
  function updateTelemetry(t, forceSyncCharts = false) {
    if (!t) return;
    document.getElementById("kpi-total-detected").textContent = t.total_detections ?? 0;
    const uniq = t.unique_tracked_total ?? t.total_detections ?? 0;
    document.getElementById("kpi-unique-sub").textContent = `${uniq} tracked total`;

    document.getElementById("kpi-inside-roi").textContent = t.inside_roi_now ?? 0;
    document.getElementById("kpi-roi-entered").textContent =
      t.region_unique_entered ?? t.inside_roi_now ?? 0;

    const confPct = Math.round(Number(t.avg_confidence || 0) * 100);
    document.getElementById("kpi-avg-conf").textContent = `${confPct}%`;

    const cDeltaEl = document.getElementById("kpi-conf-delta");
    const cd = Number(t.conf_delta || 0) * 100;
    if (Math.abs(cd) >= 0.1 && state.enhancement.method !== "none") {
      cDeltaEl.textContent = `${cd > 0 ? "+" : ""}${cd.toFixed(1)}%`;
      cDeltaEl.className = `badge-pill num ${cd > 0 ? "" : "neg"}`;
    } else {
      cDeltaEl.textContent = "Live";
      cDeltaEl.className = "badge-pill neutral num";
    }

    // Throttle heavier canvas/DOM updates to ~2.4 Hz during live video streaming
    const now = performance.now();
    if (!forceSyncCharts && now - state.lastChartSyncMs < 420) return;
    state.lastChartSyncMs = now;

    if (charts) {
      const mergedColors = { ...state.classColorMap, ...(t.class_colors || {}) };
      charts.updateLiveDistribution(t.current_counts_by_class || {}, mergedColors);
    }
    if (t.region_counter_log) renderZoneEntries(t.region_counter_log);
  }

  function renderZoneEntries(rows) {
    const el = document.getElementById("roi-entries-list");
    if (!el) return;
    if (!rows || !rows.length) {
      el.innerHTML = `<div class="row-item"><span class="row-sub">No zone entries recorded yet</span></div>`;
      return;
    }
    el.innerHTML = rows
      .slice(-30)
      .reverse()
      .map(
        (r) => `
        <div class="row-item">
          <div class="row-left">
            <div class="row-icon num">${esc(r.track_id)}</div>
            <div>
              <div class="row-title">${esc(r.class_name)}</div>
              <div class="row-sub num">${Number(r.first_entered_sec || 0).toFixed(2)}s</div>
            </div>
          </div>
          <span class="badge-pill num">${Math.round(Number(r.confidence || 0) * 100)}%</span>
        </div>`
      )
      .join("");
  }

  function renderArchive(items) {
    const el = document.getElementById("reports-archive-list");
    if (!el) return;
    if (!items || !items.length) {
      el.innerHTML = `<div class="row-item"><span class="row-sub">No exported files yet</span></div>`;
      return;
    }
    el.innerHTML = items
      .slice(0, 20)
      .map(
        (it) => `
        <div class="row-item">
          <div class="row-left">
            <div class="row-icon" style="color:${it.kind === "excel" ? "#b8f69c" : "#c7b5ff"}">${
          it.kind === "excel" ? "XLS" : "MP4"
        }</div>
            <div style="min-width:0">
              <div class="row-title">${esc(it.filename)}</div>
              <div class="row-sub num">${it.size_kb} KB • ${it.created_at}</div>
            </div>
          </div>
          <a href="${it.download_url}" download="${esc(
          it.filename
        )}" class="pill-btn" style="text-decoration:none;padding:5px 12px">Download</a>
        </div>`
      )
      .join("");
  }

  // ---------------------------------------------------------------------------
  // Image, Batch & Video Inference
  // ---------------------------------------------------------------------------
  function setLoader(on) {
    document.getElementById("viewport-loader").classList.toggle("hidden", !on);
  }

  function showFrame(b64) {
    const img = document.getElementById("main-viewport-img");
    if (img.classList.contains("hidden")) {
      img.classList.remove("hidden");
      img.onload = () => {
        if (roiEditor) roiEditor.syncSize();
        img.onload = null;
      };
    }
    img.src = b64;
    state.lastAnnotatedB64 = b64;
  }

  async function runImage(path) {
    if (!path) return;
    state.activeSource = path;
    setLoader(true);
    try {
      const res = await api("/api/detect/image", {
        method: "POST",
        body: JSON.stringify({ source_path: path }),
      });
      showFrame(res.annotated_image);
      document.getElementById("hud-fps").textContent = `${res.width}×${res.height}`;
      document.getElementById("hud-latency").textContent = `${Math.round(res.latency_ms)} ms`;
      updateTelemetry(res.telemetry, true);
      if (charts && res.frame_detail_log) charts.updateTimeline(res.frame_detail_log);
    } catch (e) {
      showToast(e.message);
    } finally {
      setLoader(false);
    }
  }

  async function runBatch(paths) {
    if (!paths || !paths.length) return;
    setLoader(true);
    try {
      const res = await api("/api/detect/batch", {
        method: "POST",
        body: JSON.stringify({ source_paths: paths }),
      });
      if (res.after_image) showFrame(res.after_image);
      const sm = res.summary || {};
      document.getElementById("hud-fps").textContent = `${(res.batch_items || []).length} files`;
      document.getElementById("hud-latency").textContent = `${Math.round(res.latency_ms)} ms`;

      const cmap = {};
      (res.per_class_table || []).forEach((r) => (cmap[r.class_name] = r.after_count));
      updateTelemetry(
        {
          total_detections: sm.after_total,
          unique_tracked_total: sm.after_total,
          inside_roi_now: sm.after_inside_roi,
          region_unique_entered: sm.after_inside_roi,
          avg_confidence: sm.after_avg_conf,
          conf_delta: sm.delta_conf,
          current_counts_by_class: cmap,
          region_counter_log: res.region_counter_log || [],
        },
        true
      );
      if (charts) charts.updateTimeline(res.frame_detail_log || []);

      const strip = document.getElementById("batch-filmstrip-panel");
      strip.innerHTML = (res.batch_items || [])
        .map(
          (it, idx) => `
          <button data-idx="${idx}" class="batch-thumb">
            <img src="${it.after_image}" alt="" />
            <div style="text-align:left">
              <div style="font-size:11px;font-weight:600">${esc(it.filename.slice(0, 18))}</div>
              <div class="num" style="font-size:10px;color:#b8f69c">${it.after_total} obj</div>
            </div>
          </button>`
        )
        .join("");

      strip.querySelectorAll(".batch-thumb").forEach((b) => {
        b.addEventListener("click", () => {
          const item = res.batch_items[Number(b.dataset.idx)];
          if (item && item.after_image) showFrame(item.after_image);
        });
      });
    } catch (e) {
      showToast(e.message);
    } finally {
      setLoader(false);
    }
  }

  function stopVideo() {
    if (state.ws) {
      try {
        state.ws.send(JSON.stringify({ action: "stop" }));
        state.ws.close();
      } catch (_) {}
      state.ws = null;
    }
    if (state.prevBlobUrl) {
      URL.revokeObjectURL(state.prevBlobUrl);
      state.prevBlobUrl = null;
    }
    state.videoPaused = false;
    document.getElementById("label-vid-playpause").textContent = "Pause";
  }

  function startVideo(path) {
    if (!path) return;
    stopVideo();
    state.activeSource = path;
    setLoader(true);

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/video?source=${encodeURIComponent(path)}&binary=1`);
    ws.binaryType = "arraybuffer";
    state.ws = ws;

    ws.onmessage = (evt) => {
      // Fast Zero-GC Binary JPEG Frame Path (45–60 FPS)
      if (evt.data instanceof ArrayBuffer) {
        setLoader(false);
        const blob = new Blob([evt.data], { type: "image/jpeg" });
        const nextUrl = URL.createObjectURL(blob);
        showFrame(nextUrl);
        if (state.prevBlobUrl) {
          URL.revokeObjectURL(state.prevBlobUrl);
        }
        state.prevBlobUrl = nextUrl;
        return;
      }

      const msg = JSON.parse(evt.data);
      if (msg.type === "error") {
        setLoader(false);
        showToast(msg.message);
        return;
      }
      if (msg.type === "ended") {
        state.videoPaused = true;
        document.getElementById("label-vid-playpause").textContent = "Play";
        if (charts && msg.frame_detail_log) charts.updateTimeline(msg.frame_detail_log);
        return;
      }
      if (msg.type === "hud") {
        document.getElementById("hud-fps").textContent = `${Number(msg.fps).toFixed(1)} fps`;
        document.getElementById("hud-latency").textContent = `${Math.round(msg.latency_ms)} ms`;
        const sl = document.getElementById("slider-vid-progress");
        const tot = Math.max(1, msg.total_frames || 100);
        sl.max = tot;
        if (!state.isSeekingSlider) sl.value = msg.frame_idx;
        document.getElementById("vid-frame-label").textContent = `${msg.frame_idx} / ${tot}`;
        if (msg.telemetry) updateTelemetry(msg.telemetry, false);
        return;
      }
      if (msg.type === "frame") {
        setLoader(false);
        showFrame(msg.image);
        document.getElementById("hud-fps").textContent = `${Number(msg.fps).toFixed(1)} fps`;
        document.getElementById("hud-latency").textContent = `${Math.round(msg.latency_ms)} ms`;

        const sl = document.getElementById("slider-vid-progress");
        const tot = Math.max(1, msg.total_frames || 100);
        sl.max = tot;
        if (!state.isSeekingSlider) sl.value = msg.frame_idx;
        document.getElementById("vid-frame-label").textContent = `${msg.frame_idx} / ${tot}`;

        updateTelemetry(msg.telemetry, false);
      }
    };

    ws.onerror = () => setLoader(false);
    ws.onclose = () => setLoader(false);
  }

  // ---------------------------------------------------------------------------
  // Compare Tab
  // ---------------------------------------------------------------------------
  async function runCompare() {
    const loader = document.getElementById("lab-loader");
    loader.classList.remove("hidden");
    syncEnhanceSections();

    try {
      const step = Number(document.getElementById("lab-sample-step").value || 10);
      const body = { sample_every_n: step };
      if (state.inputMode === "batch" && state.batchSources.length > 1) {
        body.source_paths = state.batchSources;
      } else {
        body.source_path =
          state.activeSource ||
          (state.samples.images && state.samples.images[0] && state.samples.images[0].path);
      }

      const res = await api("/api/compare", { method: "POST", body: JSON.stringify(body) });
      const sm = res.summary || {};

      if (res.before_image && res.after_image) {
        document.getElementById("lab-swipe-before").src = res.before_image;
        document.getElementById("lab-swipe-after").src = res.after_image;
        document.getElementById("lab-split-before").src = res.before_image;
        document.getElementById("lab-split-after").src = res.after_image;
      }

      document.getElementById("lab-after-tag").textContent = `After (${
        sm.enhancement_method || "none"
      })`;

      const dc = Number(sm.delta_count || 0);
      const dp = Number(sm.delta_pct || 0);
      const dPill = document.getElementById("lab-delta-pill");
      dPill.textContent = `${dc >= 0 ? "+" : ""}${dc} (${dp >= 0 ? "+" : ""}${dp.toFixed(1)}%)`;
      dPill.className = `badge-pill num ${dc < 0 ? "neg" : dc === 0 ? "neutral" : ""}`;

      const cd = Number(sm.delta_conf || 0) * 100;
      document.getElementById("lab-conf-pill").textContent = `Conf ${
        cd >= 0 ? "+" : ""
      }${cd.toFixed(1)}%`;

      const rows = res.per_class_table || [];
      if (charts) {
        charts.updateBeforeAfter(rows);
        charts.updateTimeline(res.frame_detail_log || []);
      }

      const listEl = document.getElementById("lab-class-rows");
      if (!rows.length) {
        listEl.innerHTML = `<div class="row-item"><span class="row-sub">No objects above threshold</span></div>`;
      } else {
        listEl.innerHTML = rows
          .map((r) => {
            const pCls = r.delta_count > 0 ? "" : r.delta_count < 0 ? "neg" : "neutral";
            return `
            <div class="row-item">
              <div class="row-left">
                <div class="row-icon" style="background:${r.color_hex}22;color:${r.color_hex}">●</div>
                <div>
                  <div class="row-title">${esc(r.class_name)}</div>
                  <div class="row-sub num">${r.before_count} → ${r.after_count}  •  Conf ${Math.round(
              r.after_avg_conf * 100
            )}%</div>
                </div>
              </div>
              <span class="badge-pill num ${pCls}">${r.delta_count >= 0 ? "+" : ""}${
              r.delta_count
            }</span>
            </div>`;
          })
          .join("");
      }
    } catch (e) {
      showToast(e.message);
    } finally {
      loader.classList.add("hidden");
    }
  }

  // ---------------------------------------------------------------------------
  // Settings & Exports
  // ---------------------------------------------------------------------------
  function collectSettings() {
    const ids = Array.from(state.enabledIds);
    return {
      conf_threshold: Number(document.getElementById("slider-conf").value),
      iou_threshold: Number(document.getElementById("slider-iou").value),
      enabled_class_ids: ids,
      all_classes_enabled: ids.length === state.classes.length && state.classes.length > 0,
      enhancement: {
        method: document.getElementById("enhancement-method").value,
        clahe_clip_limit: Number(document.getElementById("slider-clahe-clip").value),
        clahe_tile_grid: Number(document.getElementById("select-clahe-grid").value),
        cs_low_percentile: Number(document.getElementById("slider-cs-low").value),
        cs_high_percentile: Number(document.getElementById("slider-cs-high").value),
        brightness: Number(document.getElementById("slider-brightness").value),
        contrast_alpha: Number(document.getElementById("slider-contrast").value),
        gamma: Number(document.getElementById("slider-gamma").value),
      },
      render_options: {
        show_boxes: document.getElementById("chk-show-boxes").checked,
        show_track_ids: document.getElementById("chk-show-ids").checked,
        show_trails: document.getElementById("chk-show-trails").checked,
        show_roi: document.getElementById("chk-show-roi").checked,
      },
    };
  }

  function scheduleSettings() {
    syncEnhanceSections();
    clearTimeout(state.settingsTimer);
    state.settingsTimer = setTimeout(async () => {
      try {
        const st = await api("/api/settings", {
          method: "POST",
          body: JSON.stringify(collectSettings()),
        });
        state.enhancement = st.enhancement;
        if (state.activeTab === "compare") runCompare();
        else if (state.inputMode === "image" && state.activeSource) runImage(state.activeSource);
        else if (state.inputMode === "batch" && state.batchSources.length)
          runBatch(state.batchSources);
      } catch (e) {
        showToast(e.message);
      }
    }, 180);
  }

  function downloadUrl(url, name) {
    const a = document.createElement("a");
    a.href = url;
    if (name) a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  async function exportExcel() {
    const lbl = document.getElementById("label-export-excel");
    lbl.textContent = "Exporting...";
    try {
      const res = await api("/api/export/excel", {
        method: "POST",
        body: JSON.stringify({ source_path: state.activeSource }),
      });
      if (res.reports) renderArchive(res.reports);
      downloadUrl(res.download_url, res.filename);
      showToast(`Saved ${res.filename}`);
    } catch (e) {
      showToast(e.message);
    } finally {
      lbl.textContent = "Export .xlsx";
    }
  }

  async function refreshReports() {
    try {
      const r = await api("/api/reports");
      renderArchive(r.reports || []);
    } catch (_) {}
  }

  // ---------------------------------------------------------------------------
  // Event Binding
  // ---------------------------------------------------------------------------
  function bindEvents() {
    document.querySelectorAll(".nav-tab").forEach((b) => {
      b.addEventListener("click", () => switchTab(b.dataset.tab));
    });

    document.getElementById("btn-mode-image").addEventListener("click", () => switchMode("image"));
    document.getElementById("btn-mode-video").addEventListener("click", () => switchMode("video"));
    document.getElementById("btn-mode-batch").addEventListener("click", () => switchMode("batch"));

    // Tuning Popover
    const pop = document.getElementById("tune-popover");
    const tuneBtn = document.getElementById("btn-toggle-tune");
    tuneBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      pop.classList.toggle("hidden");
      tuneBtn.classList.toggle("active", !pop.classList.contains("hidden"));
    });
    document.addEventListener("click", (e) => {
      if (!pop.classList.contains("hidden") && !pop.contains(e.target) && e.target !== tuneBtn) {
        pop.classList.add("hidden");
        tuneBtn.classList.remove("active");
      }
    });

    // Model Switch & Upload
    document.getElementById("model-selector").addEventListener("change", async (e) => {
      try {
        const st = await api("/api/model/switch", {
          method: "POST",
          body: JSON.stringify({ model_name: e.target.value }),
        });
        applyStatus(st);
        showToast(`Model: ${st.model.active_model}`);
        if (state.inputMode === "image" && state.activeSource) runImage(state.activeSource);
      } catch (err) {
        showToast(err.message);
      }
    });

    const mFile = document.getElementById("input-model-file");
    document.getElementById("btn-upload-model").addEventListener("click", () => mFile.click());
    const bannerUploadBtn = document.getElementById("btn-banner-upload-pt");
    if (bannerUploadBtn) bannerUploadBtn.addEventListener("click", () => mFile.click());
    mFile.addEventListener("change", async () => {
      if (!mFile.files[0]) return;
      const fd = new FormData();
      fd.append("file", mFile.files[0]);
      showToast("Loading model...");
      try {
        const res = await fetch("/api/model/upload", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Upload failed");
        applyStatus(data);
        showToast(`Loaded ${data.model.active_model}`);
        if (state.inputMode === "image" && state.activeSource) runImage(state.activeSource);
      } catch (err) {
        showToast(err.message);
      } finally {
        mFile.value = "";
      }
    });

    // Sample & Media Upload / Drag-Drop on Stage
    document.getElementById("sample-selector").addEventListener("change", (e) => {
      const v = e.target.value;
      if (!v) return;
      if (state.inputMode === "video") startVideo(v);
      else runImage(v);
    });

    const mediaInput = document.getElementById("input-media-files");
    document.getElementById("btn-upload-media").addEventListener("click", () => mediaInput.click());
    const vidInlineBtn = document.getElementById("btn-vid-upload-inline");
    if (vidInlineBtn) vidInlineBtn.addEventListener("click", () => mediaInput.click());

    const uploadMediaFiles = async (files) => {
      if (!files || !files.length) return;
      showToast("Loading media...");
      const fd = new FormData();
      Array.from(files).forEach((f) => fd.append("files", f));
      try {
        const r = await fetch("/api/upload/media", { method: "POST", body: fd });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || "Upload error");
        state.samples = d.samples;
        const up = d.uploaded || [];
        if (!up.length) return;
        if (state.activeTab !== "live") switchTab("live");
        const vid = up.find((u) => u.media_type === "video");
        if (vid) {
          state.activeSource = vid.path;
          switchMode("video");
        } else if (up.length > 1) {
          state.batchSources = up.map((u) => u.path);
          switchMode("batch");
          runBatch(state.batchSources);
        } else {
          state.activeSource = up[0].path;
          switchMode("image");
        }
      } catch (err) {
        showToast(err.message);
      } finally {
        mediaInput.value = "";
      }
    };

    mediaInput.addEventListener("change", () => uploadMediaFiles(mediaInput.files));
    const stage = document.getElementById("viewport-stage");
    stage.addEventListener("dragover", (e) => {
      e.preventDefault();
      stage.classList.add("drop-active");
    });
    stage.addEventListener("dragleave", () => stage.classList.remove("drop-active"));
    stage.addEventListener("drop", (e) => {
      e.preventDefault();
      stage.classList.remove("drop-active");
      uploadMediaFiles(e.dataTransfer.files);
    });

    // Sliders & Switches
    document.getElementById("enhancement-method").addEventListener("change", scheduleSettings);
    document.getElementById("select-clahe-grid").addEventListener("change", scheduleSettings);

    const bindRange = (id, lblId, fmt) => {
      const el = document.getElementById(id);
      const lb = document.getElementById(lblId);
      el.addEventListener("input", () => {
        lb.textContent = fmt(Number(el.value));
        scheduleSettings();
      });
    };
    bindRange("slider-conf", "val-conf", (v) => v.toFixed(2));
    bindRange("slider-iou", "val-iou", (v) => v.toFixed(2));
    bindRange("slider-clahe-clip", "val-clahe-clip", (v) => v.toFixed(1));
    bindRange("slider-cs-low", "val-cs-low", (v) => `${v.toFixed(1)}%`);
    bindRange("slider-cs-high", "val-cs-high", (v) => `${v.toFixed(1)}%`);
    bindRange("slider-brightness", "val-brightness", (v) => (v >= 0 ? `+${v}` : `${v}`));
    bindRange("slider-contrast", "val-contrast", (v) => v.toFixed(2));
    bindRange("slider-gamma", "val-gamma", (v) => v.toFixed(2));

    ["chk-show-boxes", "chk-show-ids", "chk-show-trails", "chk-show-roi"].forEach((id) =>
      document.getElementById(id).addEventListener("change", scheduleSettings)
    );

    document.getElementById("btn-classes-all").addEventListener("click", () => {
      state.classes.forEach((c) => state.enabledIds.add(Number(c.id)));
      renderClassChips();
      scheduleSettings();
    });
    document.getElementById("btn-classes-none").addEventListener("click", () => {
      state.enabledIds.clear();
      renderClassChips();
      scheduleSettings();
    });

    // ROI Zone Controls
    const syncZoneUI = (editing) => {
      document.getElementById("roi-edit-subpanel").classList.toggle("hidden", !editing);
      document.getElementById("label-toggle-roi-edit").textContent = editing ? "Done" : "Edit Zone";
    };
    document.getElementById("btn-toggle-roi-edit").addEventListener("click", async () => {
      if (!roiEditor.isEditing) {
        roiEditor.setEditing(true);
        syncZoneUI(true);
      } else {
        if (!roiEditor.finishFreshDrawIfValid()) return;
        roiEditor.setEditing(false);
        syncZoneUI(false);
        await saveRoi(roiEditor.getPoints());
      }
    });
    document
      .getElementById("btn-roi-draw-new")
      .addEventListener("click", () => roiEditor.startFreshDraw());
    document.getElementById("btn-roi-reset").addEventListener("click", async () => {
      const r = await api("/api/roi/reset", { method: "POST" });
      roiEditor.setPoints(r.points);
      if (state.inputMode === "image" && state.activeSource) runImage(state.activeSource);
    });

    // Stage Actions
    document.getElementById("btn-rerun-image").addEventListener("click", () => {
      if (state.activeSource) runImage(state.activeSource);
    });
    document
      .getElementById("btn-open-in-lab")
      .addEventListener("click", () => switchTab("compare"));
    document.getElementById("btn-save-annotated-img").addEventListener("click", () => {
      if (state.lastAnnotatedB64) downloadUrl(state.lastAnnotatedB64, `snapshot_${Date.now()}.jpg`);
    });

    // Video Controls
    document.getElementById("btn-vid-playpause").addEventListener("click", () => {
      if (!state.ws) {
        if (state.activeSource) startVideo(state.activeSource);
        return;
      }
      state.videoPaused = !state.videoPaused;
      state.ws.send(JSON.stringify({ action: state.videoPaused ? "pause" : "resume" }));
      document.getElementById("label-vid-playpause").textContent = state.videoPaused
        ? "Play"
        : "Pause";
    });
    document.getElementById("btn-vid-restart").addEventListener("click", () => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.videoPaused = false;
        document.getElementById("label-vid-playpause").textContent = "Pause";
        state.ws.send(JSON.stringify({ action: "restart" }));
      } else if (state.activeSource) startVideo(state.activeSource);
    });
    document.getElementById("btn-vid-stop").addEventListener("click", stopVideo);

    const vProg = document.getElementById("slider-vid-progress");
    vProg.addEventListener("mousedown", () => (state.isSeekingSlider = true));
    vProg.addEventListener("change", () => {
      state.isSeekingSlider = false;
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ action: "seek", frame: Number(vProg.value) }));
      }
    });

    document.getElementById("btn-export-mp4").addEventListener("click", async () => {
      if (!state.activeSource) return;
      showToast("Saving MP4...");
      try {
        const r = await api("/api/video/export", {
          method: "POST",
          body: JSON.stringify({ source_path: state.activeSource }),
        });
        if (r.reports) renderArchive(r.reports);
        downloadUrl(r.download_url, r.filename);
      } catch (e) {
        showToast(e.message);
      }
    });

    // Compare Controls & Swipe Handle
    document.getElementById("btn-lab-swipe").addEventListener("click", () => {
      document.getElementById("lab-swipe-wrapper").classList.remove("hidden");
      document.getElementById("lab-split-wrapper").classList.add("hidden");
      document.getElementById("btn-lab-swipe").classList.add("active");
      document.getElementById("btn-lab-split").classList.remove("active");
    });
    document.getElementById("btn-lab-split").addEventListener("click", () => {
      document.getElementById("lab-swipe-wrapper").classList.add("hidden");
      document.getElementById("lab-split-wrapper").classList.remove("hidden");
      document.getElementById("btn-lab-split").classList.add("active");
      document.getElementById("btn-lab-swipe").classList.remove("active");
    });
    document.getElementById("btn-run-comparison").addEventListener("click", runCompare);
    document.getElementById("btn-quick-clahe").addEventListener("click", async () => {
      document.getElementById("enhancement-method").value = "clahe";
      syncEnhanceSections();
      await api("/api/settings", { method: "POST", body: JSON.stringify(collectSettings()) });
      runCompare();
    });

    const sw = document.getElementById("lab-swipe-wrapper");
    let swDrag = false;
    const moveSwipe = (cx) => {
      const r = sw.getBoundingClientRect();
      const p = Math.max(2, Math.min(98, ((cx - r.left) / r.width) * 100));
      sw.style.setProperty("--swipe-pct", `${p.toFixed(1)}%`);
    };
    sw.addEventListener("mousedown", (e) => {
      swDrag = true;
      moveSwipe(e.clientX);
    });
    window.addEventListener("mousemove", (e) => {
      if (swDrag) moveSwipe(e.clientX);
    });
    window.addEventListener("mouseup", () => (swDrag = false));

    // Excel Exports
    ["btn-export-excel", "btn-top-export", "btn-lab-export", "btn-rep-export"].forEach((id) =>
      document.getElementById(id).addEventListener("click", exportExcel)
    );
    document.getElementById("btn-refresh-reports").addEventListener("click", refreshReports);
  }

  async function saveRoi(points) {
    try {
      await api("/api/roi", { method: "POST", body: JSON.stringify({ points }) });
      if (state.inputMode === "image" && state.activeSource) runImage(state.activeSource);
      else if (state.activeTab === "compare") runCompare();
    } catch (e) {
      showToast(e.message);
    }
  }

  async function init() {
    charts = new window.PasarChartsManager();
    charts.initAll();

    roiEditor = new window.ROIPolygonEditor({
      canvasEl: document.getElementById("roi-editor-canvas"),
      mediaEl: document.getElementById("main-viewport-img"),
      stageEl: document.getElementById("viewport-stage"),
      onSavePoints: (pts) => saveRoi(pts),
      notify: (m) => showToast(m),
    });

    bindEvents();

    try {
      const [modelSt, st] = await Promise.all([
        api("/api/model/status"),
        api("/api/status"),
      ]);
      applyStatus(st);
      applyModelStatus(modelSt);
      if (st.samples && st.samples.images && st.samples.images.length) {
        const first = st.samples.images[0].path;
        state.activeSource = first;
        document.getElementById("sample-selector").value = first;
        await runImage(first);
      }
    } catch (e) {
      showToast(e.message);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
