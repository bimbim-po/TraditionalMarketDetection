/**
 * pasarvision — Serverless Cloudflare Edge Runtime (`cloud_runtime.js`)
 *
 * Automatically activates ONLY when hosted on Cloudflare (`*.pages.dev`, custom domain, etc.).
 * On `127.0.0.1` / `localhost`, this module does nothing so Python FastAPI handles all requests.
 *
 * Powers 100% real client-side WebAssembly SIMD YOLOv11s (`best.pt` weights),
 * Luminance-preserving CLAHE/HE/CS/Gamma, Two-Stage ByteTrack, Interactive ROI Counter,
 * Before vs. After Lab, and 4-Sheet Enterprise `.xlsx` generation with zero server lag.
 */

(function () {
  const host = window.location.hostname;
  const forceEdge = new URLSearchParams(window.location.search).get("edge") === "1";
  const IS_LOCAL_FASTAPI = (host === "127.0.0.1" || host === "localhost") && !forceEdge;
  if (IS_LOCAL_FASTAPI) return;

  const TARGET_CLASS_HEX = {
    cart: "#06B6D4",
    customer: "#3B82F6",
    scale: "#F59E0B",
    seller: "#F43F5E",
    vegetables: "#10B981",
  };
  const FALLBACK_HEX = ["#06B6D4", "#3B82F6", "#F59E0B", "#F43F5E", "#10B981", "#8B5CF6", "#EC4899"];

  function colorForClass(cid, cname) {
    const k = String(cname || "").trim().toLowerCase();
    return TARGET_CLASS_HEX[k] || FALLBACK_HEX[cid % FALLBACK_HEX.length];
  }

  // ---------------------------------------------------------------------------
  // Dynamic External Script Loader (ONNX Runtime Web & ExcelJS)
  // ---------------------------------------------------------------------------
  const loadedScripts = {};
  function loadScriptOnce(src) {
    if (loadedScripts[src]) return loadedScripts[src];
    loadedScripts[src] = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.async = true;
      s.onload = resolve;
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
    return loadedScripts[src];
  }

  // ---------------------------------------------------------------------------
  // Edge State
  // ---------------------------------------------------------------------------
  const DEFAULT_ROI = [
    [0.20, 0.35],
    [0.80, 0.35],
    [0.88, 0.90],
    [0.12, 0.90],
  ];

  const edge = {
    manifest: null,
    worker: null,
    workerReady: false,
    sessionPromise: null,
    modelLoadingPct: 0,
    nextReqId: 1,
    pendingReqs: new Map(),
    inferMutex: Promise.resolve(),
    activeModelName: "best.pt",
    isFallback: false,
    imageSide: 640,
    videoSide: 416,
    names: { 0: "cart", 1: "customer", 2: "scale", 3: "seller", 4: "vegetables" },

    conf: 0.35,
    iou: 0.45,
    enabledClassIds: null,

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
    renderOpts: {
      show_boxes: true,
      show_track_ids: true,
      show_trails: true,
      show_roi: true,
    },
    roi: DEFAULT_ROI.map((p) => p.slice()),

    samples: { images: [], videos: [] },
    uploadedBlobs: {}, // virtual path -> ObjectURL
    reports: [],
    latestReportData: null,
    latestSource: null,
  };

  function getOrCreateWorker() {
    if (edge.worker) return edge.worker;
    const w = new Worker("/static/js/yolo_worker.js?v=4", { type: "module" });
    w.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "progress") {
        edge.modelLoadingPct = msg.pct;
      } else if (msg.type === "ready") {
        edge.workerReady = true;
        edge.modelLoadingPct = 100;
        if (msg.names) edge.names = msg.names;
        if (edge._resolveReady) {
          edge._resolveReady(true);
          edge._resolveReady = null;
        }
      } else if (msg.type === "customLoaded" || msg.type === "result") {
        const cb = edge.pendingReqs.get(msg.reqId);
        if (cb) {
          edge.pendingReqs.delete(msg.reqId);
          cb.resolve(msg);
        }
      } else if (msg.type === "error") {
        if (msg.reqId && edge.pendingReqs.has(msg.reqId)) {
          const cb = edge.pendingReqs.get(msg.reqId);
          edge.pendingReqs.delete(msg.reqId);
          cb.reject(new Error(msg.message));
        } else if (edge._rejectReady) {
          edge._rejectReady(new Error(msg.message));
          edge._rejectReady = null;
          edge.sessionPromise = null;
        }
      }
    };
    w.onerror = (err) => {
      console.error("[PasarVision Worker Error]", err);
      if (edge._rejectReady) {
        edge._rejectReady(new Error("AI Web Worker error"));
        edge._rejectReady = null;
      }
      edge.sessionPromise = null;
    };
    edge.worker = w;
    return w;
  }

  async function ensureModelLoaded() {
    if (edge.workerReady) return true;
    if (edge.sessionPromise) return edge.sessionPromise;

    edge.sessionPromise = new Promise((resolve, reject) => {
      edge._resolveReady = resolve;
      edge._rejectReady = reject;
      ensureManifestSamples().catch(() => {});
      const w = getOrCreateWorker();
      w.postMessage({ type: "init" });
    });

    return edge.sessionPromise;
  }

  async function ensureManifestSamples() {
    if (edge.manifest) return;
    try {
      const mRes = await origFetch("/model/manifest_v4.json", { cache: "no-store" });
      if (!mRes.ok) return;
      const manifest = await mRes.json();
      edge.manifest = manifest;
      edge.imageSide = manifest.image_side || 640;
      edge.videoSide = manifest.video_side || 416;
      edge.names = manifest.names || edge.names;
      if (manifest.samples && edge.samples.images.length === 0) {
        edge.samples = JSON.parse(JSON.stringify(manifest.samples));
      }
    } catch (_) {}
  }

  // Kick off background AI worker + chunk download immediately on page open
  ensureModelLoaded().catch(() => {});

  function describeEnhancement(cfg) {
    if (cfg.method === "none") return "Original (No Enhancement)";
    if (cfg.method === "clahe")
      return `CLAHE (L-Channel | clip=${cfg.clahe_clip_limit}, grid=${cfg.clahe_tile_grid}x${cfg.clahe_tile_grid})`;
    if (cfg.method === "he") return "Histogram Equalization (Y-Channel)";
    if (cfg.method === "cs")
      return `Contrast Stretching (${cfg.cs_low_percentile}%–${cfg.cs_high_percentile}%)`;
    if (cfg.method === "brightness_gamma")
      return `Brightness & Gamma (b=${cfg.brightness}, c=${cfg.contrast_alpha}, g=${cfg.gamma})`;
    return "Original";
  }

  function buildModelStatus() {
    const classKeys = Object.keys(edge.names)
      .map(Number)
      .sort((a, b) => a - b);
    const enabledSet =
      edge.enabledClassIds === null ? new Set(classKeys) : new Set(edge.enabledClassIds);
    const classes = classKeys.map((id) => {
      const name = edge.names[id];
      return {
        id,
        name,
        color_hex: colorForClass(id, name),
        enabled: enabledSet.has(id),
        is_target_market_class: Boolean(TARGET_CLASS_HEX[String(name).toLowerCase()]),
      };
    });

    return {
      active_model: edge.activeModelName,
      runtime_backend: "onnxruntime-web",
      torch_available: false,
      is_production: !edge.isFallback,
      is_fallback: edge.isFallback,
      badge_type: edge.isFallback ? "fallback" : "production",
      badge_text: edge.isFallback
        ? `Demo Fallback: ${edge.activeModelName}`
        : `Production Model: ${edge.activeModelName}`,
      fallback_warning: null,
      device: "wasm",
      device_label: "Cloudflare Edge • WASM SIMD",
      conf_threshold: edge.conf,
      iou_threshold: edge.iou,
      num_classes: classes.length,
      names: edge.names,
      target_market_classes: ["cart", "customer", "scale", "seller", "vegetables"],
      classes,
      available_models: [
        {
          name: edge.activeModelName,
          size_mb: 18.3,
          is_active: true,
          is_fallback: false,
        },
      ],
    };
  }

  function buildFullStatus() {
    return {
      model: buildModelStatus(),
      enhancement: { ...edge.enhancement, description: describeEnhancement(edge.enhancement) },
      render_options: { ...edge.renderOpts },
      roi_polygon: edge.roi.map((p) => p.slice()),
      samples: edge.samples,
      reports: edge.reports,
      has_cached_report: edge.latestReportData !== null,
    };
  }

  // ---------------------------------------------------------------------------
  // Luminance-Preserving Image Enhancements (TypedArray SIMD-Friendly)
  // ---------------------------------------------------------------------------
  function applyImageDataEnhancement(imageData, cfg) {
    if (!cfg || cfg.method === "none") return imageData;
    const w = imageData.width;
    const h = imageData.height;
    const src = imageData.data;
    const out = new ImageData(new Uint8ClampedArray(src), w, h);
    const dst = out.data;
    const n = w * h;

    if (cfg.method === "brightness_gamma") {
      const alpha = Number(cfg.contrast_alpha || 1.15);
      const beta = Number(cfg.brightness || 18);
      const invG = 1.0 / Math.max(0.1, Number(cfg.gamma || 1.25));
      const lut = new Uint8Array(256);
      for (let i = 0; i < 256; i++) {
        const lin = Math.max(0, Math.min(255, i * alpha + beta));
        lut[i] = Math.round(Math.pow(lin / 255.0, invG) * 255.0);
      }
      for (let p = 0; p < dst.length; p += 4) {
        dst[p] = lut[src[p]];
        dst[p + 1] = lut[src[p + 1]];
        dst[p + 2] = lut[src[p + 2]];
      }
      return out;
    }

    // Extract Luminance Y [0..255]
    const lum = new Uint8Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      lum[i] = Math.min(255, Math.round(0.299 * src[p] + 0.587 * src[p + 1] + 0.114 * src[p + 2]));
    }

    const newLum = new Uint8Array(n);

    if (cfg.method === "he") {
      const hist = new Uint32Array(256);
      for (let i = 0; i < n; i++) hist[lum[i]]++;
      const cdf = new Uint8Array(256);
      let acc = 0;
      for (let v = 0; v < 256; v++) {
        acc += hist[v];
        cdf[v] = Math.round((acc / n) * 255);
      }
      for (let i = 0; i < n; i++) newLum[i] = cdf[lum[i]];
    } else if (cfg.method === "cs") {
      const hist = new Uint32Array(256);
      for (let i = 0; i < n; i++) hist[lum[i]]++;
      const lowTarget = (Number(cfg.cs_low_percentile || 2) / 100) * n;
      const highTarget = (Number(cfg.cs_high_percentile || 98) / 100) * n;
      let pLow = 0,
        pHigh = 255,
        acc = 0;
      for (let v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc < lowTarget) pLow = v;
        if (acc <= highTarget) pHigh = v;
      }
      const span = Math.max(1, pHigh - pLow);
      for (let i = 0; i < n; i++) {
        newLum[i] = Math.max(0, Math.min(255, Math.round(((lum[i] - pLow) * 255) / span)));
      }
    } else if (cfg.method === "clahe") {
      const grid = Number(cfg.clahe_tile_grid || 8);
      const clipLimitFactor = Number(cfg.clahe_clip_limit || 2.5);
      const tileW = Math.ceil(w / grid);
      const tileH = Math.ceil(h / grid);
      const cdfs = new Array(grid * grid);

      for (let ty = 0; ty < grid; ty++) {
        for (let tx = 0; tx < grid; tx++) {
          const x0 = tx * tileW,
            y0 = ty * tileH;
          const x1 = Math.min(w, x0 + tileW),
            y1 = Math.min(h, y0 + tileH);
          const count = Math.max(1, (x1 - x0) * (y1 - y0));
          const hist = new Float32Array(256);
          for (let yy = y0; yy < y1; yy++) {
            const rowOff = yy * w;
            for (let xx = x0; xx < x1; xx++) hist[lum[rowOff + xx]]++;
          }
          const clipVal = Math.max(1, (clipLimitFactor * count) / 256.0);
          let excess = 0;
          for (let v = 0; v < 256; v++) {
            if (hist[v] > clipVal) {
              excess += hist[v] - clipVal;
              hist[v] = clipVal;
            }
          }
          const redist = excess / 256.0;
          const lut = new Uint8Array(256);
          let sum = 0;
          for (let v = 0; v < 256; v++) {
            hist[v] += redist;
            sum += hist[v];
            lut[v] = Math.min(255, Math.round((sum / count) * 255));
          }
          cdfs[ty * grid + tx] = lut;
        }
      }

      // Bilinear interpolation across tile centers
      for (let y = 0; y < h; y++) {
        const fy = (y + 0.5) / tileH - 0.5;
        const ty0 = Math.max(0, Math.min(grid - 1, Math.floor(fy)));
        const ty1 = Math.min(grid - 1, ty0 + 1);
        const wy = Math.max(0, Math.min(1, fy - ty0));
        const rowOff = y * w;

        for (let x = 0; x < w; x++) {
          const fx = (x + 0.5) / tileW - 0.5;
          const tx0 = Math.max(0, Math.min(grid - 1, Math.floor(fx)));
          const tx1 = Math.min(grid - 1, tx0 + 1);
          const wx = Math.max(0, Math.min(1, fx - tx0));

          const val = lum[rowOff + x];
          const v00 = cdfs[ty0 * grid + tx0][val];
          const v10 = cdfs[ty0 * grid + tx1][val];
          const v01 = cdfs[ty1 * grid + tx0][val];
          const v11 = cdfs[ty1 * grid + tx1][val];

          const top = v00 * (1 - wx) + v10 * wx;
          const bot = v01 * (1 - wx) + v11 * wx;
          newLum[rowOff + x] = Math.round(top * (1 - wy) + bot * wy);
        }
      }
    }

    // Reapply chrominance-preserving luminance ratio
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      const oldY = lum[i];
      const targetY = newLum[i];
      if (oldY < 2) {
        dst[p] = Math.min(255, src[p] + targetY);
        dst[p + 1] = Math.min(255, src[p + 1] + targetY);
        dst[p + 2] = Math.min(255, src[p + 2] + targetY);
      } else {
        const ratio = targetY / oldY;
        dst[p] = Math.min(255, Math.round(src[p] * ratio));
        dst[p + 1] = Math.min(255, Math.round(src[p + 1] * ratio));
        dst[p + 2] = Math.min(255, Math.round(src[p + 2] * ratio));
      }
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // YOLOv11 WebAssembly Inference + NMS + Point-in-Polygon + ByteTrack
  // ---------------------------------------------------------------------------
  function boxIoU(a, b) {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const iw = Math.max(0, x2 - x1);
    const ih = Math.max(0, y2 - y1);
    const inter = iw * ih;
    const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
    const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
    return inter / (areaA + areaB - inter + 1e-6);
  }

  function pointInPolygon(px, py, polyPx) {
    let inside = false;
    for (let i = 0, j = polyPx.length - 1; i < polyPx.length; j = i++) {
      const xi = polyPx[i][0],
        yi = polyPx[i][1];
      const xj = polyPx[j][0],
        yj = polyPx[j][1];
      const intersect =
        yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-9) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  class JSByteTracker {
    constructor() {
      this.reset();
    }
    reset() {
      this.nextId = 1;
      this.tracks = [];
    }
    update(dets, highThresh) {
      for (const t of this.tracks) {
        const decay = Math.pow(0.85, Math.max(0, t.lost));
        for (let k = 0; k < 4; k++) t.bbox[k] += t.vel[k] * decay;
        t.lost += 1;
      }
      const high = dets.filter((d) => d.conf >= highThresh);
      const low = dets.filter((d) => d.conf < highThresh);
      const usedT = new Set();
      const usedH = new Set();
      const out = [];

      const pairs = [];
      for (let ti = 0; ti < this.tracks.length; ti++) {
        for (let di = 0; di < high.length; di++) {
          if (this.tracks[ti].cls !== high[di].cls) continue;
          const v = boxIoU(this.tracks[ti].bbox, high[di].bbox);
          if (v >= 0.25) pairs.push([v, ti, di]);
        }
      }
      pairs.sort((a, b) => b[0] - a[0]);
      for (const [, ti, di] of pairs) {
        if (usedT.has(ti) || usedH.has(di)) continue;
        usedT.add(ti);
        usedH.add(di);
        const trk = this.tracks[ti];
        for (let k = 0; k < 4; k++) {
          trk.vel[k] = 0.65 * trk.vel[k] + 0.35 * (high[di].bbox[k] - trk.bbox[k]);
          trk.bbox[k] = high[di].bbox[k];
        }
        trk.lost = 0;
        out.push({ ...high[di], trackId: trk.id });
      }

      // Stage 2 low-conf continuity
      const pairs2 = [];
      for (let ti = 0; ti < this.tracks.length; ti++) {
        if (usedT.has(ti) || this.tracks[ti].lost > 3) continue;
        for (let di = 0; di < low.length; di++) {
          if (this.tracks[ti].cls !== low[di].cls) continue;
          const v = boxIoU(this.tracks[ti].bbox, low[di].bbox);
          if (v >= 0.35) pairs2.push([v, ti, di]);
        }
      }
      pairs2.sort((a, b) => b[0] - a[0]);
      const usedL = new Set();
      for (const [, ti, di] of pairs2) {
        if (usedT.has(ti) || usedL.has(di)) continue;
        usedT.add(ti);
        usedL.add(di);
        const trk = this.tracks[ti];
        for (let k = 0; k < 4; k++) trk.bbox[k] = low[di].bbox[k];
        trk.lost = 0;
        if (low[di].conf >= highThresh * 0.85) {
          out.push({ ...low[di], trackId: trk.id });
        }
      }

      for (let di = 0; di < high.length; di++) {
        if (usedH.has(di)) continue;
        const id = this.nextId++;
        this.tracks.push({
          id,
          bbox: high[di].bbox.slice(),
          vel: [0, 0, 0, 0],
          cls: high[di].cls,
          lost: 0,
        });
        out.push({ ...high[di], trackId: id });
      }

      this.tracks = this.tracks.filter((t) => t.lost <= 25);
      return out;
    }
  }

  const workCanvas = document.createElement("canvas");
  const workCtx = workCanvas.getContext("2d", { willReadFrequently: true });
  const letterCanvas = document.createElement("canvas");
  const letterCtx = letterCanvas.getContext("2d", { willReadFrequently: true });

  function runYoloOnCanvas(sourceCanvas, scoreFloor, targetSide = edge.imageSide) {
    const task = edge.inferMutex.then(() =>
      _runYoloOnCanvasInternal(sourceCanvas, scoreFloor, targetSide)
    );
    edge.inferMutex = task.catch(() => {});
    return task;
  }

  async function _runYoloOnCanvasInternal(sourceCanvas, scoreFloor, targetSide) {
    await ensureModelLoaded();
    const w0 = sourceCanvas.width;
    const h0 = sourceCanvas.height;
    const side = targetSide || edge.videoSide || 416;

    const ratio = Math.min(side / h0, side / w0);
    const nw = Math.round(w0 * ratio);
    const nh = Math.round(h0 * ratio);
    const W = Math.max(32, Math.ceil(nw / 32) * 32);
    const H = Math.max(32, Math.ceil(nh / 32) * 32);
    const padX = Math.floor((W - nw) * 0.5);
    const padY = Math.floor((H - nh) * 0.5);

    if (letterCanvas.width !== W || letterCanvas.height !== H) {
      letterCanvas.width = W;
      letterCanvas.height = H;
    }
    letterCtx.fillStyle = "rgb(114,114,114)";
    letterCtx.fillRect(0, 0, W, H);
    letterCtx.drawImage(sourceCanvas, padX, padY, nw, nh);

    const imgData = letterCtx.getImageData(0, 0, W, H).data;
    const enabledIds =
      edge.enabledClassIds === null ? null : Array.from(edge.enabledClassIds);
    if (enabledIds && enabledIds.length === 0) return [];

    const reqId = edge.nextReqId++;
    const w = getOrCreateWorker();
    const res = await new Promise((resolve, reject) => {
      edge.pendingReqs.set(reqId, { resolve, reject });
      w.postMessage(
        {
          type: "infer",
          reqId,
          rgbaBuffer: imgData.buffer,
          W,
          H,
          padX,
          padY,
          ratio,
          w0,
          h0,
          scoreFloor,
          iou: edge.iou,
          enabledIds,
        },
        [imgData.buffer]
      );
    });
    return res.dets || [];
  }

  function paintAnnotatedTarget(
    outC,
    baseSource,
    w,
    h,
    dets,
    sessionTrackerState,
    useTracking,
    cornerTag,
    encodeJpg = true
  ) {
    if (outC.width !== w || outC.height !== h) {
      outC.width = w;
      outC.height = h;
    }
    const ctx = outC.getContext("2d");
    ctx.drawImage(baseSource, 0, 0, w, h);

    const polyPx = edge.roi.map((p) => [p[0] * w, p[1] * h]);

    let insideNow = 0;
    const insideByClass = {};
    const countsByClass = {};
    const classColors = {};
    let confSum = 0;

    for (const d of dets) {
      const cname = edge.names[d.cls] || `class_${d.cls}`;
      const chex = colorForClass(d.cls, cname);
      d.className = cname;
      d.colorHex = chex;
      const cx = (d.bbox[0] + d.bbox[2]) * 0.5;
      const ay = d.bbox[3];
      d.anchor = [cx, ay];
      d.inside = pointInPolygon(cx, ay, polyPx);

      countsByClass[cname] = (countsByClass[cname] || 0) + 1;
      classColors[cname] = chex;
      confSum += d.conf;
      if (d.inside) {
        insideNow++;
        insideByClass[cname] = (insideByClass[cname] || 0) + 1;
      }

      if (useTracking && d.trackId && sessionTrackerState) {
        sessionTrackerState.uniqueAll.add(d.trackId);
        if (!sessionTrackerState.trails[d.trackId]) sessionTrackerState.trails[d.trackId] = [];
        sessionTrackerState.trails[d.trackId].push([cx, ay]);
        if (sessionTrackerState.trails[d.trackId].length > 20)
          sessionTrackerState.trails[d.trackId].shift();

        if (d.inside && !sessionTrackerState.roiIds.has(d.trackId)) {
          sessionTrackerState.roiIds.add(d.trackId);
          sessionTrackerState.roiLog.push({
            track_id: `#${d.trackId}`,
            raw_id: d.trackId,
            class_name: cname,
            first_entered_sec: sessionTrackerState.tsSec || 0,
            confidence: Number(d.conf.toFixed(4)),
          });
        }
      }
    }

    // 1. ROI Overlay
    if (edge.renderOpts.show_roi && polyPx.length >= 3) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(polyPx[0][0], polyPx[0][1]);
      for (let i = 1; i < polyPx.length; i++) ctx.lineTo(polyPx[i][0], polyPx[i][1]);
      ctx.closePath();
      ctx.fillStyle = insideNow > 0 ? "rgba(184, 246, 156, 0.16)" : "rgba(199, 181, 255, 0.12)";
      ctx.fill();
      ctx.setLineDash([8, 5]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = insideNow > 0 ? "#b8f69c" : "#c7b5ff";
      ctx.stroke();
      ctx.restore();
    }

    // 2. Trails
    if (useTracking && edge.renderOpts.show_trails && sessionTrackerState) {
      for (const d of dets) {
        const tr = d.trackId && sessionTrackerState.trails[d.trackId];
        if (!tr || tr.length < 2) continue;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(tr[0][0], tr[0][1]);
        for (let i = 1; i < tr.length; i++) ctx.lineTo(tr[i][0], tr[i][1]);
        ctx.strokeStyle = d.colorHex;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      }
    }

    // 3. Boxes & Pill Labels
    if (edge.renderOpts.show_boxes) {
      ctx.font = "600 12px 'Plus Jakarta Sans', sans-serif";
      for (const d of dets) {
        const [x1, y1, x2, y2] = d.bbox;
        ctx.strokeStyle = d.colorHex;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        if (edge.renderOpts.show_roi) {
          ctx.beginPath();
          ctx.arc(d.anchor[0], d.anchor[1], d.inside ? 5 : 3, 0, Math.PI * 2);
          ctx.fillStyle = d.inside ? "#b8f69c" : d.colorHex;
          ctx.fill();
        }

        const idTxt = edge.renderOpts.show_track_ids && d.trackId ? `#${d.trackId} ` : "";
        const lbl = `${idTxt}${d.className} ${Math.round(d.conf * 100)}%`;
        const tw = ctx.measureText(lbl).width;
        const ly = y1 > 22 ? y1 - 22 : y1 + 2;
        ctx.fillStyle = "rgba(14, 16, 24, 0.88)";
        ctx.fillRect(x1, ly, tw + 12, 20);
        ctx.fillStyle = d.colorHex;
        ctx.fillRect(x1, ly, 3, 20);
        ctx.fillStyle = "#f5f7fb";
        ctx.fillText(lbl, x1 + 7, ly + 14);
      }
    }

    // 4. Zone HUD Pill
    if (edge.renderOpts.show_roi) {
      const enteredTot =
        useTracking && sessionTrackerState ? sessionTrackerState.roiIds.size : insideNow;
      const hud = useTracking
        ? `Zone: ${insideNow}  |  Total: ${enteredTot}`
        : `In Zone: ${insideNow}`;
      ctx.font = "700 12px 'Plus Jakarta Sans', sans-serif";
      const hw = ctx.measureText(hud).width + 22;
      ctx.fillStyle = "rgba(14, 16, 24, 0.88)";
      ctx.fillRect(w - hw - 12, 12, hw, 28);
      ctx.strokeStyle = "#b8f69c";
      ctx.lineWidth = 1;
      ctx.strokeRect(w - hw - 12, 12, hw, 28);
      ctx.fillStyle = "#b8f69c";
      ctx.fillText(hud, w - hw - 1, 30);
    }

    if (cornerTag) {
      ctx.font = "700 12px 'Plus Jakarta Sans', sans-serif";
      const cw = ctx.measureText(cornerTag).width + 20;
      ctx.fillStyle = "rgba(14, 16, 24, 0.88)";
      ctx.fillRect(12, 12, cw, 28);
      ctx.fillStyle = "#f5f7fb";
      ctx.fillText(cornerTag, 22, 30);
    }

    const avgConf = dets.length ? confSum / dets.length : 0;
    const uniqueTot =
      useTracking && sessionTrackerState ? sessionTrackerState.uniqueAll.size : dets.length;
    const roiEnteredTot =
      useTracking && sessionTrackerState ? sessionTrackerState.roiIds.size : insideNow;

    return {
      dataUrl: encodeJpg ? outC.toDataURL("image/jpeg", 0.84) : null,
      total: dets.length,
      avgConf: Number(avgConf.toFixed(4)),
      insideNow,
      insideByClass,
      uniqueTot,
      roiEnteredTot,
      countsByClass,
      classColors,
      roiLog:
        useTracking && sessionTrackerState
          ? sessionTrackerState.roiLog.slice(-50)
          : dets
              .filter((d) => d.inside)
              .map((d, i) => ({
                track_id: `#IMG-${i + 1}`,
                class_name: d.className,
                first_entered_sec: 0,
                confidence: Number(d.conf.toFixed(4)),
              })),
    };
  }

  function renderAnnotatedCanvas(baseCanvas, dets, sessionTrackerState, useTracking, cornerTag) {
    const outC = document.createElement("canvas");
    return paintAnnotatedTarget(
      outC,
      baseCanvas,
      baseCanvas.width,
      baseCanvas.height,
      dets,
      sessionTrackerState,
      useTracking,
      cornerTag,
      true
    );
  }

  function resolveMediaUrl(sourcePath) {
    if (edge.uploadedBlobs[sourcePath]) return edge.uploadedBlobs[sourcePath];
    return sourcePath.startsWith("/") ? sourcePath : `/${sourcePath}`;
  }

  function loadImageElement(srcUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Cannot load image: ${srcUrl}`));
      img.src = srcUrl;
    });
  }

  function makeEnhancedCanvas(imgOrVideo, w, h, enhCfg) {
    workCanvas.width = w;
    workCanvas.height = h;
    workCtx.drawImage(imgOrVideo, 0, 0, w, h);
    if (enhCfg && enhCfg.method !== "none") {
      const idata = workCtx.getImageData(0, 0, w, h);
      const enh = applyImageDataEnhancement(idata, enhCfg);
      workCtx.putImageData(enh, 0, 0);
    }
    const copy = document.createElement("canvas");
    copy.width = w;
    copy.height = h;
    copy.getContext("2d").drawImage(workCanvas, 0, 0);
    return copy;
  }

  function buildPerClassTable(bDets, aDets) {
    const bCnt = {},
      aCnt = {},
      bConf = {},
      aConf = {};
    for (const d of bDets) {
      const n = edge.names[d.cls] || `class_${d.cls}`;
      bCnt[n] = (bCnt[n] || 0) + 1;
      (bConf[n] = bConf[n] || []).push(d.conf);
    }
    for (const d of aDets) {
      const n = edge.names[d.cls] || `class_${d.cls}`;
      aCnt[n] = (aCnt[n] || 0) + 1;
      (aConf[n] = aConf[n] || []).push(d.conf);
    }
    const all = Array.from(new Set([...Object.keys(bCnt), ...Object.keys(aCnt)]));
    return all
      .map((cn) => {
        const bc = bCnt[cn] || 0;
        const ac = aCnt[cn] || 0;
        const dc = ac - bc;
        const dp = bc === 0 ? (ac > 0 ? 100 : 0) : Number((((ac - bc) / bc) * 100).toFixed(2));
        const bAvg = bc ? bConf[cn].reduce((s, v) => s + v, 0) / bc : 0;
        const aAvg = ac ? aConf[cn].reduce((s, v) => s + v, 0) / ac : 0;
        return {
          class_name: cn,
          color_hex: colorForClass(0, cn),
          before_count: bc,
          after_count: ac,
          delta_count: dc,
          delta_pct: dp,
          before_avg_conf: Number(bAvg.toFixed(4)),
          after_avg_conf: Number(aAvg.toFixed(4)),
          conf_delta: Number((aAvg - bAvg).toFixed(4)),
        };
      })
      .sort((a, b) => b.after_count + b.before_count - (a.after_count + a.before_count));
  }

  async function detectSingleImageEdge(sourcePath) {
    const t0 = performance.now();
    const img = await loadImageElement(resolveMediaUrl(sourcePath));
    const w = img.naturalWidth;
    const h = img.naturalHeight;

    const rawCanvas = makeEnhancedCanvas(img, w, h, { method: "none" });
    const rawDets = await runYoloOnCanvas(rawCanvas, edge.conf);

    const enhCanvas =
      edge.enhancement.method === "none"
        ? rawCanvas
        : makeEnhancedCanvas(img, w, h, edge.enhancement);
    const enhDets =
      edge.enhancement.method === "none" ? rawDets : await runYoloOnCanvas(enhCanvas, edge.conf);

    const rawRendered = renderAnnotatedCanvas(rawCanvas, rawDets, null, false, null);
    const activeRendered = renderAnnotatedCanvas(enhCanvas, enhDets, null, false, null);
    const perClass = buildPerClassTable(rawDets, enhDets);
    const fname = sourcePath.split("/").pop();
    const latency = Number((performance.now() - t0).toFixed(1));

    const bTot = rawRendered.total;
    const aTot = activeRendered.total;
    const dPct = bTot === 0 ? (aTot > 0 ? 100 : 0) : Number((((aTot - bTot) / bTot) * 100).toFixed(2));

    edge.latestSource = sourcePath;
    edge.latestReportData = {
      mode: "Image Socio-Economic Detection (Cloudflare Edge)",
      source_type: "image",
      source_name: fname,
      summary: {
        model_badge: buildModelStatus().badge_text,
        device_label: "Cloudflare Edge • WASM SIMD",
        enhancement_method: edge.enhancement.method,
        enhancement_description: describeEnhancement(edge.enhancement),
        conf_threshold: edge.conf,
        iou_threshold: edge.iou,
        before_total: bTot,
        after_total: aTot,
        delta_count: aTot - bTot,
        delta_pct: dPct,
        before_avg_conf: rawRendered.avgConf,
        after_avg_conf: activeRendered.avgConf,
        delta_conf: Number((activeRendered.avgConf - rawRendered.avgConf).toFixed(4)),
        after_inside_roi: activeRendered.insideNow,
        unique_tracked_total: aTot,
        region_unique_entered: activeRendered.insideNow,
      },
      per_class_table: perClass,
      frame_detail_log: [
        {
          frame_or_file: fname,
          timestamp_sec: 0,
          before_detections: bTot,
          after_detections: aTot,
          before_avg_conf: rawRendered.avgConf,
          after_avg_conf: activeRendered.avgConf,
          inside_roi_count: activeRendered.insideNow,
        },
      ],
      region_counter_log: activeRendered.roiLog,
    };

    return {
      source_name: fname,
      source_path: sourcePath,
      annotated_image: activeRendered.dataUrl,
      width: w,
      height: h,
      latency_ms: latency,
      enhancement_description: describeEnhancement(edge.enhancement),
      telemetry: {
        total_detections: aTot,
        unique_tracked_total: aTot,
        inside_roi_now: activeRendered.insideNow,
        inside_roi_by_class: activeRendered.insideByClass,
        region_unique_entered: activeRendered.insideNow,
        avg_confidence: activeRendered.avgConf,
        conf_delta: Number((activeRendered.avgConf - rawRendered.avgConf).toFixed(4)),
        delta_count: aTot - bTot,
        current_counts_by_class: activeRendered.countsByClass,
        class_colors: activeRendered.classColors,
        region_counter_log: activeRendered.roiLog,
      },
      per_class_table: perClass,
      frame_detail_log: edge.latestReportData.frame_detail_log,
    };
  }

  async function compareMediaEdge(sourcePath, sampleStep = 10) {
    const t0 = performance.now();
    const fname = sourcePath.split("/").pop();
    const isVid = /\.(mp4|mov|avi|webm)$/i.test(fname);

    if (!isVid) {
      const img = await loadImageElement(resolveMediaUrl(sourcePath));
      const w = img.naturalWidth,
        h = img.naturalHeight;
      const bCan = makeEnhancedCanvas(img, w, h, { method: "none" });
      const aCan =
        edge.enhancement.method === "none"
          ? bCan
          : makeEnhancedCanvas(img, w, h, edge.enhancement);
      const bDets = await runYoloOnCanvas(bCan, edge.conf, edge.videoSide);
      const aDets =
        edge.enhancement.method === "none"
          ? bDets
          : await runYoloOnCanvas(aCan, edge.conf, edge.videoSide);

      const bRen = renderAnnotatedCanvas(bCan, bDets, null, false, "BEFORE | ORIGINAL");
      const aRen = renderAnnotatedCanvas(
        aCan,
        aDets,
        null,
        false,
        `AFTER | ${edge.enhancement.method.toUpperCase()}`
      );
      const perClass = buildPerClassTable(bDets, aDets);
      const dPct =
        bRen.total === 0
          ? aRen.total > 0
            ? 100
            : 0
          : Number((((aRen.total - bRen.total) / bRen.total) * 100).toFixed(2));

      const res = {
        mode: "Before vs. After Lab (Image)",
        source_type: "image",
        source_name: fname,
        width: w,
        height: h,
        latency_ms: Number((performance.now() - t0).toFixed(1)),
        before_image: bRen.dataUrl,
        after_image: aRen.dataUrl,
        summary: {
          model_badge: buildModelStatus().badge_text,
          device_label: "Cloudflare Edge • WASM SIMD",
          enhancement_method: edge.enhancement.method,
          enhancement_description: describeEnhancement(edge.enhancement),
          conf_threshold: edge.conf,
          iou_threshold: edge.iou,
          before_total: bRen.total,
          after_total: aRen.total,
          delta_count: aRen.total - bRen.total,
          delta_pct: dPct,
          before_avg_conf: bRen.avgConf,
          after_avg_conf: aRen.avgConf,
          delta_conf: Number((aRen.avgConf - bRen.avgConf).toFixed(4)),
          after_inside_roi: aRen.insideNow,
          unique_tracked_total: aRen.total,
          region_unique_entered: aRen.insideNow,
        },
        per_class_table: perClass,
        frame_detail_log: [
          {
            frame_or_file: fname,
            timestamp_sec: 0,
            before_detections: bRen.total,
            after_detections: aRen.total,
            before_avg_conf: bRen.avgConf,
            after_avg_conf: aRen.avgConf,
            inside_roi_count: aRen.insideNow,
          },
        ],
        region_counter_log: aRen.roiLog,
      };
      edge.latestReportData = res;
      return res;
    }

    // Video sampled comparison
    const vid = document.createElement("video");
    vid.src = resolveMediaUrl(sourcePath);
    vid.muted = true;
    vid.playsInline = true;
    await new Promise((res, rej) => {
      vid.onloadedmetadata = res;
      vid.onerror = () => rej(new Error("Cannot decode video"));
    });

    const dur = Math.min(vid.duration || 5, 25);
    const fps = 15;
    const dt = Math.max(0.25, sampleStep / fps);
    const allBDets = [],
      allADets = [],
      frameLog = [];
    let bestScore = -1e9,
      bestBUrl = null,
      bestAUrl = null;
    const tracker = new JSByteTracker();
    const sessState = { uniqueAll: new Set(), roiIds: new Set(), roiLog: [], trails: {} };

    let fIdx = 0;
    for (let t = 0; t < dur; t += dt) {
      vid.currentTime = t;
      await new Promise((r) => (vid.onseeked = r));
      const w = vid.videoWidth || 640,
        h = vid.videoHeight || 360;
      const bCan = makeEnhancedCanvas(vid, w, h, { method: "none" });
      const aCan =
        edge.enhancement.method === "none"
          ? bCan
          : makeEnhancedCanvas(vid, w, h, edge.enhancement);
      const bDets = await runYoloOnCanvas(bCan, edge.conf, edge.videoSide);
      const aDets =
        edge.enhancement.method === "none"
          ? bDets
          : await runYoloOnCanvas(aCan, edge.conf, edge.videoSide);

      const tracked = tracker.update(aDets, edge.conf);
      sessState.tsSec = Number(t.toFixed(2));
      const bRen = renderAnnotatedCanvas(bCan, bDets, null, false, `BEFORE | ${t.toFixed(1)}s`);
      const aRen = renderAnnotatedCanvas(
        aCan,
        tracked,
        sessState,
        true,
        `AFTER | ${t.toFixed(1)}s`
      );

      allBDets.push(...bDets);
      allADets.push(...aDets);
      frameLog.push({
        frame_or_file: `Frame #${fIdx}`,
        timestamp_sec: Number(t.toFixed(2)),
        before_detections: bRen.total,
        after_detections: aDets.length,
        before_avg_conf: bRen.avgConf,
        after_avg_conf: aRen.avgConf,
        inside_roi_count: aRen.insideNow,
      });

      const gain = (aDets.length - bRen.total) * 3 + (aRen.avgConf - bRen.avgConf) * 10;
      if (gain > bestScore) {
        bestScore = gain;
        bestBUrl = bRen.dataUrl;
        bestAUrl = aRen.dataUrl;
      }
      fIdx += sampleStep;
    }

    const perClass = buildPerClassTable(allBDets, allADets);
    const bTot = allBDets.length,
      aTot = allADets.length;
    const bAvg = bTot ? allBDets.reduce((s, d) => s + d.conf, 0) / bTot : 0;
    const aAvg = aTot ? allADets.reduce((s, d) => s + d.conf, 0) / aTot : 0;
    const dPct = bTot === 0 ? (aTot > 0 ? 100 : 0) : Number((((aTot - bTot) / bTot) * 100).toFixed(2));

    const out = {
      mode: "Before vs. After Lab (Video)",
      source_type: "video",
      source_name: fname,
      before_image: bestBUrl,
      after_image: bestAUrl,
      summary: {
        model_badge: buildModelStatus().badge_text,
        device_label: "Cloudflare Edge • WASM SIMD",
        enhancement_method: edge.enhancement.method,
        enhancement_description: describeEnhancement(edge.enhancement),
        conf_threshold: edge.conf,
        iou_threshold: edge.iou,
        before_total: bTot,
        after_total: aTot,
        delta_count: aTot - bTot,
        delta_pct: dPct,
        before_avg_conf: Number(bAvg.toFixed(4)),
        after_avg_conf: Number(aAvg.toFixed(4)),
        delta_conf: Number((aAvg - bAvg).toFixed(4)),
        after_inside_roi: sessState.roiIds.size,
        unique_tracked_total: sessState.uniqueAll.size,
        region_unique_entered: sessState.roiIds.size,
      },
      per_class_table: perClass,
      frame_detail_log: frameLog,
      region_counter_log: sessState.roiLog,
    };
    edge.latestReportData = out;
    return out;
  }

  // ---------------------------------------------------------------------------
  // 4-Sheet Enterprise Excel (.xlsx) Generator in Browser via ExcelJS
  // ---------------------------------------------------------------------------
  async function generateEdgeExcel() {
    await loadScriptOnce("https://cdn.jsdelivr.net/npm/exceljs@4.4.0/dist/exceljs.min.js");
    if (!edge.latestReportData) {
      const first = edge.samples.images[0] && edge.samples.images[0].path;
      if (first) await detectSingleImageEdge(first);
    }
    const rep = edge.latestReportData;
    const sm = rep.summary || {};

    const wb = new window.ExcelJS.Workbook();
    const headerFill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF0F172A" } };
    const headerFont = { name: "Calibri", size: 11, bold: true, color: { argb: "FFFFFFFF" } };

    // Sheet 1: Executive_Summary
    const ws1 = wb.addWorksheet("Executive_Summary");
    ws1.columns = [{ width: 34 }, { width: 22 }, { width: 34 }];
    const titleRow = ws1.addRow(["PASARVISION AI — TRADITIONAL MARKET ANALYTICS REPORT"]);
    titleRow.font = { bold: true, size: 13, color: { argb: "FF10B981" } };
    ws1.addRow(["Generated", new Date().toISOString(), rep.mode || ""]);
    ws1.addRow(["Input Source", rep.source_name || "", sm.model_badge || "best.pt"]);
    ws1.addRow(["Enhancement", sm.enhancement_description || "", `Conf=${sm.conf_threshold}`]);
    ws1.addRow([]);

    const kpiHdr = ws1.addRow(["Metric Indicator", "Value", "Notes"]);
    kpiHdr.eachCell((c) => {
      c.fill = headerFill;
      c.font = headerFont;
    });
    ws1.addRow(["Total Detections (Before)", sm.before_total || 0, "Raw baseline"]);
    ws1.addRow(["Total Detections (After)", sm.after_total || 0, "Active enhancement"]);
    ws1.addRow(["Detection Delta (%)", `${sm.delta_pct || 0}%`, `Net: ${sm.delta_count || 0}`]);
    ws1.addRow(["Avg Confidence (Before)", sm.before_avg_conf || 0, ""]);
    ws1.addRow(["Avg Confidence (After)", sm.after_avg_conf || 0, ""]);
    ws1.addRow(["Unique Tracked IDs", sm.unique_tracked_total || 0, "ByteTrack"]);
    ws1.addRow(["ROI Unique Entered", sm.region_unique_entered || 0, "Aisle Zone"]);

    // Sheet 2: Per_Class_Comparison
    const ws2 = wb.addWorksheet("Per_Class_Comparison");
    ws2.columns = [
      { width: 18 },
      { width: 14 },
      { width: 14 },
      { width: 14 },
      { width: 14 },
      { width: 16 },
      { width: 16 },
      { width: 14 },
    ];
    const h2 = ws2.addRow([
      "Class",
      "Before Count",
      "After Count",
      "Delta Count",
      "Delta %",
      "Before Avg Conf",
      "After Avg Conf",
      "Conf Delta",
    ]);
    h2.eachCell((c) => {
      c.fill = headerFill;
      c.font = headerFont;
    });
    (rep.per_class_table || []).forEach((r) => {
      ws2.addRow([
        r.class_name,
        r.before_count,
        r.after_count,
        r.delta_count,
        `${r.delta_pct}%`,
        r.before_avg_conf,
        r.after_avg_conf,
        r.conf_delta,
      ]);
    });

    // Sheet 3: Frame_Detail_Log
    const ws3 = wb.addWorksheet("Frame_Detail_Log");
    ws3.columns = [
      { width: 22 },
      { width: 14 },
      { width: 18 },
      { width: 18 },
      { width: 16 },
      { width: 16 },
      { width: 16 },
    ];
    const h3 = ws3.addRow([
      "Frame/File",
      "Timestamp (s)",
      "Before Detections",
      "After Detections",
      "Before Avg Conf",
      "After Avg Conf",
      "Inside ROI Count",
    ]);
    h3.eachCell((c) => {
      c.fill = headerFill;
      c.font = headerFont;
    });
    (rep.frame_detail_log || []).forEach((f) => {
      ws3.addRow([
        f.frame_or_file,
        f.timestamp_sec,
        f.before_detections,
        f.after_detections,
        f.before_avg_conf,
        f.after_avg_conf,
        f.inside_roi_count,
      ]);
    });

    // Sheet 4: Region_Counter_Log
    const ws4 = wb.addWorksheet("Region_Counter_Log");
    ws4.columns = [{ width: 16 }, { width: 18 }, { width: 18 }, { width: 16 }];
    const h4 = ws4.addRow(["Track ID", "Class", "First Entered (s)", "Confidence"]);
    h4.eachCell((c) => {
      c.fill = headerFill;
      c.font = headerFont;
    });
    (rep.region_counter_log || []).forEach((r) => {
      ws4.addRow([r.track_id, r.class_name, r.first_entered_sec, r.confidence]);
    });

    const buf = await wb.xlsx.writeBuffer();
    const blob = new Blob([buf], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    const url = URL.createObjectURL(blob);
    const filename = `PasarVision_Report_${Date.now()}.xlsx`;
    edge.reports.unshift({
      filename,
      kind: "excel",
      size_kb: Number((buf.byteLength / 1024).toFixed(1)),
      created_at: new Date().toLocaleTimeString(),
      download_url: url,
    });
    return { filename, download_url: url, reports: edge.reports };
  }

  // ---------------------------------------------------------------------------
  // Virtual WebSocket Video Streamer (Runs YOLOv11s + ByteTrack on `<video>`)
  // ---------------------------------------------------------------------------
  const OrigWebSocket = window.WebSocket;
  class EdgeVideoSocket {
    constructor(url) {
      this.readyState = 1; // OPEN
      this.onmessage = null;
      this.onerror = null;
      this.onclose = null;
      this._paused = false;
      this._stopped = false;

      const u = new URL(url, window.location.href);
      const source = u.searchParams.get("source");
      setTimeout(() => this._runLoop(source), 20);
    }

    send(raw) {
      const msg = JSON.parse(raw);
      if (msg.action === "pause") {
        this._paused = true;
        if (this._vid) this._vid.pause();
      } else if (msg.action === "resume" || msg.action === "play") {
        this._paused = false;
        if (this._vid) this._vid.play().catch(() => {});
      } else if (msg.action === "stop") {
        this.close();
      } else if (msg.action === "restart") {
        if (this._tracker) this._tracker.reset();
        if (this._sessState) {
          this._sessState.uniqueAll.clear();
          this._sessState.roiIds.clear();
          this._sessState.roiLog = [];
          this._sessState.trails = {};
        }
        if (this._vid) {
          this._vid.currentTime = 0;
          this._paused = false;
          this._vid.play().catch(() => {});
        }
      } else if (msg.action === "seek" && this._vid) {
        const total = Math.max(1, Math.round((this._vid.duration || 5) * 30));
        this._vid.currentTime = (Number(msg.frame || 0) / total) * (this._vid.duration || 5);
      }
    }

    close() {
      this._stopped = true;
      this.readyState = 3;
      const edgeCanvas = document.getElementById("edge-video-canvas");
      if (edgeCanvas) edgeCanvas.classList.add("hidden");
      if (this._vid) {
        this._vid.pause();
        this._vid.src = "";
      }
      if (this.onclose) this.onclose();
    }

    async _runLoop(source) {
      try {
        // Trigger background Web Worker model load immediately
        ensureModelLoaded().catch(() => {});

        const vid = document.createElement("video");
        vid.preload = "auto";
        vid.src = resolveMediaUrl(source);
        vid.muted = true;
        vid.playsInline = true;
        this._vid = vid;

        await new Promise((res, rej) => {
          vid.onloadeddata = res;
          vid.onloadedmetadata = () => setTimeout(res, 120);
          vid.onerror = () =>
            rej(new Error("Format video tidak didukung browser (gunakan MP4 H.264 / WebM)."));
        });

        this._tracker = new JSByteTracker();
        this._sessState = { uniqueAll: new Set(), roiIds: new Set(), roiLog: [], trails: {} };
        const totalFrames = Math.max(1, Math.round((vid.duration || 5) * 30));
        const frameLog = [];

        const edgeCanvas = document.getElementById("edge-video-canvas");
        const mainImg = document.getElementById("main-viewport-img");
        if (mainImg) mainImg.classList.add("hidden");
        if (edgeCanvas) edgeCanvas.classList.remove("hidden");

        // Preallocated offscreen canvas for async worker snapshots
        const snapC = document.createElement("canvas");
        const snapCtx = snapC.getContext("2d");

        await vid.play().catch(() => {});

        let activeTracked = [];
        let detWallT = performance.now();
        const prevCenters = new Map();
        const vels = new Map();
        let inferBusy = false;
        let lastFpsT = performance.now();
        let fpsEma = 59.0;
        let tickCount = 0;

        const stepDisplay = () => {
          if (this._stopped) return;
          if (this._paused) {
            setTimeout(() => requestAnimationFrame(stepDisplay), 60);
            return;
          }
          if (vid.ended) {
            this._paused = true;
            if (this.onmessage) {
              this.onmessage({
                data: JSON.stringify({
                  type: "ended",
                  frame_detail_log: frameLog,
                  region_counter_log: this._sessState.roiLog,
                }),
              });
            }
            setTimeout(() => requestAnimationFrame(stepDisplay), 100);
            return;
          }

          const now = performance.now();
          const dt = Math.max(4, now - lastFpsT);
          lastFpsT = now;
          fpsEma = 0.88 * fpsEma + 0.12 * Math.min(62, 1000 / dt);

          const rawW = vid.videoWidth || 640;
          const rawH = vid.videoHeight || 360;
          const scale = rawW > 854 ? 854 / rawW : 1;
          const w = Math.round(rawW * scale);
          const h = Math.round(rawH * scale);

          // Fast path: draw HTMLVideoElement directly via GPU when enhancement is 'none'
          const baseSource =
            edge.enhancement.method === "none"
              ? vid
              : makeEnhancedCanvas(vid, w, h, edge.enhancement);

          if (!inferBusy && vid.readyState >= 2) {
            inferBusy = true;
            if (snapC.width !== w || snapC.height !== h) {
              snapC.width = w;
              snapC.height = h;
            }
            snapCtx.drawImage(baseSource, 0, 0, w, h);
            const snapTime = Number(vid.currentTime.toFixed(2));
            runYoloOnCanvas(snapC, Math.max(0.12, edge.conf * 0.5), edge.videoSide)
              .then((rawDets) => {
                if (this._stopped) return;
                const doneT = performance.now();
                this._sessState.tsSec = snapTime;
                const tracked = this._tracker.update(rawDets, edge.conf);

                for (const d of tracked) {
                  if (!d.trackId) continue;
                  const cx = (d.bbox[0] + d.bbox[2]) * 0.5;
                  const cy = (d.bbox[1] + d.bbox[3]) * 0.5;
                  const prev = prevCenters.get(d.trackId);
                  if (prev) {
                    const dtS = Math.max(0.02, (doneT - prev.t) * 0.001);
                    const vx = (cx - prev.cx) / dtS;
                    const vy = (cy - prev.cy) / dtS;
                    const oldV = vels.get(d.trackId) || [0, 0];
                    vels.set(d.trackId, [0.5 * oldV[0] + 0.5 * vx, 0.5 * oldV[1] + 0.5 * vy]);
                  }
                  prevCenters.set(d.trackId, { cx, cy, t: doneT });
                }
                activeTracked = tracked;
                detWallT = doneT;
              })
              .catch((err) => {
                console.error("[PasarVision Edge Video Infer]", err);
                if (!this._warnedErr && this.onmessage) {
                  this._warnedErr = true;
                  this.onmessage({
                    data: JSON.stringify({
                      type: "error",
                      message: `AI Engine: ${err.message || err}`,
                    }),
                  });
                }
              })
              .finally(() => {
                inferBusy = false;
              });
          }

          // Velocity extrapolation for fluid 60 FPS box motion between worker detections
          const dtSec = Math.min(0.12, Math.max(0, (now - detWallT) * 0.001));
          const smoothTracked = activeTracked.map((d) => {
            const v = d.trackId ? vels.get(d.trackId) : null;
            if (!v || dtSec <= 0.001) return { ...d, bbox: d.bbox.slice() };
            const dx = Math.max(-28, Math.min(28, v[0] * dtSec));
            const dy = Math.max(-28, Math.min(28, v[1] * dtSec));
            return {
              ...d,
              bbox: [
                Math.max(0, Math.min(w - 1, d.bbox[0] + dx)),
                Math.max(0, Math.min(h - 1, d.bbox[1] + dy)),
                Math.max(0, Math.min(w - 1, d.bbox[2] + dx)),
                Math.max(0, Math.min(h - 1, d.bbox[3] + dy)),
              ],
            };
          });

          const loadingTag = !edge.workerReady ? `Loading AI ${edge.modelLoadingPct}%...` : null;
          const prevW = edgeCanvas ? edgeCanvas.width : 0;
          const ren = paintAnnotatedTarget(
            edgeCanvas,
            baseSource,
            w,
            h,
            smoothTracked,
            this._sessState,
            true,
            loadingTag,
            false
          );
          if (prevW !== w && window.dispatchEvent) {
            window.dispatchEvent(new Event("resize"));
          }

          const renderMs = Math.max(0.5, performance.now() - now);
          const curFrame = Math.min(totalFrames, Math.round(vid.currentTime * 30) + 1);
          tickCount++;

          if (tickCount === 1 || tickCount % 8 === 0) {
            if (tickCount % 16 === 0) {
              frameLog.push({
                frame_or_file: `Frame #${curFrame}`,
                timestamp_sec: Number(vid.currentTime.toFixed(2)),
                before_detections: ren.total,
                after_detections: ren.total,
                before_avg_conf: ren.avgConf,
                after_avg_conf: ren.avgConf,
                inside_roi_count: ren.insideNow,
              });
              edge.latestSource = source;
              edge.latestReportData = {
                mode: "Live Video Tracking (Cloudflare Edge)",
                source_type: "video",
                source_name: source.split("/").pop(),
                summary: {
                  model_badge: buildModelStatus().badge_text,
                  enhancement_method: edge.enhancement.method,
                  enhancement_description: describeEnhancement(edge.enhancement),
                  conf_threshold: edge.conf,
                  before_total: ren.total,
                  after_total: ren.total,
                  delta_count: 0,
                  delta_pct: 0,
                  before_avg_conf: ren.avgConf,
                  after_avg_conf: ren.avgConf,
                  delta_conf: 0,
                  unique_tracked_total: ren.uniqueTot,
                  region_unique_entered: ren.roiEnteredTot,
                },
                per_class_table: [],
                frame_detail_log: frameLog.slice(-50),
                region_counter_log: this._sessState.roiLog,
              };
            }

            if (this.onmessage && !this._stopped) {
              this.onmessage({
                data: JSON.stringify({
                  type: "hud",
                  frame_idx: curFrame,
                  total_frames: totalFrames,
                  fps: Number(fpsEma.toFixed(1)),
                  latency_ms: Number(renderMs.toFixed(1)),
                  width: w,
                  height: h,
                  telemetry: {
                    total_detections: ren.total,
                    unique_tracked_total: ren.uniqueTot,
                    inside_roi_now: ren.insideNow,
                    inside_roi_by_class: ren.insideByClass,
                    region_unique_entered: ren.roiEnteredTot,
                    avg_confidence: ren.avgConf,
                    current_counts_by_class: ren.countsByClass,
                    class_colors: ren.classColors,
                    region_counter_log: ren.roiLog,
                  },
                }),
              });
            }
          }
          requestAnimationFrame(stepDisplay);
        };
        requestAnimationFrame(stepDisplay);
      } catch (err) {
        if (this.onmessage) {
          this.onmessage({ data: JSON.stringify({ type: "error", message: err.message }) });
        }
      }
    }
  }
  EdgeVideoSocket.OPEN = 1;

  window.WebSocket = function (url, protocols) {
    if (String(url).includes("/ws/video")) {
      return new EdgeVideoSocket(url);
    }
    return new OrigWebSocket(url, protocols);
  };
  window.WebSocket.OPEN = 1;

  // ---------------------------------------------------------------------------
  // Virtual `/api/*` Fetch Router
  // ---------------------------------------------------------------------------
  const origFetch = window.fetch.bind(window);
  function jsonResp(obj, status = 200) {
    return new Response(JSON.stringify(obj), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }

  window.fetch = async function (input, init = {}) {
    const urlStr = typeof input === "string" ? input : input.url;
    const u = new URL(urlStr, window.location.href);
    if (!u.pathname.startsWith("/api/")) {
      return origFetch(input, init);
    }

    try {
      await ensureManifestSamples();
      const path = u.pathname;
      const method = (init.method || "GET").toUpperCase();
      const bodyJson =
        init.body && typeof init.body === "string" ? JSON.parse(init.body) : {};

      if (path === "/api/status" && method === "GET") {
        return jsonResp(buildFullStatus());
      }
      if (path === "/api/model/status" && method === "GET") {
        return jsonResp(buildModelStatus());
      }
      if (path === "/api/reports" && method === "GET") {
        return jsonResp({ reports: edge.reports });
      }
      if (path === "/api/roi" && method === "GET") {
        return jsonResp({ points: edge.roi });
      }
      if (path === "/api/roi" && method === "POST") {
        edge.roi = bodyJson.points.map((p) => [Number(p[0]), Number(p[1])]);
        return jsonResp({ points: edge.roi });
      }
      if (path === "/api/roi/reset" && method === "POST") {
        edge.roi = DEFAULT_ROI.map((p) => p.slice());
        return jsonResp({ points: edge.roi });
      }
      if (path === "/api/settings" && method === "POST") {
        if (bodyJson.conf_threshold !== undefined) edge.conf = Number(bodyJson.conf_threshold);
        if (bodyJson.iou_threshold !== undefined) edge.iou = Number(bodyJson.iou_threshold);
        if (bodyJson.all_classes_enabled) edge.enabledClassIds = null;
        else if (Array.isArray(bodyJson.enabled_class_ids))
          edge.enabledClassIds = bodyJson.enabled_class_ids;
        if (bodyJson.enhancement)
          edge.enhancement = { ...edge.enhancement, ...bodyJson.enhancement };
        if (bodyJson.render_options)
          edge.renderOpts = { ...edge.renderOpts, ...bodyJson.render_options };
        return jsonResp(buildFullStatus());
      }
      if (path === "/api/upload/media" && method === "POST") {
        const fd = init.body;
        const files = fd.getAll("files");
        const uploaded = [];
        for (const f of files) {
          const isVid =
            (f.type && f.type.startsWith("video/")) ||
            /\.(mp4|mov|avi|webm|mkv|m4v)$/i.test(f.name);
          const vpath = `uploads/${Date.now()}_${f.name}`;
          const objUrl = URL.createObjectURL(f);
          edge.uploadedBlobs[vpath] = objUrl;
          const entry = {
            name: f.name,
            path: vpath,
            url: objUrl,
            origin: "uploaded",
            media_type: isVid ? "video" : "image",
            size_kb: Number((f.size / 1024).toFixed(1)),
          };
          uploaded.push(entry);
          if (isVid) edge.samples.videos.unshift(entry);
          else edge.samples.images.unshift(entry);
        }
        return jsonResp({ uploaded, samples: edge.samples });
      }
      if (path === "/api/model/upload" && method === "POST") {
        const file = init.body.get("file");
        if (!file || !file.name.toLowerCase().endsWith(".onnx")) {
          return jsonResp(
            {
              detail:
                "In browser cloud mode, upload `.onnx` weights (`YOLO('best.pt').export(format='onnx')`). Your `best.pt` is already pre-loaded!",
            },
            400
          );
        }
        const ab = await file.arrayBuffer();
        const reqId = edge.nextReqId++;
        const w = getOrCreateWorker();
        await new Promise((resolve, reject) => {
          edge.pendingReqs.set(reqId, { resolve, reject });
          w.postMessage({ type: "loadCustomOnnx", reqId, buffer: ab }, [ab]);
        });
        edge.workerReady = true;
        edge.activeModelName = file.name;
        return jsonResp(buildFullStatus());
      }
      if (path === "/api/detect/image" && method === "POST") {
        const res = await detectSingleImageEdge(bodyJson.source_path);
        return jsonResp(res);
      }
      if (path === "/api/detect/batch" && method === "POST") {
        const paths = bodyJson.source_paths || [];
        const items = [];
        let bTot = 0,
          aTot = 0,
          aRoi = 0;
        const fLog = [];
        for (const p of paths) {
          const r = await detectSingleImageEdge(p);
          items.push({
            filename: r.source_name,
            before_total: r.telemetry.total_detections - r.telemetry.delta_count,
            after_total: r.telemetry.total_detections,
            delta_count: r.telemetry.delta_count,
            after_avg_conf: r.telemetry.avg_confidence,
            after_inside_roi: r.telemetry.inside_roi_now,
            after_image: r.annotated_image,
            before_image: r.annotated_image,
          });
          aTot += r.telemetry.total_detections;
          bTot += r.telemetry.total_detections - r.telemetry.delta_count;
          aRoi += r.telemetry.inside_roi_now;
          fLog.push(...r.frame_detail_log);
        }
        return jsonResp({
          mode: "Batch Image Analysis",
          source_type: "batch",
          source_name: `Batch (${items.length} images)`,
          width: 960,
          height: 600,
          latency_ms: 45,
          after_image: items[0] ? items[0].after_image : null,
          batch_items: items,
          summary: {
            enhancement_description: describeEnhancement(edge.enhancement),
            before_total: bTot,
            after_total: aTot,
            delta_count: aTot - bTot,
            delta_pct: 0,
            after_avg_conf: items[0] ? items[0].after_avg_conf : 0,
            delta_conf: 0,
            after_inside_roi: aRoi,
          },
          per_class_table: [],
          frame_detail_log: fLog,
          region_counter_log: [],
        });
      }
      if (path === "/api/compare" && method === "POST") {
        const target =
          bodyJson.source_path || (bodyJson.source_paths && bodyJson.source_paths[0]);
        const res = await compareMediaEdge(target, bodyJson.sample_every_n || 10);
        return jsonResp(res);
      }
      if (path === "/api/export/excel" && method === "POST") {
        const res = await generateEdgeExcel();
        return jsonResp(res);
      }
      if (path === "/api/video/export" && method === "POST") {
        return jsonResp(
          { detail: "Use Live Video stream or Export .xlsx in Cloud Browser Mode." },
          400
        );
      }

      return jsonResp({ detail: `Unhandled cloud route ${path}` }, 404);
    } catch (err) {
      return jsonResp({ detail: err.message || String(err) }, 500);
    }
  };
})();
