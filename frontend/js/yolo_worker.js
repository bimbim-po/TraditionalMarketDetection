/**
 * PasarVision AI — Dedicated Multi-Threaded WASM YOLOv11s Web Worker (`yolo_worker.js`)
 *
 * Offloads all neural-network inference, NCHW tensor packing, and NMS off the
 * browser main thread so video rendering runs at a locked 60 FPS.
 */

import * as ort from "/static/ort/ort.wasm.min.mjs";

let session = null;
let initPromise = null;
let names = { 0: "cart", 1: "customer", 2: "scale", 3: "seller", 4: "vegetables" };

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

async function ensureSession() {
  if (session) return session;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      ort.env.wasm.wasmPaths = "/static/ort/";
      ort.env.wasm.simd = true;
      ort.env.wasm.numThreads = self.crossOriginIsolated
        ? Math.min(4, navigator.hardwareConcurrency || 4)
        : 1;

      const mRes = await fetch("/model/manifest_v4.json", { cache: "no-store" });
      if (!mRes.ok) throw new Error(`manifest_v4.json HTTP ${mRes.status}`);
      const manifest = await mRes.json();
      if (manifest.names) names = manifest.names;

      let loaded = 0;
      const totalChunks = manifest.chunks.length;
      const buffers = await Promise.all(
        manifest.chunks.map(async (c) => {
          const r = await fetch(c.file);
          if (!r.ok) throw new Error(`Chunk ${c.file} HTTP ${r.status}`);
          const ab = await r.arrayBuffer();
          loaded++;
          self.postMessage({
            type: "progress",
            pct: Math.round((loaded / totalChunks) * 100),
          });
          return ab;
        })
      );

      const totalBytes = buffers.reduce((s, b) => s + b.byteLength, 0);
      const merged = new Uint8Array(totalBytes);
      let off = 0;
      for (const b of buffers) {
        merged.set(new Uint8Array(b), off);
        off += b.byteLength;
      }

      session = await ort.InferenceSession.create(merged, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      self.postMessage({
        type: "ready",
        threads: ort.env.wasm.numThreads,
        names,
      });
      return session;
    } catch (err) {
      initPromise = null;
      throw err;
    }
  })();

  return initPromise;
}

async function runInfer(msg) {
  const sess = await ensureSession();
  const { reqId, rgbaBuffer, W, H, padX, padY, ratio, w0, h0, scoreFloor, iou, enabledIds } = msg;

  const rgba = new Uint8ClampedArray(rgbaBuffer);
  const plane = W * H;
  const floatData = new Float32Array(3 * plane);
  const inv255 = 1.0 / 255.0;
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    floatData[i] = rgba[p] * inv255;
    floatData[plane + i] = rgba[p + 1] * inv255;
    floatData[2 * plane + i] = rgba[p + 2] * inv255;
  }

  const inputName = sess.inputNames[0];
  const tensor = new ort.Tensor("float32", floatData, [1, 3, H, W]);
  const outMap = await sess.run({ [inputName]: tensor });
  const outTensor = outMap[sess.outputNames[0]];
  const dims = outTensor.dims;
  const data = outTensor.data;

  const nc = dims[1] - 4;
  const anchors = dims[2];
  const enabledSet = enabledIds ? new Set(enabledIds) : null;
  if (enabledSet && enabledSet.size === 0) {
    self.postMessage({ type: "result", reqId, dets: [] });
    return;
  }

  const candidates = [];
  for (let a = 0; a < anchors; a++) {
    let bestC = -1;
    let bestS = scoreFloor;
    for (let c = 0; c < nc; c++) {
      if (enabledSet && !enabledSet.has(c)) continue;
      const sc = data[(4 + c) * anchors + a];
      if (sc >= bestS) {
        bestS = sc;
        bestC = c;
      }
    }
    if (bestC === -1) continue;

    const cx = data[a];
    const cy = data[anchors + a];
    const bw = data[2 * anchors + a];
    const bh = data[3 * anchors + a];

    const x1 = Math.max(0, Math.min(w0 - 1, (cx - bw * 0.5 - padX) / ratio));
    const y1 = Math.max(0, Math.min(h0 - 1, (cy - bh * 0.5 - padY) / ratio));
    const x2 = Math.max(0, Math.min(w0 - 1, (cx + bw * 0.5 - padX) / ratio));
    const y2 = Math.max(0, Math.min(h0 - 1, (cy + bh * 0.5 - padY) / ratio));
    candidates.push({ bbox: [x1, y1, x2, y2], conf: bestS, cls: bestC });
  }

  candidates.sort((a, b) => b.conf - a.conf);
  const kept = [];
  for (const cand of candidates) {
    let sup = false;
    for (const k of kept) {
      if (k.cls === cand.cls && boxIoU(k.bbox, cand.bbox) > iou) {
        sup = true;
        break;
      }
    }
    if (!sup) kept.push(cand);
  }

  self.postMessage({ type: "result", reqId, dets: kept });
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await ensureSession();
    } else if (msg.type === "loadCustomOnnx") {
      ort.env.wasm.wasmPaths = "/static/ort/";
      ort.env.wasm.simd = true;
      ort.env.wasm.numThreads = self.crossOriginIsolated
        ? Math.min(4, navigator.hardwareConcurrency || 4)
        : 1;
      session = await ort.InferenceSession.create(new Uint8Array(msg.buffer), {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      initPromise = Promise.resolve(session);
      self.postMessage({ type: "customLoaded", reqId: msg.reqId });
    } else if (msg.type === "infer") {
      await runInfer(msg);
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      reqId: msg.reqId,
      message: err && err.message ? err.message : String(err),
    });
  }
};
