/**
 * PasarVision AI — Lightweight Interactive ROI Polygon Canvas (`roi_canvas.js`)
 * Zero `shadowBlur` overhead for instant 60fps pointer responsiveness.
 */

class ROIPolygonEditor {
  constructor({ canvasEl, mediaEl, stageEl, onSavePoints, notify }) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    this.mediaEl = mediaEl;
    this.stageEl = stageEl;
    this.onSavePoints = onSavePoints;
    this.notify = notify || (() => {});

    this.points = [
      [0.20, 0.35],
      [0.80, 0.35],
      [0.88, 0.90],
      [0.12, 0.90],
    ];

    this.isEditing = false;
    this.isDrawingNew = false;
    this.draftPoints = [];
    this.hoverCursor = null;
    this.draggingIndex = -1;
    this.hoveredVertex = -1;
    this.hoveredMidpoint = -1;

    this._bindEvents();
  }

  setPoints(normPoints) {
    if (Array.isArray(normPoints) && normPoints.length >= 3) {
      this.points = normPoints.map((p) => [
        Math.max(0, Math.min(1, Number(p[0]))),
        Math.max(0, Math.min(1, Number(p[1]))),
      ]);
      if (this.isEditing) this.render();
    }
  }

  getPoints() {
    return this.points.map((p) => [Number(p[0].toFixed(4)), Number(p[1].toFixed(4))]);
  }

  setEditing(active) {
    this.isEditing = Boolean(active);
    if (!this.isEditing) {
      this.isDrawingNew = false;
      this.draftPoints = [];
      this.draggingIndex = -1;
      this.canvas.classList.remove("editing-active");
    } else {
      this.canvas.classList.add("editing-active");
    }
    this.syncSize();
    this.render();
  }

  startFreshDraw() {
    if (!this.isEditing) this.setEditing(true);
    this.isDrawingNew = true;
    this.draftPoints = [];
    this.render();
    this.notify("Click 3+ points on frame, then click Done.");
  }

  finishFreshDrawIfValid() {
    if (this.isDrawingNew) {
      if (this.draftPoints.length >= 3) {
        this.points = this.draftPoints.slice();
        this.isDrawingNew = false;
        this.draftPoints = [];
        this.render();
        return true;
      }
      this.notify("Add at least 3 points.");
      return false;
    }
    return true;
  }

  syncSize() {
    if (!this.stageEl || !this.canvas) return;
    const rect = this.stageEl.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));

    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.isEditing) this.render();
    else this.ctx.clearRect(0, 0, w, h);
  }

  _getMediaBox() {
    const stageRect = this.stageEl.getBoundingClientRect();
    const sw = Math.max(1, stageRect.width);
    const sh = Math.max(1, stageRect.height);
    const natW = (this.mediaEl && this.mediaEl.naturalWidth) || 960;
    const natH = (this.mediaEl && this.mediaEl.naturalHeight) || 540;

    if (!this.mediaEl || this.mediaEl.classList.contains("hidden") || !natW || !natH) {
      return { offsetX: 0, offsetY: 0, width: sw, height: sh };
    }
    const scale = Math.min(sw / natW, sh / natH);
    const rw = natW * scale;
    const rh = natH * scale;
    return { offsetX: (sw - rw) * 0.5, offsetY: (sh - rh) * 0.5, width: rw, height: rh };
  }

  _normToPx(pt, box) {
    return [box.offsetX + pt[0] * box.width, box.offsetY + pt[1] * box.height];
  }

  _pxToNorm(x, y, box) {
    return [
      Math.max(0, Math.min(1, (x - box.offsetX) / Math.max(1, box.width))),
      Math.max(0, Math.min(1, (y - box.offsetY) / Math.max(1, box.height))),
    ];
  }

  _getPointerPos(evt) {
    const rect = this.canvas.getBoundingClientRect();
    return [evt.clientX - rect.left, evt.clientY - rect.top];
  }

  _bindEvents() {
    window.addEventListener("resize", () => this.syncSize());

    this.canvas.addEventListener("contextmenu", (e) => {
      if (!this.isEditing) return;
      e.preventDefault();
      if (this.isDrawingNew) {
        this.draftPoints.pop();
        this.render();
        return;
      }
      const [mx, my] = this._getPointerPos(e);
      const vIdx = this._hitTestVertex(mx, my);
      if (vIdx !== -1 && this.points.length > 3) {
        this.points.splice(vIdx, 1);
        this.render();
        if (this.onSavePoints) this.onSavePoints(this.getPoints());
      }
    });

    this.canvas.addEventListener("mousedown", (e) => {
      if (!this.isEditing || e.button !== 0) return;
      const [mx, my] = this._getPointerPos(e);
      const box = this._getMediaBox();

      if (this.isDrawingNew) {
        if (this.draftPoints.length >= 3) {
          const [p0x, p0y] = this._normToPx(this.draftPoints[0], box);
          if (Math.hypot(mx - p0x, my - p0y) <= 15) {
            this.points = this.draftPoints.slice();
            this.isDrawingNew = false;
            this.draftPoints = [];
            this.render();
            if (this.onSavePoints) this.onSavePoints(this.getPoints());
            return;
          }
        }
        this.draftPoints.push(this._pxToNorm(mx, my, box));
        this.render();
        return;
      }

      const vIdx = this._hitTestVertex(mx, my);
      if (vIdx !== -1) {
        this.draggingIndex = vIdx;
        return;
      }

      const midIdx = this._hitTestMidpoint(mx, my);
      if (midIdx !== -1) {
        const p1 = this.points[midIdx];
        const p2 = this.points[(midIdx + 1) % this.points.length];
        this.points.splice(midIdx + 1, 0, [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5]);
        this.draggingIndex = midIdx + 1;
        this.render();
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isEditing) return;
      const [mx, my] = this._getPointerPos(e);
      const box = this._getMediaBox();
      this.hoverCursor = [mx, my];

      if (this.draggingIndex !== -1) {
        this.points[this.draggingIndex] = this._pxToNorm(mx, my, box);
        this.render();
        return;
      }

      if (this.isDrawingNew) {
        this.render();
        return;
      }

      const hv = this._hitTestVertex(mx, my);
      const hm = hv === -1 ? this._hitTestMidpoint(mx, my) : -1;
      if (hv !== this.hoveredVertex || hm !== this.hoveredMidpoint) {
        this.hoveredVertex = hv;
        this.hoveredMidpoint = hm;
        this.canvas.style.cursor = hv !== -1 ? "grab" : hm !== -1 ? "copy" : "default";
        this.render();
      }
    });

    window.addEventListener("mouseup", () => {
      if (!this.isEditing) return;
      if (this.draggingIndex !== -1) {
        this.draggingIndex = -1;
        this.render();
        if (this.onSavePoints) this.onSavePoints(this.getPoints());
      }
    });
  }

  _hitTestVertex(mx, my) {
    const box = this._getMediaBox();
    for (let i = 0; i < this.points.length; i++) {
      const [px, py] = this._normToPx(this.points[i], box);
      if (Math.hypot(mx - px, my - py) <= 13) return i;
    }
    return -1;
  }

  _hitTestMidpoint(mx, my) {
    const box = this._getMediaBox();
    const n = this.points.length;
    for (let i = 0; i < n; i++) {
      const [x1, y1] = this._normToPx(this.points[i], box);
      const [x2, y2] = this._normToPx(this.points[(i + 1) % n], box);
      if (Math.hypot(mx - (x1 + x2) * 0.5, my - (y1 + y2) * 0.5) <= 11) return i;
    }
    return -1;
  }

  render() {
    const rect = this.stageEl.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);
    if (!this.isEditing) return;

    const box = this._getMediaBox();
    const activePts = this.isDrawingNew ? this.draftPoints : this.points;
    if (!activePts || activePts.length === 0) return;

    const pxPts = activePts.map((p) => this._normToPx(p, box));

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(pxPts[0][0], pxPts[0][1]);
    for (let i = 1; i < pxPts.length; i++) ctx.lineTo(pxPts[i][0], pxPts[i][1]);
    if (!this.isDrawingNew) {
      ctx.closePath();
      ctx.fillStyle = "rgba(184, 246, 156, 0.18)";
      ctx.fill();
    } else if (this.hoverCursor) {
      ctx.lineTo(this.hoverCursor[0], this.hoverCursor[1]);
    }
    ctx.setLineDash([7, 5]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#b8f69c";
    ctx.stroke();
    ctx.restore();

    if (!this.isDrawingNew && pxPts.length >= 3) {
      for (let i = 0; i < pxPts.length; i++) {
        const p1 = pxPts[i];
        const p2 = pxPts[(i + 1) % pxPts.length];
        const mx = (p1[0] + p2[0]) * 0.5;
        const my = (p1[1] + p2[1]) * 0.5;
        ctx.beginPath();
        ctx.arc(mx, my, this.hoveredMidpoint === i ? 7 : 5, 0, Math.PI * 2);
        ctx.fillStyle = "#c7b5ff";
        ctx.fill();
      }
    }

    for (let i = 0; i < pxPts.length; i++) {
      const [vx, vy] = pxPts[i];
      ctx.beginPath();
      ctx.arc(vx, vy, this.hoveredVertex === i ? 9 : 7, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#08090e";
      ctx.stroke();
    }
  }
}

window.ROIPolygonEditor = ROIPolygonEditor;
