/**
 * PasarVision AI — Lightweight Zero-Lag Chart.js Visuals (`charts.js`)
 *
 * Styled directly after the Ethereal Bento reference:
 * 1. `Class split` Doughnut Ring (`#chart-live-classes`) with 74% cutout + 2-column legend
 * 2. `Class flow` Rounded Lavender Vertical Bars (`#chart-ba-comparison`)
 * 3. `Activity` Smooth Lavender/Mint Spline (`#chart-frame-timeline`)
 *
 * Zero animation overhead during live video updates (`animation: false`).
 */

const ETHEREAL_RING_COLORS = [
  "#c7b5ff", // Soft Lavender
  "#88f4d8", // Pastel Mint
  "#fcd34d", // Warm Gold
  "#ff8580", // Soft Coral
  "#60a5fa", // Sky Blue
  "#ccf779", // Lime
];

class PasarChartsManager {
  constructor() {
    this.donutChart = null;
    this.beforeAfterChart = null;
    this.timelineChart = null;

    if (typeof Chart !== "undefined") {
      Chart.defaults.color = "#666e85";
      Chart.defaults.font.family = "'Plus Jakarta Sans', 'Inter', sans-serif";
      Chart.defaults.borderColor = "rgba(255, 255, 255, 0.04)";
    }
  }

  initAll() {
    if (typeof Chart === "undefined") return;
    this._initDonutChart();
    this._initBeforeAfterChart();
    this._initTimelineChart();
  }

  _initDonutChart() {
    const el = document.getElementById("chart-live-classes");
    if (!el) return;
    this.donutChart = new Chart(el, {
      type: "doughnut",
      data: {
        labels: ["Idle"],
        datasets: [
          {
            data: [1],
            backgroundColor: ["#222636"],
            borderWidth: 0,
            hoverOffset: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "74%",
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#161926",
            titleColor: "#f5f7fb",
            bodyColor: "#b8f69c",
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1,
          },
        },
      },
    });
  }

  _initBeforeAfterChart() {
    const el = document.getElementById("chart-ba-comparison");
    if (!el) return;
    this.beforeAfterChart = new Chart(el, {
      type: "bar",
      data: {
        labels: [],
        datasets: [
          {
            label: "Before",
            data: [],
            backgroundColor: "#2b2740",
            borderRadius: 12,
            borderSkipped: false,
            maxBarThickness: 34,
          },
          {
            label: "After",
            data: [],
            backgroundColor: "#c7b5ff",
            borderRadius: 12,
            borderSkipped: false,
            maxBarThickness: 34,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        plugins: {
          legend: {
            position: "top",
            align: "end",
            labels: { boxWidth: 10, usePointStyle: true, pointStyle: "circle", color: "#9ca3b8" },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: "#9ca3b8", font: { size: 11, weight: "600" } },
          },
          y: {
            beginAtZero: true,
            ticks: { precision: 0, font: { size: 11 } },
            grid: { color: "rgba(255,255,255,0.04)" },
          },
        },
      },
    });
  }

  _initTimelineChart() {
    const el = document.getElementById("chart-frame-timeline");
    if (!el) return;
    this.timelineChart = new Chart(el, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Detected",
            data: [],
            borderColor: "#c7b5ff",
            backgroundColor: "rgba(199, 181, 255, 0.12)",
            fill: true,
            tension: 0.35,
            pointRadius: 2,
          },
          {
            label: "In Zone",
            data: [],
            borderColor: "#88f4d8",
            fill: false,
            tension: 0.35,
            pointRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            position: "top",
            align: "end",
            labels: { boxWidth: 10, usePointStyle: true, pointStyle: "circle", color: "#9ca3b8" },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { maxTicksLimit: 8, font: { size: 10 } },
          },
          y: {
            beginAtZero: true,
            ticks: { precision: 0, font: { size: 10 } },
            grid: { color: "rgba(255,255,255,0.04)" },
          },
        },
      },
    });
  }

  updateLiveDistribution(countsByClass = {}, classColors = {}) {
    if (!this.donutChart) return;
    const entries = Object.entries(countsByClass)
      .filter(([, cnt]) => cnt > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);

    const total = entries.reduce((acc, [, v]) => acc + v, 0);
    const centerTotalEl = document.getElementById("donut-total-num");
    if (centerTotalEl) centerTotalEl.textContent = String(total);

    const legendEl = document.getElementById("split-legend-grid");

    if (entries.length === 0) {
      this.donutChart.data.labels = ["Empty"];
      this.donutChart.data.datasets[0].data = [1];
      this.donutChart.data.datasets[0].backgroundColor = ["#222636"];
      this.donutChart.update("none");
      if (legendEl) {
        legendEl.innerHTML = `<div class="legend-item"><span class="legend-name">No objects</span><span class="legend-val num">0%</span></div>`;
      }
      return;
    }

    const labels = entries.map(([k]) => k);
    const values = entries.map(([, v]) => v);
    const colors = labels.map(
      (k, idx) => classColors[k] || ETHEREAL_RING_COLORS[idx % ETHEREAL_RING_COLORS.length]
    );

    this.donutChart.data.labels = labels;
    this.donutChart.data.datasets[0].data = values;
    this.donutChart.data.datasets[0].backgroundColor = colors;
    this.donutChart.update("none");

    if (legendEl) {
      legendEl.innerHTML = entries
        .map(([name, cnt], idx) => {
          const pct = total > 0 ? Math.round((cnt / total) * 100) : 0;
          const col = colors[idx];
          return `
            <div class="legend-item">
              <span class="legend-name">${name}</span>
              <span class="legend-val num">
                <i class="legend-bar" style="background:${col}"></i>
                ${pct}% <small style="color:#666e85;font-weight:500">(${cnt})</small>
              </span>
            </div>
          `;
        })
        .join("");
    }
  }

  updateBeforeAfter(perClassTable = []) {
    if (!this.beforeAfterChart) return;
    const rows = perClassTable
      .filter((r) => (Number(r.before_count || 0) + Number(r.after_count || 0)) > 0)
      .slice(0, 10);
    this.beforeAfterChart.data.labels = rows.map((r) => r.class_name);
    this.beforeAfterChart.data.datasets[0].data = rows.map((r) => r.before_count);
    this.beforeAfterChart.data.datasets[1].data = rows.map((r) => r.after_count);
    this.beforeAfterChart.update("none");
  }

  updateTimeline(frameLog = []) {
    if (!this.timelineChart) return;
    const sliced = frameLog.slice(-60);
    this.timelineChart.data.labels = sliced.map((f) => f.frame_or_file);
    this.timelineChart.data.datasets[0].data = sliced.map((f) => f.after_detections);
    this.timelineChart.data.datasets[1].data = sliced.map((f) => f.inside_roi_count);
    this.timelineChart.update("none");
  }
}

window.PasarChartsManager = PasarChartsManager;
