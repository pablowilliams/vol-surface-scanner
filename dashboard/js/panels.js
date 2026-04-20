// Renders the new panels: term structure, vega heatmap, Heston, backtest,
// regime badge. All canvases include a data-table fallback for screen reader
// users; we update those tables here so non visual users can also read the
// underlying numbers.

import { announcePolite } from "./live-region.js";

function clear(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
}

function setupCanvas(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = canvas.clientWidth || canvas.width;
  const h = canvas.clientHeight || canvas.height;
  if (canvas.width !== Math.round(w * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

function drawLineSeries(canvas, xs, ys, opts = {}) {
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  clear(ctx, w, h);
  if (!xs || xs.length === 0 || !ys || ys.length === 0) return;
  const pad = 30;
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const xspan = Math.max(1e-9, xmax - xmin);
  const yspan = Math.max(1e-9, ymax - ymin);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  // gridlines
  ctx.strokeStyle = "rgba(148, 163, 184, 0.10)";
  for (let g = 1; g < 4; g++) {
    const y = pad + ((h - 2 * pad) * g) / 4;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }
  ctx.strokeStyle = opts.colour || "#38BDF8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const x = pad + ((xs[i] - xmin) / xspan) * (w - 2 * pad);
    const y = h - pad - ((ys[i] - ymin) / yspan) * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    ctx.fillStyle = opts.colour || "#38BDF8";
    ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
  }
  ctx.stroke();
  if (opts.title) {
    ctx.fillStyle = "rgba(241, 245, 249, 0.78)";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText(opts.title, pad + 6, pad + 14);
  }
  // axis tick labels (min/max)
  ctx.fillStyle = "rgba(148, 163, 184, 0.85)";
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillText(ymax.toFixed(3), 4, pad + 8);
  ctx.fillText(ymin.toFixed(3), 4, h - pad);
  ctx.fillText(xmin.toFixed(2), pad, h - 8);
  ctx.fillText(xmax.toFixed(2), w - pad - 24, h - 8);
}

function drawTwoLine(canvas, xs, ya, yb, labels = ["A", "B"]) {
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  clear(ctx, w, h);
  if (!xs || xs.length === 0) return;
  const allY = ya.concat(yb);
  const pad = 32;
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...allY), ymax = Math.max(...allY);
  const xspan = Math.max(1e-9, xmax - xmin);
  const yspan = Math.max(1e-9, ymax - ymin);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.08)";
  for (let g = 1; g < 4; g++) {
    const y = pad + ((h - 2 * pad) * g) / 4;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }
  const drawSeries = (vals, colour) => {
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const x = pad + ((xs[i] - xmin) / xspan) * (w - 2 * pad);
      const y = h - pad - ((vals[i] - ymin) / yspan) * (h - 2 * pad);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  drawSeries(ya, "#38BDF8");
  drawSeries(yb, "#F59E0B");
  ctx.fillStyle = "rgba(241, 245, 249, 0.85)";
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillRect(w - pad - 110, pad + 6, 8, 8);
  ctx.fillStyle = "#38BDF8";
  ctx.fillRect(w - pad - 110, pad + 6, 8, 8);
  ctx.fillStyle = "rgba(241, 245, 249, 0.85)";
  ctx.fillText(labels[0], w - pad - 96, pad + 14);
  ctx.fillStyle = "#F59E0B";
  ctx.fillRect(w - pad - 50, pad + 6, 8, 8);
  ctx.fillStyle = "rgba(241, 245, 249, 0.85)";
  ctx.fillText(labels[1], w - pad - 36, pad + 14);
}

function drawHeatmap(canvas, xs, ys, matrix) {
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  clear(ctx, w, h);
  if (!matrix || matrix.length === 0) return;
  const flat = matrix.flat();
  const vmin = Math.min(...flat);
  const vmax = Math.max(...flat);
  const span = Math.max(1e-9, vmax - vmin);
  const pad = 36;
  const cellW = (w - 2 * pad) / xs.length;
  const cellH = (h - 2 * pad) / ys.length;
  for (let i = 0; i < ys.length; i++) {
    for (let j = 0; j < xs.length; j++) {
      const v = (matrix[i][j] - vmin) / span;
      const r = Math.round(20 + 200 * v);
      const g = Math.round(20 + 100 * v);
      const b = Math.round(150 - 100 * v);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b + 60})`;
      ctx.fillRect(pad + j * cellW, h - pad - (i + 1) * cellH, cellW + 1, cellH + 1);
    }
  }
  ctx.strokeStyle = "rgba(148, 163, 184, 0.32)";
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  ctx.fillStyle = "rgba(241, 245, 249, 0.78)";
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillText(`min ${vmin.toFixed(3)}`, pad, h - 6);
  ctx.fillText(`max ${vmax.toFixed(3)}`, w - pad - 70, h - 6);
  ctx.fillText("strike ->", pad, pad - 6);
  ctx.save();
  ctx.translate(8, h - pad);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("tenor ->", 0, 10);
  ctx.restore();
}

function drawBars(canvas, labels, values, opts = {}) {
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  clear(ctx, w, h);
  if (values.length === 0) return;
  const pad = 32;
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const span = Math.max(1e-9, max - min);
  const barW = (w - 2 * pad) / values.length * 0.7;
  const stepW = (w - 2 * pad) / values.length;
  const zeroY = h - pad - ((0 - min) / span) * (h - 2 * pad);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  ctx.beginPath();
  ctx.moveTo(pad, zeroY);
  ctx.lineTo(w - pad, zeroY);
  ctx.stroke();
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const x = pad + i * stepW + (stepW - barW) / 2;
    const top = h - pad - ((Math.max(v, 0) - min) / span) * (h - 2 * pad);
    const bot = h - pad - ((Math.min(v, 0) - min) / span) * (h - 2 * pad);
    ctx.fillStyle = v >= 0 ? (opts.colour || "#38BDF8") : "#F87171";
    ctx.fillRect(x, top, barW, bot - top);
    ctx.fillStyle = "rgba(241, 245, 249, 0.85)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillText(String(labels[i]), x, h - 6);
  }
}

function fillTable(tableEl, header, rows) {
  if (!tableEl) return;
  const head = tableEl.querySelector("thead tr");
  const body = tableEl.querySelector("tbody");
  if (head && header) {
    head.innerHTML = header.map((h) => `<th scope="col">${h}</th>`).join("");
  }
  if (body) {
    body.innerHTML = rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("");
  }
}

function setRegimeBadge(regime) {
  const el = document.getElementById("regime-badge");
  if (!el) return;
  const labels = { calm: "Calm", stressed: "Stressed", crash: "Crash", unknown: "Unknown" };
  el.className = "regime-badge regime-" + regime;
  el.textContent = labels[regime] || regime;
}

async function fetchJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

export async function renderPanels() {
  let ts, vega, heston, backtest, regime;
  try {
    [ts, vega, heston, backtest, regime] = await Promise.all([
      fetchJson("data/term_structure.json"),
      fetchJson("data/greeks.json"),
      fetchJson("data/heston.json"),
      fetchJson("data/backtest.json"),
      fetchJson("data/regime.json"),
    ]);
  } catch (err) {
    console.error("panel data load failed", err);
    return;
  }

  // --- Term structure ----------------------------------------------------
  if (ts) {
    drawLineSeries(document.getElementById("ts-atm-canvas"), ts.tenors, ts.atm_vol, { colour: "#38BDF8", title: "ATM vol" });
    drawLineSeries(document.getElementById("ts-skew-canvas"), ts.tenors, ts.atm_skew, { colour: "#F59E0B", title: "ATM skew" });
    drawLineSeries(document.getElementById("ts-kurt-canvas"), ts.tenors, ts.atm_kurtosis, { colour: "#34D399", title: "ATM kurtosis" });
    fillTable(
      document.getElementById("ts-atm-table"),
      ["Tenor", "ATM vol"],
      ts.tenors.map((t, i) => [t.toFixed(3), ts.atm_vol[i].toFixed(4)]),
    );
    fillTable(
      document.getElementById("ts-skew-table"),
      ["Tenor", "ATM skew"],
      ts.tenors.map((t, i) => [t.toFixed(3), ts.atm_skew[i].toFixed(4)]),
    );
    fillTable(
      document.getElementById("ts-kurt-table"),
      ["Tenor", "ATM kurtosis"],
      ts.tenors.map((t, i) => [t.toFixed(3), ts.atm_kurtosis[i].toFixed(4)]),
    );
  }

  // --- Vega heatmap ------------------------------------------------------
  if (vega) {
    drawHeatmap(document.getElementById("vega-canvas"), vega.strikes, vega.tenors, vega.vega);
    const thead = document.getElementById("vega-thead");
    const tbody = document.getElementById("vega-tbody");
    if (thead && tbody) {
      thead.innerHTML = '<th scope="col">Tenor</th>' + vega.strikes.map((s) => `<th scope="col">K=${s.toFixed(2)}</th>`).join("");
      tbody.innerHTML = vega.tenors
        .map((t, i) => `<tr><th scope="row">${t.toFixed(3)}</th>` + vega.vega[i].map((v) => `<td>${v.toFixed(3)}</td>`).join("") + "</tr>")
        .join("");
    }
  }

  // --- Heston ------------------------------------------------------------
  if (heston) {
    const dl = document.getElementById("heston-params");
    if (dl) {
      dl.innerHTML = [
        ["kappa", heston.kappa],
        ["theta", heston.theta],
        ["v0", heston.v0],
        ["rho", heston.rho],
        ["xi", heston.xi],
        ["RMSE", heston.rmse],
      ]
        .map(([k, v]) => `<div class="kv-pair"><dt>${k}</dt><dd>${Number(v).toFixed(5)}</dd></div>`)
        .join("");
    }
    const m = await fetchJson("data/meta.json").catch(() => null);
    const sviPer = m?.metrics?.svi_per_tenor_rmse || [];
    const hestonPer = heston.per_tenor_rmse || [];
    if (sviPer.length && hestonPer.length) {
      const xs = Array.from({ length: sviPer.length }, (_, i) => i + 1);
      drawTwoLine(document.getElementById("heston-rmse-canvas"), xs, sviPer, hestonPer, ["SVI", "Heston"]);
      fillTable(
        document.getElementById("heston-rmse-table"),
        ["Tenor index", "SVI", "Heston"],
        xs.map((x, i) => [x, sviPer[i].toFixed(5), hestonPer[i].toFixed(5)]),
      );
    }
  }

  // --- Backtest ----------------------------------------------------------
  if (backtest) {
    const horizons = Object.keys(backtest.lead_lag).map(Number).sort((a, b) => a - b);
    const vals = horizons.map((h) => Number(backtest.lead_lag[String(h)]));
    drawBars(document.getElementById("backtest-canvas"), horizons.map(String), vals);
    const summary = document.getElementById("backtest-summary");
    if (summary) {
      summary.innerHTML = `<strong>Hit rate</strong> ${(backtest.hit_rate * 100).toFixed(1)}% over ${backtest.n_days} days. Lead lag at h=1 is ${vals[0]?.toFixed(3) ?? "n/a"}.`;
    }
    fillTable(
      document.getElementById("backtest-table"),
      ["Horizon", "Correlation"],
      horizons.map((h, i) => [String(h), vals[i].toFixed(3)]),
    );
  }

  // --- Regime badge ------------------------------------------------------
  if (regime) {
    setRegimeBadge(regime.inferred || "unknown");
    announcePolite(`Inferred regime ${regime.inferred}.`);
  }
}

renderPanels();
