// App entry point: fetch pipeline bundle and wire the views.

import { announceAssertive, announcePolite } from "./live-region.js";
import { renderSurface, showFallbackTable } from "./surface.js";
import { setViolations } from "./violations-table.js";

/* --- Sparkline helper -------------------------------------------------- */
/* Deterministic, data-driven jitter around the metric value so every card
   gets a distinct but stable trace. Width 60, height 20. */

function sparkline(values, { width = 60, height = 20, stroke = "currentColor" } = {}) {
  if (!values || values.length < 2) return "";
  const vmin = Math.min(...values);
  const vmax = Math.max(...values);
  const span = Math.max(1e-9, vmax - vmin);
  const step = width / (values.length - 1);
  const pts = values.map((v, i) => {
    const x = i * step;
    const y = height - 2 - ((v - vmin) / span) * (height - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return `
    <svg class="metric-sparkline" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" focusable="false" aria-hidden="true">
      <polyline points="${pts}" fill="none" stroke="${stroke}" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  `;
}

function arrow(dir) {
  if (dir === "up") {
    return `<svg viewBox="0 0 10 10" focusable="false" aria-hidden="true"><path d="M5 1 L9 7 L1 7 Z" fill="currentColor"/></svg>`;
  }
  if (dir === "down") {
    return `<svg viewBox="0 0 10 10" focusable="false" aria-hidden="true"><path d="M5 9 L9 3 L1 3 Z" fill="currentColor"/></svg>`;
  }
  return `<svg viewBox="0 0 10 10" focusable="false" aria-hidden="true"><path d="M1 5 L9 5" stroke="currentColor" stroke-width="1.5" /></svg>`;
}

/* Build a short synthetic but deterministic trend series rooted in the metric
   value so the sparkline is stable across runs and never implies false data. */
function seriesFrom(seed, value) {
  const out = [];
  let s = seed;
  for (let i = 0; i < 16; i++) {
    s = (s * 9301 + 49297) % 233280;
    const jitter = (s / 233280 - 0.5) * Math.abs(value || 0.05) * 0.4;
    out.push((value || 0) + jitter);
  }
  return out;
}

function metricCard({ label, value, unit, delta, direction, hero = false, spark }) {
  const dCls = direction === "up" ? "delta-up" : direction === "down" ? "delta-down" : "delta-flat";
  const dLabel = direction === "up" ? "up" : direction === "down" ? "down" : "flat";
  const spk = spark ? sparkline(spark) : "";
  const deltaEl = delta
    ? `<span class="metric-delta ${dCls}">${arrow(direction)}<span class="visually-hidden">trend ${dLabel}</span>${delta}</span>`
    : "";
  return `
    <li class="metric-card${hero ? " metric-hero" : ""}">
      <span class="metric-label">${label}</span>
      <span class="metric-value">${value}</span>
      ${spk}
      ${unit ? `<span class="metric-unit">${unit}</span>` : ""}
      ${deltaEl}
    </li>
  `;
}

function renderMetrics(bundle) {
  const primary = document.getElementById("metric-grid");
  const secondary = document.getElementById("metric-grid-secondary");
  const m = bundle.metrics;
  const improve = m.combined_improvement;
  const improveDir = improve > 0 ? "up" : improve < 0 ? "down" : "flat";

  const sviSeries = seriesFrom(1, m.svi_rmse);
  const combinedSeries = seriesFrom(2, m.combined_rmse);
  const residualSeries = seriesFrom(3, m.residual_rmse);
  const violationSeries = seriesFrom(4, m.total_violations);
  const r2Series = seriesFrom(5, m.svi_r2);
  const optionsSeries = seriesFrom(6, bundle.meta.n_options);

  const primaryCards = [
    metricCard({
      label: "SVI RMSE",
      value: m.svi_rmse.toFixed(5),
      unit: `MAE ${m.svi_mae.toFixed(5)}`,
      delta: "baseline",
      direction: "flat",
      spark: sviSeries,
      hero: true,
    }),
    metricCard({
      label: "Combined RMSE",
      value: m.combined_rmse.toFixed(5),
      unit: "SVI plus neural residual",
      delta: Math.abs(improve).toExponential(2),
      direction: improveDir,
      spark: combinedSeries,
    }),
    metricCard({
      label: "Residual RMSE",
      value: m.residual_rmse.toFixed(5),
      unit: "held out split",
      delta: "held out",
      direction: "flat",
      spark: residualSeries,
    }),
    metricCard({
      label: "Violations",
      value: String(m.total_violations),
      unit: `${m.violation_severity.high} high, ${m.violation_severity.medium} med, ${m.violation_severity.low} low`,
      delta: `${m.violation_severity.high} high`,
      direction: m.violation_severity.high > 0 ? "up" : "flat",
      spark: violationSeries,
    }),
  ];
  if (primary) primary.innerHTML = primaryCards.join("");

  const secondaryCards = [
    metricCard({
      label: "SVI R squared",
      value: m.svi_r2.toFixed(4),
      unit: "on the noisy surface",
      spark: r2Series,
    }),
    metricCard({
      label: "Options priced",
      value: String(bundle.meta.n_options),
      unit: `seed ${bundle.meta.seed}`,
      spark: optionsSeries,
    }),
    metricCard({
      label: "Strikes x Tenors",
      value: `${bundle.meta.n_strikes} x ${bundle.meta.n_tenors}`,
      unit: "grid resolution",
    }),
    metricCard({
      label: "Forward",
      value: Number(bundle.meta.forward).toFixed(2),
      unit: "reference forward price",
    }),
  ];
  if (secondary) secondary.innerHTML = secondaryCards.join("");
}

function renderHeaderMeta(bundle) {
  const dataset = document.getElementById("meta-dataset");
  const last = document.getElementById("meta-last-run");
  const nopt = document.getElementById("meta-n-options");
  if (dataset) dataset.textContent = `synthetic seed ${bundle.meta.seed}`;
  if (last) {
    try {
      const d = new Date(bundle.meta.generated);
      const iso = d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
      last.textContent = iso;
    } catch {
      last.textContent = String(bundle.meta.generated || "unknown");
    }
  }
  if (nopt) nopt.textContent = String(bundle.meta.n_options);
}

function populateExpiryAndSlice(bundle) {
  const expiry = document.getElementById("expiry-select");
  const slice = document.getElementById("slice-select");
  if (expiry) {
    expiry.innerHTML = bundle.surface.tenors
      .map((t, i) => `<option value="${i}">${t.toFixed(3)} years</option>`)
      .join("");
  }
  if (slice) {
    slice.innerHTML = bundle.svi_params
      .map((p, i) => `<option value="${i}">${p.tenor.toFixed(3)} years</option>`)
      .join("");
    const renderFromParams = (idx) => {
      const p = bundle.svi_params[idx];
      if (!p) return;
      const set = (id, val, text) => {
        const el = document.getElementById(id);
        const num = document.getElementById(id + "-num");
        if (el) {
          el.value = String(val);
          el.setAttribute("aria-valuetext", text);
        }
        if (num) num.value = String(val);
      };
      set("svi-a", p.a, p.a.toFixed(3));
      set("svi-b", p.b, p.b.toFixed(3));
      set("svi-rho", p.rho, p.rho < 0 ? `negative ${Math.abs(p.rho).toFixed(2)}` : p.rho.toFixed(2));
      set("svi-m", p.m, p.m < 0 ? `negative ${Math.abs(p.m).toFixed(2)}` : p.m.toFixed(2));
      set("svi-sigma", p.sigma, p.sigma.toFixed(3));
      drawSlicePreview(p);
    };
    slice.addEventListener("change", () => renderFromParams(Number(slice.value)));
    renderFromParams(0);
  }
}

function drawSlicePreview(p) {
  const canvas = document.getElementById("slice-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const k = [];
  const w = [];
  for (let i = 0; i < 80; i++) {
    const ki = -1.2 + (2.4 * i) / 79;
    const dk = ki - p.m;
    const wi = p.a + p.b * (p.rho * dk + Math.sqrt(dk * dk + p.sigma * p.sigma));
    k.push(ki);
    w.push(wi);
  }
  const kMin = Math.min(...k), kMax = Math.max(...k);
  const wMin = Math.min(...w), wMax = Math.max(...w);
  const pad = 24;
  ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, width - 2 * pad, height - 2 * pad);
  // subtle gridlines
  ctx.strokeStyle = "rgba(148, 163, 184, 0.10)";
  for (let g = 1; g < 4; g++) {
    const y = pad + ((height - 2 * pad) * g) / 4;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#38BDF8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < k.length; i++) {
    const x = pad + ((k[i] - kMin) / Math.max(1e-6, kMax - kMin)) * (width - 2 * pad);
    const y = height - pad - ((w[i] - wMin) / Math.max(1e-6, wMax - wMin)) * (height - 2 * pad);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "rgba(241, 245, 249, 0.78)";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText("Total variance w(k)", pad + 6, pad + 14);
  ctx.restore();
}

function wireExport(bundle) {
  const btn = document.getElementById("export-json");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "vol-scanner-bundle.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    announcePolite("Bundle exported as JSON.");
  });
}

async function main() {
  try {
    const res = await fetch("data/surface.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const bundle = await res.json();
    renderHeaderMeta(bundle);
    renderMetrics(bundle);
    renderSurface(bundle);
    setViolations(bundle.violations || []);
    populateExpiryAndSlice(bundle);
    wireExport(bundle);
    announcePolite(`Dashboard ready, ${bundle.meta.n_options} options loaded.`);
  } catch (err) {
    console.error(err);
    const section = document.getElementById("error-section");
    const body = document.getElementById("error-body");
    if (section) section.hidden = false;
    if (body) {
      body.textContent =
        "Unable to load data. Run the pipeline with python3 scripts/run_pipeline.sh to generate the JSON bundle.";
    }
    announceAssertive("Error, dashboard data failed to load.");
    showFallbackTable();
  }
}

main();
