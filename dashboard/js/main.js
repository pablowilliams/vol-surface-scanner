// App entry point: fetch pipeline bundle and wire the views.

import { announceAssertive, announcePolite } from "./live-region.js";
import { renderSurface, showFallbackTable } from "./surface.js";
import { setViolations } from "./violations-table.js";

function renderMetrics(bundle) {
  const grid = document.getElementById("metric-grid");
  if (!grid) return;
  const m = bundle.metrics;
  const improve = m.combined_improvement;
  const card = (label, value, sub, deltaClass) => `
    <li class="metric-card">
      <span class="metric-label">${label}</span>
      <span class="metric-value">${value}</span>
      <span class="metric-delta ${deltaClass}">${sub}</span>
    </li>
  `;
  const entries = [
    card("SVI RMSE", m.svi_rmse.toFixed(5), `MAE ${m.svi_mae.toFixed(5)}`, "delta-flat"),
    card("Combined RMSE", m.combined_rmse.toFixed(5), (improve >= 0 ? "down " : "up ") + Math.abs(improve).toExponential(2), improve >= 0 ? "delta-up" : "delta-down"),
    card("Residual RMSE", m.residual_rmse.toFixed(5), "held out split", "delta-flat"),
    card("Violations", String(m.total_violations), `High ${m.violation_severity.high}, Med ${m.violation_severity.medium}, Low ${m.violation_severity.low}`, "delta-flat"),
    card("SVI R squared", m.svi_r2.toFixed(4), "on the noisy surface", "delta-flat"),
    card("Options priced", String(bundle.meta.n_options), `seed ${bundle.meta.seed}`, "delta-flat"),
  ];
  grid.innerHTML = entries.join("");
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
  const pad = 20;
  ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, width - 2 * pad, height - 2 * pad);
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
  ctx.fillStyle = "rgba(241, 245, 249, 0.85)";
  ctx.font = "12px 'JetBrains Mono', monospace";
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
