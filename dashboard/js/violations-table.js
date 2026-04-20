// Accessible sortable violations table with type filter, severity toggles,
// and a result-count indicator.

import { announcePolite } from "./live-region.js";

let data = [];
let filtered = [];
let sortKey = "severity";
let sortDir = "descending";
let severityFilters = new Set(); // empty means all severities

const severityOrder = { low: 0, medium: 1, high: 2 };

function compare(a, b, key, dir) {
  let av = a[key];
  let bv = b[key];
  if (key === "severity") {
    av = severityOrder[av] ?? -1;
    bv = severityOrder[bv] ?? -1;
  }
  if (typeof av === "string") {
    const cmp = av.localeCompare(bv);
    return dir === "ascending" ? cmp : -cmp;
  }
  if (av === bv) return 0;
  return dir === "ascending" ? (av < bv ? -1 : 1) : (av < bv ? 1 : -1);
}

function severityIcon(level) {
  if (level === "high") {
    return `<svg viewBox="0 0 10 10" focusable="false" aria-hidden="true"><path d="M5 0.5 L9.5 8.5 L0.5 8.5 Z" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linejoin="round"/><path d="M5 4 L5 6" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/><circle cx="5" cy="7.2" r="0.5" fill="currentColor"/></svg>`;
  }
  if (level === "medium") {
    return `<svg viewBox="0 0 10 10" focusable="false" aria-hidden="true"><circle cx="5" cy="5" r="3.6" fill="none" stroke="currentColor" stroke-width="1.1"/><circle cx="5" cy="5" r="1.4" fill="currentColor"/></svg>`;
  }
  return `<svg viewBox="0 0 10 10" focusable="false" aria-hidden="true"><circle cx="5" cy="5" r="2" fill="currentColor"/></svg>`;
}

function severityChip(level) {
  const labels = { high: "High", medium: "Medium", low: "Low" };
  return `<span class="severity-chip severity-${level}">${severityIcon(level)}<span>${labels[level] || level}</span></span>`;
}

function render() {
  const tbody = document.getElementById("violations-tbody");
  const empty = document.getElementById("violations-empty");
  const count = document.getElementById("violation-result-count");
  if (!tbody) return;
  tbody.innerHTML = "";
  if (count) {
    count.innerHTML = `<strong>${filtered.length}</strong> ${filtered.length === 1 ? "row" : "rows"}`;
  }
  if (filtered.length === 0) {
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;
  const frag = document.createDocumentFragment();
  for (const v of filtered) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${v.type}</td>
      <td>${severityChip(v.severity)}</td>
      <td>${Number(v.strike).toFixed(2)}</td>
      <td>${Number(v.tenor).toFixed(3)}</td>
      <td>${Number(v.magnitude).toExponential(2)}</td>
      <td>${v.description}</td>
    `;
    frag.appendChild(tr);
  }
  tbody.appendChild(frag);
}

function applyFilterSort() {
  const f = document.getElementById("violation-filter");
  const selected = f ? f.value : "all";
  let out = selected === "all" ? [...data] : data.filter((v) => v.type === selected);
  if (severityFilters.size > 0) {
    out = out.filter((v) => severityFilters.has(v.severity));
  }
  out.sort((a, b) => compare(a, b, sortKey, sortDir));
  filtered = out;
  render();
}

function wireHeaders() {
  const headers = document.querySelectorAll("#violations-table th[aria-sort]");
  headers.forEach((th) => {
    const btn = th.querySelector("button");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const key = btn.dataset.sortKey;
      if (sortKey === key) {
        sortDir = sortDir === "ascending" ? "descending" : "ascending";
      } else {
        sortKey = key;
        sortDir = "ascending";
      }
      headers.forEach((h) => h.setAttribute("aria-sort", "none"));
      th.setAttribute("aria-sort", sortDir);
      applyFilterSort();
      announcePolite(`Sorted by ${key}, ${sortDir}. ${filtered.length} rows.`);
    });
  });
}

function wireFilter() {
  const f = document.getElementById("violation-filter");
  if (!f) return;
  f.addEventListener("change", () => {
    applyFilterSort();
    announcePolite(`Filter set to ${f.value}. ${filtered.length} rows.`);
  });
}

function wireSeverityToggles() {
  const buttons = document.querySelectorAll("[data-severity-filter]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const level = btn.dataset.severityFilter;
      const pressed = btn.getAttribute("aria-pressed") === "true";
      const next = !pressed;
      btn.setAttribute("aria-pressed", String(next));
      if (next) severityFilters.add(level);
      else severityFilters.delete(level);
      applyFilterSort();
      const active = Array.from(severityFilters);
      const label = active.length === 0 ? "all severities" : active.join(", ");
      announcePolite(`Severity filter ${label}. ${filtered.length} rows.`);
    });
  });
}

export function setViolations(rows) {
  data = rows.slice();
  // Default sort: severity descending.
  const highHeader = document.querySelector('#violations-table th[aria-sort] button[data-sort-key="severity"]');
  if (highHeader && highHeader.closest("th")) {
    highHeader.closest("th").setAttribute("aria-sort", "descending");
  }
  applyFilterSort();
}

wireHeaders();
wireFilter();
wireSeverityToggles();
