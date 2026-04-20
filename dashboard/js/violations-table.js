// Accessible sortable violations table.

import { announcePolite } from "./live-region.js";

let data = [];
let filtered = [];
let sortKey = "severity";
let sortDir = "descending";

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

function severityChip(level) {
  const labels = { high: "High", medium: "Medium", low: "Low" };
  const icons = {
    high: "\u26A0\uFE0F",
    medium: "\u25CE",
    low: "\u2022",
  };
  return `<span class="severity-chip severity-${level}"><span aria-hidden="true">${icons[level] || ""}</span> ${labels[level] || level}</span>`;
}

function render() {
  const tbody = document.getElementById("violations-tbody");
  const empty = document.getElementById("violations-empty");
  if (!tbody) return;
  tbody.innerHTML = "";
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
  filtered = selected === "all" ? [...data] : data.filter((v) => v.type === selected);
  filtered.sort((a, b) => compare(a, b, sortKey, sortDir));
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
