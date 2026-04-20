// Slider and select controls, theme toggle, keyboard interactions.

import { announcePolite } from "./live-region.js";

const themeKey = "vol-scanner-theme";

function initTheme() {
  const stored = localStorage.getItem(themeKey);
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initial = stored || (prefersDark ? "dark" : "dark");
  applyTheme(initial);
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", (event) => {
      if (!localStorage.getItem(themeKey)) {
        applyTheme(event.matches ? "dark" : "light");
      }
    });
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  const label = document.getElementById("theme-toggle-label");
  const isDark = theme === "dark";
  btn.setAttribute("aria-pressed", String(isDark));
  if (label) label.textContent = isDark ? "Dark" : "Light";
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  localStorage.setItem(themeKey, next);
  applyTheme(next);
  announcePolite(next === "dark" ? "Dark theme active" : "Light theme active");
}

function wireThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", toggleTheme);
}

function bindSliderNumberPair(rangeId, numberId, format) {
  const range = document.getElementById(rangeId);
  const number = document.getElementById(numberId);
  if (!range || !number) return;
  const update = (source) => {
    const val = source.valueAsNumber;
    range.value = String(val);
    number.value = String(val);
    range.setAttribute("aria-valuetext", format(val));
  };
  range.addEventListener("input", () => update(range));
  number.addEventListener("input", () => update(number));
  range.addEventListener("keydown", (event) => {
    const step = parseFloat(range.step) || 1;
    const pageStep = step * 10;
    let handled = true;
    if (event.key === "PageUp") {
      range.value = String(range.valueAsNumber + pageStep);
    } else if (event.key === "PageDown") {
      range.value = String(range.valueAsNumber - pageStep);
    } else if (event.key === "Home") {
      range.value = String(parseFloat(range.min));
    } else if (event.key === "End") {
      range.value = String(parseFloat(range.max));
    } else {
      handled = false;
    }
    if (handled) {
      event.preventDefault();
      update(range);
    }
  });
}

function formatSigned(n) {
  const v = Number(n);
  if (v < 0) return `negative ${Math.abs(v).toFixed(2)}`;
  return v.toFixed(2);
}
function formatThree(n) { return Number(n).toFixed(3); }

export function initControls() {
  initTheme();
  wireThemeToggle();
  bindSliderNumberPair("svi-a", "svi-a-num", formatThree);
  bindSliderNumberPair("svi-b", "svi-b-num", formatThree);
  bindSliderNumberPair("svi-rho", "svi-rho-num", formatSigned);
  bindSliderNumberPair("svi-m", "svi-m-num", formatSigned);
  bindSliderNumberPair("svi-sigma", "svi-sigma-num", formatThree);
}

initControls();
